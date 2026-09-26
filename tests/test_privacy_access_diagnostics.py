"""Tests for the read-only per-record privacy access chain diagnostics.

Covers `GET /machines/{machine_id}/privacy-accesses/diagnostics`: GET-only
405, ``invalid_query`` 422 for any query parameter or carried body before the
machine lookup (and without reading access records), ``not_found`` 404 with
no diagnostic data, ``internal_error`` 500 on a machine/record read fault
with no partial result, empty-machine ``true/0/[]``, chain ordering by the
actual UTC instant of ``accessed_at`` then id, 1-based gap-free positions,
the empty-string predecessor on the first record, all seven anomaly codes,
multiple anomalies retained on one record, later records still listed after
the first anomaly, per-machine isolation, tolerant handling of damaged
values, read-only byte-identical stability, and persistence across restarts.
"""
import hashlib
import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability import privacy_chain

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "machine_id",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
)

MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def canonical_content_hash(record: dict) -> str:
    document = json.dumps(
        {key: record[key] for key in CONTENT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


def create_machine(client, external_id="machine-1"):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def diagnostics_url(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/diagnostics"


def register(client, machine_id, accessed_at=T0, result="success", matches_count=1,
             window_start=T1, window_end=T2):
    response = client.post(
        f"/machines/{machine_id}/privacy-accesses",
        json={
            "accessed_at": accessed_at,
            "window_start": window_start,
            "window_end": window_end,
            "result": result,
            "matches_count": matches_count,
        },
    )
    assert response.status_code == 201
    return response.json()


def fetch_rows(client, machine_id):
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM privacy_accesses WHERE machine_id = :m"),
            {"m": machine_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def tamper(db_path, statement, parameters=()):
    connection = sqlite3.connect(db_path)
    connection.execute(statement, parameters)
    connection.commit()
    connection.close()


def diagnose(client, machine_id):
    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 200
    return response.json()


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


def test_only_get_is_accepted_on_diagnostics_path(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(diagnostics_url(machine_id))
        assert response.status_code == 405


@pytest.mark.parametrize("query", ["?unexpected=1", "?from_accessed_at=" + T0])
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(diagnostics_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    response = client.get(diagnostics_url(MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        diagnostics_url(machine_id),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_machine_lookup(client):
    response = client.request(
        "GET",
        diagnostics_url(MISSING_ID),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_diagnostic_data(client):
    response = client.get(diagnostics_url(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content
    assert b"valid" not in response.content


def test_access_read_failure_is_500_with_no_partial_result(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content
    assert b"checked_count" not in response.content


def test_machine_read_failure_is_500_without_partial_machine_object(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_validation_and_method_errors_do_not_read_records(client):
    # With the records table dropped, any record read would 500;
    # validation-phase and routing errors must still come back as their codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    assert client.get(diagnostics_url(machine_id) + "?x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)(diagnostics_url(machine_id)).status_code == 405


# --------------------------------------------------------------------------- #
# Sound chains
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid_with_zero_records(client):
    machine_id = create_machine(client)
    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 0,
        "records": [],
    }


def test_sound_chain_reports_positions_predecessors_and_digests(client):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2, result="failed",
                     matches_count=0)

    result = diagnose(client, machine_id)
    assert result["machine_id"] == machine_id
    assert result["valid"] is True
    assert result["checked_count"] == 3

    rows = {row["id"]: row for row in fetch_rows(client, machine_id)}
    expected_ids = [first["id"], second["id"], third["id"]]
    assert [record["id"] for record in result["records"]] == expected_ids

    previous_chain_hash = ""
    for position, (record, record_id) in enumerate(
        zip(result["records"], expected_ids, strict=True), start=1
    ):
        row = rows[record_id]
        assert record["position"] == position
        assert record["previous_access_id"] == (
            None if position == 1 else expected_ids[position - 2]
        )
        assert record["content_hash"] == row["content_hash"]
        assert record["chain_hash"] == row["chain_hash"]
        assert HEX64_RE.match(record["content_hash"])
        assert HEX64_RE.match(record["chain_hash"])
        assert record["content_hash"] == canonical_content_hash(row)
        assert record["chain_hash"] == chain_hash(
            previous_chain_hash, record["content_hash"]
        )
        assert record["errors"] == []
        previous_chain_hash = record["chain_hash"]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = register(client, machine_id, accessed_at="2026-03-01T00:00:00.5Z")
    exact = register(client, machine_id, accessed_at=T0)

    result = diagnose(client, machine_id)
    assert [r["id"] for r in result["records"]] == [exact["id"], fractional["id"]]
    assert result["records"][0]["position"] == 1
    assert result["records"][0]["previous_access_id"] is None
    assert result["records"][1]["position"] == 2
    assert result["records"][1]["previous_access_id"] == exact["id"]
    assert all(record["errors"] == [] for record in result["records"])
    assert result["valid"] is True


def test_out_of_order_registration_is_diagnosed_in_chain_order(client):
    machine_id = create_machine(client)
    later = register(client, machine_id, accessed_at=T2)
    earlier = register(client, machine_id, accessed_at=T0)

    result = diagnose(client, machine_id)
    assert [r["id"] for r in result["records"]] == [earlier["id"], later["id"]]
    assert [r["position"] for r in result["records"]] == [1, 2]
    assert result["records"][1]["previous_access_id"] == earlier["id"]
    assert all(record["errors"] == [] for record in result["records"])


# --------------------------------------------------------------------------- #
# Per-record anomaly codes
# --------------------------------------------------------------------------- #


def test_content_tamper_flags_content_and_breaks_successor_chain(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET matches_count = 42 WHERE id = ?",
        (first["id"],),
    )

    result = diagnose(client, machine_id)
    assert result["valid"] is False
    assert result["checked_count"] == 2
    # Both records are still listed, positions unchanged.
    assert [r["position"] for r in result["records"]] == [1, 2]
    assert result["records"][0]["errors"] == [
        "bad_content_hash",
        "bad_chain_hash",
    ]
    # The successor's stored chain digest was chained through the original
    # first digest; the recomputed continuation differs.
    assert result["records"][1]["errors"] == ["bad_chain_hash"]


def test_missing_predecessor_link(client, tmp_path):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)
    register(client, machine_id, accessed_at=T2)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = NULL WHERE id = ?",
        (second["id"],),
    )

    result = diagnose(client, machine_id)
    assert result["valid"] is False
    by_id = {r["id"]: r for r in result["records"]}
    # Content and both chain digests still verify; only the link is missing.
    assert by_id[second["id"]]["errors"] == ["missing_previous"]
    assert all(
        r["errors"] == [] for r in result["records"] if r["id"] != second["id"]
    )


def test_bad_previous_pointer(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        (first["id"], third["id"]),
    )

    result = diagnose(client, machine_id)
    by_id = {r["id"]: r for r in result["records"]}
    assert by_id[third["id"]]["errors"] == ["bad_previous"]
    assert all(
        r["errors"] == [] for r in result["records"] if r["id"] != third["id"]
    )


def test_first_record_with_a_predecessor_is_bad_previous(client, tmp_path):
    machine_id = create_machine(client)
    only = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        ("11111111-1111-1111-1111-111111111111", only["id"]),
    )

    result = diagnose(client, machine_id)
    assert result["records"][0]["errors"] == ["bad_previous"]


def test_cross_machine_previous_pointer_is_bad_previous(client, tmp_path):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    foreign = register(client, machine_one, accessed_at=T0)
    register(client, machine_two, accessed_at=T0)
    second = register(client, machine_two, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        (foreign["id"], second["id"]),
    )

    result_two = diagnose(client, machine_two)
    by_id = {r["id"]: r for r in result_two["records"]}
    assert by_id[second["id"]]["errors"] == ["bad_previous"]
    # The other machine stays fully sound.
    assert diagnose(client, machine_one)["valid"] is True


def test_bad_chain_hash_does_not_falsely_flag_successors(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"],),
    )

    result = diagnose(client, machine_id)
    # Only the stored chain digest of the first record is wrong; its content
    # is unchanged, so the expected continuation through it still matches the
    # second record's stored digest.
    assert result["records"][0]["errors"] == ["bad_chain_hash"]
    assert result["records"][1]["errors"] == []
    assert result["valid"] is False


def test_unparseable_accessed_at_is_bad_time_and_sorts_last(client, tmp_path):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    damaged = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET accessed_at = 'not-a-time' WHERE id = ?",
        (damaged["id"],),
    )

    result = diagnose(client, machine_id)
    # The damaged stamp sorts after every parseable record.
    assert [r["id"] for r in result["records"]] == [
        result["records"][0]["id"],
        damaged["id"],
    ]
    damaged_record = result["records"][1]
    assert damaged_record["position"] == 2
    assert "bad_time" in damaged_record["errors"]
    assert "bad_content_hash" in damaged_record["errors"]
    assert "bad_chain_hash" in damaged_record["errors"]


def test_non_hex_stored_hashes_do_not_crash(client, tmp_path):
    machine_id = create_machine(client)
    damaged = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET content_hash = 'junk' WHERE id = ?",
        (damaged["id"],),
    )

    result = diagnose(client, machine_id)
    assert result["valid"] is False
    # The business fields are unchanged, so the recomputed chain digest over
    # them still matches the stored chain hash: only the content digest
    # column is wrong.
    assert result["records"][0]["errors"] == ["bad_content_hash"]


def test_malformed_row_id_is_bad_id(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)

    # A directly-inserted legacy/damaged row with a non-UUID identifier.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO privacy_accesses (id, machine_id, accessed_at, "
                "window_start, window_end, result, matches_count, "
                "previous_access_id, content_hash, chain_hash) VALUES "
                "(:id, :m, :a, :ws, :we, :r, :c, NULL, NULL, NULL)"
            ),
            {
                "id": "not-a-uuid",
                "m": machine_id,
                "a": T3,
                "ws": T1,
                "we": T2,
                "r": "success",
                "c": 1,
            },
        )

    result = diagnose(client, machine_id)
    assert result["valid"] is False
    damaged_record = result["records"][-1]
    assert damaged_record["id"] == "not-a-uuid"
    assert "bad_id" in damaged_record["errors"]
    assert "missing_previous" in damaged_record["errors"]
    assert "bad_content_hash" in damaged_record["errors"]
    assert "bad_chain_hash" in damaged_record["errors"]
    # The error codes appear in the documented fixed order.
    assert damaged_record["errors"] == sorted(
        damaged_record["errors"],
        key=[
            "missing_previous",
            "bad_previous",
            "bad_content_hash",
            "bad_chain_hash",
            "bad_time",
            "bad_id",
            "bad_ownership",
        ].index,
    )


def test_first_anomaly_sets_invalid_but_later_records_still_listed(client, tmp_path):
    machine_id = create_machine(client)
    records = [
        register(client, machine_id, accessed_at=T0),
        register(client, machine_id, accessed_at=T1),
        register(client, machine_id, accessed_at=T2),
    ]

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET result = 'failed' WHERE id = ?",
        (records[1]["id"],),
    )

    result = diagnose(client, machine_id)
    assert result["valid"] is False
    assert result["checked_count"] == 3
    assert [r["id"] for r in result["records"]] == [r["id"] for r in records]
    assert result["records"][0]["errors"] == []
    assert "bad_content_hash" in result["records"][1]["errors"]
    # The third record is still emitted, in its position, exactly as stored.
    assert result["records"][2]["position"] == 3
    assert result["records"][2]["previous_access_id"] == records[1]["id"]


def test_other_machines_damaged_records_never_affect_result(client, tmp_path):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, accessed_at=T0)
    damaged = register(client, machine_two, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET window_end = ? WHERE id = ?",
        (T4, damaged["id"]),
    )

    one = diagnose(client, machine_one)
    assert one == {
        "machine_id": machine_one,
        "valid": True,
        "checked_count": 1,
        "records": one["records"],
    }
    assert one["records"][0]["errors"] == []
    assert diagnose(client, machine_two)["valid"] is False


# --------------------------------------------------------------------------- #
# Serialization, read-only, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_single_newline(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    # Compact: no JSON whitespace after separators.
    assert b", " not in response.content
    assert b": " not in response.content


def test_top_level_and_record_field_order_is_fixed(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)

    raw = client.get(diagnostics_url(machine_id)).content.decode("utf-8")
    assert raw.index('"machine_id"') < raw.index('"valid"')
    assert raw.index('"valid"') < raw.index('"checked_count"')
    assert raw.index('"checked_count"') < raw.index('"records"')
    record_part = raw[raw.index('"id"'):]
    assert (
        record_part.index('"id"')
        < record_part.index('"position"')
        < record_part.index('"previous_access_id"')
        < record_part.index('"content_hash"')
        < record_part.index('"chain_hash"')
        < record_part.index('"errors"')
    )


def test_diagnostics_are_read_only_and_byte_identical(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    before = fetch_rows(client, machine_id)

    first = client.get(diagnostics_url(machine_id))
    second = client.get(diagnostics_url(machine_id))
    assert first.content == second.content
    assert fetch_rows(client, machine_id) == before


def test_results_survive_restart_byte_identical(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        register(first, machine_id, accessed_at=T1)
        register(first, machine_id, accessed_at=T0)
        body_before = first.get(diagnostics_url(machine_id)).content

    with TestClient(app) as second:
        body_after = second.get(diagnostics_url(machine_id)).content
        assert body_after == body_before
        result = json.loads(body_after)
        assert result["valid"] is True
        assert result["checked_count"] == 2


def test_error_code_constants_are_the_seven_documented_codes():
    assert set(privacy_chain._ERROR_CODE_ORDER) == {
        "missing_previous",
        "bad_previous",
        "bad_content_hash",
        "bad_chain_hash",
        "bad_time",
        "bad_id",
        "bad_ownership",
    }
    assert len(privacy_chain._ERROR_CODE_ORDER) == 7


@pytest.mark.parametrize(
    "column, value",
    [
        # SQLite stores these despite the declared column types (dynamic
        # typing): a non-integer count and a non-string result. The
        # diagnostics scan must flag them as content/chain anomalies rather
        # than crash.
        ("matches_count", "junk"),
        ("result", 123),
        ("content_hash", None),
        ("chain_hash", None),
    ],
)
def test_damaged_values_never_crash_and_are_emitted_verbatim(
    client, tmp_path, column, value
):
    machine_id = create_machine(client)
    damaged = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        f"UPDATE privacy_accesses SET {column} = ? WHERE id = ?",
        (value, damaged["id"]),
    )

    # Never raises; always lists the record in its 1-based position.
    result = diagnose(client, machine_id)
    assert result["checked_count"] == 1
    record = result["records"][0]
    assert record["id"] == damaged["id"]
    assert record["position"] == 1
    assert record["errors"]
    # Digest columns are emitted exactly as stored, including damaged nulls.
    if column in ("content_hash", "chain_hash"):
        assert record[column] is None


def test_a_second_record_with_a_damaged_null_link_reports_missing_previous(
    client, tmp_path
):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = NULL WHERE id = ?",
        (second["id"],),
    )

    result = diagnose(client, machine_id)
    record = next(r for r in result["records"] if r["id"] == second["id"])
    # The damaged null is emitted verbatim, not rewritten to an empty string.
    assert record["previous_access_id"] is None
    assert record["errors"] == ["missing_previous"]
