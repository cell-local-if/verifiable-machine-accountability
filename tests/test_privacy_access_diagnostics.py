"""Tests for the privacy access per-record chain diagnostics query.

Covers the read-only ``GET /machines/{machine_id}/privacy-accesses/diagnostics``
endpoint: GET-only 405, ``invalid_query`` 422 for any query parameter or body
before the machine lookup, ``not_found`` 404, ``internal_error`` 500 on read
failures with no partial result, the ``{machine_id, valid, checked_count,
records}`` envelope, 1-based positions in (accessed-at instant, id) order with
damaged stamps last, stored-value verbatim output, the seven-code anomaly
taxonomy, per-machine isolation, read-only byte-identical stability, and
persistence across restarts.
"""
import hashlib
import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

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


def accesses_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def diagnostics_url(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/diagnostics"


def register(client, machine_id, accessed_at=T0, result="success", matches_count=1,
             window_start=T1, window_end=T2):
    response = client.post(
        accesses_path(machine_id),
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


def test_missing_machine_returns_404_without_diagnostics(client):
    response = client.get(diagnostics_url(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content
    assert b"valid" not in response.content


def test_read_failure_is_500_with_no_partial_diagnostics(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content
    assert b"valid" not in response.content


def test_machine_read_failure_is_500_with_no_machine_object(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"machine_id" not in response.content


def test_validation_and_method_errors_do_not_read_accesses(client):
    # With the table dropped, any record read would 500; validation-phase
    # and routing errors must still come back as their own codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    assert client.get(diagnostics_url(machine_id) + "?x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    assert client.get(diagnostics_url(MISSING_ID)).status_code == 404
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)(diagnostics_url(machine_id)).status_code == 405


# --------------------------------------------------------------------------- #
# Envelope, ordering, and sound chains
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


def test_sound_chain_lists_every_record_with_empty_anomalies(client):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2, result="failed",
                     matches_count=0)

    rows = {row["id"]: row for row in fetch_rows(client, machine_id)}
    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 200
    payload = response.json()
    assert payload["machine_id"] == machine_id
    assert payload["valid"] is True
    assert payload["checked_count"] == 3
    assert [record["position"] for record in payload["records"]] == [1, 2, 3]

    expected_previous = None
    for record, registered in zip(payload["records"], (first, second, third)):
        stored = rows[registered["id"]]
        assert record["id"] == registered["id"]
        assert record["previous_access_id"] == expected_previous
        assert record["content_hash"] == stored["content_hash"]
        assert HEX64_RE.match(record["content_hash"])
        assert record["content_hash"] == canonical_content_hash(stored)
        assert record["chain_hash"] == stored["chain_hash"]
        assert HEX64_RE.match(record["chain_hash"])
        assert record["anomaly_codes"] == []
        expected_previous = registered["id"]


def test_positions_follow_access_instant_then_id(client):
    machine_id = create_machine(client)
    later = register(client, machine_id, accessed_at=T2)
    fractional = register(client, machine_id, accessed_at="2026-03-01T00:00:00.500000Z")
    exact = register(client, machine_id, accessed_at=T0)

    payload = client.get(diagnostics_url(machine_id)).json()
    # The exact-second record sorts before the fractional one of the same
    # second, and both before the later record, regardless of insert order.
    assert [record["id"] for record in payload["records"]] == [
        exact["id"],
        fractional["id"],
        later["id"],
    ]
    assert [record["position"] for record in payload["records"]] == [1, 2, 3]
    assert payload["records"][0]["previous_access_id"] is None
    assert payload["records"][1]["previous_access_id"] == exact["id"]
    assert payload["records"][2]["previous_access_id"] == fractional["id"]
    assert payload["valid"] is True


def test_response_is_compact_json_with_single_trailing_newline(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)

    response = client.get(diagnostics_url(machine_id))
    assert response.status_code == 200
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b'": ' not in response.content
    assert b", " not in response.content


# --------------------------------------------------------------------------- #
# Anomaly taxonomy
# --------------------------------------------------------------------------- #


def test_tampered_content_flags_content_and_chain_onward(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET matches_count = 42 WHERE id = ?",
        (first["id"],),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["checked_count"] == 2
    # The tampered record keeps every one of its problems; the chain
    # expectation is recomputed, so the follower's stored chain hash no
    # longer verifies either.
    assert payload["records"][0]["id"] == first["id"]
    assert payload["records"][0]["anomaly_codes"] == [
        "bad_content_hash",
        "bad_chain_hash",
    ]
    assert payload["records"][1]["id"] == second["id"]
    assert payload["records"][1]["anomaly_codes"] == ["bad_chain_hash"]
    # Stored values are emitted verbatim, never repaired: the tampered row
    # keeps its original (now unverifiable) digests in the output.
    stored = {row["id"]: row for row in fetch_rows(client, machine_id)}
    assert payload["records"][0]["content_hash"] == stored[first["id"]]["content_hash"]
    assert payload["records"][0]["chain_hash"] == stored[first["id"]]["chain_hash"]


def test_missing_previous_link_is_flagged(client, tmp_path):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = NULL WHERE id = ?",
        (second["id"],),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["records"][0]["anomaly_codes"] == []
    assert payload["records"][1]["previous_access_id"] is None
    assert payload["records"][1]["anomaly_codes"] == ["missing_previous"]


def test_wrong_previous_link_is_flagged(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        (first["id"], third["id"]),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["records"][2]["anomaly_codes"] == ["wrong_previous"]
    assert payload["records"][2]["previous_access_id"] == first["id"]


def test_first_record_with_a_predecessor_is_wrong_previous(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        (second["id"], first["id"]),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["records"][0]["anomaly_codes"] == ["wrong_previous"]
    assert payload["records"][0]["previous_access_id"] == second["id"]
    # The follower now points at its true predecessor and is otherwise sound.
    assert payload["records"][1]["anomaly_codes"] == []


def test_tampered_chain_hash_is_flagged(client, tmp_path):
    machine_id = create_machine(client)
    damaged = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET chain_hash = ? WHERE id = ?",
        ("0" * 64, damaged["id"]),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["records"][0]["anomaly_codes"] == ["bad_chain_hash"]
    # The stored (damaged) digest is emitted verbatim, not rewritten.
    assert payload["records"][0]["chain_hash"] == "0" * 64


def test_unparseable_accessed_at_sorts_last_and_flags_time(client, tmp_path):
    machine_id = create_machine(client)
    sound = register(client, machine_id, accessed_at=T0)
    damaged = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET accessed_at = 'not-a-time' WHERE id = ?",
        (damaged["id"],),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    # The damaged stamp sorts after every parseable instant, never crashes.
    assert [record["id"] for record in payload["records"]] == [
        sound["id"],
        damaged["id"],
    ]
    assert payload["records"][1]["position"] == 2
    assert payload["records"][1]["anomaly_codes"] == [
        "bad_content_hash",
        "bad_chain_hash",
        "bad_time",
    ]


def test_non_uuid_id_is_flagged(client, tmp_path):
    machine_id = create_machine(client)
    damaged = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET id = 'not-a-uuid' WHERE id = ?",
        (damaged["id"],),
    )

    payload = client.get(diagnostics_url(machine_id)).json()
    assert payload["valid"] is False
    assert payload["records"][0]["id"] == "not-a-uuid"
    assert payload["records"][0]["anomaly_codes"] == [
        "bad_content_hash",
        "bad_chain_hash",
        "bad_id",
    ]


def test_other_machines_damage_never_enters_the_result(client, tmp_path):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, accessed_at=T0)
    damaged = register(client, machine_two, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET content_hash = 'junk' WHERE id = ?",
        (damaged["id"],),
    )

    payload_one = client.get(diagnostics_url(machine_one)).json()
    assert payload_one["valid"] is True
    assert payload_one["checked_count"] == 1
    assert payload_one["records"][0]["anomaly_codes"] == []

    payload_two = client.get(diagnostics_url(machine_two)).json()
    assert payload_two["valid"] is False
    assert payload_two["records"][0]["content_hash"] == "junk"
    # The business fields are untouched, so the recomputed chain expectation
    # still matches the stored chain hash; only the digest itself is broken.
    assert payload_two["records"][0]["anomaly_codes"] == ["bad_content_hash"]


# --------------------------------------------------------------------------- #
# Read-only, stable, persistent
# --------------------------------------------------------------------------- #


def test_diagnostics_is_read_only_and_byte_identical(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    before = fetch_rows(client, machine_id)

    first_response = client.get(diagnostics_url(machine_id))
    second_response = client.get(diagnostics_url(machine_id))

    assert first_response.content == second_response.content
    assert first_response.content.endswith(b"\n")
    assert fetch_rows(client, machine_id) == before


def test_diagnostics_survives_restart_byte_identical(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        register(first, machine_id, accessed_at=T0)
        register(first, machine_id, accessed_at=T1)
        rows_before = fetch_rows(first, machine_id)
        body_before = first.get(diagnostics_url(machine_id)).content

    with TestClient(app) as second:
        # Records persist across the restart and the diagnostics output is
        # byte-identical; the restart itself issued no writes to the chain.
        assert fetch_rows(second, machine_id) == rows_before
        assert second.get(diagnostics_url(machine_id)).content == body_before
