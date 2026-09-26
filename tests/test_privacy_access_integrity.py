"""Tests for the privacy access tamper-evident hash chain and integrity audit.

Covers the chain fields stored by `POST /machines/{machine_id}/privacy-accesses`
and the read-only `GET /machines/{machine_id}/privacy-accesses/integrity`
endpoint: GET-only 405, ``invalid_query`` 422 for any query parameter before
the machine lookup, ``not_found`` 404, empty-chain ``true/0/null``, chain
ordering by the actual UTC instant of ``accessed_at`` then id (exact second
before fractional second), tamper detection (content, previous link, chain
hash, cross-machine/duplicate pointers), per-machine isolation, read-only
stability, concurrent registration safety, restart without writes, and
startup backfill of pre-chain records.
"""
import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

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


def accesses_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def integrity_url(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/integrity"


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
            text(
                "SELECT * FROM privacy_accesses WHERE machine_id = :m"
            ),
            {"m": machine_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def chain_ordered(rows):
    from accountability.privacy_chain import _accessed_instant

    return sorted(rows, key=lambda row: (_accessed_instant(row["accessed_at"]), row["id"]))


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


def test_only_get_is_accepted_on_integrity_path(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine_id))
        assert response.status_code == 405


@pytest.mark.parametrize("query", ["?unexpected=1", "?from_accessed_at=" + T0])
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    response = client.get(integrity_url(MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        integrity_url(machine_id),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_machine_lookup(client):
    response = client.request(
        "GET",
        integrity_url(MISSING_ID),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_conclusion(client):
    response = client.get(integrity_url(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_read_failure_is_500_with_no_partial_conclusion(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    response = client.get(integrity_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"valid" not in response.content
    assert b"checked_count" not in response.content


def test_validation_and_method_errors_do_not_read_accesses(client):
    # With the table dropped, any record read would 500; validation-phase
    # and routing errors must still come back as their own codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE privacy_accesses"))

    assert client.get(integrity_url(machine_id) + "?x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    assert client.get(integrity_url(MISSING_ID)).status_code == 404
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)(integrity_url(machine_id)).status_code == 405


# --------------------------------------------------------------------------- #
# Chain shape and integrity results
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid_with_zero_count(client):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_access_id": None,
    }


def test_registration_response_keeps_original_fields(client):
    machine_id = create_machine(client)
    record = register(client, machine_id)
    assert set(record.keys()) == {
        "id",
        "machine_id",
        "accessed_at",
        "window_start",
        "window_end",
        "result",
        "matches_count",
    }


def test_chain_fields_are_stored_and_linked_in_access_order(client):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2, result="failed",
                     matches_count=0)

    rows = {row["id"]: row for row in fetch_rows(client, machine_id)}
    ordered = [rows[record["id"]] for record in (first, second, third)]

    previous = None
    previous_chain_hash = ""
    for row in ordered:
        assert row["previous_access_id"] == (previous["id"] if previous else None)
        assert HEX64_RE.match(row["content_hash"])
        assert HEX64_RE.match(row["chain_hash"])
        assert row["content_hash"] == canonical_content_hash(row)
        assert row["chain_hash"] == chain_hash(previous_chain_hash, row["content_hash"])
        previous = row
        previous_chain_hash = row["chain_hash"]

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 3,
        "broken_access_id": None,
    }


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = register(client, machine_id, accessed_at="2026-03-01T00:00:00.500000Z")
    exact = register(client, machine_id, accessed_at=T0)

    rows = {row["id"]: row for row in fetch_rows(client, machine_id)}
    # The exact-second record is the chain head even though it registered
    # second and its ISO text sorts after the fractional stamp.
    assert rows[exact["id"]]["previous_access_id"] is None
    assert rows[fractional["id"]]["previous_access_id"] == exact["id"]
    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_access_id": None,
    }


def test_out_of_order_registration_keeps_chain_valid(client):
    machine_id = create_machine(client)
    later = register(client, machine_id, accessed_at=T2)
    earlier = register(client, machine_id, accessed_at=T0)

    rows = {row["id"]: row for row in fetch_rows(client, machine_id)}
    assert rows[earlier["id"]]["previous_access_id"] is None
    assert rows[later["id"]]["previous_access_id"] == earlier["id"]
    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_access_id": None,
    }


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, accessed_at=T0)
    register(client, machine_two, accessed_at=T0)

    rows_one = fetch_rows(client, machine_one)
    rows_two = fetch_rows(client, machine_two)
    assert rows_one[0]["previous_access_id"] is None
    assert rows_two[0]["previous_access_id"] is None
    assert rows_one[0]["chain_hash"] != rows_two[0]["chain_hash"]

    for machine_id in (machine_one, machine_two):
        assert client.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_access_id": None,
        }


def test_duplicate_registration_is_still_rejected_and_writes_nothing(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0, matches_count=3)

    response = client.post(
        accesses_path(machine_id),
        json={
            "accessed_at": T0,
            "window_start": T1,
            "window_end": T2,
            "result": "success",
            "matches_count": 99,
        },
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_access"}}
    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_access_id": None,
    }


# --------------------------------------------------------------------------- #
# Tamper detection
# --------------------------------------------------------------------------- #


def tamper(db_path, statement, parameters=()):
    connection = sqlite3.connect(db_path)
    connection.execute(statement, parameters)
    connection.commit()
    connection.close()


def test_integrity_detects_tampered_content(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET matches_count = 42 WHERE id = ?",
        (first["id"],),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": first["id"],
    }


def test_integrity_detects_tampered_previous_link(client, tmp_path):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    second = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = NULL WHERE id = ?",
        (second["id"],),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": second["id"],
    }


def test_integrity_detects_tampered_chain_hash(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": first["id"],
    }


def test_integrity_detects_cross_machine_previous_pointer(client, tmp_path):
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

    assert client.get(integrity_url(machine_two)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": second["id"],
    }
    # The other machine's chain is untouched.
    assert client.get(integrity_url(machine_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_access_id": None,
    }


def test_integrity_detects_duplicate_previous_pointer(client, tmp_path):
    machine_id = create_machine(client)
    first = register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    third = register(client, machine_id, accessed_at=T2, result="failed",
                     matches_count=0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET previous_access_id = ? WHERE id = ?",
        (first["id"], third["id"]),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_access_id": third["id"],
    }


def test_integrity_reports_first_broken_record(client, tmp_path):
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
    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET result = 'failed' WHERE id = ?",
        (records[2]["id"],),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_access_id": records[1]["id"],
    }


def test_integrity_ignores_other_machines(client, tmp_path):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, accessed_at=T0)
    damaged = register(client, machine_two, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET window_end = ? WHERE id = ?",
        (T4, damaged["id"]),
    )

    assert client.get(integrity_url(machine_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_access_id": None,
    }
    assert client.get(integrity_url(machine_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_access_id": damaged["id"],
    }


def test_integrity_tolerates_unparseable_accessed_at(client, tmp_path):
    # A damaged stamp sorts after every parseable record and is judged by its
    # recomputed content hash — never a crash, never a repair.
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    damaged = register(client, machine_id, accessed_at=T1)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET accessed_at = 'not-a-time' WHERE id = ?",
        (damaged["id"],),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": damaged["id"],
    }


def test_integrity_tolerates_non_hex_stored_hashes(client, tmp_path):
    machine_id = create_machine(client)
    damaged = register(client, machine_id, accessed_at=T0)

    tamper(
        tmp_path / "test.db",
        "UPDATE privacy_accesses SET content_hash = 'junk' WHERE id = ?",
        (damaged["id"],),
    )

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_access_id": damaged["id"],
    }


# --------------------------------------------------------------------------- #
# Read-only, concurrent, persistent
# --------------------------------------------------------------------------- #


def test_integrity_is_read_only_and_stable(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T0)
    register(client, machine_id, accessed_at=T1)
    before = fetch_rows(client, machine_id)

    first_response = client.get(integrity_url(machine_id))
    second_response = client.get(integrity_url(machine_id))

    # Repeat calls over unchanged data are byte-identical compact JSON.
    assert first_response.content == second_response.content
    assert first_response.content.endswith(b"\n")
    assert first_response.json() == second_response.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_access_id": None,
    }
    assert fetch_rows(client, machine_id) == before


def test_concurrent_registrations_keep_chain_unbroken(client):
    machine_id = create_machine(client)

    def do_register(index):
        return client.post(
            accesses_path(machine_id),
            json={
                "accessed_at": f"2026-03-01T00:00:{index:02d}Z",
                "window_start": T1,
                "window_end": T2,
                "result": "success",
                "matches_count": index,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_register, range(20)))

    assert all(response.status_code == 201 for response in responses)

    rows = chain_ordered(fetch_rows(client, machine_id))
    assert len(rows) == 20
    assert rows[0]["previous_access_id"] is None
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_access_id"] == previous["id"]

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 20,
        "broken_access_id": None,
    }


def test_chain_survives_restart_without_writes(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        register(first, machine_id, accessed_at=T0)
        register(first, machine_id, accessed_at=T1)
        rows_before = fetch_rows(first, machine_id)
        integrity = first.get(integrity_url(machine_id)).json()

    with TestClient(app) as second:
        # Restart over a complete database issues no writes and changes nothing.
        assert fetch_rows(second, machine_id) == rows_before
        assert second.get(integrity_url(machine_id)).json() == integrity == {
            "valid": True,
            "checked_count": 2,
            "broken_access_id": None,
        }


def test_startup_backfills_pre_chain_records(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        register(first, machine_id, accessed_at=T1)
        register(first, machine_id, accessed_at=T0)

    # Simulate a database written before the chain feature existed.
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE privacy_accesses SET "
        "previous_access_id = NULL, content_hash = NULL, chain_hash = NULL"
    )
    connection.commit()
    connection.close()

    with TestClient(app) as second:
        rows = chain_ordered(fetch_rows(second, machine_id))
        assert rows[0]["previous_access_id"] is None
        assert rows[1]["previous_access_id"] == rows[0]["id"]
        for row in rows:
            assert HEX64_RE.match(row["content_hash"])
            assert HEX64_RE.match(row["chain_hash"])
            assert row["content_hash"] == canonical_content_hash(row)
        assert rows[0]["chain_hash"] == chain_hash("", rows[0]["content_hash"])
        assert rows[1]["chain_hash"] == chain_hash(
            rows[0]["chain_hash"], rows[1]["content_hash"]
        )
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_access_id": None,
        }
