"""Tests for the privacy access tamper-evident chain and integrity audit.

Covers the chain fields added to privacy access registrations and
`GET /machines/{machine_id}/privacy-accesses/integrity`:

- GET-only ``405`` without reading data, ``422 invalid_query`` for any extra
  query parameter checked before the machine lookup, and ``404 not_found``
  for a missing machine with no integrity conclusion;
- an empty machine reports ``{valid: true, checked_count: 0,
  broken_access_id: null}`` and ``checked_count`` only ever counts the path
  machine's records;
- a sound chain verifies in the stable (accessed_at instant, id) order —
  exact-second stamps before fractional-second stamps of the same second —
  across out-of-order registration;
- a corrupted previous-access link, content hash, or chain hash, a missing
  link, a cross-machine predecessor, or a repeated pointer reports the first
  broken record;
- registration and chain linking commit atomically, concurrent registrations
  never break the chain, duplicates still write nothing, startup backfills
  pre-chain databases in the stable order, and a complete chain is not
  rewritten on restart.
"""
import hashlib
import json
import threading
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


MISSING_ID = "00000000-0000-0000-0000-000000000064"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def accesses_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def integrity_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/integrity"


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


def register(client, machine_id, accessed_at, result="success", matches_count=3):
    response = client.post(
        accesses_path(machine_id),
        json={
            "accessed_at": accessed_at,
            "window_start": T1,
            "window_end": T2,
            "result": result,
            "matches_count": matches_count if result == "success" else 0,
        },
    )
    assert response.status_code == 201
    return response.json()


def chain_rows(client, machine_id):
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM privacy_accesses WHERE machine_id = :m "
                "ORDER BY accessed_at, id"
            ),
            {"m": machine_id},
        ).mappings().all()
    return [dict(row) for row in rows]


def expected_content_hash(row):
    document = json.dumps(
        {
            "id": row["id"],
            "machine_id": row["machine_id"],
            "accessed_at": row["accessed_at"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "result": row["result"],
            "matches_count": row["matches_count"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def update_row(client, access_id, **values):
    assignments = ", ".join(f"{name} = :{name}" for name in values)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE privacy_accesses SET {assignments} WHERE id = :id"
            ),
            {**values, "id": access_id},
        )


def test_empty_machine_chain_is_valid(client):
    machine_id = create_machine(client)
    response = client.get(integrity_path(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_access_id": None,
    }


def test_only_get_is_accepted_on_integrity_path(client):
    machine_id = create_machine(client)
    register(client, machine_id, T0)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_path(machine_id))
        assert response.status_code == 405
    # The rejected methods read nothing and changed nothing.
    assert client.get(integrity_path(machine_id)).json()["valid"] is True


def test_any_query_parameter_is_invalid_query_before_machine_lookup(client):
    machine_id = create_machine(client)
    for target in (machine_id, MISSING_ID):
        response = client.get(f"{integrity_path(target)}?from_accessed_at={T0}")
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}
        response = client.get(f"{integrity_path(target)}?machine_id={machine_id}")
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_not_found_with_no_conclusion(client):
    response = client.get(integrity_path(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_sound_chain_verifies_in_accessed_at_instant_order(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:02.500000Z"
    # Register out of order, mixing an exact-second and a fractional stamp.
    register(client, machine_id, T4)
    first = register(client, machine_id, T0)
    register(client, machine_id, fractional)
    register(client, machine_id, T2)

    response = client.get(integrity_path(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 4,
        "broken_access_id": None,
    }

    # The stored chain follows the (instant, id) order: the exact-second T2
    # record sorts before the fractional record of the same second.
    rows = chain_rows(client, machine_id)
    by_id = {row["id"]: row for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: (
            datetime.fromisoformat(row["accessed_at"][:-1] + "+00:00"),
            row["id"],
        ),
    )
    assert [row["accessed_at"] for row in ordered] == [T0, T2, fractional, T4]
    previous_id = None
    previous_chain_hash = ""
    for row in ordered:
        assert row["previous_access_id"] == previous_id
        assert row["content_hash"] == expected_content_hash(row)
        expected_chain = hashlib.sha256(
            f"{previous_chain_hash}:{row['content_hash']}".encode("utf-8")
        ).hexdigest()
        assert row["chain_hash"] == expected_chain
        previous_id = row["id"]
        previous_chain_hash = row["chain_hash"]
    assert ordered[0]["id"] == first["id"]
    assert by_id[first["id"]]["previous_access_id"] is None


def test_checked_count_only_counts_path_machine_records(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, T0)
    register(client, machine_one, T1)
    register(client, machine_two, T0)

    # Damage machine two's only record; machine one's audit is unaffected.
    rows = chain_rows(client, machine_two)
    update_row(client, rows[0]["id"], content_hash="0" * 64)

    assert client.get(integrity_path(machine_one)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_access_id": None,
    }
    assert client.get(integrity_path(machine_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_access_id": rows[0]["id"],
    }


def test_corrupted_content_is_reported_at_first_broken_record(client):
    machine_id = create_machine(client)
    register(client, machine_id, T0)
    middle = register(client, machine_id, T1)
    register(client, machine_id, T2)

    update_row(client, middle["id"], matches_count=99)

    assert client.get(integrity_path(machine_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_access_id": middle["id"],
    }


def test_missing_or_cross_machine_previous_link_is_broken(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register(client, machine_one, T0)
    second = register(client, machine_one, T1)
    foreign = register(client, machine_two, T0)

    update_row(client, second["id"], previous_access_id=foreign["id"])
    assert client.get(integrity_path(machine_one)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": second["id"],
    }


def test_repeated_pointer_is_broken(client):
    machine_id = create_machine(client)
    first = register(client, machine_id, T0)
    register(client, machine_id, T1)
    third = register(client, machine_id, T2)

    # Point the third record at the first, repeating a predecessor.
    update_row(client, third["id"], previous_access_id=first["id"])
    result = client.get(integrity_path(machine_id)).json()
    assert result["valid"] is False
    assert result["checked_count"] == 3


def test_corrupted_chain_hash_is_broken(client):
    machine_id = create_machine(client)
    register(client, machine_id, T0)
    second = register(client, machine_id, T1)

    update_row(client, second["id"], chain_hash="f" * 64)
    assert client.get(integrity_path(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_access_id": second["id"],
    }


def test_duplicate_registration_still_rejected_and_writes_nothing(client):
    machine_id = create_machine(client)
    register(client, machine_id, T0)
    response = client.post(
        accesses_path(machine_id),
        json={
            "accessed_at": T0,
            "window_start": T1,
            "window_end": T2,
            "result": "success",
            "matches_count": 9,
        },
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_access"}}
    assert client.get(integrity_path(machine_id)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_access_id": None,
    }


def test_concurrent_registrations_never_break_the_chain(client):
    machine_id = create_machine(client)
    errors = []

    def register_one(index):
        try:
            register(
                client,
                machine_id,
                f"2026-03-01T00:00:{10 + index:02d}Z",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=register_one, args=(index,)) for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert client.get(integrity_path(machine_id)).json() == {
        "valid": True,
        "checked_count": 8,
        "broken_access_id": None,
    }


def test_startup_backfills_pre_chain_database(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'old.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first_client:
        machine_id = create_machine(first_client)
        register(first_client, machine_id, T0)
        register(first_client, machine_id, T1)
        # Simulate a database written before the chain feature.
        with first_client.app.state.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE privacy_accesses SET previous_access_id = NULL, "
                    "content_hash = NULL, chain_hash = NULL"
                )
            )

    with TestClient(app) as restarted:
        response = restarted.get(integrity_path(machine_id))
        assert response.json() == {
            "valid": True,
            "checked_count": 2,
            "broken_access_id": None,
        }
        rows = chain_rows(restarted, machine_id)
        assert all(row["content_hash"] for row in rows)
        assert all(row["chain_hash"] for row in rows)


def test_complete_chain_is_not_rewritten_on_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'complete.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first_client:
        machine_id = create_machine(first_client)
        register(first_client, machine_id, T0)
        register(first_client, machine_id, T1)
        before = chain_rows(first_client, machine_id)

    with TestClient(app) as restarted:
        after = chain_rows(restarted, machine_id)
        assert after == before
        assert restarted.get(integrity_path(machine_id)).json()["valid"] is True
