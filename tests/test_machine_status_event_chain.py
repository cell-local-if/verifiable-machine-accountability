"""Tests for the machine status-history tamper-evident chain.

Covers both the chain fields on ``GET /machines/{machine_id}/status-history``
and the read-only verification sub-entry
``GET /machines/{machine_id}/status-history/integrity``:

- every accepted status change appends one record carrying
  ``previous_status_event_id`` (``null`` on the machine's first record),
  ``content_hash`` = SHA-256 of the compact key-sorted JSON of the five
  transition fields, and ``chain_hash`` =
  SHA-256(previous_chain_hash + ":" + content_hash);
- per-machine chains independent across machines, ordered by the actual UTC
  instant of ``created_at`` then id;
- the integrity conclusion ``{valid, checked_count, broken_status_event_id}``
  for empty, sound, and damaged chains, reporting the first broken record;
- damaged creation moment, identifier, ownership, status edge (illegal,
  self loop, non-continuing), predecessor, content digest, and chain digest
  all count toward the total and break the chain without a crash;
- 422 ``invalid_query`` (extra parameter or a body) before any machine read,
  404 ``not_found``, 405 for non-GET (including HEAD), and 500
  ``internal_error`` on a real read failure, with no partial conclusion;
- read-only byte-identical results, restart stability, legacy backfill, and
  concurrent same-target serialization leaving one unbroken chain.
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
    "from_status",
    "to_status",
    "created_at",
)

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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
    message = f"{previous_chain_hash}:{content_hash}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


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


def set_status(client, machine_id, status):
    return client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )


def history(client, machine_id):
    response = client.get(f"/machines/{machine_id}/status-history")
    assert response.status_code == 200
    return response.json()


def integrity_url(machine_id):
    return f"/machines/{machine_id}/status-history/integrity"


def integrity(client, machine_id):
    return client.get(integrity_url(machine_id))


def assert_valid(records):
    previous_chain = ""
    previous_id = None
    for record in records:
        assert record["previous_status_event_id"] == previous_id
        assert record["content_hash"] == canonical_content_hash(record)
        assert record["chain_hash"] == chain_hash(
            previous_chain, record["content_hash"]
        )
        previous_chain = record["chain_hash"]
        previous_id = record["id"]


# --- chain construction ------------------------------------------------------


def test_change_appends_chain_fields(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")

    entry = history(client, machine_id)[0]

    assert set(entry.keys()) == {
        "id",
        "machine_id",
        "from_status",
        "to_status",
        "created_at",
        "previous_status_event_id",
        "content_hash",
        "chain_hash",
    }
    assert entry["previous_status_event_id"] is None
    assert HEX64_RE.match(entry["content_hash"])
    assert HEX64_RE.match(entry["chain_hash"])
    assert entry["content_hash"] == canonical_content_hash(entry)
    assert entry["chain_hash"] == chain_hash("", entry["content_hash"])


def test_records_link_in_created_order(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")

    entries = history(client, machine_id)

    assert len(entries) == 2
    first, second = entries
    assert first["previous_status_event_id"] is None
    assert second["previous_status_event_id"] == first["id"]
    assert second["chain_hash"] == chain_hash(
        first["chain_hash"], second["content_hash"]
    )
    assert_valid(entries)


def test_alternating_chain_is_continuous(client):
    machine_id = create_machine(client)
    for _ in range(4):
        set_status(client, machine_id, "suspended")
        set_status(client, machine_id, "active")

    records = history(client, machine_id)
    assert len(records) == 8
    assert [(r["from_status"], r["to_status"]) for r in records] == [
        ("active", "suspended") if i % 2 == 0 else ("suspended", "active")
        for i in range(8)
    ]
    assert_valid(records)


def test_chains_are_independent_per_machine(client):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    set_status(client, one, "suspended")
    set_status(client, two, "suspended")

    for machine_id in (one, two):
        entries = history(client, machine_id)
        assert entries[0]["previous_status_event_id"] is None
        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_status_event_id": None,
        }


def test_failed_change_appends_no_chain_record(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")

    assert set_status(client, machine_id, "suspended").status_code == 409

    assert len(history(client, machine_id)) == 1
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }


# --- integrity conclusions ---------------------------------------------------


def test_empty_chain_is_valid(client):
    machine_id = create_machine(client)

    response = integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
    assert response.content == (
        b'{"valid":true,"checked_count":0,"broken_status_event_id":null}\n'
    )


def test_complete_chain_is_valid(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")

    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }


def test_missing_machine_returns_404(client):
    response = integrity(client, MISSING_ID)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"checked_count" not in response.content


def test_extra_query_param_returns_422_before_machine_lookup(client):
    response = client.get(integrity_url(MISSING_ID), params={"from": "x"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_returns_422_before_machine_lookup(client):
    response = client.request("GET", integrity_url(MISSING_ID), content=b"{}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_non_get_methods_return_405(client, method):
    machine_id = create_machine(client)

    response = getattr(client, method)(integrity_url(machine_id))

    assert response.status_code == 405
    # The rejected method neither reads nor writes anything.
    assert history(client, machine_id) == []


def test_read_failure_returns_500_without_partial_conclusion(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")

    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    response = integrity(client, machine_id)

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"checked_count" not in response.content


# --- tamper detection --------------------------------------------------------


def test_detects_tampered_content(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    first = history(client, machine_id)[0]

    # A genuine content change: turn the stored edge into a self loop (which
    # also breaks the content digest and the edge continuity).
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET to_status = 'active' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_detects_tampered_previous_link(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    second = history(client, machine_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET previous_status_event_id = NULL "
        "WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_detects_tampered_chain_hash(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    first = history(client, machine_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_detects_tampered_content_hash(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    first = history(client, machine_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET content_hash = ? WHERE id = ?",
        ("1" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_reports_first_broken_record_only(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    set_status(client, machine_id, "suspended")
    records = history(client, machine_id)

    connection = sqlite3.connect(tmp_path / "test.db")
    for record in records[1:]:
        connection.execute(
            "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
            ("f" * 64, record["id"]),
        )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_status_event_id": records[1]["id"],
    }


def test_damaged_created_at_counts_and_breaks_without_crash(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    second = history(client, machine_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET created_at = 'not-a-time' WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_self_loop_edge_is_invalid(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    first = history(client, machine_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET from_status = 'suspended' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json()["broken_status_event_id"] == first[
        "id"
    ]


def test_illegal_status_value_is_invalid(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    first = history(client, machine_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET to_status = 'deleted' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    result = integrity(client, machine_id).json()
    assert result == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_non_continuing_edge_is_invalid(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    records = history(client, machine_id)

    # Two active->suspended edges in a row: the second cannot continue.
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET from_status = 'active', "
        "to_status = 'suspended' WHERE id = ?",
        (records[1]["id"],),
    )
    connection.commit()
    connection.close()

    result = integrity(client, machine_id).json()
    assert result["valid"] is False
    assert result["checked_count"] == 2
    assert result["broken_status_event_id"] == records[1]["id"]


def test_other_machines_damage_does_not_affect_result(client, tmp_path):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    set_status(client, one, "suspended")
    set_status(client, two, "suspended")
    damaged = history(client, two)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
        ("9" * 64, damaged["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, one).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
    assert integrity(client, two).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": damaged["id"],
    }


def test_damaged_record_is_not_repaired_or_recomputed(client, tmp_path):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    first = history(client, machine_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machine_status_events SET content_hash = ? WHERE id = ?",
        ("7" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    # Repeated audits never repair; the list also keeps the stored digest.
    for _ in range(3):
        assert integrity(client, machine_id).json()["valid"] is False
    assert history(client, machine_id)[0]["content_hash"] == "7" * 64


# --- read-only, stability, restart, legacy -----------------------------------


def test_read_only_and_byte_identical(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    listed_before = history(client, machine_id)

    first = integrity(client, machine_id)
    second = integrity(client, machine_id)

    assert first.content == second.content
    assert first.content.endswith(b"\n") and not first.content.endswith(b"\n\n")
    assert history(client, machine_id) == listed_before


def test_conclusion_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        set_status(first, machine_id, "suspended")
        set_status(first, machine_id, "active")
        conclusion = integrity(first, machine_id).json()
        listed = history(first, machine_id)

    with TestClient(app) as second:
        assert integrity(second, machine_id).json() == conclusion == {
            "valid": True,
            "checked_count": 2,
            "broken_status_event_id": None,
        }
        # The restart backfilled nothing: stored chain data is unchanged.
        assert history(second, machine_id) == listed


def test_legacy_records_are_backfilled_on_startup(tmp_path, monkeypatch):
    machine_id = "11111111-1111-1111-1111-111111111111"
    db_path = tmp_path / "legacy.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE machines (
            id VARCHAR(36) PRIMARY KEY, external_id VARCHAR UNIQUE,
            display_name VARCHAR, public_key VARCHAR, status VARCHAR,
            version INTEGER, created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE machine_status_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            from_status VARCHAR, to_status VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    legacy_rows = [
        (
            "aaaaaaaa-0000-0000-0000-000000000002",
            machine_id,
            "suspended",
            "active",
            "2026-01-02T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            machine_id,
            "active",
            "suspended",
            "2026-01-01T00:00:00.000000Z",
        ),
    ]
    connection.executemany(
        "INSERT INTO machine_status_events VALUES (?,?,?,?,?)", legacy_rows
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        entries = history(client, machine_id)

        # Ordered by (created_at, id), not insertion order.
        assert [entry["from_status"] for entry in entries] == ["active", "suspended"]
        assert entries[0]["previous_status_event_id"] is None
        assert entries[1]["previous_status_event_id"] == entries[0]["id"]
        assert_valid(entries)

        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_status_event_id": None,
        }


def test_empty_database_is_directly_usable(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}"
    )
    with TestClient(app) as client:
        machine_id = create_machine(client)
        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 0,
            "broken_status_event_id": None,
        }
        set_status(client, machine_id, "suspended")
        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_status_event_id": None,
        }


# --- concurrency -------------------------------------------------------------


def test_concurrent_same_target_leaves_one_unbroken_chain(client):
    machine_id = create_machine(client)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: set_status(client, machine_id, "suspended"), range(8)
            )
        )

    assert sum(r.status_code == 200 for r in responses) == 1

    connection = sqlite3.connect(client.app.state.engine.url.database)
    rows = connection.execute(
        "SELECT id, previous_status_event_id FROM machine_status_events "
        "WHERE machine_id = ?",
        (machine_id,),
    ).fetchall()
    connection.close()
    assert len(rows) == 1
    assert rows[0][1] is None
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }


def test_chain_of_many_serialized_changes_is_unbroken(client):
    machine_id = create_machine(client)
    count = 10
    for index in range(count):
        target = "suspended" if index % 2 == 0 else "active"
        assert set_status(client, machine_id, target).status_code == 200

    connection = sqlite3.connect(client.app.state.engine.url.database)
    rows = connection.execute(
        "SELECT id, previous_status_event_id FROM machine_status_events "
        "WHERE machine_id = ?",
        (machine_id,),
    ).fetchall()
    connection.close()

    assert len(rows) == count
    previous_ids = [row[1] for row in rows]
    assert previous_ids.count(None) == 1
    ids = {row[0] for row in rows}
    assert set(pid for pid in previous_ids if pid is not None) <= ids
    assert len(set(previous_ids)) == count
    assert_valid(history(client, machine_id))
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": count,
        "broken_status_event_id": None,
    }
