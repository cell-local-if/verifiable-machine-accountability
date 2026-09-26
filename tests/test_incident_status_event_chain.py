import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

CONTENT_KEYS = (
    "id",
    "machine_id",
    "event_id",
    "incident_id",
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


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def create_incident(client, machine_id, event_id, summary="something happened"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": "breach", "summary": summary},
    )
    assert response.status_code == 201
    return response.json()["id"]


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def history(client, machine_id, event_id, incident_id):
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status-history"
    )
    assert response.status_code == 200
    return response.json()


def integrity_url(machine_id):
    return f"/machines/{machine_id}/incident-status-events/integrity"


def integrity(client, machine_id):
    return client.get(integrity_url(machine_id))


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_id = create_incident(client, machine_id, event_id)
    return machine_id, event_id, incident_id


def test_transition_appends_chain_fields(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")

    entries = history(client, machine_id, event_id, incident_id)

    assert len(entries) == 1
    entry = entries[0]
    assert set(entry.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_id",
        "from_status",
        "to_status",
        "created_at",
        "previous_status_event_id",
        "content_hash",
        "chain_hash",
    }
    assert UUID_RE.match(entry["id"])
    assert entry["previous_status_event_id"] is None
    assert HEX64_RE.match(entry["content_hash"])
    assert HEX64_RE.match(entry["chain_hash"])
    assert entry["content_hash"] == canonical_content_hash(entry)
    assert entry["chain_hash"] == chain_hash("", entry["content_hash"])


def test_records_link_in_created_order(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")

    entries = history(client, machine_id, event_id, incident_id)

    assert len(entries) == 2
    first, second = entries
    assert first["previous_status_event_id"] is None
    assert second["previous_status_event_id"] == first["id"]
    assert second["content_hash"] == canonical_content_hash(second)
    assert second["chain_hash"] == chain_hash(
        first["chain_hash"], second["content_hash"]
    )


def test_chain_spans_incidents_of_one_machine(client, incident):
    machine_id, event_id, incident_id = incident
    other_incident = create_incident(client, machine_id, event_id, summary="two")
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, other_incident, "acknowledged")

    first = history(client, machine_id, event_id, incident_id)[0]
    second = history(client, machine_id, event_id, other_incident)[0]

    # The chain is per machine, not per incident: the second incident's first
    # record links to the first incident's record.
    assert first["previous_status_event_id"] is None
    assert second["previous_status_event_id"] == first["id"]


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)

    transition(client, machine_one, event_one, incident_one, "acknowledged")
    transition(client, machine_two, event_two, incident_two, "acknowledged")

    for machine_id, event_id, incident_id in (
        (machine_one, event_one, incident_one),
        (machine_two, event_two, incident_two),
    ):
        entries = history(client, machine_id, event_id, incident_id)
        assert entries[0]["previous_status_event_id"] is None
        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_status_event_id": None,
        }


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


def test_complete_chain_is_valid(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")

    response = integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }


def test_missing_machine_returns_404(client):
    response = integrity(client, MISSING_ID)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_extra_query_param_returns_422_before_machine_lookup(client):
    response = client.get(integrity_url(MISSING_ID), params={"from": "x"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_returns_422_before_machine_lookup(client):
    response = client.request("GET", integrity_url(MISSING_ID), content=b"{}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_non_get_methods_return_405(client, incident):
    machine_id, _, _ = incident
    url = integrity_url(machine_id)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405, method


def test_detects_tampered_content(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    first = history(client, machine_id, event_id, incident_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET to_status = 'resolved' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_detects_tampered_previous_link(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    second = history(client, machine_id, event_id, incident_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET previous_status_event_id = NULL "
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


def test_detects_tampered_chain_hash(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    first = history(client, machine_id, event_id, incident_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_reports_first_broken_record_only(client, tmp_path):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_ids = [
        create_incident(client, machine_id, event_id, summary=f"case-{n}")
        for n in range(3)
    ]
    for incident_id in incident_ids:
        transition(client, machine_id, event_id, incident_id, "acknowledged")
    entries = [
        history(client, machine_id, event_id, incident_id)[0]
        for incident_id in incident_ids
    ]

    connection = sqlite3.connect(tmp_path / "test.db")
    for entry in entries[1:]:
        connection.execute(
            "UPDATE incident_status_events SET to_status = 'resolved' WHERE id = ?",
            (entry["id"],),
        )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_status_event_id": entries[1]["id"],
    }


def test_damaged_created_at_counts_and_breaks_without_crash(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    second = history(client, machine_id, event_id, incident_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET created_at = 'not-a-time' WHERE id = ?",
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


def test_other_machines_damage_does_not_affect_result(client, tmp_path):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)
    transition(client, machine_one, event_one, incident_one, "acknowledged")
    transition(client, machine_two, event_two, incident_two, "acknowledged")
    damaged = history(client, machine_two, event_two, incident_two)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET to_status = 'resolved' WHERE id = ?",
        (damaged["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_one).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
    assert integrity(client, machine_two).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": damaged["id"],
    }


def test_read_only_and_byte_identical(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    listed_before = history(client, machine_id, event_id, incident_id)

    first = integrity(client, machine_id)
    second = integrity(client, machine_id)

    assert first.content == second.content
    assert first.content.endswith(b"\n") and not first.content.endswith(b"\n\n")
    assert first.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }
    assert history(client, machine_id, event_id, incident_id) == listed_before


def test_conclusion_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident_id = create_incident(first, machine_id, event_id)
        transition(first, machine_id, event_id, incident_id, "acknowledged")
        transition(first, machine_id, event_id, incident_id, "resolved")
        conclusion = integrity(first, machine_id).json()
        listed = history(first, machine_id, event_id, incident_id)

    with TestClient(app) as second:
        assert integrity(second, machine_id).json() == conclusion == {
            "valid": True,
            "checked_count": 2,
            "broken_status_event_id": None,
        }
        # The restart backfilled nothing: the stored chain data is unchanged.
        assert history(second, machine_id, event_id, incident_id) == listed


def test_legacy_records_are_backfilled_on_startup(tmp_path, monkeypatch):
    machine_id = "11111111-1111-1111-1111-111111111111"
    event_id = "22222222-2222-2222-2222-222222222222"
    incident_id = "33333333-3333-3333-3333-333333333333"
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
        CREATE TABLE authorization_decision_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            action_type VARCHAR, resource VARCHAR, allowed BOOLEAN,
            reason VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE authorization_decision_incidents (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            event_id VARCHAR(36), incident_type VARCHAR, summary VARCHAR,
            status VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE incident_status_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            event_id VARCHAR(36), incident_id VARCHAR(36),
            from_status VARCHAR, to_status VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_events VALUES (?,?,?,?,?,?,?)",
        (event_id, machine_id, "read", "res/x", 1, "allowed_by_policy", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_incidents VALUES (?,?,?,?,?,?,?)",
        (incident_id, machine_id, event_id, "breach", "legacy", "resolved", "t0"),
    )
    legacy_rows = [
        (
            "aaaaaaaa-0000-0000-0000-000000000002",
            machine_id,
            event_id,
            incident_id,
            "acknowledged",
            "resolved",
            "2026-01-02T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            machine_id,
            event_id,
            incident_id,
            "open",
            "acknowledged",
            "2026-01-01T00:00:00.000000Z",
        ),
    ]
    connection.executemany(
        "INSERT INTO incident_status_events VALUES (?,?,?,?,?,?,?)", legacy_rows
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        entries = history(client, machine_id, event_id, incident_id)

        # Ordered by (created_at, id), not insertion order.
        assert [entry["from_status"] for entry in entries] == [
            "open",
            "acknowledged",
        ]
        assert entries[0]["previous_status_event_id"] is None
        assert entries[1]["previous_status_event_id"] == entries[0]["id"]
        for entry in entries:
            assert HEX64_RE.match(entry["content_hash"])
            assert HEX64_RE.match(entry["chain_hash"])
            assert entry["content_hash"] == canonical_content_hash(entry)
        assert entries[0]["chain_hash"] == chain_hash(
            "", entries[0]["content_hash"]
        )
        assert entries[1]["chain_hash"] == chain_hash(
            entries[0]["chain_hash"], entries[1]["content_hash"]
        )

        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_status_event_id": None,
        }


def test_concurrent_transitions_form_one_unbroken_chain(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    count = 12
    incident_ids = [
        create_incident(client, machine_id, event_id, summary=f"case-{n}")
        for n in range(count)
    ]

    def acknowledge(incident_id):
        return client.post(
            f"/machines/{machine_id}/authorization-decision-events/{event_id}"
            f"/incidents/{incident_id}/status",
            json={"status": "acknowledged"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(acknowledge, incident_ids))

    assert all(response.status_code == 200 for response in responses)

    connection = sqlite3.connect(
        client.app.state.engine.url.database
    )
    rows = connection.execute(
        "SELECT id, previous_status_event_id FROM incident_status_events "
        "WHERE machine_id = ?",
        (machine_id,),
    ).fetchall()
    connection.close()
    assert len(rows) == count
    assert len({row[0] for row in rows}) == count

    # Exactly one record has no predecessor and the links form a single chain.
    previous_ids = [row[1] for row in rows]
    assert previous_ids.count(None) == 1
    ids = {row[0] for row in rows}
    assert set(pid for pid in previous_ids if pid is not None) <= ids
    assert len(set(previous_ids)) == count

    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": count,
        "broken_status_event_id": None,
    }


def test_failed_transition_appends_no_chain_record(client, incident):
    machine_id, event_id, incident_id = incident

    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status",
        json={"status": "resolved"},
    )

    assert response.status_code == 409
    assert history(client, machine_id, event_id, incident_id) == []
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
