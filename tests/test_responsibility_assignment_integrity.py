import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party",
    "role",
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


def assignments_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    )


def integrity_url(machine_id):
    return f"/machines/{machine_id}/responsibility-assignments/integrity"


def assign(client, machine_id, event_id, incident_id, party, role):
    response = client.post(
        assignments_url(machine_id, event_id, incident_id),
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_id = create_incident(client, machine_id, event_id)
    return machine_id, event_id, incident_id


def test_create_returns_chain_fields(client, incident):
    machine_id, event_id, incident_id = incident

    record = assign(client, machine_id, event_id, incident_id, "alice", "owner")

    assert record["previous_assignment_id"] is None
    assert HEX64_RE.match(record["content_hash"])
    assert HEX64_RE.match(record["chain_hash"])
    assert record["content_hash"] == canonical_content_hash(record)
    assert record["chain_hash"] == chain_hash("", record["content_hash"])


def test_chain_links_in_creation_order_across_incidents(client, incident):
    machine_id, event_id, incident_id = incident
    other_incident = create_incident(client, machine_id, event_id, summary="two")

    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    second = assign(client, machine_id, event_id, other_incident, "bob", "reviewer")
    third = assign(client, machine_id, event_id, incident_id, "carol", "auditor")

    records = [first, second, third]
    assert first["previous_assignment_id"] is None
    for previous, current in zip(records, records[1:]):
        assert current["previous_assignment_id"] == previous["id"]
        assert current["content_hash"] == canonical_content_hash(current)
        assert current["chain_hash"] == chain_hash(
            previous["chain_hash"], current["content_hash"]
        )


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)

    first = assign(client, machine_one, event_one, incident_one, "alice", "owner")
    second = assign(client, machine_two, event_two, incident_two, "bob", "owner")

    assert first["previous_assignment_id"] is None
    assert second["previous_assignment_id"] is None
    assert first["chain_hash"] != second["chain_hash"]


def test_integrity_missing_machine_returns_404(client):
    response = client.get(integrity_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_empty_chain_is_valid(client):
    machine_id = create_machine(client)

    response = client.get(integrity_url(machine_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_assignment_id": None,
    }


def test_integrity_complete_chain_is_valid(client, incident):
    machine_id, event_id, incident_id = incident
    for n in range(3):
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")

    response = client.get(integrity_url(machine_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_assignment_id": None,
    }


def test_integrity_ignores_other_machines(client, tmp_path):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)
    assign(client, machine_one, event_one, incident_one, "alice", "owner")
    damaged = assign(client, machine_two, event_two, incident_two, "bob", "owner")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (damaged["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
    assert client.get(integrity_url(machine_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_assignment_id": damaged["id"],
    }


def test_integrity_detects_tampered_content(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_integrity_detects_tampered_previous_link(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    second = assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments "
        "SET previous_assignment_id = NULL WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": second["id"],
    }


def test_integrity_detects_tampered_chain_hash(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_integrity_reports_first_broken_record(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")
        for n in range(3)
    ]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (records[1]["id"],),
    )
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (records[2]["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": records[1]["id"],
    }


def test_integrity_is_read_only_and_stable(client, incident):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")
    listed_before = client.get(
        assignments_url(machine_id, event_id, incident_id)
    ).json()

    first = client.get(integrity_url(machine_id)).json()
    second = client.get(integrity_url(machine_id)).json()

    assert first == second == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }
    assert (
        client.get(assignments_url(machine_id, event_id, incident_id)).json()
        == listed_before
    )


def test_concurrent_creates_keep_chain_unbroken(client, incident):
    machine_id, event_id, incident_id = incident

    def do_assign(index):
        return client.post(
            assignments_url(machine_id, event_id, incident_id),
            json={"party": f"party-{index}", "role": f"role-{index}"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_assign, range(20)))

    assert all(response.status_code == 201 for response in responses)

    listed = client.get(assignments_url(machine_id, event_id, incident_id)).json()
    assert len(listed) == 20
    ordered = sorted(listed, key=lambda r: (r["created_at"], r["id"]))
    assert ordered[0]["previous_assignment_id"] is None
    for previous, current in zip(ordered, ordered[1:]):
        assert current["previous_assignment_id"] == previous["id"]

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 20,
        "broken_assignment_id": None,
    }


def test_chain_survives_restart_without_writes(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident_id = create_incident(first, machine_id, event_id)
        assign(first, machine_id, event_id, incident_id, "alice", "owner")
        assign(first, machine_id, event_id, incident_id, "bob", "reviewer")
        listed = first.get(assignments_url(machine_id, event_id, incident_id)).json()
        integrity = first.get(integrity_url(machine_id)).json()

    with TestClient(app) as second:
        # Restart over a complete database issues no writes and changes nothing.
        assert (
            second.get(assignments_url(machine_id, event_id, incident_id)).json()
            == listed
        )
        assert second.get(integrity_url(machine_id)).json() == integrity == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }


def test_startup_backfills_pre_chain_records(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident_id = create_incident(first, machine_id, event_id)
        assign(first, machine_id, event_id, incident_id, "alice", "owner")
        assign(first, machine_id, event_id, incident_id, "bob", "reviewer")

    # Simulate a database written before the chain feature existed.
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments SET "
        "previous_assignment_id = NULL, content_hash = NULL, chain_hash = NULL"
    )
    connection.commit()
    connection.close()

    with TestClient(app) as second:
        listed = second.get(assignments_url(machine_id, event_id, incident_id)).json()
        ordered = sorted(listed, key=lambda r: (r["created_at"], r["id"]))
        assert ordered[0]["previous_assignment_id"] is None
        assert ordered[1]["previous_assignment_id"] == ordered[0]["id"]
        for record in ordered:
            assert HEX64_RE.match(record["content_hash"])
            assert HEX64_RE.match(record["chain_hash"])
            assert record["content_hash"] == canonical_content_hash(record)
        assert ordered[0]["chain_hash"] == chain_hash("", ordered[0]["content_hash"])
        assert ordered[1]["chain_hash"] == chain_hash(
            ordered[0]["chain_hash"], ordered[1]["content_hash"]
        )
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }
