import hashlib
import json
import re
import sqlite3

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


def assign(client, machine_id, event_id, incident_id, party, role):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


def chain_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        "/responsibility-assignments/integrity"
    )


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_id = create_incident(client, machine_id, event_id)
    return machine_id, event_id, incident_id


def test_empty_chain_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.get(chain_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_assignment_id": None,
    }
    assert response.content == (
        b'{"valid":true,"checked_count":0,"broken_assignment_id":null}\n'
    )


def test_complete_chain_is_valid(client, incident):
    machine_id, event_id, incident_id = incident
    for n in range(3):
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")

    response = client.get(chain_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_assignment_id": None,
    }


def test_scope_is_whole_machine_chain_not_event(client, incident):
    machine_id, event_id, incident_id = incident
    other_event = record_event(client, machine_id, resource="res/y")
    other_incident = create_incident(client, machine_id, other_event, summary="two")
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, other_event, other_incident, "bob", "reviewer")

    # Querying via either event audits the same whole-machine chain.
    assert client.get(chain_url(machine_id, event_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }
    assert client.get(chain_url(machine_id, other_event)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }


def test_missing_machine_returns_404(client):
    response = client.get(chain_url(MISSING_ID, MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_returns_404(client):
    machine_id = create_machine(client)

    response = client.get(chain_url(machine_id, MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_foreign_event_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)

    response = client.get(chain_url(machine_one, event_two))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_extra_query_param_returns_422_before_machine_lookup(client):
    response = client.get(chain_url(MISSING_ID, MISSING_ID), params={"from": "x"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_returns_422_before_machine_lookup(client):
    response = client.request(
        "GET", chain_url(MISSING_ID, MISSING_ID), content=b"{}"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_non_get_methods_return_405(client, incident):
    machine_id, event_id, _ = incident
    url = chain_url(machine_id, event_id)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405, method


def test_detects_tampered_content(client, incident, tmp_path):
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

    assert client.get(chain_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_detects_tampered_previous_link(client, incident, tmp_path):
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

    assert client.get(chain_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": second["id"],
    }


def test_detects_tampered_chain_hash(client, incident, tmp_path):
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

    assert client.get(chain_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_reports_first_broken_record_only(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")
        for n in range(3)
    ]

    connection = sqlite3.connect(tmp_path / "test.db")
    for record in records[1:]:
        connection.execute(
            "UPDATE incident_responsibility_assignments SET party = 'forged' "
            "WHERE id = ?",
            (record["id"],),
        )
    connection.commit()
    connection.close()

    assert client.get(chain_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": records[1]["id"],
    }


def test_damaged_created_at_counts_and_breaks_without_crash(
    client, incident, tmp_path
):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    second = assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET created_at = 'not-a-time' "
        "WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(chain_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": second["id"],
    }


def test_other_machines_damage_does_not_affect_result(client, tmp_path):
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

    assert client.get(chain_url(machine_one, event_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
    assert client.get(chain_url(machine_two, event_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_assignment_id": damaged["id"],
    }


def test_read_only_and_byte_identical(client, incident):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")
    listed_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    ).json()

    first = client.get(chain_url(machine_id, event_id))
    second = client.get(chain_url(machine_id, event_id))

    assert first.content == second.content
    assert first.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }
    listed_after = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    ).json()
    assert listed_after == listed_before


def test_conclusion_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident_id = create_incident(first, machine_id, event_id)
        assign(first, machine_id, event_id, incident_id, "alice", "owner")
        assign(first, machine_id, event_id, incident_id, "bob", "reviewer")
        conclusion = first.get(chain_url(machine_id, event_id)).json()

    with TestClient(app) as second:
        assert second.get(chain_url(machine_id, event_id)).json() == conclusion == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }


def test_existing_machine_level_integrity_unchanged(client, incident):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")

    assert client.get(
        f"/machines/{machine_id}/responsibility-assignments/integrity"
    ).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
