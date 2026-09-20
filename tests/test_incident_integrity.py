import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


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


def create_incident(
    client, machine_id, event_id, incident_type="breach", summary="something happened"
):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def status_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status"
    )


def assignment_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    )


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "incidents/integrity"
    )


def get_integrity(client, machine_id):
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_path_of(client):
    return client.app.state.engine.url.database


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        status_url(machine_id, event_id, incident_id), json={"status": status}
    )
    assert response.status_code == 200
    return response.json()


def assign(client, machine_id, event_id, incident_id, party, role):
    response = client.post(
        assignment_url(machine_id, event_id, incident_id),
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


def execute_db(client, sql, params=()):
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(sql, params)


def test_missing_machine_returns_404(client):
    response = client.get(integrity_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_no_incidents_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 0,
        "broken_incident_id": None,
    }


def test_open_incident_without_assignments_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


def test_full_valid_lifecycle_with_assignment_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


def test_only_counts_incidents_of_path_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    create_incident(client, machine_two, event_two)

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 0,
        "broken_incident_id": None,
    }
    assert get_integrity(client, machine_two)["checked_count"] == 1


def test_dangling_event_reference_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    execute_db(
        client,
        "DELETE FROM authorization_decision_events WHERE id = ?",
        (event_id,),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_reference_to_event_of_other_machine_is_broken(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_event = record_event(client, machine_two)

    broken_id = str(uuid.uuid4())
    execute_db(
        client,
        "INSERT INTO authorization_decision_incidents "
        "(id, machine_id, event_id, incident_type, summary, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            broken_id,
            machine_one,
            foreign_event,
            "breach",
            "note",
            "open",
            "2026-01-01T00:00:00Z",
        ),
    )

    assert get_integrity(client, machine_one) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": broken_id,
    }
    # The foreign machine's own audit is untouched.
    assert get_integrity(client, machine_two) == {
        "valid": True,
        "checked_count": 0,
        "broken_incident_id": None,
    }


def test_open_incident_with_a_history_record_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    # Fabricate a stray open -> acknowledged history row without changing the
    # incident's stored status.
    execute_db(
        client,
        "INSERT INTO incident_status_events "
        "(id, machine_id, event_id, incident_id, from_status, to_status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            machine_id,
            event_id,
            incident["id"],
            "open",
            "acknowledged",
            "2026-01-01T00:00:00Z",
        ),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_acknowledged_incident_without_history_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    # Drop the history row while leaving the incident acknowledged.
    execute_db(
        client,
        "DELETE FROM incident_status_events WHERE incident_id = ?",
        (incident["id"],),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_acknowledged_incident_with_wrong_history_edge_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    execute_db(
        client,
        "UPDATE incident_status_events SET from_status = ?, to_status = ? "
        "WHERE incident_id = ?",
        ("acknowledged", "resolved", incident["id"]),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_resolved_incident_missing_one_history_step_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    # Remove the second (acknowledged -> resolved) history row.
    execute_db(
        client,
        "DELETE FROM incident_status_events WHERE to_status = 'resolved'"
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_history_entry_with_wrong_machine_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    execute_db(
        client,
        "UPDATE incident_status_events SET machine_id = ? WHERE incident_id = ?",
        (str(uuid.uuid4()), incident["id"]),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_history_entry_with_wrong_event_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    other_event = record_event(client, machine_id, resource="res/other")
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    execute_db(
        client,
        "UPDATE incident_status_events SET event_id = ? WHERE incident_id = ?",
        (other_event, incident["id"]),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_history_entry_with_wrong_incident_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    execute_db(
        client,
        "UPDATE incident_status_events SET incident_id = ? WHERE incident_id = ?",
        (str(uuid.uuid4()), incident["id"]),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_assignment_with_wrong_machine_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET machine_id = ? "
        "WHERE incident_id = ?",
        (str(uuid.uuid4()), incident["id"]),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_assignment_with_wrong_event_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    other_event = record_event(client, machine_id, resource="res/other")
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET event_id = ? "
        "WHERE incident_id = ?",
        (other_event, incident["id"]),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_assignment_with_wrong_incident_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET incident_id = ? "
        "WHERE incident_id = ?",
        (str(uuid.uuid4()), incident["id"]),
    )

    # With its only assignment detached, the resolved incident also has no
    # valid responsibility record.
    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_blank_party_assignment_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET party = '   ' "
        "WHERE incident_id = ?",
        (incident["id"],),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_blank_role_assignment_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET role = '\t\n' "
        "WHERE incident_id = ?",
        (incident["id"],),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_duplicate_party_role_pair_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    # The API rejects an exact duplicate, so fabricate a second row whose
    # stored values differ only by surrounding whitespace: after trimming they
    # are the same (party, role) pair.
    execute_db(
        client,
        "INSERT INTO incident_responsibility_assignments "
        "(id, machine_id, event_id, incident_id, party, role, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            machine_id,
            event_id,
            incident["id"],
            "  alice\t",
            "owner\n",
            "2026-01-01T00:00:00Z",
        ),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_resolved_incident_without_any_assignment_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_resolved_incident_with_only_invalid_assignment_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    assign(client, machine_id, event_id, incident["id"], "alice", "owner")

    # Corrupt the sole assignment, so the resolved incident has no sound
    # responsibility record left.
    execute_db(
        client,
        "UPDATE incident_responsibility_assignments SET party = '' "
        "WHERE incident_id = ?",
        (incident["id"],),
    )

    assert get_integrity(client, machine_id)["broken_incident_id"] == incident["id"]


def test_acknowledged_incident_needs_no_assignment(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


def test_first_broken_incident_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    good = create_incident(client, machine_id, event_one, summary="good")

    first_broken = create_incident(
        client, machine_id, event_two, summary="broken-early"
    )
    # Force the first broken incident to sort before the good one.
    execute_db(
        client,
        "UPDATE authorization_decision_incidents SET created_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00Z", first_broken["id"]),
    )
    execute_db(
        client,
        "DELETE FROM authorization_decision_events WHERE id = ?",
        (event_two,),
    )

    second_broken = create_incident(
        client, machine_id, event_one, summary="broken-late"
    )
    execute_db(
        client,
        "UPDATE authorization_decision_incidents SET created_at = ? WHERE id = ?",
        ("2020-01-02T00:00:00Z", second_broken["id"]),
    )
    transition(client, machine_id, event_one, second_broken["id"], "acknowledged")
    transition(client, machine_id, event_one, second_broken["id"], "resolved")
    # No responsibility assignment -> the resolved closure is open.

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_incident_id": first_broken["id"],
    }
    assert result["broken_incident_id"] != good["id"]


def test_other_machine_damage_does_not_fail_this_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    incident_one = create_incident(client, machine_one, event_one)
    incident_two = create_incident(client, machine_two, event_two)

    # Resolve machine two's incident without any responsibility assignment.
    transition(client, machine_two, event_two, incident_two["id"], "acknowledged")
    transition(client, machine_two, event_two, incident_two["id"], "resolved")

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }
    assert get_integrity(client, machine_two)["valid"] is False
    assert (
        get_integrity(client, machine_two)["broken_incident_id"]
        == incident_two["id"]
    )


def test_integrity_check_writes_nothing_and_is_repeatable(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    # Resolved but never assigned -> broken.

    incidents_url = (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"
    )
    incidents_before = client.get(incidents_url).json()
    history_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident['id']}/status-history"
    ).json()
    assignments_before = client.get(
        assignment_url(machine_id, event_id, incident["id"])
    ).json()

    first = get_integrity(client, machine_id)
    assert first == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }
    assert get_integrity(client, machine_id) == first

    assert client.get(incidents_url).json() == incidents_before
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/{event_id}"
            f"/incidents/{incident['id']}/status-history"
        ).json()
        == history_before
    )
    assert (
        client.get(
            assignment_url(machine_id, event_id, incident["id"])
        ).json()
        == assignments_before
        == []
    )


def test_integrity_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        transition(first, machine_id, event_id, incident["id"], "acknowledged")
        assert get_integrity(first, machine_id) == {
            "valid": True,
            "checked_count": 1,
            "broken_incident_id": None,
        }

        # Resolve without a responsibility assignment; the reopened process
        # reads the same stored rows and reports the broken closure.
        transition(first, machine_id, event_id, incident["id"], "resolved")
        expected_bad = get_integrity(first, machine_id)
        assert expected_bad == {
            "valid": False,
            "checked_count": 1,
            "broken_incident_id": incident["id"],
        }

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == expected_bad
