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


def incidents_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        "/incidents"
    )


def create_incident(
    client, machine_id, event_id, incident_type="breach", summary="something happened"
):
    response = client.post(
        incidents_url(machine_id, event_id),
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def transition_status(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"{incidents_url(machine_id, event_id)}/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def assign_responsibility(
    client, machine_id, event_id, incident_id, party="alice", role="owner"
):
    response = client.post(
        f"{incidents_url(machine_id, event_id)}/{incident_id}"
        "/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


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


def test_open_incident_without_history_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


def test_acknowledged_incident_with_exact_history_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


def test_resolved_incident_with_history_and_assignment_is_valid(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")
    transition_status(client, machine_id, event_id, incident["id"], "resolved")
    assign_responsibility(
        client, machine_id, event_id, incident["id"], party=" alice ", role="owner"
    )

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

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "DELETE FROM authorization_decision_events WHERE id = ?", (event_id,)
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_reference_to_event_of_other_machine_is_broken(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    foreign_event = record_event(client, machine_two)
    incident = create_incident(client, machine_one, event_one)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_incidents SET event_id = ? WHERE id = ?",
            (foreign_event, incident["id"]),
        )

    assert get_integrity(client, machine_one) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }
    # The foreign machine's own audit is untouched.
    assert get_integrity(client, machine_two) == {
        "valid": True,
        "checked_count": 0,
        "broken_incident_id": None,
    }


def test_open_incident_with_history_record_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
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


def test_acknowledged_incident_missing_history_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "DELETE FROM incident_status_events WHERE incident_id = ?",
            (incident["id"],),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_history_record_with_wrong_transition_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE incident_status_events SET from_status = 'resolved' "
            "WHERE incident_id = ?",
            (incident["id"],),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_history_record_with_foreign_machine_id_is_broken(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE incident_status_events SET machine_id = ? "
            "WHERE incident_id = ?",
            (other_machine, incident["id"]),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_unknown_status_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_incidents SET status = 'closed' "
            "WHERE id = ?",
            (incident["id"],),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_resolved_incident_without_assignment_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")
    transition_status(client, machine_id, event_id, incident["id"], "resolved")

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_assignment_with_blank_party_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    assignment = assign_responsibility(
        client, machine_id, event_id, incident["id"]
    )

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE incident_responsibility_assignments SET party = '   ' "
            "WHERE id = ?",
            (assignment["id"],),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_assignment_with_wrong_event_id_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id, resource="res/1")
    other_event = record_event(client, machine_id, resource="res/2")
    incident = create_incident(client, machine_id, event_id)
    assignment = assign_responsibility(
        client, machine_id, event_id, incident["id"]
    )

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE incident_responsibility_assignments SET event_id = ? "
            "WHERE id = ?",
            (other_event, assignment["id"]),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_duplicate_party_role_combination_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    assign_responsibility(
        client, machine_id, event_id, incident["id"], party="alice", role="owner"
    )

    # Same trimmed (party, role) pair, stored with different surrounding
    # whitespace so the database unique constraint does not reject it.
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "INSERT INTO incident_responsibility_assignments "
            "(id, machine_id, event_id, incident_id, party, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                machine_id,
                event_id,
                incident["id"],
                " alice",
                "owner ",
                "2026-01-01T00:00:00Z",
            ),
        )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_first_broken_incident_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = create_incident(
        client, machine_id, event_id, summary="first", incident_type="t1"
    )
    second = create_incident(
        client, machine_id, event_id, summary="second", incident_type="t2"
    )
    good = create_incident(
        client, machine_id, event_id, summary="third", incident_type="t3"
    )

    # Both earlier incidents are damaged; the earliest in (created_at, id)
    # order must be the one reported.
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_incidents SET status = 'closed' "
            "WHERE id IN (?, ?)",
            (first["id"], second["id"]),
        )

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_incident_id": first["id"],
    }
    assert result["broken_incident_id"] != good["id"]


def test_other_machine_damage_does_not_fail_this_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    create_incident(client, machine_one, event_one)
    damaged = create_incident(client, machine_two, event_two)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_incidents SET status = 'closed' "
            "WHERE id = ?",
            (damaged["id"],),
        )

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }
    assert get_integrity(client, machine_two)["valid"] is False


def test_integrity_check_writes_nothing_and_is_repeatable(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition_status(client, machine_id, event_id, incident["id"], "acknowledged")

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE incident_status_events SET from_status = 'resolved' "
            "WHERE incident_id = ?",
            (incident["id"],),
        )

    incidents_before = client.get(incidents_url(machine_id, event_id)).json()
    history_before = client.get(
        f"{incidents_url(machine_id, event_id)}/{incident['id']}/status-history"
    ).json()

    first = get_integrity(client, machine_id)
    assert first == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }
    assert get_integrity(client, machine_id) == first

    # Nothing was repaired or deleted: the incident and its damaged history
    # record are still stored exactly as they were.
    assert (
        client.get(incidents_url(machine_id, event_id)).json() == incidents_before
    )
    assert (
        client.get(
            f"{incidents_url(machine_id, event_id)}/{incident['id']}/status-history"
        ).json()
        == history_before
    )
    with sqlite3.connect(db_path_of(client)) as conn:
        stored_from = conn.execute(
            "SELECT from_status FROM incident_status_events WHERE incident_id = ?",
            (incident["id"],),
        ).fetchone()[0]
    assert stored_from == "resolved"


def test_integrity_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        transition_status(
            first, machine_id, event_id, incident["id"], "acknowledged"
        )
        transition_status(first, machine_id, event_id, incident["id"], "resolved")
        assign_responsibility(first, machine_id, event_id, incident["id"])
        expected_good = get_integrity(first, machine_id)
        assert expected_good == {
            "valid": True,
            "checked_count": 1,
            "broken_incident_id": None,
        }

        # Damage the stored history after the first process checked it; a
        # restarted process reads the same stored bytes and reports the damage.
        with sqlite3.connect(db_path_of(first)) as conn:
            conn.execute(
                "DELETE FROM incident_status_events WHERE incident_id = ?",
                (incident["id"],),
            )
        expected_bad = get_integrity(first, machine_id)
        assert expected_bad == {
            "valid": False,
            "checked_count": 1,
            "broken_incident_id": incident["id"],
        }

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == expected_bad
