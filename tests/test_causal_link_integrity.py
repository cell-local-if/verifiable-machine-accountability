import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        test_client.db_path = tmp_path / "test.db"
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


def record_event(client, machine_id, resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201
    return response.json()


def create_link(client, machine_id, cause_event_id, effect_event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{cause_event_id}/causal-links",
        json={"effect_event_id": effect_event_id},
    )
    assert response.status_code == 201
    return response.json()


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "causal-links/integrity"
    )


def get_integrity(client, machine_id):
    return client.get(integrity_url(machine_id))


def insert_link_row(db_path, machine_id, cause_event_id, effect_event_id, created_at):
    """Insert a causal link directly, bypassing the API's validation."""
    link_id = str(uuid.uuid4())
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO authorization_decision_causal_links "
            "(id, machine_id, cause_event_id, effect_event_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (link_id, machine_id, cause_event_id, effect_event_id, created_at),
        )
    return link_id


def snapshot_links(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT id, machine_id, cause_event_id, effect_event_id, created_at "
            "FROM authorization_decision_causal_links ORDER BY created_at, id"
        ).fetchall()


def test_unknown_machine_returns_404(client):
    response = get_integrity(client, str(uuid.uuid4()))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_graph_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_link_id": None,
    }


def test_consistent_links_are_valid(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")
    create_link(client, machine_id, event_a["id"], event_b["id"])
    create_link(client, machine_id, event_b["id"], event_c["id"])

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_link_id": None,
    }


def test_other_machines_links_are_not_counted(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    event_a = record_event(client, machine_one, resource="res/a")
    event_b = record_event(client, machine_one, resource="res/b")
    other_a = record_event(client, machine_two, resource="res/a")
    other_b = record_event(client, machine_two, resource="res/b")
    create_link(client, machine_one, event_a["id"], event_b["id"])
    create_link(client, machine_two, other_a["id"], other_b["id"])
    # A dangling link on machine two must not affect machine one's audit.
    insert_link_row(
        client.db_path, machine_two, other_a["id"], str(uuid.uuid4()), "2026-01-03T00:00:00Z"
    )

    response = get_integrity(client, machine_one)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 1,
        "broken_link_id": None,
    }


def test_missing_effect_event_is_broken(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    create_link(client, machine_id, event_a["id"], event_b["id"])
    broken_id = insert_link_row(
        client.db_path, machine_id, event_b["id"], str(uuid.uuid4()),
        "2026-01-02T00:00:00Z",
    )

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_link_id": broken_id,
    }


def test_missing_cause_event_is_broken(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id)
    broken_id = insert_link_row(
        client.db_path, machine_id, str(uuid.uuid4()), event_a["id"],
        "2026-01-01T00:00:00Z",
    )

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": broken_id,
    }


def test_endpoint_from_another_machine_is_broken(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    event_a = record_event(client, machine_one, resource="res/a")
    foreign = record_event(client, machine_two, resource="res/b")
    broken_id = insert_link_row(
        client.db_path, machine_one, event_a["id"], foreign["id"],
        "2026-01-01T00:00:00Z",
    )

    response = get_integrity(client, machine_one)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": broken_id,
    }


def test_self_link_is_broken(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id)
    broken_id = insert_link_row(
        client.db_path, machine_id, event_a["id"], event_a["id"],
        "2026-01-01T00:00:00Z",
    )

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": broken_id,
    }


def test_cycle_reports_the_closing_link(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")
    insert_link_row(
        client.db_path, machine_id, event_a["id"], event_b["id"],
        "2026-01-01T00:00:00Z",
    )
    insert_link_row(
        client.db_path, machine_id, event_b["id"], event_c["id"],
        "2026-01-02T00:00:00Z",
    )
    closing_id = insert_link_row(
        client.db_path, machine_id, event_c["id"], event_a["id"],
        "2026-01-03T00:00:00Z",
    )

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 3,
        "broken_link_id": closing_id,
    }


def test_first_anomaly_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    first_broken = insert_link_row(
        client.db_path, machine_id, event_a["id"], str(uuid.uuid4()),
        "2026-01-01T00:00:00Z",
    )
    insert_link_row(
        client.db_path, machine_id, event_b["id"], event_b["id"],
        "2026-01-02T00:00:00Z",
    )

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_link_id": first_broken,
    }


def test_audit_is_read_only(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    insert_link_row(
        client.db_path, machine_id, event_a["id"], event_b["id"],
        "2026-01-01T00:00:00Z",
    )
    insert_link_row(
        client.db_path, machine_id, event_b["id"], event_a["id"],
        "2026-01-02T00:00:00Z",
    )
    before = snapshot_links(client.db_path)

    response = get_integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json()["valid"] is False
    # The broken data is left exactly as it was: nothing repaired or deleted.
    assert snapshot_links(client.db_path) == before


def test_result_survives_restart(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    insert_link_row(
        client.db_path, machine_id, event_a["id"], event_b["id"],
        "2026-01-01T00:00:00Z",
    )
    closing_id = insert_link_row(
        client.db_path, machine_id, event_b["id"], event_a["id"],
        "2026-01-02T00:00:00Z",
    )
    expected = {
        "valid": False,
        "checked_count": 2,
        "broken_link_id": closing_id,
    }
    assert get_integrity(client, machine_id).json() == expected

    # A fresh app instance over the same database file reports the same audit.
    with TestClient(app) as restarted:
        response = restarted.get(integrity_url(machine_id))

    assert response.status_code == 200
    assert response.json() == expected
