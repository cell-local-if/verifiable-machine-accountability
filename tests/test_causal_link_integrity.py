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
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_path_of(client):
    return client.app.state.engine.url.database


def insert_link_row(db_path, *, link_id, machine_id, cause_event_id,
                    effect_event_id, created_at):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO authorization_decision_causal_links "
            "(id, machine_id, cause_event_id, effect_event_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (link_id, machine_id, cause_event_id, effect_event_id, created_at),
        )


def delete_event_row(db_path, event_id):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM authorization_decision_events WHERE id = ?",
            (event_id,),
        )


def test_missing_machine_returns_404(client):
    response = client.get(
        integrity_url("00000000-0000-0000-0000-000000000000")
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_graph_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_integrity(client, machine_id) == {
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
    create_link(client, machine_id, event_a["id"], event_c["id"])

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 3,
        "broken_link_id": None,
    }


def test_only_counts_links_of_path_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    cause = record_event(client, machine_two, resource="res/a")
    effect = record_event(client, machine_two, resource="res/b")
    create_link(client, machine_two, cause["id"], effect["id"])

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 0,
        "broken_link_id": None,
    }
    assert get_integrity(client, machine_two)["checked_count"] == 1


def test_missing_effect_event_is_broken(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, cause["id"], effect["id"])

    delete_event_row(db_path_of(client), effect["id"])

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": link["id"],
    }


def test_missing_cause_event_is_broken(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, cause["id"], effect["id"])

    delete_event_row(db_path_of(client), cause["id"])

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": link["id"],
    }


def test_endpoint_owned_by_other_machine_is_broken(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    cause = record_event(client, machine_one, resource="res/a")
    foreign = record_event(client, machine_two, resource="res/b")

    link_id = str(uuid.uuid4())
    insert_link_row(
        db_path_of(client),
        link_id=link_id,
        machine_id=machine_one,
        cause_event_id=cause["id"],
        effect_event_id=foreign["id"],
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_one) == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": link_id,
    }
    # The foreign machine's own graph is untouched.
    assert get_integrity(client, machine_two) == {
        "valid": True,
        "checked_count": 0,
        "broken_link_id": None,
    }


def test_self_link_is_broken(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    link_id = str(uuid.uuid4())
    insert_link_row(
        db_path_of(client),
        link_id=link_id,
        machine_id=machine_id,
        cause_event_id=event["id"],
        effect_event_id=event["id"],
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_link_id": link_id,
    }


def test_directed_cycle_is_broken_at_closing_edge(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")
    create_link(client, machine_id, event_a["id"], event_b["id"])
    create_link(client, machine_id, event_b["id"], event_c["id"])

    closing_id = str(uuid.uuid4())
    insert_link_row(
        db_path_of(client),
        link_id=closing_id,
        machine_id=machine_id,
        cause_event_id=event_c["id"],
        effect_event_id=event_a["id"],
        created_at="2999-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 3,
        "broken_link_id": closing_id,
    }


def test_cycle_broken_link_follows_scan_order(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")

    # Insert c -> a with the earliest timestamp so it is scanned first; the
    # cycle is then closed by the later b -> c edge in scan order. All three
    # edges are written directly because the create endpoint rejects cycles.
    db_path = db_path_of(client)
    early_id = str(uuid.uuid4())
    insert_link_row(
        db_path,
        link_id=early_id,
        machine_id=machine_id,
        cause_event_id=event_c["id"],
        effect_event_id=event_a["id"],
        created_at="2020-01-01T00:00:00Z",
    )
    middle_id = str(uuid.uuid4())
    insert_link_row(
        db_path,
        link_id=middle_id,
        machine_id=machine_id,
        cause_event_id=event_a["id"],
        effect_event_id=event_b["id"],
        created_at="2020-01-02T00:00:00Z",
    )
    closing_id = str(uuid.uuid4())
    insert_link_row(
        db_path,
        link_id=closing_id,
        machine_id=machine_id,
        cause_event_id=event_b["id"],
        effect_event_id=event_c["id"],
        created_at="2020-01-03T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 3,
        "broken_link_id": closing_id,
    }


def test_first_broken_link_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    good = create_link(client, machine_id, event_a["id"], event_b["id"])

    first_broken_id = str(uuid.uuid4())
    insert_link_row(
        db_path_of(client),
        link_id=first_broken_id,
        machine_id=machine_id,
        cause_event_id=event_a["id"],
        effect_event_id="00000000-0000-0000-0000-000000000000",
        created_at="2026-01-01T00:00:00Z",
    )
    second_broken_id = str(uuid.uuid4())
    insert_link_row(
        db_path_of(client),
        link_id=second_broken_id,
        machine_id=machine_id,
        cause_event_id=event_b["id"],
        effect_event_id="00000000-0000-0000-0000-000000000001",
        created_at="2026-01-02T00:00:00Z",
    )

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_link_id": first_broken_id,
    }
    assert result["broken_link_id"] != good["id"]


def test_integrity_check_writes_nothing(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    create_link(client, machine_id, event_a["id"], event_b["id"])
    insert_link_row(
        db_path_of(client),
        link_id=str(uuid.uuid4()),
        machine_id=machine_id,
        cause_event_id=event_b["id"],
        effect_event_id="00000000-0000-0000-0000-000000000000",
        created_at="2999-01-01T00:00:00Z",
    )

    links_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_a['id']}/causal-links"
    ).json()
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    chain_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    first = get_integrity(client, machine_id)
    assert first["valid"] is False
    second = get_integrity(client, machine_id)
    assert second == first

    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event_a['id']}/causal-links"
        ).json()
        == links_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        == events_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).json()
        == chain_before
    )


def test_integrity_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_a = record_event(first, machine_id, resource="res/a")
        event_b = record_event(first, machine_id, resource="res/b")
        create_link(first, machine_id, event_a["id"], event_b["id"])
        broken_id = str(uuid.uuid4())
        insert_link_row(
            db_path_of(first),
            link_id=broken_id,
            machine_id=machine_id,
            cause_event_id=event_b["id"],
            effect_event_id=event_a["id"],
            created_at="2999-01-01T00:00:00Z",
        )
        expected = get_integrity(first, machine_id)
        assert expected == {
            "valid": False,
            "checked_count": 2,
            "broken_link_id": broken_id,
        }

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == expected
