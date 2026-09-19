import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HASH_A = "a" * 64
HASH_B = "b" * 64

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


def evidence_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/evidence"
    )


def create_evidence(client, machine_id, event_id, evidence_type="log",
                    content_hash=HASH_A):
    response = client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )
    assert response.status_code == 201
    return response.json()


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "evidence/integrity"
    )


def get_integrity(client, machine_id):
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_path_of(client):
    return client.app.state.engine.url.database


def insert_evidence_row(db_path, *, evidence_id, machine_id, event_id,
                        evidence_type, content_hash, created_at):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO authorization_decision_evidence "
            "(id, machine_id, event_id, evidence_type, content_hash, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (evidence_id, machine_id, event_id, evidence_type, content_hash,
             created_at),
        )


def delete_event_row(db_path, event_id):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM authorization_decision_events WHERE id = ?",
            (event_id,),
        )


def test_missing_machine_returns_404(client):
    response = client.get(integrity_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_no_evidence_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


def test_consistent_evidence_is_valid(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    create_evidence(client, machine_id, event_a, content_hash=HASH_A)
    create_evidence(client, machine_id, event_a, content_hash=HASH_B)
    create_evidence(client, machine_id, event_b, evidence_type="signature",
                    content_hash="0123456789abcdef" * 4)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 3,
        "broken_evidence_id": None,
    }


def test_only_counts_evidence_of_path_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    create_evidence(client, machine_two, event_two)

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }
    assert get_integrity(client, machine_two)["checked_count"] == 1


def test_broken_evidence_of_other_machine_does_not_fail(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    create_evidence(client, machine_one, event_one)

    # A corrupted record owned by another machine must not affect this
    # machine's result.
    insert_evidence_row(
        db_path_of(client),
        evidence_id=str(uuid.uuid4()),
        machine_id=machine_two,
        event_id=MISSING_ID,
        evidence_type="log",
        content_hash=HASH_A,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }
    assert get_integrity(client, machine_two)["valid"] is False


def test_missing_event_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    evidence = create_evidence(client, machine_id, event_id)

    delete_event_row(db_path_of(client), event_id)

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence["id"],
    }


def test_event_owned_by_other_machine_is_broken(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_event = record_event(client, machine_two)

    evidence_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=evidence_id,
        machine_id=machine_one,
        event_id=foreign_event,
        evidence_type="log",
        content_hash=HASH_A,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_one) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence_id,
    }
    # The other machine's own evidence view is untouched.
    assert get_integrity(client, machine_two) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


def test_blank_evidence_type_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    evidence_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=evidence_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type="   ",
        content_hash=HASH_A,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence_id,
    }


def test_uppercase_content_hash_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    evidence_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=evidence_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type="log",
        content_hash="A" * 64,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence_id,
    }


def test_wrong_length_content_hash_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    evidence_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=evidence_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type="log",
        content_hash="a" * 63,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence_id,
    }


def test_first_broken_record_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    good = create_evidence(client, machine_id, event_id, content_hash=HASH_A)

    db_path = db_path_of(client)
    first_broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path,
        evidence_id=first_broken_id,
        machine_id=machine_id,
        event_id=MISSING_ID,
        evidence_type="log",
        content_hash=HASH_B,
        created_at="2026-01-01T00:00:00Z",
    )
    second_broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path,
        evidence_id=second_broken_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type="",
        content_hash=HASH_B,
        created_at="2026-01-02T00:00:00Z",
    )

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": first_broken_id,
    }
    assert result["broken_evidence_id"] != good["id"]


def test_integrity_check_writes_nothing(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_evidence(client, machine_id, event_id, content_hash=HASH_A)
    insert_evidence_row(
        db_path_of(client),
        evidence_id=str(uuid.uuid4()),
        machine_id=machine_id,
        event_id=MISSING_ID,
        evidence_type="log",
        content_hash=HASH_B,
        created_at="2999-01-01T00:00:00Z",
    )

    evidence_before = client.get(evidence_url(machine_id, event_id)).json()
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

    assert client.get(evidence_url(machine_id, event_id)).json() == evidence_before
    assert (
        client.get(f"/machines/{machine_id}/authorization-decision-events").json()
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
        event_id = record_event(first, machine_id)
        create_evidence(first, machine_id, event_id, content_hash=HASH_A)
        broken_id = str(uuid.uuid4())
        insert_evidence_row(
            db_path_of(first),
            evidence_id=broken_id,
            machine_id=machine_id,
            event_id=event_id,
            evidence_type="log",
            content_hash="F" * 64,
            created_at="2999-01-01T00:00:00Z",
        )
        expected = get_integrity(first, machine_id)
        assert expected == {
            "valid": False,
            "checked_count": 2,
            "broken_evidence_id": broken_id,
        }

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == expected
