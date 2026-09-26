import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

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


def assign(client, machine_id, event_id, incident_id, party, role):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


def check_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        "/responsibility-assignments/integrity"
    )


@pytest.fixture
def machine_event(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    return machine_id, event_id


def test_empty_chain_is_valid(client, machine_event):
    machine_id, event_id = machine_event

    response = client.get(check_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_assignment_id": None,
    }
    assert response.content == (
        b'{"valid":true,"checked_count":0,"broken_assignment_id":null}\n'
    )


def test_complete_chain_is_valid(client, machine_event):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    for n in range(3):
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")

    response = client.get(check_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_assignment_id": None,
    }


def test_scope_is_machine_chain_event_only_confirms_context(client, machine_event):
    machine_id, event_id = machine_event
    other_event = record_event(client, machine_id, resource="res/y")
    incident_one = create_incident(client, machine_id, event_id, summary="one")
    incident_two = create_incident(client, machine_id, other_event, summary="two")
    assign(client, machine_id, event_id, incident_one, "alice", "owner")
    assign(client, machine_id, other_event, incident_two, "bob", "reviewer")

    # The event id does not filter the chain: both events see the machine's
    # complete two-record chain and the identical conclusion.
    for path_event in (event_id, other_event):
        assert client.get(check_url(machine_id, path_event)).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }


def test_extra_query_param_is_422_before_machine_lookup(client):
    response = client.get(check_url(MISSING_ID, MISSING_ID) + "?verbose=true")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_is_422_before_machine_lookup(client):
    response = client.request(
        "GET", check_url(MISSING_ID, MISSING_ID), content=b'{"x": 1}'
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_non_get_methods_return_405(client, machine_event):
    machine_id, event_id = machine_event

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(check_url(machine_id, event_id))
        assert response.status_code == 405


def test_missing_machine_returns_404(client):
    response = client.get(check_url(MISSING_ID, MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_returns_404(client, machine_event):
    machine_id, _ = machine_event

    response = client.get(check_url(machine_id, MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_event_of_another_machine_returns_404(client, machine_event):
    machine_id, _ = machine_event
    other_machine = create_machine(client, external_id="machine-2")
    other_event = record_event(client, other_machine)

    response = client.get(check_url(machine_id, other_event))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_detects_tampered_content(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_detects_tampered_previous_link(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
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

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": second["id"],
    }


def test_detects_tampered_chain_hash(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_assignment_id": first["id"],
    }


def test_first_broken_record_wins(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")
        for n in range(3)
    ]

    connection = sqlite3.connect(tmp_path / "test.db")
    for record in records[1:]:
        connection.execute(
            "UPDATE incident_responsibility_assignments "
            "SET party = 'forged' WHERE id = ?",
            (record["id"],),
        )
    connection.commit()
    connection.close()

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": records[1]["id"],
    }


def test_other_machines_damage_does_not_leak(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    assign(client, machine_id, event_id, incident_id, "alice", "owner")

    other_machine = create_machine(client, external_id="machine-2")
    other_event = record_event(client, other_machine)
    other_incident = create_incident(client, other_machine, other_event)
    damaged = assign(client, other_machine, other_event, other_incident, "bob", "owner")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (damaged["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
    assert client.get(check_url(other_machine, other_event)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_assignment_id": damaged["id"],
    }


def test_damaged_created_at_does_not_crash(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    record = assign(client, machine_id, event_id, incident_id, "alice", "owner")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET created_at = 'garbage' "
        "WHERE id = ?",
        (record["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(check_url(machine_id, event_id))

    # The damaged stamp is neither repaired nor recomputed: the record no
    # longer matches its stored digests and is reported, without a crash.
    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_assignment_id": record["id"],
    }


def test_check_order_uses_actual_utc_instant(client, machine_event, tmp_path):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    first = assign(client, machine_id, event_id, incident_id, "alice", "owner")
    second = assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    # Rewrite the stamps so text ordering and instant ordering disagree:
    # "…:00.5Z" sorts before "…:00Z" lexicographically ('.' < 'Z') but the
    # exact-second stamp is the earlier instant. Rebuild the chain so it is
    # sound only when checked in (instant, id) order: the exact-second record
    # is the head.
    first["created_at"] = "2026-01-01T00:00:00.500000Z"
    second["created_at"] = "2026-01-01T00:00:00Z"
    second["previous_assignment_id"] = None
    second["content_hash"] = canonical_content_hash(second)
    second["chain_hash"] = chain_hash("", second["content_hash"])
    first["previous_assignment_id"] = second["id"]
    first["content_hash"] = canonical_content_hash(first)
    first["chain_hash"] = chain_hash(second["chain_hash"], first["content_hash"])

    connection = sqlite3.connect(tmp_path / "test.db")
    for record in (first, second):
        connection.execute(
            "UPDATE incident_responsibility_assignments SET created_at = ?, "
            "previous_assignment_id = ?, content_hash = ?, chain_hash = ? "
            "WHERE id = ?",
            (
                record["created_at"],
                record["previous_assignment_id"],
                record["content_hash"],
                record["chain_hash"],
                record["id"],
            ),
        )
    connection.commit()
    connection.close()

    assert client.get(check_url(machine_id, event_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }


def test_read_only_and_byte_identical(client, machine_event):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")
    listed_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    ).json()

    first = client.get(check_url(machine_id, event_id))
    second = client.get(check_url(machine_id, event_id))

    assert first.content == second.content
    assert first.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }
    # The verification wrote nothing: the stored records are untouched.
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/{event_id}"
            f"/incidents/{incident_id}/responsibility-assignments"
        ).json()
        == listed_before
    )


def test_conclusion_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident_id = create_incident(first, machine_id, event_id)
        assign(first, machine_id, event_id, incident_id, "alice", "owner")
        assign(first, machine_id, event_id, incident_id, "bob", "reviewer")
        conclusion = first.get(check_url(machine_id, event_id)).content

    with TestClient(app) as second:
        assert second.get(check_url(machine_id, event_id)).content == conclusion
        assert json.loads(conclusion) == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }


def test_read_failure_returns_500_without_partial_result(
    client, machine_event, tmp_path
):
    machine_id, event_id = machine_event
    incident_id = create_incident(client, machine_id, event_id)
    assign(client, machine_id, event_id, incident_id, "alice", "owner")

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute("DROP TABLE incident_responsibility_assignments")
    connection.commit()
    connection.close()

    response = client.get(check_url(machine_id, event_id))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
