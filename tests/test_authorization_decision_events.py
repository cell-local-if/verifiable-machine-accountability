import re
import uuid

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)


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


def create_rule(
    client,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=0,
):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )
    assert response.status_code == 201
    return response


def declare(client, machine_id, action_type="read", resource_pattern="res/*", enabled=True):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def evaluate(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": action_type, "resource": resource},
    )


def test_create_event_allowed_returns_201_with_full_record(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    response = record_event(client, machine_id)

    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["action_type"] == "read"
    assert body["resource"] == "res/x"
    assert body["allowed"] is True
    assert body["reason"] == "allowed_by_policy"
    assert RFC3339_Z_RE.match(body["created_at"])


def test_create_event_denied_records_denial(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="deny", priority=0)
    create_rule(client, effect="allow", priority=5)

    response = record_event(client, machine_id)

    assert response.status_code == 201
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == "denied_by_policy"


@pytest.mark.parametrize(
    "expected_reason",
    ["no_enabled_declaration", "no_matching_policy"],
)
def test_create_event_records_every_reason(client, expected_reason):
    machine_id = create_machine(client)
    if expected_reason == "no_enabled_declaration":
        declare(client, machine_id, enabled=False)
        create_rule(client)
    else:
        declare(client, machine_id)

    response = record_event(client, machine_id)

    assert response.status_code == 201
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == expected_reason


def test_create_event_strips_whitespace_and_uses_stripped_values(client):
    machine_id = create_machine(client)
    declare(client, machine_id, action_type="read", resource_pattern="res/*")
    create_rule(client)

    response = record_event(client, machine_id, action_type="  read\t", resource=" res/x ")

    assert response.status_code == 201
    body = response.json()
    assert body["action_type"] == "read"
    assert body["resource"] == "res/x"
    assert body["allowed"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource": "r"},
        {"action_type": "a", "resource": "\t"},
        {"action_type": "   ", "resource": "   "},
        {"action_type": "a"},
        {"resource": "r"},
        {},
        {"action_type": 1, "resource": "r"},
        {"action_type": "a", "resource": 2},
        {"action_type": None, "resource": "r"},
    ],
)
def test_create_event_rejects_invalid_payload_with_422(client, payload):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events", json=payload
    )

    assert response.status_code == 422


def test_create_event_missing_machine_returns_404(client):
    response = record_event(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_event_invalid_payload_even_for_missing_machine_is_422(client):
    # Body validation happens before the handler; malformed input is 422
    # regardless of whether the machine exists.
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/authorization-decision-events",
        json={"action_type": "  ", "resource": "r"},
    )

    assert response.status_code == 422


def test_list_events_empty_returns_empty_list(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    )

    assert response.status_code == 200
    assert response.json() == []


def test_list_events_returns_events_for_machine_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    recorded = []
    for resource in ["res/a", "res/b", "res/c"]:
        body = record_event(client, machine_id, resource=resource).json()
        recorded.append(body)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    )

    assert response.status_code == 200
    events = response.json()
    expected = sorted(recorded, key=lambda e: (e["created_at"], e["id"]))
    assert [e["id"] for e in events] == [e["id"] for e in expected]
    assert [e["resource"] for e in events] == [e["resource"] for e in expected]
    ordering_key = [(e["created_at"], e["id"]) for e in events]
    assert ordering_key == sorted(ordering_key)
    for event in events:
        assert event["machine_id"] == machine_id
        assert set(event.keys()) == {
            "id",
            "machine_id",
            "action_type",
            "resource",
            "allowed",
            "reason",
            "created_at",
        }


def test_list_events_is_scoped_per_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    declare(client, machine_one)
    declare(client, machine_two)
    create_rule(client)

    event_one = record_event(client, machine_one, resource="res/one").json()
    record_event(client, machine_two, resource="res/two")

    response = client.get(
        f"/machines/{machine_one}/authorization-decision-events"
    )

    assert response.status_code == 200
    events = response.json()
    assert [e["id"] for e in events] == [event_one["id"]]
    assert events[0]["resource"] == "res/one"


def test_list_events_missing_machine_returns_404(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/authorization-decision-events"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_evaluation_endpoint_does_not_create_audit_events(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    evaluate(client, machine_id)
    evaluate(client, machine_id)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    )

    assert response.status_code == 200
    assert response.json() == []


def test_events_reflect_request_inputs_and_result_at_time(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)

    allowed_body = record_event(client, machine_id, resource="res/a").json()
    assert allowed_body["allowed"] is True

    # A newly added same-priority deny for the exact resource makes deny win
    # on subsequent decisions, but must not alter the already-stored event.
    create_rule(client, resource_pattern="res/a", effect="deny", priority=0)
    denied_body = record_event(client, machine_id, resource="res/a").json()
    assert denied_body["allowed"] is False

    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert events[0]["id"] == allowed_body["id"]
    assert events[0]["allowed"] is True
    assert events[0]["reason"] == "allowed_by_policy"
    assert events[1]["id"] == denied_body["id"]
    assert events[1]["allowed"] is False
    assert events[1]["reason"] == "denied_by_policy"


def test_events_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id, resource="res/persist").json()

    with TestClient(app) as second:
        response = second.get(
            f"/machines/{machine_id}/authorization-decision-events"
        )

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0] == event
    assert events[0]["resource"] == "res/persist"


def test_each_event_has_unique_id(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    ids = {
        record_event(client, machine_id, resource=f"res/{i}").json()["id"]
        for i in range(3)
    }

    assert len(ids) == 3
    for event_id in ids:
        uuid.UUID(event_id)
