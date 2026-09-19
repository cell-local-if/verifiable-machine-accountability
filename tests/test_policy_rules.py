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


def create_rule(
    client,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=0,
):
    return client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )


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


def evaluate(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": action_type, "resource": resource},
    )


def test_create_policy_rule_returns_201_with_full_record(client):
    response = create_rule(client, priority=3)

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "res/*"
    assert body["effect"] == "allow"
    assert body["priority"] == 3
    assert body["created_at"].endswith("Z")
    assert body["updated_at"].endswith("Z")
    assert body["created_at"] == body["updated_at"]


def test_create_policy_rule_strips_whitespace(client):
    response = create_rule(client, action_type="  read \t", resource_pattern=" res/* ")

    assert response.status_code == 201
    body = response.json()
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "res/*"


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource_pattern": "r", "effect": "allow", "priority": 0},
        {"action_type": "a", "resource_pattern": "", "effect": "allow", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "maybe", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "ALLOW", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": -1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": True},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": 1.5},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": "0"},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow"},
        {"action_type": "a", "resource_pattern": "r", "priority": 0},
    ],
)
def test_create_policy_rule_rejects_invalid_payload(client, payload):
    response = client.post("/policy-rules", json=payload)

    assert response.status_code == 422


def test_duplicate_policy_rule_returns_409_and_adds_nothing(client):
    assert create_rule(client, priority=1).status_code == 201

    response = create_rule(client, effect="deny", priority=1)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_policy_rule"}}


def test_same_action_resource_with_different_priority_allowed(client):
    assert create_rule(client, priority=1).status_code == 201
    assert create_rule(client, priority=2).status_code == 201


def test_policy_rules_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        assert create_rule(first).status_code == 201

    with TestClient(app) as second:
        response = evaluate(second, machine_id)

    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_evaluation_missing_machine_returns_404(client):
    response = evaluate(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource": "r"},
        {"action_type": "a", "resource": "\t"},
        {"action_type": "a"},
        {"resource": "r"},
        {"action_type": 1, "resource": "r"},
    ],
)
def test_evaluation_rejects_invalid_payload(client, payload):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-evaluations", json=payload
    )

    assert response.status_code == 422


def test_evaluation_no_enabled_declaration(client):
    machine_id = create_machine(client)
    declare(client, machine_id, enabled=False)
    assert create_rule(client).status_code == 201

    response = evaluate(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_declaration_pattern_must_match_resource(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/a/*")
    assert create_rule(client).status_code == 201

    response = evaluate(client, machine_id, resource="res/b/x")

    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_evaluation_no_matching_policy(client):
    machine_id = create_machine(client)
    declare(client, machine_id)

    response = evaluate(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_policy_action_type_must_match(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    assert create_rule(client, action_type="write").status_code == 201

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_policy_pattern_literal_except_star(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/x")
    assert create_rule(client, resource_pattern="res.?").status_code == 201

    response = evaluate(client, machine_id, resource="res/x")

    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_star_matches_any_string(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="*")
    assert create_rule(client, resource_pattern="res/*/x").status_code == 201

    response = evaluate(client, machine_id, resource="res/a/b/x")

    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_lowest_priority_deny_wins(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    assert create_rule(client, effect="deny", priority=0).status_code == 201
    assert create_rule(client, effect="allow", priority=5).status_code == 201

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "denied_by_policy"}


def test_higher_priority_deny_ignored(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    assert create_rule(client, effect="allow", priority=0).status_code == 201
    assert create_rule(client, effect="deny", priority=5).status_code == 201

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_evaluation_does_not_modify_data(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    assert create_rule(client).status_code == 201

    before = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    evaluate(client, machine_id)
    evaluate(client, machine_id)
    after = client.get(f"/machines/{machine_id}/behavior-declarations").json()

    assert before == after
