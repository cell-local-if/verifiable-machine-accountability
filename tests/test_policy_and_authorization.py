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


def create_declaration(
    client, machine_id, action_type="read", resource_pattern="doc/*", enabled=True
):
    return client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )


def create_rule(
    client,
    action_type="read",
    resource_pattern="doc/*",
    effect="allow",
    priority=100,
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


# ---------------------------------------------------------------------------
# POST /policy-rules
# ---------------------------------------------------------------------------


def test_create_policy_rule_returns_201_with_full_record(client):
    response = create_rule(client)

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "doc/*"
    assert body["effect"] == "allow"
    assert body["priority"] == 100
    assert body["created_at"].endswith("Z")
    assert body["updated_at"].endswith("Z")
    assert body["created_at"] == body["updated_at"]


def test_create_policy_rule_accepts_deny_effect(client):
    response = create_rule(client, effect="deny")

    assert response.status_code == 201
    assert response.json()["effect"] == "deny"


def test_create_policy_rule_accepts_priority_zero(client):
    response = create_rule(client, priority=0)

    assert response.status_code == 201
    assert response.json()["priority"] == 0


def test_create_policy_rule_strips_surrounding_whitespace(client):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": " read \t",
            "resource_pattern": "  doc/* ",
            "effect": "allow",
            "priority": 1,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "doc/*"


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "   ", "resource_pattern": "r", "effect": "allow", "priority": 1},
        {"action_type": "a", "resource_pattern": "\t", "effect": "allow", "priority": 1},
        {"action_type": "a", "resource_pattern": "r", "effect": "permit", "priority": 1},
        {"action_type": "a", "resource_pattern": "r", "effect": "ALLOW", "priority": 1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": -1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": True},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": False},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": 1.5},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": "1"},
        {"resource_pattern": "r", "effect": "allow", "priority": 1},
        {"action_type": "a", "effect": "allow", "priority": 1},
        {"action_type": "a", "resource_pattern": "r", "priority": 1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow"},
    ],
)
def test_create_policy_rule_rejects_invalid_payload(client, payload):
    response = client.post("/policy-rules", json=payload)

    assert response.status_code == 422


def test_duplicate_policy_rule_returns_409_and_adds_nothing(client):
    assert create_rule(client, effect="allow").status_code == 201

    response = create_rule(client, effect="deny")

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_policy_rule"}}

    # The conflicting tuple must not have been written: a distinct priority on
    # the same action/resource still creates its own unique row.
    other = create_rule(client, effect="deny", priority=200)
    assert other.status_code == 201
    assert other.json()["effect"] == "deny"


def test_same_action_resource_different_priority_allowed(client):
    assert create_rule(client, priority=1).status_code == 201
    assert create_rule(client, priority=2).status_code == 201


def test_policy_rules_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        created = create_rule(first).json()

    with TestClient(app) as second:
        machine_id = create_machine(second)
        create_declaration(second, machine_id)
        response = second.post(
            f"/machines/{machine_id}/authorization-evaluations",
            json={"action_type": "read", "resource": "doc/a"},
        )

    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}
    assert created["effect"] == "allow"


# ---------------------------------------------------------------------------
# POST /machines/{machine_id}/authorization-evaluations
# ---------------------------------------------------------------------------


def evaluate(client, machine_id, action_type="read", resource="doc/a"):
    return client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": action_type, "resource": resource},
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "   ", "resource": "r"},
        {"action_type": "a", "resource": "\t"},
        {"resource": "r"},
        {"action_type": "a"},
        {},
    ],
)
def test_evaluation_rejects_invalid_payload(client, payload):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-evaluations", json=payload
    )

    assert response.status_code == 422


def test_evaluation_missing_machine_returns_404(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/authorization-evaluations",
        json={"action_type": "read", "resource": "doc/a"},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_evaluation_no_declaration_at_all_returns_no_enabled_declaration(client):
    machine_id = create_machine(client)
    create_rule(client)

    response = evaluate(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_evaluation_disabled_declaration_returns_no_enabled_declaration(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, enabled=False)
    create_rule(client)

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_evaluation_declaration_for_other_action_returns_no_enabled_declaration(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="write")
    create_rule(client, action_type="read")

    response = evaluate(client, machine_id, action_type="read")

    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_evaluation_non_matching_declaration_returns_no_enabled_declaration(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="other/*")
    create_rule(client)

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "no_enabled_declaration"}


def test_evaluation_star_pattern_matches_any_resource(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*")

    response = evaluate(client, machine_id, resource="anything/here")

    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_evaluation_literal_pattern_matches_exactly(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="doc/a")
    create_rule(client, resource_pattern="doc/a")

    assert evaluate(client, machine_id, resource="doc/a").json() == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }
    assert evaluate(client, machine_id, resource="doc/b").json() == {
        "allowed": False,
        "reason": "no_enabled_declaration",
    }


def test_evaluation_enabled_declaration_but_no_policy_returns_no_matching_policy(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id)

    response = evaluate(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_evaluation_policy_for_other_action_returns_no_matching_policy(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read")
    create_rule(client, action_type="write")

    response = evaluate(client, machine_id, action_type="read")

    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_evaluation_policy_non_matching_resource_returns_no_matching_policy(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="doc/*")
    create_rule(client, resource_pattern="other/*")

    response = evaluate(client, machine_id, resource="doc/a")

    assert response.json() == {"allowed": False, "reason": "no_matching_policy"}


def test_evaluation_deny_rule_denies(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id)
    create_rule(client, effect="deny")

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "denied_by_policy"}


def test_evaluation_lowest_priority_deny_wins(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=10)
    create_rule(client, resource_pattern="*", effect="deny", priority=5)

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": False, "reason": "denied_by_policy"}


def test_evaluation_higher_priority_deny_ignored_when_lower_allows(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=1)
    create_rule(client, resource_pattern="*", effect="deny", priority=2)

    response = evaluate(client, machine_id)

    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_evaluation_deny_in_lowest_group_wins_over_allow_same_priority(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, resource_pattern="*")
    # Two distinct patterns at the same priority both match; the deny wins.
    create_rule(client, resource_pattern="doc/*", effect="allow", priority=3)
    create_rule(client, resource_pattern="*", effect="deny", priority=3)

    response = evaluate(client, machine_id, resource="doc/a")

    assert response.json() == {"allowed": False, "reason": "denied_by_policy"}


def test_evaluation_does_not_modify_data(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id).json()
    rule = create_rule(client).json()

    before_machine = client.get(f"/machines/{machine_id}").json()
    for _ in range(3):
        assert evaluate(client, machine_id).json() == {
            "allowed": True,
            "reason": "allowed_by_policy",
        }
    after_machine = client.get(f"/machines/{machine_id}").json()
    after_declarations = client.get(
        f"/machines/{machine_id}/behavior-declarations"
    ).json()

    assert after_machine == before_machine
    assert after_declarations == [declaration]
    assert rule["created_at"] == rule["updated_at"]


def test_evaluation_strips_input_whitespace(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read", resource_pattern="*")
    create_rule(client, action_type="read", resource_pattern="*")

    response = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "  read ", "resource": "\tx/y "},
    )

    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}
