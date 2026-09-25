import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=1):
    """Insert a policy rule directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rule_id,
                "action_type": "read",
                "resource_pattern": "res/*",
                "effect": "allow",
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at or created_at,
            },
        )


def test_empty_policy_rules_returns_empty_array(client):
    response = client.get("/policy-rules")

    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_created_rules(client):
    first = create_rule(client, priority=3).json()
    second = create_rule(client, action_type="write", effect="deny", priority=7).json()

    response = client.get("/policy-rules")

    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()] == [first["id"], second["id"]]


def test_list_items_have_exactly_the_persisted_fields(client):
    created = create_rule(client, priority=4).json()

    [rule] = client.get("/policy-rules").json()

    assert set(rule.keys()) == {
        "id",
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
    }
    # The listing keeps only the visible fields; the creation response also
    # carries the chain fields, which are exposed on /policy-rules/chain.
    assert rule == {
        key: created[key]
        for key in (
            "id",
            "action_type",
            "resource_pattern",
            "effect",
            "priority",
            "created_at",
            "updated_at",
        )
    }


def test_list_values_are_persisted_without_normalization(client, tmp_path):
    # The POST endpoint trims its inputs; the listing must echo exactly what is
    # stored, including values the write path would never produce.
    rule_id = rid(1)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rule_id,
                "action_type": "  Read ",
                "resource_pattern": " res/* ",
                "effect": "ALLOW",
                "priority": 9,
                "created_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-04-01T00:00:00Z",
            },
        )

    [rule] = client.get("/policy-rules").json()

    assert rule["id"] == rule_id
    assert rule["action_type"] == "  Read "
    assert rule["resource_pattern"] == " res/* "
    assert rule["effect"] == "ALLOW"
    assert rule["priority"] == 9
    assert rule["created_at"] == "2026-03-01T00:00:00Z"
    assert rule["updated_at"] == "2026-04-01T00:00:00Z"


def test_rules_ordered_by_created_at_instant_then_id(client):
    # Rows deliberately inserted out of order, with one exact/fractional pair.
    insert_rule_row(client, rid(30), "2026-03-01T00:00:03Z", priority=30)
    insert_rule_row(client, rid(21), "2026-03-01T00:00:02Z", priority=21)
    insert_rule_row(client, rid(20), "2026-03-01T00:00:02Z", priority=20)
    insert_rule_row(client, rid(10), "2026-03-01T00:00:01Z", priority=10)
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z", priority=1)
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00.500000Z", priority=2)

    response = client.get("/policy-rules")

    assert [rule["id"] for rule in response.json()] == [
        rid(1),   # exact second sorts before the fractional stamp
        rid(2),   # same wall-clock second, 0.5s later
        rid(10),
        rid(20),  # tie at :02 breaks by id
        rid(21),
        rid(30),
    ]


def test_equal_instant_tie_breaks_by_id(client):
    stamp = "2026-03-01T00:00:00.250000Z"
    insert_rule_row(client, rid(21), stamp, priority=21)
    insert_rule_row(client, rid(20), stamp, priority=20)

    response = client.get("/policy-rules")

    assert [rule["id"] for rule in response.json()] == [rid(20), rid(21)]


def test_list_is_read_only_and_deterministic(client):
    create_rule(client, priority=1)
    create_rule(client, priority=2)

    first = client.get("/policy-rules").json()
    second = client.get("/policy-rules").json()
    third = client.get("/policy-rules").json()

    assert first == second == third
    assert len(first) == 2
    for rule in first:
        assert rule["created_at"] == rule["updated_at"]


def test_list_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        create_rule(first, priority=1)
        create_rule(first, priority=2)
        expected = first.get("/policy-rules").json()

    with TestClient(app) as second:
        response = second.get("/policy-rules")

    assert response.status_code == 200
    assert response.json() == expected
    assert len(response.json()) == 2


def test_listing_does_not_change_authorization_evaluation(client):
    # The listing endpoint must never participate in evaluation: calling it
    # before and after an evaluation leaves the decision unchanged.
    machine_response = client.post(
        "/machines",
        json={
            "external_id": "machine-1",
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    )
    machine_id = machine_response.json()["id"]
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )
    create_rule(client, effect="allow", priority=0)

    client.get("/policy-rules")
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    client.get("/policy-rules")

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}
