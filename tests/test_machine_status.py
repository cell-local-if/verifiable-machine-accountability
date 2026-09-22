from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


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
    return response.json()


def set_status(client, machine_id, status):
    return client.post(f"/machines/{machine_id}/status", json={"status": status})


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


def evaluate(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": action_type, "resource": resource},
    )


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


# --------------------------------------------------------------------------- #
# Validation: 422 before any machine lookup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": None},
        {"status": 1},
        {"status": True},
        {"status": ["active"]},
        {"status": {"value": "active"}},
        {"status": "ACTIVE"},
        {"status": "suspended "},
        {"status": "paused"},
        {"status": ""},
        {"other": "active"},
        ["active"],
        "active",
        42,
        None,
    ],
)
def test_invalid_body_returns_422_before_lookup(client, payload):
    # Even against a non-existent machine, body problems surface as 422,
    # never 404.
    response = client.post(f"/machines/{MISSING_MACHINE}/status", json=payload)
    assert response.status_code == 422


def test_invalid_body_against_existing_machine_is_422(client):
    machine = create_machine(client)
    response = set_status(client, machine["id"], "paused")
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 404 / 409 / 200 semantics
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = set_status(client, MISSING_MACHINE, "suspended")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_suspend_active_machine_returns_full_record(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == machine["id"]
    assert body["external_id"] == machine["external_id"]
    assert body["display_name"] == machine["display_name"]
    assert body["public_key"] == "key-1"
    assert body["status"] == "suspended"
    assert body["version"] == 1
    assert body["created_at"] == machine["created_at"]
    assert body["updated_at"] >= machine["updated_at"]
    assert set(body.keys()) == {
        "id",
        "external_id",
        "display_name",
        "public_key",
        "status",
        "version",
        "created_at",
        "updated_at",
    }

    fetched = client.get(f"/machines/{machine['id']}").json()
    assert fetched == body


def test_reactivate_suspended_machine(client):
    machine = create_machine(client)
    assert set_status(client, machine["id"], "suspended").status_code == 200

    suspended = client.get(f"/machines/{machine['id']}").json()

    response = set_status(client, machine["id"], "active")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "active"
    assert body["public_key"] == "key-1"
    assert body["version"] == 1
    assert body["created_at"] == machine["created_at"]
    assert body["updated_at"] >= suspended["updated_at"]


def test_same_status_is_409_and_writes_nothing(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "active")
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    unchanged = client.get(f"/machines/{machine['id']}").json()
    assert unchanged == machine

    assert set_status(client, machine["id"], "suspended").status_code == 200
    suspended = client.get(f"/machines/{machine['id']}").json()

    response = set_status(client, machine["id"], "suspended")
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    assert client.get(f"/machines/{machine['id']}").json() == suspended


def test_transition_changes_only_status_and_updated_at(client):
    machine = create_machine(client)
    # Rotate the key first so version/public_key are non-initial; the status
    # change must leave both, plus created_at, exactly as they were.
    rotated = client.post(
        f"/machines/{machine['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    ).json()
    assert rotated["version"] == 2

    response = set_status(client, machine["id"], "suspended")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "suspended"
    assert body["version"] == 2
    assert body["public_key"] == "key-2"
    assert body["created_at"] == rotated["created_at"]
    assert body["external_id"] == rotated["external_id"]
    assert body["display_name"] == rotated["display_name"]


# --------------------------------------------------------------------------- #
# Concurrency: at most one request to the same target state succeeds
# --------------------------------------------------------------------------- #


def test_concurrent_suspend_allows_at_most_one_success(client):
    machine = create_machine(client)

    def suspend(_):
        return set_status(client, machine["id"], "suspended")

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(suspend, range(4)))

    assert sum(r.status_code == 200 for r in responses) == 1
    assert all(r.status_code in (200, 409) for r in responses)
    assert client.get(f"/machines/{machine['id']}").json()["status"] == "suspended"


def test_concurrent_reactivate_allows_at_most_one_success(client):
    machine = create_machine(client)
    assert set_status(client, machine["id"], "suspended").status_code == 200

    def activate(_):
        return set_status(client, machine["id"], "active")

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(activate, range(4)))

    assert sum(r.status_code == 200 for r in responses) == 1
    assert all(r.status_code in (200, 409) for r in responses)
    assert client.get(f"/machines/{machine['id']}").json()["status"] == "active"


# --------------------------------------------------------------------------- #
# Persistence across restarts
# --------------------------------------------------------------------------- #


def test_status_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        assert set_status(first, machine["id"], "suspended").status_code == 200
        suspended = first.get(f"/machines/{machine['id']}").json()

    with TestClient(app) as second:
        response = second.get(f"/machines/{machine['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "suspended"
    assert body == suspended


# --------------------------------------------------------------------------- #
# Suspended-machine authorization behavior
# --------------------------------------------------------------------------- #


def test_suspended_evaluation_is_denied_without_declarations_or_policy(client):
    machine = create_machine(client)
    # No declarations and no policy exist at all: the suspended verdict must
    # still be machine_suspended, not no_enabled_declaration.
    assert set_status(client, machine["id"], "suspended").status_code == 200

    response = evaluate(client, machine["id"])
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "machine_suspended"}


def test_suspended_denies_even_when_policy_would_allow(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client, effect="allow", priority=0)

    assert set_status(client, machine["id"], "suspended").status_code == 200

    response = evaluate(client, machine["id"])
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "machine_suspended"}


def test_active_evaluation_follows_existing_rules(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)

    assert evaluate(client, machine["id"]).json() == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }


def test_suspended_decision_event_is_persisted_and_chained(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)

    assert set_status(client, machine["id"], "suspended").status_code == 200

    response = record_event(client, machine["id"], resource="res/x")
    assert response.status_code == 201
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == "machine_suspended"
    assert body["machine_id"] == machine["id"]
    assert body["previous_event_id"] is None
    assert len(body["content_hash"]) == 64
    assert len(body["chain_hash"]) == 64

    # The denial is persisted by the existing rules.
    events = client.get(
        f"/machines/{machine['id']}/authorization-decision-events"
    ).json()
    assert len(events) == 1
    assert events[0]["id"] == body["id"]
    assert events[0]["allowed"] is False
    assert events[0]["reason"] == "machine_suspended"

    integrity = client.get(
        f"/machines/{machine['id']}/authorization-decision-events/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 1, "broken_event_id": None}


def test_suspended_events_extend_the_same_hash_chain(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)

    before = record_event(client, machine["id"], resource="res/a").json()
    assert before["allowed"] is True

    assert set_status(client, machine["id"], "suspended").status_code == 200
    during = record_event(client, machine["id"], resource="res/b").json()
    assert during["allowed"] is False
    assert during["reason"] == "machine_suspended"
    assert during["previous_event_id"] == before["id"]

    integrity = client.get(
        f"/machines/{machine['id']}/authorization-decision-events/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 2, "broken_event_id": None}


def test_reactivation_restores_existing_evaluation_rules(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)

    assert set_status(client, machine["id"], "suspended").status_code == 200
    assert evaluate(client, machine["id"]).json() == {
        "allowed": False,
        "reason": "machine_suspended",
    }

    assert set_status(client, machine["id"], "active").status_code == 200
    assert evaluate(client, machine["id"]).json() == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }

    # New events after reactivation are decided by the normal rules.
    after = record_event(client, machine["id"], resource="res/after").json()
    assert after["allowed"] is True
    assert after["reason"] == "allowed_by_policy"


def test_reactivation_does_not_change_prior_events(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)

    allowed_event = record_event(client, machine["id"], resource="res/a").json()
    assert set_status(client, machine["id"], "suspended").status_code == 200
    suspended_event = record_event(client, machine["id"], resource="res/b").json()
    assert set_status(client, machine["id"], "active").status_code == 200

    events = client.get(
        f"/machines/{machine['id']}/authorization-decision-events"
    ).json()
    assert [e["id"] for e in events] == [allowed_event["id"], suspended_event["id"]]
    assert events[0] == allowed_event
    assert events[1] == suspended_event
    assert events[0]["allowed"] is True
    assert events[1]["allowed"] is False
    assert events[1]["reason"] == "machine_suspended"

    integrity = client.get(
        f"/machines/{machine['id']}/authorization-decision-events/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 2, "broken_event_id": None}


def test_suspended_evaluation_does_not_persist_anything(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)
    assert set_status(client, machine["id"], "suspended").status_code == 200

    evaluate(client, machine["id"])
    evaluate(client, machine["id"])

    assert (
        client.get(
            f"/machines/{machine['id']}/authorization-decision-events"
        ).json()
        == []
    )


def test_suspension_keeps_other_interfaces_compatible(client):
    # Existing machine/rotation/declaration interfaces keep working while
    # suspended; only authorization decisions are short-circuited.
    machine = create_machine(client)
    declare(client, machine["id"])
    assert set_status(client, machine["id"], "suspended").status_code == 200

    rotated = client.post(
        f"/machines/{machine['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )
    assert rotated.status_code == 200
    assert rotated.json()["status"] == "suspended"
    assert rotated.json()["version"] == 2

    declarations = client.get(
        f"/machines/{machine['id']}/behavior-declarations"
    ).json()
    assert len(declarations) == 1
