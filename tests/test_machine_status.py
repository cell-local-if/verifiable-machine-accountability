import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

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
    return response.json()


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


def set_status(client, machine_id, status):
    return client.post(f"/machines/{machine_id}/status", json={"status": status})


# --- status endpoint basics ------------------------------------------------


def test_new_machine_is_active(client):
    machine = create_machine(client)
    assert machine["status"] == "active"


def test_suspend_updates_only_status_and_updated_at(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "suspended"
    assert body["id"] == machine["id"]
    assert body["external_id"] == machine["external_id"]
    assert body["display_name"] == machine["display_name"]
    assert body["public_key"] == machine["public_key"]
    assert body["version"] == machine["version"] == 1
    assert body["created_at"] == machine["created_at"]
    assert RFC3339_Z_RE.match(body["updated_at"])
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


def test_reactivate_after_suspend(client):
    machine = create_machine(client)

    assert set_status(client, machine["id"], "suspended").status_code == 200
    response = set_status(client, machine["id"], "active")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "active"
    assert body["version"] == 1
    assert body["public_key"] == "key-1"
    assert body["created_at"] == machine["created_at"]

    assert client.get(f"/machines/{machine['id']}").json()["status"] == "active"


def test_suspend_when_already_suspended_returns_409_and_writes_nothing(client):
    machine = create_machine(client)
    suspended = set_status(client, machine["id"], "suspended").json()

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    assert client.get(f"/machines/{machine['id']}").json() == suspended


def test_activate_when_already_active_returns_409_and_writes_nothing(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "active")

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    assert client.get(f"/machines/{machine['id']}").json() == machine


def test_status_missing_machine_returns_404(client):
    response = set_status(
        client, "00000000-0000-0000-0000-000000000000", "suspended"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize(
    "send",
    [
        lambda c, m: c.post(f"/machines/{m}/status", json={}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"status": None}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"status": 1}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"status": True}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"status": "deleted"}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"status": "Suspended"}),
        lambda c, m: c.post(f"/machines/{m}/status", json={"other": "suspended"}),
        lambda c, m: c.post(f"/machines/{m}/status", json=["suspended"]),
        lambda c, m: c.post(f"/machines/{m}/status", json="suspended"),
        lambda c, m: c.post(f"/machines/{m}/status", content=""),
    ],
)
def test_invalid_status_payload_returns_422_before_lookup(client, send):
    machine_id = create_machine(client)["id"]

    assert send(client, machine_id).status_code == 422
    assert client.get(f"/machines/{machine_id}").json()["status"] == "active"


def test_invalid_status_payload_is_422_even_for_missing_machine(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/status",
        json={"status": "deleted"},
    )
    assert response.status_code == 422

    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/status",
        json={"status": 1},
    )
    assert response.status_code == 422


def test_concurrent_suspend_allows_at_most_one_success(client):
    machine = create_machine(client)

    def suspend(_):
        return set_status(client, machine["id"], "suspended")

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(suspend, range(4)))

    assert sum(r.status_code == 200 for r in responses) == 1
    assert sorted(r.status_code for r in responses) == [200, 409, 409, 409]
    assert client.get(f"/machines/{machine['id']}").json()["status"] == "suspended"


def test_concurrent_suspend_when_already_suspended_all_fail(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: set_status(client, machine["id"], "suspended"), range(4))
        )

    assert {r.status_code for r in responses} == {409}


def test_concurrent_reactivate_allows_at_most_one_success(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: set_status(client, machine["id"], "active"), range(4))
        )

    assert sum(r.status_code == 200 for r in responses) == 1
    assert client.get(f"/machines/{machine['id']}").json()["status"] == "active"


def test_status_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        assert set_status(first, machine["id"], "suspended").status_code == 200

    with TestClient(app) as second:
        fetched = second.get(f"/machines/{machine['id']}").json()
        assert fetched["status"] == "suspended"
        # A suspended machine keeps being denied after a restart.
        decision = evaluate(second, machine["id"])
        assert decision.status_code == 200
        assert decision.json() == {"allowed": False, "reason": "machine_suspended"}
        # Reactivation persists too.
        assert set_status(second, machine["id"], "active").status_code == 200

    with TestClient(app) as third:
        assert third.get(f"/machines/{machine['id']}").json()["status"] == "active"


# --- suspended authorization behavior --------------------------------------


def test_suspended_evaluation_denies_even_when_policy_allows(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client, effect="allow")
    assert evaluate(client, machine["id"]).json()["allowed"] is True

    set_status(client, machine["id"], "suspended")

    response = evaluate(client, machine["id"])
    assert response.status_code == 200
    assert response.json() == {"allowed": False, "reason": "machine_suspended"}


def test_suspended_evaluation_short_circuits_other_reasons(client):
    # Without suspension this machine would be denied for
    # no_enabled_declaration; suspension takes precedence.
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    response = evaluate(client, machine["id"])
    assert response.json() == {"allowed": False, "reason": "machine_suspended"}


def test_suspended_decision_event_is_recorded_on_chain(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client, effect="allow")
    # An event from before the suspension is later shown unchanged.
    before = record_event(client, machine["id"], resource="res/before").json()
    assert before["allowed"] is True

    set_status(client, machine["id"], "suspended")

    response = record_event(client, machine["id"], resource="res/after")
    assert response.status_code == 201
    body = response.json()
    assert body["allowed"] is False
    assert body["reason"] == "machine_suspended"
    assert body["previous_event_id"] == before["id"]
    assert body["machine_id"] == machine["id"]

    events = client.get(
        f"/machines/{machine['id']}/authorization-decision-events"
    ).json()
    assert [e["id"] for e in events] == [before["id"], body["id"]]
    # The earlier event's result is immutable.
    assert events[0]["allowed"] is True
    assert events[0]["reason"] == "allowed_by_policy"
    assert events[1]["allowed"] is False
    assert events[1]["reason"] == "machine_suspended"

    integrity = client.get(
        f"/machines/{machine['id']}/authorization-decision-events/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 2, "broken_event_id": None}


def test_suspended_evaluation_creates_no_events(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client)
    set_status(client, machine["id"], "suspended")

    evaluate(client, machine["id"])
    evaluate(client, machine["id"])

    events = client.get(
        f"/machines/{machine['id']}/authorization-decision-events"
    ).json()
    assert events == []


def test_reactivation_resumes_normal_evaluation_rules(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client, effect="allow")
    set_status(client, machine["id"], "suspended")
    assert evaluate(client, machine["id"]).json()["reason"] == "machine_suspended"

    set_status(client, machine["id"], "active")

    response = evaluate(client, machine["id"])
    assert response.status_code == 200
    assert response.json() == {"allowed": True, "reason": "allowed_by_policy"}


def test_events_from_suspension_window_stay_denied_after_reactivation(client):
    machine = create_machine(client)
    declare(client, machine["id"])
    create_rule(client, effect="allow")
    set_status(client, machine["id"], "suspended")
    suspended_event = record_event(client, machine["id"]).json()
    set_status(client, machine["id"], "active")
    after_event = record_event(client, machine["id"]).json()

    events = client.get(
        f"/machines/{machine['id']}/authorization-decision-events"
    ).json()
    by_id = {e["id"]: e for e in events}
    assert by_id[suspended_event["id"]]["allowed"] is False
    assert by_id[suspended_event["id"]]["reason"] == "machine_suspended"
    assert by_id[after_event["id"]]["allowed"] is True
    assert by_id[after_event["id"]]["reason"] == "allowed_by_policy"

    integrity = client.get(
        f"/machines/{machine['id']}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True
    assert integrity["checked_count"] == 2


def test_suspended_decision_event_on_missing_machine_is_404(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/authorization-decision-events",
        json={"action_type": "read", "resource": "res/x"},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --- compatibility with key rotation ---------------------------------------


def test_key_rotation_preserves_suspended_status(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    response = client.post(
        f"/machines/{machine['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "suspended"
    assert body["version"] == 2
    assert body["public_key"] == "key-2"

    fetched = client.get(f"/machines/{machine['id']}").json()
    assert fetched["status"] == "suspended"
    assert fetched["version"] == 2


def test_status_change_does_not_bump_version_after_rotation(client):
    machine = create_machine(client)
    client.post(
        f"/machines/{machine['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 2
    assert body["public_key"] == "key-2"
    assert body["status"] == "suspended"
