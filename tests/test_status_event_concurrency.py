"""Concurrency between machine status changes and authorization decision events.

The machine lookup, suspended-status check, declaration/policy decision, chain
tail read, and event insert must commit as one indivisible authorization
write. Under concurrent suspension and decision-event requests there is
therefore one definite serial order: events that land before the suspension
keep their active-state result, every event after it is a
``machine_suspended`` denial, the chain never loses or forks an event, and a
later status change never rewrites an earlier event.
"""

from concurrent.futures import ThreadPoolExecutor

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
    return response.json()


def allow_everything(client, machine_id):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )
    assert response.status_code == 201
    response = client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )
    assert response.status_code == 201


def list_events(client, machine_id):
    return client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()


def assert_chain_valid(client, machine_id, expected_count):
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True
    assert integrity["checked_count"] == expected_count
    assert integrity["broken_event_id"] is None


def record_event(client, machine_id, index):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": f"res/{index}"},
    )


def set_status(client, machine_id, status):
    return client.post(f"/machines/{machine_id}/status", json={"status": status})


def race_suspend_against_events(client, machine_id, event_count, worker_count):
    """Fire one suspend alongside many event appends; return the event bodies."""
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(record_event, client, machine_id, i)
            for i in range(event_count)
        ]
        suspend = pool.submit(set_status, client, machine_id, "suspended")
        responses = [future.result() for future in futures]
        suspend_response = suspend.result()

    assert suspend_response.status_code == 200
    for response in responses:
        assert response.status_code == 201
    return [response.json() for response in responses]


def test_suspend_racing_events_has_one_definite_serial_order(client):
    machine = create_machine(client)
    machine_id = machine["id"]
    allow_everything(client, machine_id)

    # An event recorded before the race guarantees at least one active allow;
    # the suspended state after the race guarantees at least one denial.
    before = record_event(client, machine_id, "before").json()
    assert before["allowed"] is True

    recorded = race_suspend_against_events(client, machine_id, 24, 8)

    assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"
    after = record_event(client, machine_id, "after").json()
    assert after["allowed"] is False
    assert after["reason"] == "machine_suspended"

    stored = list_events(client, machine_id)
    assert len(stored) == 26
    by_id = {event["id"]: event for event in stored}
    assert set(by_id) == {before["id"], after["id"]} | {
        event["id"] for event in recorded
    }

    # The single chain order gives one cut point: active allows first, then
    # suspended denials. No event can reflect a state different from the order.
    reasons = [event["reason"] for event in stored]
    assert set(reasons) <= {"allowed_by_policy", "machine_suspended"}
    cut = next(
        (i for i, reason in enumerate(reasons) if reason == "machine_suspended"),
        len(reasons),
    )
    assert reasons[:cut] == ["allowed_by_policy"] * cut
    assert reasons[cut:] == ["machine_suspended"] * (len(reasons) - cut)
    assert cut >= 1  # the pre-race event is always an active-state allow
    for event in stored:
        assert event["allowed"] is (event["reason"] == "allowed_by_policy")

    assert_chain_valid(client, machine_id, len(stored))

    # Committed events are immutable: a later reactivation neither recomputes
    # nor rewrites the suspended denials.
    assert set_status(client, machine_id, "active").status_code == 200
    assert list_events(client, machine_id) == stored
    assert_chain_valid(client, machine_id, len(stored))


def test_repeated_suspend_reactivate_races_never_lose_events(client):
    machine = create_machine(client)
    machine_id = machine["id"]
    allow_everything(client, machine_id)

    total = 0
    for _ in range(4):
        # Each round starts active, the race ends suspended.
        assert set_status(client, machine_id, "active").status_code in (200, 409)
        recorded = race_suspend_against_events(client, machine_id, 12, 6)
        total += len(recorded)

    stored = list_events(client, machine_id)
    assert len(stored) == total
    assert_chain_valid(client, machine_id, total)
    assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"


def test_events_while_suspended_racing_each_other_are_all_denied(client):
    machine = create_machine(client)
    machine_id = machine["id"]
    allow_everything(client, machine_id)
    assert set_status(client, machine_id, "suspended").status_code == 200

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda i: record_event(client, machine_id, i), range(24)))

    for response in responses:
        assert response.status_code == 201
        body = response.json()
        assert body["allowed"] is False
        assert body["reason"] == "machine_suspended"

    stored = list_events(client, machine_id)
    assert len(stored) == 24
    assert {event["reason"] for event in stored} == {"machine_suspended"}
    assert_chain_valid(client, machine_id, 24)


def test_event_before_suspend_is_not_rewritten_by_later_suspend(client):
    machine = create_machine(client)
    machine_id = machine["id"]
    allow_everything(client, machine_id)

    before = record_event(client, machine_id, 0).json()
    assert before["allowed"] is True
    assert set_status(client, machine_id, "suspended").status_code == 200
    after = record_event(client, machine_id, 1).json()
    assert after["allowed"] is False
    assert after["reason"] == "machine_suspended"
    assert after["previous_event_id"] == before["id"]

    # Repeated same-target status requests all fail and write nothing.
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: set_status(client, machine_id, "suspended"), range(4))
        )
    assert {response.status_code for response in responses} == {409}

    stored = list_events(client, machine_id)
    assert len(stored) == 2
    assert stored[0]["id"] == before["id"]
    assert stored[0]["allowed"] is True
    assert stored[0]["reason"] == "allowed_by_policy"
    assert stored[1]["id"] == after["id"]
    assert stored[1]["reason"] == "machine_suspended"
    assert_chain_valid(client, machine_id, 2)


def test_status_change_never_appends_an_event(client):
    machine = create_machine(client)
    machine_id = machine["id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: set_status(client, machine_id, "suspended"), range(4))
        )

    assert sorted(response.status_code for response in responses) == [200, 409, 409, 409]
    assert list_events(client, machine_id) == []
