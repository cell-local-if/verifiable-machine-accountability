"""Concurrency tests for the status-change / decision-event atomic write.

The status determination, the declaration/policy decision, and the
hash-chain append run in one locked write transaction that is serialized
against machine status changes, so a concurrent status change and decision
event always have one definite serial order:

* status change first — the later event uses the new status (a suspension
  yields ``machine_suspended``, never the stale active-era result);
* event append first — the event keeps its pre-change result and the later
  status change never rewrites it.

The deterministic tests park one request at a precise point and drive the
other against it; the burst tests assert the serial-order invariants hold
under real concurrency without any patching.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app
from accountability import authorization, chain


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


def allow_read_setup(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )


def record_event(client, machine_id, index):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": f"res/{index}"},
    )


def events(client, machine_id):
    return client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()


def integrity(client, machine_id):
    return client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()


# --- deterministic serial orders -------------------------------------------


def test_status_change_committed_first_forces_suspended_decision(
    client, monkeypatch
):
    """The event request starts first but has not taken the write lock; the
    suspension commits first, so the event must decide against suspended.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)

    gate = threading.Event()
    ready = threading.Event()
    real_append = chain.append_decision_event

    def delayed_append(engine, **kwargs):
        # Park after request validation but before the locked transaction.
        ready.set()
        assert gate.wait(timeout=10)
        return real_append(engine, **kwargs)

    monkeypatch.setattr(chain, "append_decision_event", delayed_append)

    def suspend():
        assert ready.wait(timeout=10)
        time.sleep(0.2)
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        event_future = pool.submit(record_event, client, machine_id, 0)
        suspend_response = pool.submit(suspend).result()
        assert suspend_response.status_code == 200
        assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"
        gate.set()
        event_response = event_future.result()

    assert event_response.status_code == 201
    assert event_response.json()["allowed"] is False
    assert event_response.json()["reason"] == "machine_suspended"

    stored = events(client, machine_id)
    assert len(stored) == 1
    assert stored[0]["reason"] == "machine_suspended"
    assert integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_event_id": None,
    }


def test_event_committed_first_keeps_pre_change_result_and_is_not_rewritten(
    client, monkeypatch
):
    """The event holds the write lock after reading ``active``; a concurrent
    suspension cannot commit until the event finishes, so the event keeps its
    allowed result and the later suspension must not rewrite it.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)

    hold = threading.Event()
    entered = threading.Event()
    real_decide = authorization.decide

    def stall_inside_lock(executor, mid, status, action_type, resource):
        if status == "active":
            entered.set()
            assert hold.wait(timeout=10)
        return real_decide(executor, mid, status, action_type, resource)

    monkeypatch.setattr(authorization, "decide", stall_inside_lock)

    def suspend():
        assert entered.wait(timeout=10)
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        event_future = pool.submit(record_event, client, machine_id, 0)
        suspend_future = pool.submit(suspend)
        # The event transaction holds the lock; the suspension is serialized
        # behind it and must not have committed.
        assert entered.wait(timeout=10)
        time.sleep(0.3)
        assert not suspend_future.done()
        assert client.get(f"/machines/{machine_id}").json()["status"] == "active"
        hold.set()
        event_response = event_future.result()
        suspend_response = suspend_future.result()

    assert event_response.status_code == 201
    assert event_response.json()["allowed"] is True
    assert event_response.json()["reason"] == "allowed_by_policy"
    assert suspend_response.status_code == 200
    assert suspend_response.json()["status"] == "suspended"

    stored = events(client, machine_id)
    assert len(stored) == 1
    # The committed event is never recomputed or rewritten by the later
    # status change.
    assert stored[0]["allowed"] is True
    assert stored[0]["reason"] == "allowed_by_policy"
    assert integrity(client, machine_id)["valid"] is True


def test_event_after_committed_suspension_in_same_lock_denies(client, monkeypatch):
    """Decision and status read happen on the same locked connection: once
    suspension is visible inside the transaction, the event cannot fall back
    to declarations/policy even though they would allow.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)

    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )

    # Drive many concurrent post-suspension events: all must be suspended.
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(lambda i: record_event(client, machine_id, i), range(20))
        )

    assert all(r.status_code == 201 for r in responses)
    stored = events(client, machine_id)
    assert len(stored) == 20
    assert {e["reason"] for e in stored} == {"machine_suspended"}
    assert all(e["allowed"] is False for e in stored)
    assert integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 20,
        "broken_event_id": None,
    }


# --- burst invariants -------------------------------------------------------


def _assert_one_unbroken_chain(stored, expected_count):
    assert len(stored) == expected_count
    assert len({e["id"] for e in stored}) == expected_count
    previous_ids = [e["previous_event_id"] for e in stored]
    ids = [e["id"] for e in stored]
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]


def test_concurrent_suspend_and_events_have_one_serial_order(client):
    """A burst of decision events races one suspension. The persisted chain
    must be equivalent to one serial schedule: a prefix of active-era
    decisions followed by suspended decisions, with no later event reverting
    to the stale active result.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    count = 40

    gate = threading.Event()

    def event(index):
        gate.wait()
        return record_event(client, machine_id, index)

    def suspend():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        event_futures = [pool.submit(event, i) for i in range(count)]
        suspend_future = pool.submit(suspend)
        gate.set()
        responses = [f.result() for f in event_futures]
        suspend_response = suspend_future.result()

    assert all(r.status_code == 201 for r in responses)
    assert suspend_response.status_code == 200
    assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"

    stored = events(client, machine_id)
    _assert_one_unbroken_chain(stored, count)

    reasons = [e["reason"] for e in stored]
    assert set(reasons) <= {"allowed_by_policy", "machine_suspended"}
    # Serializability: no allowed decision may appear after the first
    # suspended one (that would be a stale active-era result).
    first_suspended = next(
        (i for i, reason in enumerate(reasons) if reason == "machine_suspended"),
        count,
    )
    assert reasons[first_suspended:] == ["machine_suspended"] * (
        count - first_suspended
    )
    assert all(e["allowed"] is False for e in stored[first_suspended:])
    assert all(e["allowed"] is True for e in stored[:first_suspended])

    assert integrity(client, machine_id) == {
        "valid": True,
        "checked_count": count,
        "broken_event_id": None,
    }


def test_concurrent_reactivate_and_events_have_one_serial_order(client):
    """Mirror case: events racing a reactivation form a suspended prefix
    followed by an allowed suffix, never the reverse.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )
    count = 40

    gate = threading.Event()

    def event(index):
        gate.wait()
        return record_event(client, machine_id, index)

    def reactivate():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "active"}
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        event_futures = [pool.submit(event, i) for i in range(count)]
        reactivate_future = pool.submit(reactivate)
        gate.set()
        responses = [f.result() for f in event_futures]
        reactivate_response = reactivate_future.result()

    assert all(r.status_code == 201 for r in responses)
    assert reactivate_response.status_code == 200

    stored = events(client, machine_id)
    _assert_one_unbroken_chain(stored, count)

    reasons = [e["reason"] for e in stored]
    assert set(reasons) <= {"allowed_by_policy", "machine_suspended"}
    first_allowed = reasons.index("allowed_by_policy")
    assert reasons[:first_allowed] == ["machine_suspended"] * first_allowed
    assert reasons[first_allowed:] == ["allowed_by_policy"] * (
        count - first_allowed
    )
    assert integrity(client, machine_id)["valid"] is True


def test_same_target_suspend_in_burst_succeeds_at_most_once(client):
    """Two suspensions racing the same machine still serialize: at most one
    success, and no half-finished machine or event state remains.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)

    gate = threading.Event()

    def suspend():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    def event():
        gate.wait()
        return record_event(client, machine_id, 0)

    with ThreadPoolExecutor(max_workers=5) as pool:
        # Submit every participant before collecting any result so all five
        # reach the barrier together.
        suspend_futures = [pool.submit(suspend) for _ in range(4)]
        event_future = pool.submit(event)
        gate.set()
        status_responses = [f.result() for f in suspend_futures]
        event_response = event_future.result()

    assert sum(r.status_code == 200 for r in status_responses) == 1
    assert sorted(r.status_code for r in status_responses) == [200, 409, 409, 409]
    assert all(
        r.json() == {"error": {"code": "invalid_status_transition"}}
        for r in status_responses
        if r.status_code == 409
    )
    assert event_response.status_code == 201

    stored = events(client, machine_id)
    assert len(stored) == 1
    # The event is serialized either just before the winning suspension
    # (allowed) or just after it (suspended); both are valid single orders,
    # but the result must match one of them exactly.
    assert stored[0]["reason"] in {
        "allowed_by_policy",
        "machine_suspended",
    }
    assert stored[0]["allowed"] == (stored[0]["reason"] == "allowed_by_policy")
    assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"
    assert integrity(client, machine_id)["valid"] is True


# --- validation and lookup semantics are unchanged under concurrency ---------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action_type": "read"},
        {"resource": "res/x"},
        {"action_type": 1, "resource": "res/x"},
        {"action_type": "read", "resource": 2},
        {"action_type": "  ", "resource": "res/x"},
        "not-an-object",
        ["read"],
    ],
)
def test_invalid_event_body_is_422_before_lookup_even_raced(client, payload):
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    gate = threading.Event()
    ready = threading.Event()

    def busy_suspend():
        ready.set()
        gate.wait(5)
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        suspend_future = pool.submit(busy_suspend)
        assert ready.wait(timeout=10)
        response = client.post(
            f"/machines/{machine_id}/authorization-decision-events", json=payload
        )
        gate.set()
        suspend_future.result()

    assert response.status_code == 422
    # A rejected body writes nothing.
    assert events(client, machine_id) == []


def test_event_missing_machine_is_404_even_under_concurrent_load(client):
    missing = "00000000-0000-0000-0000-000000000000"

    def hit(_):
        return client.post(
            f"/machines/{missing}/authorization-decision-events",
            json={"action_type": "read", "resource": "res/x"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(hit, range(16)))

    assert {r.status_code for r in responses} == {404}
    assert all(r.json() == {"error": {"code": "not_found"}} for r in responses)


def test_active_machine_policy_rules_still_decide_under_concurrency(client):
    """Active machines keep the declaration/resource/lowest-priority-deny
    semantics: same-priority deny wins, lowest numeric priority decides.
    """
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    # Same-lowest-priority deny for the exact resource must win.
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/deny",
            "effect": "deny",
            "priority": 0,
        },
    )

    gate = threading.Event()

    def hit(index):
        gate.wait()
        resource = "res/deny" if index % 2 == 0 else f"res/ok{index}"
        return client.post(
            f"/machines/{machine_id}/authorization-decision-events",
            json={"action_type": "read", "resource": resource},
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit, i) for i in range(20)]
        gate.set()
        responses = [f.result() for f in futures]

    by_resource = {r.json()["resource"]: r.json() for r in responses}
    assert by_resource["res/deny"]["reason"] == "denied_by_policy"
    assert by_resource["res/deny"]["allowed"] is False
    assert all(
        body["reason"] == "allowed_by_policy"
        for resource, body in by_resource.items()
        if resource != "res/deny"
    )
    assert integrity(client, machine_id)["valid"] is True
