"""Tests for batch consumption of one-time, short-lived authorization grants.

One more write entry sits under the machine path:

    POST /machines/{machine_id}/authorization-grants/batch-consume

It confirms several grants used in one locked write transaction: the body is
a JSON object with exactly one field ``grant_ids`` (an order-preserving
array of 1 to 100 non-blank-after-trim, non-repeating strings), every grant
is checked in input order against the single-consume eligibility rules, the
first failing grant decides the whole batch's error, and a rejected batch
writes nothing. A successful batch answers ``200`` with the
``{grant_id, use_id, consumed_at}`` objects in input order, every entry
stamped with one shared UTC consumption moment and every grant carrying
exactly one new use record. These tests cover the success shape, every
validation and lookup outcome, first-failing-grant ordering, atomicity (no
partial consumption), exactly-once consumption under concurrency against
single and batch requests, lifecycle audit events, and non-interference
with the existing single-consume, revoke, and machine-status semantics.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

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


def create_rule(client, action_type="read", resource_pattern="res/*",
                effect="allow", priority=0):
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
    return response.json()


def declare(client, machine_id, action_type="read", resource_pattern="res/*",
            enabled=True):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201
    return response.json()


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def issue(client, machine_id, event_id, ttl_seconds=300):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )
    assert response.status_code == 201
    return response


def batch_consume_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/batch-consume"


def consume_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def set_machine_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture
def issued_grants(client):
    """A machine + allow rule + three issued, still-usable grants."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    grants = []
    for index in range(3):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        assert event["allowed"] is True
        assert event["reason"] == "allowed_by_policy"
        grants.append(issue(client, machine_id, event["id"]).json())
    return machine_id, grants


def _grant_snapshot(client, grant_id):
    """The grant row, its use rows, and its lifecycle rows, verbatim."""
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, machine_id, event_id, status, issued_at, "
                "expires_at, consumed_at, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant_id)
        ).one()
        uses = conn.execute(
            text(
                "SELECT id, grant_id, machine_id, event_id, consumed_at "
                "FROM authorization_grant_uses WHERE grant_id = :id"
            ).bindparams(id=grant_id)
        ).all()
        lifecycle = conn.execute(
            text(
                "SELECT type, occurred_at FROM "
                "authorization_grant_lifecycle_events WHERE grant_id = :id "
                "ORDER BY occurred_at, id"
            ).bindparams(id=grant_id)
        ).all()
    return row, uses, [(r.type, r.occurred_at) for r in lifecycle]


def _backdated(client, grant_id):
    """Move one grant's expiry into the past without consuming it."""
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant_id)
        )


# --------------------------------------------------------------------------- #
# Batch success shape
# --------------------------------------------------------------------------- #


def test_batch_success_shape_and_shared_consumed_moment(issued_grants, client):
    machine_id, grants = issued_grants
    grant_ids = [grant["id"] for grant in grants]

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": grant_ids}
    )
    assert response.status_code == 200
    uses = response.json()
    assert [use["grant_id"] for use in uses] == grant_ids
    assert all(set(use) == {"grant_id", "use_id", "consumed_at"} for use in uses)
    # One shared UTC consumption moment for the whole batch, ending in Z.
    assert len({use["consumed_at"] for use in uses}) == 1
    assert uses[0]["consumed_at"].endswith("Z")
    # Every use gets its own fresh use id.
    assert len({use["use_id"] for use in uses}) == len(grant_ids)


def test_batch_of_one_matches_single_consume_shape(issued_grants, client):
    machine_id, grants = issued_grants

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
    )
    assert response.status_code == 200
    uses = response.json()
    assert len(uses) == 1
    assert uses[0]["grant_id"] == grants[0]["id"]
    assert set(uses[0]) == {"grant_id", "use_id", "consumed_at"}


def test_batch_input_order_is_preserved(issued_grants, client):
    machine_id, grants = issued_grants
    grant_ids = [grants[2]["id"], grants[0]["id"], grants[1]["id"]]

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": grant_ids}
    )
    assert response.status_code == 200
    assert [use["grant_id"] for use in response.json()] == grant_ids


def test_batch_consumption_is_persisted_with_one_use_per_grant(
    issued_grants, client
):
    machine_id, grants = issued_grants
    grant_ids = [grant["id"] for grant in grants]

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": grant_ids}
    )
    assert response.status_code == 200
    uses_by_grant = {use["grant_id"]: use for use in response.json()}

    for grant in grants:
        row, uses, lifecycle = _grant_snapshot(client, grant["id"])
        assert row.status == "consumed"
        assert row.consumed_at == uses_by_grant[grant["id"]]["consumed_at"]
        # Exactly one use record per grant, bound to the grant's own event.
        assert len(uses) == 1
        assert uses[0].id == uses_by_grant[grant["id"]]["use_id"]
        assert uses[0].event_id == grant["event_id"]
        assert uses[0].consumed_at == row.consumed_at
        # One consumed lifecycle event follows the issued one, reusing the
        # response's consumption moment.
        assert [event_type for event_type, _ in lifecycle] == [
            "issued",
            "consumed",
        ]
        assert lifecycle[-1][1] == row.consumed_at


def test_failed_batch_leaves_no_partial_consumption(issued_grants, client):
    machine_id, grants = issued_grants
    # The middle grant is already consumed: the batch must fail on it and
    # leave the first grant untouched.
    assert client.post(consume_url(machine_id, grants[1]["id"])).status_code == 200

    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"], grants[2]["id"]]},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}

    for grant, expected_uses in zip(grants, (0, 1, 0)):
        row, uses, lifecycle = _grant_snapshot(client, grant["id"])
        assert len(uses) == expected_uses
        if expected_uses == 0:
            assert row.status == "active"
            assert row.consumed_at is None
            assert [event_type for event_type, _ in lifecycle] == ["issued"]


# --------------------------------------------------------------------------- #
# Request validation (before any database read)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"grant_ids": []},
        {"grant_ids": ["a"] * 101},
        {"grant_ids": "not-a-list"},
        {"grant_ids": None},
        {"grant_ids": [1]},
        {"grant_ids": [True]},
        {"grant_ids": [None]},
        {"grant_ids": [{"id": "a"}]},
        {"grant_ids": ["  "]},
        {"grant_ids": ["a", "a"]},
        {"grant_ids": ["a", " a "]},
        {"grant_ids": ["a"], "extra": 1},
        {"wrong_key": ["a"]},
        ["a"],
        "a",
        1,
        None,
    ],
)
def test_invalid_body_is_invalid_grant_request(issued_grants, client, payload):
    machine_id, _grants = issued_grants
    response = client.post(batch_consume_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_unparseable_body_is_invalid_grant_request(issued_grants, client):
    machine_id, _grants = issued_grants
    response = client.post(
        batch_consume_url(machine_id),
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_boundary_of_one_hundred_ids_is_accepted(issued_grants, client):
    machine_id, _grants = issued_grants
    # 100 unknown ids pass shape validation and fail only at the lookup.
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [f"unknown-{index}" for index in range(100)]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_any_query_parameter_is_invalid_query_before_lookup(client):
    response = client.post(
        batch_consume_url("no-such-machine") + "?debug=1",
        json={"grant_ids": ["a"]},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_lookup(client):
    # An invalid body against a non-existent machine is still a 422, never
    # a 404: every body check precedes the machine and grant reads.
    response = client.post(
        batch_consume_url("no-such-machine"), json={"grant_ids": []}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_grant_ids_are_trimmed_before_use(issued_grants, client):
    machine_id, grants = issued_grants

    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [f"  {grants[0]['id']}  "]},
    )
    assert response.status_code == 200
    assert response.json()[0]["grant_id"] == grants[0]["id"]


# --------------------------------------------------------------------------- #
# Lookup outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client, issued_grants):
    _machine_id, grants = issued_grants
    response = client.post(
        batch_consume_url("no-such-machine"),
        json={"grant_ids": [grants[0]["id"]]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_grant_is_not_found(issued_grants, client):
    machine_id, grants = issued_grants
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], "no-such-grant"]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The first grant was not consumed by the failed batch.
    row, uses, _lifecycle = _grant_snapshot(client, grants[0]["id"])
    assert row.status == "active"
    assert uses == []


def test_cross_machine_grant_is_not_found(issued_grants, client):
    machine_id, grants = issued_grants
    other_machine = create_machine(client, external_id="machine-2")

    response = client.post(
        batch_consume_url(other_machine),
        json={"grant_ids": [grants[0]["id"]]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Per-grant outcome ordering (single-consume rules, first failure decides)
# --------------------------------------------------------------------------- #


def test_already_consumed_grant_is_grant_consumed(issued_grants, client):
    machine_id, grants = issued_grants
    assert client.post(consume_url(machine_id, grants[0]["id"])).status_code == 200

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}


def test_revoked_grant_is_grant_revoked(issued_grants, client):
    machine_id, grants = issued_grants
    assert client.post(revoke_url(machine_id, grants[0]["id"])).status_code == 200

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}


def test_expired_grant_is_grant_expired(issued_grants, client):
    machine_id, grants = issued_grants
    _backdated(client, grants[0]["id"])

    response = client.post(
        batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}


def test_usable_grant_on_suspended_machine_is_machine_suspended(
    issued_grants, client
):
    machine_id, grants = issued_grants
    set_machine_status(client, machine_id, "suspended")

    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "machine_suspended"}}

    # Nothing was written, and the same grants consume normally once the
    # machine returns to active.
    for grant in grants[:2]:
        row, uses, lifecycle = _grant_snapshot(client, grant["id"])
        assert row.status == "active"
        assert uses == []
        assert [event_type for event_type, _ in lifecycle] == ["issued"]

    set_machine_status(client, machine_id, "active")
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    )
    assert response.status_code == 200


def test_terminal_and_expired_outcomes_keep_precedence_when_suspended(
    issued_grants, client
):
    machine_id, grants = issued_grants
    assert client.post(consume_url(machine_id, grants[0]["id"])).status_code == 200
    assert client.post(revoke_url(machine_id, grants[1]["id"])).status_code == 200
    _backdated(client, grants[2]["id"])
    set_machine_status(client, machine_id, "suspended")

    # Even while suspended, the consumed, revoked, and expired grants keep
    # answering their own higher-precedence outcomes.
    for grant, code in zip(
        grants, ("grant_consumed", "grant_revoked", "grant_expired")
    ):
        response = client.post(
            batch_consume_url(machine_id), json={"grant_ids": [grant["id"]]}
        )
        assert response.status_code == 409
        assert response.json() == {"error": {"code": code}}


def test_first_failing_grant_in_input_order_decides(issued_grants, client):
    machine_id, grants = issued_grants
    assert client.post(revoke_url(machine_id, grants[0]["id"])).status_code == 200
    assert client.post(consume_url(machine_id, grants[1]["id"])).status_code == 200

    # The revoked grant comes first: its outcome decides, not the later
    # consumed grant's.
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}

    # Reversed input order reverses the reported outcome.
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[1]["id"], grants[0]["id"]]},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}


def test_failed_batch_appends_no_lifecycle_events(issued_grants, client):
    machine_id, grants = issued_grants
    _backdated(client, grants[1]["id"])

    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}

    with client.app.state.engine.connect() as conn:
        event_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_lifecycle_events "
                "WHERE type = 'consumed'"
            )
        ).scalar_one()
        use_count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_uses")
        ).scalar_one()
    assert event_count == 0
    assert use_count == 0


# --------------------------------------------------------------------------- #
# Interplay with the single-consume and revoke entries
# --------------------------------------------------------------------------- #


def test_single_consume_after_batch_is_grant_consumed(issued_grants, client):
    machine_id, grants = issued_grants
    assert client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    ).status_code == 200

    for grant in grants[:2]:
        response = client.post(consume_url(machine_id, grant["id"]))
        assert response.status_code == 409
        assert response.json() == {"error": {"code": "grant_consumed"}}


def test_revoke_after_batch_consume_is_grant_consumed(issued_grants, client):
    machine_id, grants = issued_grants
    assert client.post(
        batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
    ).status_code == 200

    response = client.post(revoke_url(machine_id, grants[0]["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}


def test_batch_consumed_grants_appear_in_audit_listing(issued_grants, client):
    machine_id, grants = issued_grants
    response = client.post(
        batch_consume_url(machine_id),
        json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
    )
    assert response.status_code == 200
    consumed_at = response.json()[0]["consumed_at"]

    listing = client.get(f"/machines/{machine_id}/authorization-grants")
    assert listing.status_code == 200
    listed = {grant["id"]: grant for grant in listing.json()["items"]}
    for grant in grants[:2]:
        assert listed[grant["id"]]["status"] == "consumed"
        assert listed[grant["id"]]["consumed_at"] == consumed_at
    assert listed[grants[2]["id"]]["status"] == "active"


# --------------------------------------------------------------------------- #
# Concurrency: exactly one winner, no partial consumption
# --------------------------------------------------------------------------- #


def test_concurrent_batch_and_single_have_exactly_one_winner(
    issued_grants, client
):
    machine_id, grants = issued_grants
    barrier = threading.Barrier(2)

    def hit_batch():
        barrier.wait()
        return client.post(
            batch_consume_url(machine_id),
            json={"grant_ids": [grants[0]["id"], grants[1]["id"]]},
        )

    def hit_single():
        barrier.wait()
        return client.post(consume_url(machine_id, grants[0]["id"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fn) for fn in (hit_batch, hit_single)]
        responses = [future.result() for future in futures]

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json() == {"error": {"code": "grant_consumed"}}

    # Exactly one use record per grant the winner consumed — one for the
    # single consume, two for the batch — with no duplicates or leftovers.
    winner = next(response for response in responses if response.status_code == 200)
    expected_uses = 1 if winner.request.url.path.endswith("/consume") else 2
    with client.app.state.engine.connect() as conn:
        use_count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_uses")
        ).scalar_one()
    assert use_count == expected_uses
    # The shared first grant is always consumed exactly once; the second
    # grant is consumed only when the batch won.
    row, uses, _lifecycle = _grant_snapshot(client, grants[0]["id"])
    assert row.status == "consumed"
    assert len(uses) == 1
    row, uses, _lifecycle = _grant_snapshot(client, grants[1]["id"])
    if expected_uses == 2:
        assert row.status == "consumed"
        assert len(uses) == 1
    else:
        assert row.status == "active"
        assert uses == []


def test_concurrent_overlapping_batches_have_exactly_one_winner(
    issued_grants, client
):
    machine_id, grants = issued_grants
    barrier = threading.Barrier(2)

    def hit(first, second):
        barrier.wait()
        return client.post(
            batch_consume_url(machine_id),
            json={"grant_ids": [first, second]},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = [
            future.result()
            for future in (
                pool.submit(hit, grants[0]["id"], grants[1]["id"]),
                pool.submit(hit, grants[1]["id"], grants[2]["id"]),
            )
        ]

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    winner = next(response for response in responses if response.status_code == 200)
    loser = next(response for response in responses if response.status_code == 409)
    # The loser reports the first racing grant in its own input order.
    assert loser.json() == {"error": {"code": "grant_consumed"}}

    # The winner consumed exactly its two grants, each with exactly one use
    # record; the grant only the loser named was never consumed.
    won_ids = {use["grant_id"] for use in winner.json()}
    assert len(won_ids) == 2
    for grant in grants:
        row, uses, _lifecycle = _grant_snapshot(client, grant["id"])
        if grant["id"] in won_ids:
            assert row.status == "consumed"
            assert len(uses) == 1
        else:
            assert row.status == "active"
            assert uses == []


def test_concurrent_batch_and_revoke_have_one_terminal_winner(
    issued_grants, client
):
    machine_id, grants = issued_grants
    barrier = threading.Barrier(2)

    def hit_batch():
        barrier.wait()
        return client.post(
            batch_consume_url(machine_id), json={"grant_ids": [grants[0]["id"]]}
        )

    def hit_revoke():
        barrier.wait()
        return client.post(revoke_url(machine_id, grants[0]["id"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fn) for fn in (hit_batch, hit_revoke)]
        responses = [future.result() for future in futures]

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json()["error"]["code"] in {"grant_consumed", "grant_revoked"}

    row, uses, lifecycle = _grant_snapshot(client, grants[0]["id"])
    # One definite terminal state with exactly the matching side effects.
    if row.status == "consumed":
        assert len(uses) == 1
        assert [event_type for event_type, _ in lifecycle] == [
            "issued",
            "consumed",
        ]
    else:
        assert row.status == "revoked"
        assert uses == []
        assert [event_type for event_type, _ in lifecycle] == [
            "issued",
            "revoked",
        ]


def test_non_post_methods_are_not_routed_for_batch_consume(
    issued_grants, client
):
    machine_id, _grants = issued_grants
    url = batch_consume_url(machine_id)
    assert client.get(url).status_code == 405
    assert client.put(url).status_code == 405
    assert client.delete(url).status_code == 405
    assert client.patch(url).status_code == 405
