"""Tests for batch issuance of one-time, short-lived authorization grants.

One more write entry sits under the machine path:

    POST /machines/{machine_id}/authorization-grants/batch

It signs several qualified ``allowed_by_policy`` decision events into
independent grants in one locked write transaction: the body is a JSON
object with exactly one field ``items`` (a non-empty, order-preserving
array of ``{event_id, ttl_seconds}`` objects under the single-issue field
rules, with no repeated event id), every item is checked in input order
against the single-issue eligibility rules, the first failing item decides
the whole batch's error, and a rejected batch writes nothing. A successful
batch answers ``201`` with the grant objects in input order, all stamped
with one shared UTC issue moment and each expiring that moment plus its own
``ttl_seconds``. These tests cover the success shape, every validation and
lookup outcome, first-failing-item ordering, atomicity (no partial issue),
exactly-once issue under concurrency against single and batch requests,
lifecycle audit events, and non-interference with the existing event and
basis views.
"""
import json
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


def batch_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/batch"


def grant_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants"


def consume_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def issue_batch(client, machine_id, items):
    return client.post(batch_url(machine_id), json={"items": items})


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture
def allowed_events(client):
    """A machine + enabled declaration + allow rule + three allowed events."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    events = []
    for index in range(3):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        assert event["allowed"] is True
        assert event["reason"] == "allowed_by_policy"
        events.append(event)
    return machine_id, events


def grant_count(client):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()


# --------------------------------------------------------------------------- #
# Batch success shape
# --------------------------------------------------------------------------- #


def test_batch_success_shape_and_shared_issue_moment(allowed_events, client):
    machine_id, events = allowed_events
    items = [
        {"event_id": events[0]["id"], "ttl_seconds": 1},
        {"event_id": events[1]["id"], "ttl_seconds": 90},
        {"event_id": events[2]["id"], "ttl_seconds": 300},
    ]
    response = issue_batch(client, machine_id, items)

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 3
    # The order of the input is preserved and every object carries exactly
    # the single-issue fields in the single-issue order.
    issued_ats = set()
    for grant, item, event in zip(body, items, events):
        assert list(grant.keys()) == [
            "id",
            "machine_id",
            "event_id",
            "issued_at",
            "expires_at",
            "status",
        ]
        assert grant["machine_id"] == machine_id
        assert grant["event_id"] == event["id"]
        assert grant["status"] == "active"
        assert isinstance(grant["id"], str) and grant["id"]
        for stamp in (grant["issued_at"], grant["expires_at"]):
            assert stamp.endswith("Z")
            assert _parse_z(stamp).tzinfo == timezone.utc
        issued = _parse_z(grant["issued_at"])
        expires = _parse_z(grant["expires_at"])
        assert expires - issued == timedelta(seconds=item["ttl_seconds"])
        issued_ats.add(grant["issued_at"])
    # One shared UTC issue moment for the whole batch, distinct grant ids.
    assert len(issued_ats) == 1
    assert len({grant["id"] for grant in body}) == 3


def test_batch_of_one_matches_single_issue_shape(allowed_events, client):
    machine_id, events = allowed_events
    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body) == 1
    assert body[0]["event_id"] == events[0]["id"]
    assert body[0]["status"] == "active"


def test_batch_grants_are_persisted_active(allowed_events, client):
    machine_id, events = allowed_events
    grants = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    ).json()
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, machine_id, event_id, status, consumed_at "
                "FROM authorization_grants ORDER BY event_id"
            )
        ).all()
    assert len(rows) == 2
    by_id = {row.id: row for row in rows}
    for grant in grants:
        row = by_id[grant["id"]]
        assert row.machine_id == machine_id
        assert row.event_id == grant["event_id"]
        assert row.status == "active"
        assert row.consumed_at is None


def test_batch_grants_are_consumable_and_revocable(allowed_events, client):
    machine_id, events = allowed_events
    grants = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    ).json()

    consume = client.post(consume_url(machine_id, grants[0]["id"]))
    assert consume.status_code == 200
    assert consume.json()["grant_id"] == grants[0]["id"]
    revoke = client.post(revoke_url(machine_id, grants[1]["id"]))
    assert revoke.status_code == 200
    assert revoke.json()["status"] == "revoked"

    with client.app.state.engine.connect() as conn:
        statuses = {
            row.id: row.status
            for row in conn.execute(
                text("SELECT id, status FROM authorization_grants")
            )
        }
    assert statuses[grants[0]["id"]] == "consumed"
    assert statuses[grants[1]["id"]] == "revoked"


def test_batch_appends_one_issued_lifecycle_event_per_grant(
    allowed_events, client
):
    machine_id, events = allowed_events
    grants = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    ).json()

    response = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes",
        params={"limit": 100},
    )
    assert response.status_code == 200
    records = response.json()["records"]
    assert len(records) == 2
    by_grant = {record["grant_id"]: record for record in records}
    for grant, event in zip(grants, events[:2]):
        record = by_grant[grant["id"]]
        assert record["type"] == "issued"
        assert record["authorization_event_id"] == event["id"]
        assert record["occurred_at"] == grant["issued_at"]
    integrity = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/integrity"
    ).json()
    assert integrity["valid"] is True


# --------------------------------------------------------------------------- #
# Body and query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        # Top-level shape.
        {},
        {"items": []},
        {"items": None},
        {"items": {}},
        {"items": "x"},
        {"items": 12},
        {"items": True},
        {"event_id": "e", "ttl_seconds": 60},
        {"items": [{"event_id": "e", "ttl_seconds": 60}], "extra": 1},
        # Element shape.
        {"items": [None]},
        {"items": [True]},
        {"items": ["e"]},
        {"items": [[]]},
        {"items": [{}]},
        {"items": [{"event_id": "e"}]},
        {"items": [{"ttl_seconds": 60}]},
        {"items": [{"event_id": "e", "ttl_seconds": 60, "extra": 1}]},
        # Element field values.
        {"items": [{"event_id": "e", "ttl_seconds": 0}]},
        {"items": [{"event_id": "e", "ttl_seconds": 301}]},
        {"items": [{"event_id": "e", "ttl_seconds": -1}]},
        {"items": [{"event_id": "e", "ttl_seconds": True}]},
        {"items": [{"event_id": "e", "ttl_seconds": False}]},
        {"items": [{"event_id": "e", "ttl_seconds": 1.0}]},
        {"items": [{"event_id": "e", "ttl_seconds": 60.5}]},
        {"items": [{"event_id": "e", "ttl_seconds": "60"}]},
        {"items": [{"event_id": "e", "ttl_seconds": None}]},
        {"items": [{"event_id": 123, "ttl_seconds": 60}]},
        {"items": [{"event_id": True, "ttl_seconds": 60}]},
        {"items": [{"event_id": None, "ttl_seconds": 60}]},
        {"items": [{"event_id": "", "ttl_seconds": 60}]},
        {"items": [{"event_id": "   ", "ttl_seconds": 60}]},
        # A bad element anywhere in the array fails the whole body.
        {
            "items": [
                {"event_id": "e1", "ttl_seconds": 60},
                {"event_id": "e2", "ttl_seconds": True},
            ]
        },
        # A repeated event id, even with different TTLs.
        {
            "items": [
                {"event_id": "e", "ttl_seconds": 60},
                {"event_id": "e", "ttl_seconds": 60},
            ]
        },
        {
            "items": [
                {"event_id": "e", "ttl_seconds": 60},
                {"event_id": "e", "ttl_seconds": 120},
            ]
        },
    ],
)
def test_invalid_body_is_invalid_grant_request(allowed_events, client, payload):
    machine_id, _ = allowed_events
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}
    assert grant_count(client) == 0


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_typed_body_is_invalid_grant_request(
    allowed_events, client, raw
):
    machine_id, _ = allowed_events
    response = client.post(
        batch_url(machine_id),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


@pytest.mark.parametrize("query", ["?x=1", "?ttl_seconds=60", "?=", "?x=1&x=2"])
def test_any_query_parameter_is_invalid_query_before_lookup(
    allowed_events, client, query
):
    machine_id, events = allowed_events
    missing_machine = "00000000-0000-0000-0000-000000000000"
    body = {"items": [{"event_id": events[0]["id"], "ttl_seconds": 60}]}

    # Query validation wins even over a perfectly valid body.
    response = client.post(batch_url(machine_id) + query, json=body)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # And it precedes the machine lookup: missing machine is still 422.
    response = client.post(batch_url(missing_machine) + query, json=body)
    assert response.status_code == 422
    assert grant_count(client) == 0


def test_body_validation_runs_before_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        batch_url(missing_machine),
        json={"items": [{"event_id": "anything", "ttl_seconds": True}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_grant_request"


# --------------------------------------------------------------------------- #
# Lookup and eligibility outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client, allowed_events):
    _, events = allowed_events
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = issue_batch(
        client, missing_machine, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_is_not_found(client):
    machine_id = create_machine(client)
    response = issue_batch(
        client, machine_id, [{"event_id": "no-such-event", "ttl_seconds": 60}]
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_cross_machine_event_is_not_found(client, allowed_events):
    machine_id, events = allowed_events
    other = create_machine(client, external_id="machine-2")
    response = issue_batch(
        client, other, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_denied_event_is_event_not_allowed(allowed_events, client):
    machine_id, events = allowed_events
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()
    assert denied["allowed"] is False

    response = issue_batch(
        client, machine_id, [{"event_id": denied["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}


def test_event_without_basis_snapshot_is_decision_basis_unavailable(
    allowed_events, client
):
    machine_id, events = allowed_events
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=events[0]["id"])
        )

    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_unavailable"}}


def test_event_with_failing_basis_audit_is_decision_basis_invalid(
    allowed_events, client
):
    machine_id, events = allowed_events
    with client.app.state.engine.begin() as conn:
        document = conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :id"
            ).bindparams(id=events[0]["id"])
        ).scalar_one()
        parsed = json.loads(document)
        parsed["decision"]["allowed"] = False
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id"
            ).bindparams(doc=json.dumps(parsed, separators=(",", ":")),
                         id=events[0]["id"])
        )

    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_invalid"}}


def test_event_with_existing_grant_is_grant_already_exists(
    allowed_events, client
):
    machine_id, events = allowed_events
    response = client.post(
        grant_url(machine_id),
        json={"event_id": events[0]["id"], "ttl_seconds": 60},
    )
    assert response.status_code == 201

    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}


# --------------------------------------------------------------------------- #
# First failing item in input order, and atomicity
# --------------------------------------------------------------------------- #


def test_first_failing_item_in_input_order_decides(allowed_events, client):
    machine_id, events = allowed_events
    # events[0] already has a grant; "no-such-event" does not exist.
    response = client.post(
        grant_url(machine_id),
        json={"event_id": events[0]["id"], "ttl_seconds": 60},
    )
    assert response.status_code == 201

    # The already-granted event comes first: its outcome wins.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": "no-such-event", "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}

    # The missing event comes first: its outcome wins.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": "no-such-event", "ttl_seconds": 60},
            {"event_id": events[0]["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_failed_batch_leaves_no_partial_issue(allowed_events, client):
    machine_id, events = allowed_events
    # The first item is fully qualified; the second is not allowed. The
    # batch fails and even the qualified first event gets no grant.
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()

    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": denied["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}
    assert grant_count(client) == 0

    # The qualified event is still issuable afterwards, on its own.
    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 201


def test_failed_batch_appends_no_lifecycle_events(allowed_events, client):
    machine_id, events = allowed_events
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": "no-such-event", "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 404
    records = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes",
        params={"limit": 100},
    ).json()["records"]
    assert records == []


# --------------------------------------------------------------------------- #
# One grant per event across single and batch, with concurrency
# --------------------------------------------------------------------------- #


def test_single_issue_after_batch_is_grant_already_exists(
    allowed_events, client
):
    machine_id, events = allowed_events
    assert (
        issue_batch(
            client,
            machine_id,
            [{"event_id": events[0]["id"], "ttl_seconds": 60}],
        ).status_code
        == 201
    )
    response = client.post(
        grant_url(machine_id),
        json={"event_id": events[0]["id"], "ttl_seconds": 60},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}


def test_concurrent_batch_and_single_have_exactly_one_winner(
    allowed_events, client
):
    machine_id, events = allowed_events
    count = 10
    gate = threading.Event()

    def hit_batch():
        gate.wait()
        return issue_batch(
            client,
            machine_id,
            [{"event_id": events[0]["id"], "ttl_seconds": 60}],
        )

    def hit_single():
        gate.wait()
        return client.post(
            grant_url(machine_id),
            json={"event_id": events[0]["id"], "ttl_seconds": 60},
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit_batch) for _ in range(count)] + [
            pool.submit(hit_single) for _ in range(count)
        ]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 2 * count - 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_already_exists"}}
    assert grant_count(client) == 1


def test_concurrent_overlapping_batches_have_exactly_one_winner(
    allowed_events, client
):
    machine_id, events = allowed_events
    gate = threading.Event()

    def hit(first, second):
        gate.wait()
        return issue_batch(
            client,
            machine_id,
            [
                {"event_id": events[first]["id"], "ttl_seconds": 60},
                {"event_id": events[second]["id"], "ttl_seconds": 60},
            ],
        )

    # Two batches overlapping on events[1]: only one can commit.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(hit, 0, 1), pool.submit(hit, 1, 2)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [201, 409]
    loser = next(r for r in responses if r.status_code == 409)
    assert loser.json() == {"error": {"code": "grant_already_exists"}}
    # The winner signed exactly its two events; the loser's unique event
    # was never signed (no partial issue).
    assert grant_count(client) == 2


# --------------------------------------------------------------------------- #
# Non-interference
# --------------------------------------------------------------------------- #


def test_batch_issue_never_rewrites_events_or_bases(allowed_events, client):
    machine_id, events = allowed_events
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).content
    bases_before = [
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event['id']}/decision-basis"
        ).content
        for event in events
    ]

    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert response.status_code == 201

    assert client.get(events_url).content == events_before
    for event, before in zip(events, bases_before):
        assert (
            client.get(
                f"/machines/{machine_id}/authorization-decision-events/"
                f"{event['id']}/decision-basis"
            ).content
            == before
        )
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True


def test_batch_grants_appear_in_audit_listing(allowed_events, client):
    machine_id, events = allowed_events
    grants = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    ).json()

    response = client.get(grant_url(machine_id), params={"limit": 100})
    assert response.status_code == 200
    listed = {item["id"]: item for item in response.json()["items"]}
    assert set(listed) == {grant["id"] for grant in grants}
    for grant in grants:
        item = listed[grant["id"]]
        assert item["event_id"] == grant["event_id"]
        assert item["issued_at"] == grant["issued_at"]
        assert item["expires_at"] == grant["expires_at"]
        assert item["status"] == "active"


def test_non_post_methods_are_not_routed_for_batch(allowed_events, client):
    machine_id, events = allowed_events
    for method in ("get", "put", "patch", "delete", "head"):
        response = getattr(client, method)(batch_url(machine_id))
        assert response.status_code == 405
    # The 405s wrote nothing: the events are still issuable.
    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert response.status_code == 201


# --------------------------------------------------------------------------- #
# Suspended-machine gate on batch issue
# --------------------------------------------------------------------------- #


def set_machine_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response


def _lifecycle_record_count(client, machine_id):
    response = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes",
        params={"limit": 100},
    )
    assert response.status_code == 200
    return len(response.json()["records"])


def test_suspended_machine_batch_is_machine_suspended(allowed_events, client):
    machine_id, events = allowed_events
    set_machine_status(client, machine_id, "suspended")

    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "machine_suspended"}}


def test_suspended_batch_leaves_no_partial_issue(allowed_events, client):
    machine_id, events = allowed_events
    set_machine_status(client, machine_id, "suspended")

    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert response.status_code == 409

    assert grant_count(client) == 0
    assert _lifecycle_record_count(client, machine_id) == 0
    with client.app.state.engine.connect() as conn:
        machine_status = conn.execute(
            text("SELECT status FROM machines WHERE id = :id")
            .bindparams(id=machine_id)
        ).scalar_one()
    assert machine_status == "suspended"


def test_suspended_batch_keeps_earlier_outcome_precedence(
    allowed_events, client
):
    machine_id, events = allowed_events
    # The first item's event already has a grant: with the machine
    # suspended, the existing-grant outcome still decides the batch.
    first = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert first.status_code == 201
    set_machine_status(client, machine_id, "suspended")

    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
            {"event_id": events[2]["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}

    # A fully qualified earlier item fails at its own suspended gate before
    # a later item's existing-grant outcome is ever reached.
    gated = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[1]["id"], "ttl_seconds": 60},
            {"event_id": events[0]["id"], "ttl_seconds": 60},
        ],
    )
    assert gated.status_code == 409
    assert gated.json() == {"error": {"code": "machine_suspended"}}

    missing = issue_batch(
        client,
        machine_id,
        [
            {"event_id": "no-such-event", "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    )
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "not_found"}}

    # Neither rejection added anything: only the first single-item batch's
    # grant exists.
    assert grant_count(client) == 1


def test_suspended_batch_first_item_failure_decides(allowed_events, client):
    machine_id, events = allowed_events
    set_machine_status(client, machine_id, "suspended")

    # A missing event earlier in input order still decides over the
    # suspended gate of a later item.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": "no-such-event", "ttl_seconds": 60},
            {"event_id": events[0]["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    # With every earlier item qualified, the suspended gate decides.
    gated = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert gated.status_code == 409
    assert gated.json() == {"error": {"code": "machine_suspended"}}
    assert grant_count(client) == 0


def test_batch_issue_resumes_after_reactivation(allowed_events, client):
    machine_id, events = allowed_events
    set_machine_status(client, machine_id, "suspended")
    rejected = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 90} for event in events],
    )
    assert rejected.status_code == 409
    assert rejected.json() == {"error": {"code": "machine_suspended"}}

    set_machine_status(client, machine_id, "active")
    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 90} for event in events],
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body) == 3
    assert [grant["event_id"] for grant in body] == [
        event["id"] for event in events
    ]
    assert {grant["status"] for grant in body} == {"active"}
    # One shared issue moment across the whole batch.
    assert len({grant["issued_at"] for grant in body}) == 1
    for grant in body:
        assert _parse_z(grant["expires_at"]) - _parse_z(
            grant["issued_at"]
        ) == timedelta(seconds=90)
    assert grant_count(client) == 3
    assert _lifecycle_record_count(client, machine_id) == 3


def test_suspend_racing_batch_issue_has_one_definite_serial_outcome(
    allowed_events, client
):
    machine_id, events = allowed_events
    gate = threading.Event()

    def suspend_hit():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    def batch_hit():
        gate.wait()
        return issue_batch(
            client,
            machine_id,
            [
                {"event_id": event["id"], "ttl_seconds": 60}
                for event in events
            ],
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(suspend_hit)] + [
            pool.submit(batch_hit) for _ in range(5)
        ]
        gate.set()
        responses = [f.result() for f in futures]

    batch_responses = [
        r for r in responses if "authorization-grants" in str(r.request.url.path)
    ]
    suspend_responses = [r for r in responses if r not in batch_responses]
    assert len(suspend_responses) == 1
    assert suspend_responses[0].status_code == 200
    assert len(batch_responses) == 5

    # The batch that serialized before the suspension committed all three
    # grants; every batch that serialized after saw the suspended machine
    # and wrote nothing. grant_already_exists is the loser's outcome once
    # one batch has committed.
    outcomes = {}
    for response in batch_responses:
        if response.status_code == 201:
            outcomes.setdefault("ok", 0)
            outcomes["ok"] += 1
        else:
            code = response.json()["error"]["code"]
            assert code in ("machine_suspended", "grant_already_exists")
            outcomes.setdefault(code, 0)
            outcomes[code] += 1
    assert outcomes.get("ok", 0) <= 1

    with client.app.state.engine.connect() as conn:
        final_grants = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()
        machine_status = conn.execute(
            text("SELECT status FROM machines WHERE id = :id")
            .bindparams(id=machine_id)
        ).scalar_one()
    assert machine_status == "suspended"
    assert final_grants == (3 if outcomes.get("ok") else 0)

    # After the suspension committed, no new batch ever lands: the same
    # items are rejected again (machine_suspended for unsigned events,
    # grant_already_exists for signed ones) and the count never moves.
    response = issue_batch(
        client,
        machine_id,
        [{"event_id": event["id"], "ttl_seconds": 60} for event in events],
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] in (
        "machine_suspended",
        "grant_already_exists",
    )
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one() == final_grants
