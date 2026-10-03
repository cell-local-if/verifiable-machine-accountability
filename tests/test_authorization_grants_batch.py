"""Tests for batch issue of one-time, short-lived authorization grants.

One additional write entry sits under the machine path:

    POST /machines/{machine_id}/authorization-grants/batch

It signs several eligible ``allowed_by_policy`` decision events into
independent grants inside one locked write transaction: the items are
checked in submission order, the first failure rejects the whole batch with
no partial issue, and every grant of the batch shares one UTC issue instant
with its own TTL. These tests cover the success shape, every validation and
lookup outcome, first-failure ordering, atomicity, exactly-once issue under
concurrency (batch versus batch and batch versus single), and
non-interference with the existing single-issue, consume, revoke, audit,
event, and basis behavior.
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


def grant_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants"


def batch_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/batch"


def issue(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        grant_url(machine_id),
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def issue_batch(client, machine_id, items):
    return client.post(batch_url(machine_id), json={"items": items})


def consume_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture
def allowed_events(client):
    """A machine + enabled declaration + allow rule + three allowed events."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    events = [
        record_event(client, machine_id, resource=f"res/{i}").json()
        for i in range(3)
    ]
    for event in events:
        assert event["allowed"] is True
        assert event["reason"] == "allowed_by_policy"
    return machine_id, events


# --------------------------------------------------------------------------- #
# Batch success shape
# --------------------------------------------------------------------------- #


def test_batch_success_shape_and_order(allowed_events, client):
    machine_id, events = allowed_events
    items = [
        {"event_id": events[0]["id"], "ttl_seconds": 10},
        {"event_id": events[1]["id"], "ttl_seconds": 300},
        {"event_id": events[2]["id"], "ttl_seconds": 1},
    ]
    response = issue_batch(client, machine_id, items)

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 3
    issued_stamps = set()
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
        issued_stamps.add(grant["issued_at"])
        issued = _parse_z(grant["issued_at"])
        expires = _parse_z(grant["expires_at"])
        assert expires - issued == timedelta(seconds=item["ttl_seconds"])
    # The whole batch shares one UTC issue instant.
    assert len(issued_stamps) == 1
    # The grant ids are distinct.
    assert len({grant["id"] for grant in body}) == 3


def test_batch_of_one_matches_single_issue_shape(allowed_events, client):
    machine_id, events = allowed_events
    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body) == 1
    assert list(body[0].keys()) == [
        "id",
        "machine_id",
        "event_id",
        "issued_at",
        "expires_at",
        "status",
    ]


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
    for grant, event in zip(grants, events[:2]):
        row = by_id[grant["id"]]
        assert row.machine_id == machine_id
        assert row.event_id == event["id"]
        assert row.status == "active"
        assert row.consumed_at is None


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

    listing = client.get(
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/changes"
        "?limit=100"
    )
    assert listing.status_code == 200
    items = listing.json()["records"]
    issued = [item for item in items if item["type"] == "issued"]
    assert len(issued) == 2
    by_grant = {item["grant_id"]: item for item in issued}
    for grant, event in zip(grants, events[:2]):
        lifecycle = by_grant[grant["id"]]
        assert lifecycle["authorization_event_id"] == event["id"]
        assert lifecycle["occurred_at"] == grant["issued_at"]

    integrity = client.get(
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/integrity"
    ).json()
    assert integrity["valid"] is True


# --------------------------------------------------------------------------- #
# Batch body and query validation
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
        {"items": 1},
        {"records": [{"event_id": "e", "ttl_seconds": 60}]},
        {"items": [{"event_id": "e", "ttl_seconds": 60}], "extra": 1},
        # Item shape.
        {"items": [{"event_id": "e"}]},
        {"items": [{"ttl_seconds": 60}]},
        {"items": [{"event_id": "e", "ttl_seconds": 60, "extra": 1}]},
        {"items": [[]]},
        {"items": ["x"]},
        {"items": [None]},
        {"items": [1]},
        # Item field values.
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
        # A bad second item rejects the whole batch.
        {
            "items": [
                {"event_id": "e", "ttl_seconds": 60},
                {"event_id": "f", "ttl_seconds": True},
            ]
        },
        # A repeated event id inside the batch is a format error.
        {
            "items": [
                {"event_id": "e", "ttl_seconds": 60},
                {"event_id": "e", "ttl_seconds": 30},
            ]
        },
    ],
)
def test_invalid_body_is_invalid_grant_request(allowed_events, client, payload):
    machine_id, _ = allowed_events
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_event_id_repeat_after_trimming_is_invalid(allowed_events, client):
    machine_id, _ = allowed_events
    response = client.post(
        batch_url(machine_id),
        json={
            "items": [
                {"event_id": "e", "ttl_seconds": 60},
                {"event_id": " e ", "ttl_seconds": 60},
            ]
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


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
    valid_body = {"items": [{"event_id": events[0]["id"], "ttl_seconds": 60}]}

    # Query validation wins even over a perfectly valid body.
    response = client.post(batch_url(machine_id) + query, json=valid_body)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # And it precedes the machine lookup: missing machine is still 422.
    response = client.post(batch_url(missing_machine) + query, json=valid_body)
    assert response.status_code == 422


def test_body_validation_runs_before_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        batch_url(missing_machine),
        json={"items": [{"event_id": "anything", "ttl_seconds": True}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_grant_request"


# --------------------------------------------------------------------------- #
# Lookup and eligibility outcomes, in submission order
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


def test_denied_event_is_event_not_allowed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=1)
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    event = record_event(client, machine_id, resource="res/d").json()
    assert event["allowed"] is False

    response = issue_batch(
        client, machine_id, [{"event_id": event["id"], "ttl_seconds": 60}]
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
    assert issue(client, machine_id, events[0]["id"]).status_code == 201

    response = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}


def test_first_failing_item_in_submission_order_decides(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=1)
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()
    assert denied["allowed"] is False

    # The missing event comes first: 404 wins over the later conflict.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": "no-such-event", "ttl_seconds": 60},
            {"event_id": denied["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    # The denied event comes first: its conflict wins over the later 404.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": denied["id"], "ttl_seconds": 60},
            {"event_id": "no-such-event", "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}


def test_failed_batch_leaves_no_partial_grants(allowed_events, client):
    machine_id, events = allowed_events
    # The first item is fully eligible; the second names a missing event.
    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": "no-such-event", "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 404

    with client.app.state.engine.connect() as conn:
        grant_count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()
        lifecycle_count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_lifecycle_events")
        ).scalar_one()
    assert grant_count == 0
    assert lifecycle_count == 0
    # The eligible event is still issuable afterwards.
    assert issue(client, machine_id, events[0]["id"]).status_code == 201


def test_failed_batch_leaves_other_records_unchanged(allowed_events, client):
    machine_id, events = allowed_events
    # An existing grant on another event is untouched by a rejected batch.
    existing = issue(client, machine_id, events[0]["id"]).json()
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).content

    response = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[1]["id"], "ttl_seconds": 60},
            {"event_id": events[0]["id"], "ttl_seconds": 60},
        ],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}

    assert client.get(events_url).content == events_before
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, status FROM authorization_grants")
        ).all()
    assert len(rows) == 1
    assert rows[0].id == existing["id"]
    assert rows[0].status == "active"


# --------------------------------------------------------------------------- #
# One grant per event, with concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_identical_batches_have_exactly_one_winner(
    allowed_events, client
):
    machine_id, events = allowed_events
    items = [
        {"event_id": events[0]["id"], "ttl_seconds": 60},
        {"event_id": events[1]["id"], "ttl_seconds": 60},
    ]
    count = 10
    gate = threading.Event()

    def hit():
        gate.wait()
        return issue_batch(client, machine_id, items)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit) for _ in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == count - 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_already_exists"}}

    with client.app.state.engine.connect() as conn:
        grant_count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()
    assert grant_count == 2


def test_concurrent_batch_and_single_issue_have_exactly_one_winner(
    allowed_events, client
):
    machine_id, events = allowed_events
    gate = threading.Event()

    def batch_hit():
        gate.wait()
        return issue_batch(
            client,
            machine_id,
            [
                {"event_id": events[0]["id"], "ttl_seconds": 60},
                {"event_id": events[1]["id"], "ttl_seconds": 60},
            ],
        )

    def single_hit():
        gate.wait()
        return issue(client, machine_id, events[0]["id"], ttl_seconds=60)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(batch_hit) for _ in range(5)] + [
            pool.submit(single_hit) for _ in range(5)
        ]
        gate.set()
        responses = [f.result() for f in futures]

    successes = [r for r in responses if r.status_code == 201]
    assert len(successes) == 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_already_exists"}}

    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT event_id FROM authorization_grants")
        ).all()
    event_ids = sorted(row.event_id for row in rows)
    if len(successes[0].json()) == 2:
        # The batch won: both events carry exactly one grant.
        assert event_ids == sorted([events[0]["id"], events[1]["id"]])
    else:
        # The single issue won: only its event carries a grant.
        assert event_ids == [events[0]["id"]]


def test_disjoint_batches_both_succeed(allowed_events, client):
    machine_id, events = allowed_events
    first = issue_batch(
        client, machine_id, [{"event_id": events[0]["id"], "ttl_seconds": 60}]
    )
    second = issue_batch(
        client, machine_id, [{"event_id": events[1]["id"], "ttl_seconds": 60}]
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()[0]["id"] != second.json()[0]["id"]


# --------------------------------------------------------------------------- #
# Batch-issued grants behave like singly-issued ones
# --------------------------------------------------------------------------- #


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

    consumed = client.post(consume_url(machine_id, grants[0]["id"]))
    assert consumed.status_code == 200
    assert consumed.json()["grant_id"] == grants[0]["id"]

    revoked = client.post(revoke_url(machine_id, grants[1]["id"]))
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"

    # The terminal states are kept.
    assert (
        client.post(consume_url(machine_id, grants[0]["id"])).status_code == 409
    )
    assert (
        client.post(revoke_url(machine_id, grants[1]["id"])).status_code == 409
    )


def test_batch_grants_appear_in_the_audit_listing(allowed_events, client):
    machine_id, events = allowed_events
    grants = issue_batch(
        client,
        machine_id,
        [
            {"event_id": events[0]["id"], "ttl_seconds": 60},
            {"event_id": events[1]["id"], "ttl_seconds": 60},
        ],
    ).json()

    listing = client.get(grant_url(machine_id)).json()
    by_id = {item["id"]: item for item in listing["items"]}
    assert set(by_id) == {grants[0]["id"], grants[1]["id"]}
    for grant in grants:
        item = by_id[grant["id"]]
        assert item["event_id"] == grant["event_id"]
        assert item["issued_at"] == grant["issued_at"]
        assert item["expires_at"] == grant["expires_at"]
        assert item["status"] == "active"


def test_batch_grants_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist-batch.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        events = [
            record_event(first, machine_id, resource=f"res/{i}").json()
            for i in range(2)
        ]
        grants = issue_batch(
            first,
            machine_id,
            [
                {"event_id": events[0]["id"], "ttl_seconds": 300},
                {"event_id": events[1]["id"], "ttl_seconds": 300},
            ],
        ).json()

    with TestClient(app) as second:
        # Still active and consumable after the restart, and no event can be
        # re-signed by either entry.
        assert (
            second.post(consume_url(machine_id, grants[0]["id"])).status_code
            == 200
        )
        repeat = issue(second, machine_id, events[1]["id"])
        assert repeat.status_code == 409
        assert repeat.json()["error"]["code"] == "grant_already_exists"
        repeat_batch = issue_batch(
            second,
            machine_id,
            [{"event_id": events[1]["id"], "ttl_seconds": 60}],
        )
        assert repeat_batch.status_code == 409
        assert repeat_batch.json()["error"]["code"] == "grant_already_exists"


def test_batch_never_rewrites_event_or_basis(allowed_events, client):
    machine_id, events = allowed_events
    basis_url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{events[0]['id']}/decision-basis"
    )
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    basis_before = client.get(basis_url).content
    events_before = client.get(events_url).content

    assert (
        issue_batch(
            client,
            machine_id,
            [{"event_id": events[0]["id"], "ttl_seconds": 60}],
        ).status_code
        == 201
    )

    assert client.get(basis_url).content == basis_before
    assert client.get(events_url).content == events_before
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True


def test_non_post_methods_are_not_routed_for_batch(allowed_events, client):
    machine_id, _ = allowed_events
    for method in ("get", "put", "patch", "delete", "head"):
        response = getattr(client, method)(batch_url(machine_id))
        assert response.status_code == 405
