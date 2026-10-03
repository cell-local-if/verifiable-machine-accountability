"""Tests for the read-only single-event execution accountability trace.

Covers `GET /machines/{machine_id}/authorization-decision-events/{event_id}/
execution-trace`: GET-only 405 (including HEAD) without reading records,
``invalid_query`` 422 for any query parameter, a repeated parameter, or a
carried body before the machine/event lookup, ``not_found`` 404 for a missing
machine/event and for an event owned by another machine with no execution
data, ``internal_error`` 500 on a real read fault with no event summary or
record arrays, the fixed five-group response (the event summary object plus
the grants, grant-uses, lifecycle-events, and execution-receipt arrays),
per-machine selection on each group's own event reference, child records kept
when a stored grant/use parent is damaged or missing, complete stored fields
emitted without repair or current-time status fabrication, tolerant
(instant, id) ordering per group's own business timestamp with damaged stamps
last, empty-group preservation for unsigned and denied events, machine
isolation, compact single-newline JSON, read-only byte-identical repeats, and
persistence across restarts.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T0_HALF = "2026-03-01T00:00:00.5Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
GHOST_ID = "99999999-9999-9999-9999-999999999999"

GROUP_ORDER = (
    "event_summary",
    "grants",
    "grant_uses",
    "lifecycle_events",
    "execution_receipts",
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
    return response.json()["id"]


def declare(client, machine_id, action_type="read", resource_pattern="res/*"):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": True,
        },
    )
    assert response.status_code == 201


def create_rule(client, action_type="read", resource_pattern="res/*", priority=0):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": "allow",
            "priority": priority,
        },
    )
    assert response.status_code == 201


def record_event(client, machine_id, resource="res/x", allow=False):
    if allow:
        declare(client, machine_id, resource_pattern="res/*")
        create_rule(client, resource_pattern="res/*")
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201, response.text
    return response.json()


def issue_grant(client, machine_id, event_id, ttl_seconds=300):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )
    assert response.status_code == 201, response.text
    return response.json()


def consume_grant(client, machine_id, grant_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )
    assert response.status_code == 200, response.text
    return response.json()


def revoke_grant(client, machine_id, grant_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )
    assert response.status_code == 200, response.text
    return response.json()


def register_receipt(client, machine_id, use, action_type="read",
                     resource="res/x", outcome="succeeded", digest="a" * 64):
    response = client.post(
        f"/machines/{machine_id}/execution-receipts",
        json={
            "use_id": use["use_id"],
            "action_type": action_type,
            "resource": resource,
            "outcome": outcome,
            "result_digest": digest,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def trace_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/execution-trace"
    )


def trace(client, machine_id, event_id):
    response = client.get(trace_url(machine_id, event_id))
    assert response.status_code == 200, response.text
    return response


def execute_sql(client, statement, parameters=None):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement), parameters or {})


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = getattr(client, method)(trace_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"event_summary" not in response.content


def test_method_routing_does_not_read_records(client):
    # With every table the trace could read dropped, only routing is in play.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        for table in (
            "execution_receipts",
            "authorization_grant_lifecycle_events",
            "authorization_grant_uses",
            "authorization_grants",
            "authorization_decision_events",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(
            trace_url(machine_id, event["id"])
        ).status_code == 405


@pytest.mark.parametrize(
    "query",
    ["?x=1", "?limit=10", "?x=1&x=2", "?=", "?foo"],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_repeated_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + "?x=1&x=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.request(
        "GET",
        trace_url(machine_id, event["id"]),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    response = client.get(trace_url(MISSING_ID, MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_machine_lookup(client):
    response = client.request(
        "GET",
        trace_url(MISSING_ID, MISSING_ID),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_and_method_errors_do_not_read_records(client):
    # With the tables dropped, a read would 500; validation and routing win.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grants"))
        conn.execute(text("DROP TABLE authorization_decision_events"))
    assert client.get(
        trace_url(machine_id, event["id"]) + "?x=1"
    ).status_code == 422
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(
            trace_url(machine_id, event["id"])
        ).status_code == 405


def test_missing_machine_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(MISSING_ID, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content
    assert b"execution_receipts" not in response.content


def test_missing_event_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    response = client.get(trace_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content
    assert b"grants" not in response.content


def test_event_owned_by_another_machine_is_404(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    event = record_event(client, machine_id)
    response = client.get(trace_url(other, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content


def test_event_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_events"))
    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"grants" not in response.content
    assert b"execution_receipts" not in response.content


@pytest.mark.parametrize(
    "table",
    [
        "authorization_grants",
        "authorization_grant_uses",
        "authorization_grant_lifecycle_events",
        "execution_receipts",
    ],
)
def test_associated_read_failure_is_500_with_no_trace_data(client, table):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"grant_uses" not in response.content


# --------------------------------------------------------------------------- #
# Success shape and the five fixed groups
# --------------------------------------------------------------------------- #


def test_event_with_no_grant_has_summary_and_four_empty_arrays(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)

    response = trace(client, machine_id, event["id"])
    body = response.json()
    assert list(body.keys()) == list(GROUP_ORDER)
    assert isinstance(body["event_summary"], dict)
    for group in GROUP_ORDER[1:]:
        assert body[group] == []


def test_denied_decision_returns_the_empty_execution_chain(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    assert event["allowed"] is False

    body = trace(client, machine_id, event["id"]).json()
    assert list(body.keys()) == list(GROUP_ORDER)
    assert body["event_summary"]["allowed"] is False
    assert body["event_summary"]["reason"] == "no_enabled_declaration"
    assert body["grants"] == []
    assert body["grant_uses"] == []
    assert body["lifecycle_events"] == []
    assert body["execution_receipts"] == []


def test_event_summary_carries_result_reason_moment_and_chain_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)

    summary = trace(client, machine_id, event["id"]).json()["event_summary"]
    assert list(summary.keys()) == [
        "allowed",
        "reason",
        "created_at",
        "previous_event_id",
        "content_hash",
        "chain_hash",
    ]
    assert summary["allowed"] is event["allowed"]
    assert summary["reason"] == event["reason"]
    assert summary["created_at"] == event["created_at"]
    assert summary["previous_event_id"] == event["previous_event_id"]
    assert summary["content_hash"] == event["content_hash"]
    assert summary["chain_hash"] == event["chain_hash"]
    assert "id" not in summary
    assert "machine_id" not in summary
    assert "action_type" not in summary
    assert "resource" not in summary


def test_full_consumed_flow_collects_grant_use_lifecycle_and_receipt(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = register_receipt(client, machine_id, use)

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["grants"]] == [grant["id"]]
    assert [item["id"] for item in body["grant_uses"]] == [use["use_id"]]
    lifecycle = body["lifecycle_events"]
    assert [item["type"] for item in lifecycle] == ["issued", "consumed"]
    assert {item["grant_id"] for item in lifecycle} == {grant["id"]}
    assert [item["id"] for item in body["execution_receipts"]] == [
        receipt["id"]
    ]


def test_revoked_flow_collects_grant_and_revocation_lifecycle_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    revocation = revoke_grant(client, machine_id, grant["id"])

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["grants"]] == [grant["id"]]
    assert body["grant_uses"] == []
    assert [item["type"] for item in body["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]
    assert body["execution_receipts"] == []
    stored = body["grants"][0]
    assert stored["status"] == "revoked"
    assert stored["revoked_at"] == revocation["revoked_at"]
    assert stored["consumed_at"] is None


# --------------------------------------------------------------------------- #
# Complete stored fields, verbatim
# --------------------------------------------------------------------------- #


def test_grant_emits_complete_stored_fields_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    consume_grant(client, machine_id, grant["id"])

    item = trace(client, machine_id, event["id"]).json()["grants"][0]
    assert list(item.keys()) == [
        "id",
        "machine_id",
        "event_id",
        "issued_at",
        "expires_at",
        "status",
        "consumed_at",
        "revoked_at",
    ]
    assert item["id"] == grant["id"]
    assert item["machine_id"] == machine_id
    assert item["event_id"] == event["id"]
    assert item["issued_at"] == grant["issued_at"]
    assert item["expires_at"] == grant["expires_at"]
    assert item["status"] == "consumed"
    assert item["consumed_at"] is not None
    assert item["revoked_at"] is None


def test_grant_status_is_never_derived_from_the_current_time(client):
    # A grant whose stored status is still ``active`` but whose expires_at is
    # in the past is emitted verbatim: the trace must not fabricate
    # ``expired`` from the current time or rewrite anything.
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :id",
        {"past": "2000-01-01T00:00:00Z", "id": grant["id"]},
    )

    item = trace(client, machine_id, event["id"]).json()["grants"][0]
    assert item["status"] == "active"
    assert item["expires_at"] == "2000-01-01T00:00:00Z"


def test_grant_use_emits_complete_stored_fields_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])

    item = trace(client, machine_id, event["id"]).json()["grant_uses"][0]
    assert list(item.keys()) == [
        "id",
        "grant_id",
        "machine_id",
        "event_id",
        "consumed_at",
    ]
    assert item == {
        "id": use["use_id"],
        "grant_id": grant["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "consumed_at": use["consumed_at"],
    }


def test_lifecycle_events_emit_complete_fields_and_chain_columns(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    consume_grant(client, machine_id, grant["id"])

    items = trace(client, machine_id, event["id"]).json()["lifecycle_events"]
    for item in items:
        assert list(item.keys()) == [
            "id",
            "machine_id",
            "grant_id",
            "authorization_event_id",
            "type",
            "occurred_at",
            "previous_event_id",
            "content_hash",
            "chain_hash",
        ]
        assert item["machine_id"] == machine_id
        assert item["grant_id"] == grant["id"]
        assert item["authorization_event_id"] == event["id"]
        assert item["content_hash"] is not None
        assert item["chain_hash"] is not None


def test_execution_receipt_emits_complete_fields_and_chain_columns(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = register_receipt(
        client, machine_id, use, outcome="failed", digest="b" * 64
    )

    item = trace(client, machine_id, event["id"]).json()[
        "execution_receipts"
    ][0]
    assert list(item.keys()) == [
        "id",
        "machine_id",
        "use_id",
        "grant_id",
        "authorization_event_id",
        "action_type",
        "resource",
        "outcome",
        "result_digest",
        "occurred_at",
        "previous_receipt_id",
        "content_hash",
        "chain_hash",
    ]
    assert item["id"] == receipt["id"]
    assert item["machine_id"] == machine_id
    assert item["use_id"] == use["use_id"]
    assert item["grant_id"] == grant["id"]
    assert item["authorization_event_id"] == event["id"]
    assert item["action_type"] == "read"
    assert item["resource"] == "res/x"
    assert item["outcome"] == "failed"
    assert item["result_digest"] == "b" * 64
    assert item["content_hash"] == receipt["content_hash"]
    assert item["chain_hash"] == receipt["chain_hash"]


# --------------------------------------------------------------------------- #
# Dangling parents and machine isolation
# --------------------------------------------------------------------------- #


def test_child_records_are_kept_when_their_grant_parent_is_missing(client):
    # A use, lifecycle event, and receipt selected by their own event
    # reference are still emitted when their stored grant/use parents do not
    # exist: a dangling parent never filters the child out.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :grant, :m, :e, :at)",
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "grant": GHOST_ID,
            "m": machine_id,
            "e": event["id"],
            "at": T1,
        },
    )
    execute_sql(
        client,
        "INSERT INTO authorization_grant_lifecycle_events (id, machine_id, "
        "grant_id, authorization_event_id, type, occurred_at, "
        "previous_event_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :grant, :e, 'revoked', :at, NULL, NULL, NULL)",
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "m": machine_id,
            "grant": GHOST_ID,
            "e": event["id"],
            "at": T2,
        },
    )
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :use, :grant, :e, 'read', 'res/x', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {
            "id": "33333333-3333-3333-3333-333333333333",
            "m": machine_id,
            "use": GHOST_ID,
            "grant": GHOST_ID,
            "e": event["id"],
            "digest": "c" * 64,
            "at": T3,
        },
    )

    body = trace(client, machine_id, event["id"]).json()
    assert body["grants"] == []
    assert [r["id"] for r in body["grant_uses"]] == [
        "11111111-1111-1111-1111-111111111111"
    ]
    assert body["grant_uses"][0]["grant_id"] == GHOST_ID
    assert [r["id"] for r in body["lifecycle_events"]] == [
        "22222222-2222-2222-2222-222222222222"
    ]
    assert body["lifecycle_events"][0]["grant_id"] == GHOST_ID
    assert [r["id"] for r in body["execution_receipts"]] == [
        "33333333-3333-3333-3333-333333333333"
    ]
    receipt = body["execution_receipts"][0]
    assert receipt["use_id"] == GHOST_ID
    assert receipt["grant_id"] == GHOST_ID


def test_records_of_another_event_and_another_machine_never_enter(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    # One global allow rule; each machine's own enabled declaration is what
    # its events need, so no rule is created a second time.
    create_rule(client)
    declare(client, machine_id)
    declare(client, other)
    event = record_event(client, machine_id)
    sibling = record_event(client, machine_id, resource="res/y")
    foreign = record_event(client, other)
    assert event["allowed"] is True
    assert sibling["allowed"] is True
    assert foreign["allowed"] is True

    # Real records on sibling and foreign events.
    sibling_grant = issue_grant(client, machine_id, sibling["id"])
    consume_grant(client, machine_id, sibling_grant["id"])
    foreign_grant = issue_grant(client, other, foreign["id"])
    foreign_use = consume_grant(client, other, foreign_grant["id"])
    register_receipt(
        client,
        other,
        foreign_use,
        resource="res/x",
        digest="d" * 64,
    )
    # A foreign-machine row that happens to name the selected event is still
    # other-machine data and must never enter any group.
    execute_sql(
        client,
        "INSERT INTO authorization_grants (id, machine_id, event_id, "
        "issued_at, expires_at, status, consumed_at, revoked_at) VALUES "
        "(:id, :m, :e, :at, :at, 'active', NULL, NULL)",
        {
            "id": "44444444-4444-4444-4444-444444444444",
            "m": other,
            "e": event["id"],
            "at": T1,
        },
    )
    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :grant, :m, :e, :at)",
        {
            "id": "55555555-5555-5555-5555-555555555555",
            "grant": GHOST_ID,
            "m": other,
            "e": event["id"],
            "at": T2,
        },
    )
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :ghost, :ghost, :e, 'read', 'res/x', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {
            "id": "66666666-6666-6666-6666-666666666666",
            "m": other,
            "ghost": GHOST_ID,
            "e": event["id"],
            "digest": "e" * 64,
            "at": T3,
        },
    )

    body = trace(client, machine_id, event["id"]).json()
    assert body["grants"] == []
    assert body["grant_uses"] == []
    assert body["lifecycle_events"] == []
    assert body["execution_receipts"] == []


# --------------------------------------------------------------------------- #
# Per-group ordering on each group's own business timestamp
# --------------------------------------------------------------------------- #


# Ordering tests insert several rows of one group for the same event, which
# the business UNIQUE constraints (one grant per event, one use per grant,
# one receipt per use) forbid. SQLite will not drop the index attached to an
# inline table UNIQUE constraint, so the table is recreated once with only
# its primary key and the exact columns the trace reads; foreign-key
# enforcement is off for the test databases, and each test gets a fresh
# database, so no other feature sees the unconstrained shape.
_UNCONSTRAINED_TABLE_DDL = {
    "authorization_grants": (
        "CREATE TABLE authorization_grants ("
        "id VARCHAR(36) PRIMARY KEY, "
        "machine_id VARCHAR(36) NOT NULL, "
        "event_id VARCHAR(36) NOT NULL, "
        "issued_at VARCHAR NOT NULL, "
        "expires_at VARCHAR NOT NULL, "
        "status VARCHAR NOT NULL, "
        "consumed_at VARCHAR, "
        "revoked_at VARCHAR)"
    ),
    "authorization_grant_uses": (
        "CREATE TABLE authorization_grant_uses ("
        "id VARCHAR(36) PRIMARY KEY, "
        "grant_id VARCHAR(36) NOT NULL, "
        "machine_id VARCHAR(36) NOT NULL, "
        "event_id VARCHAR(36) NOT NULL, "
        "consumed_at VARCHAR NOT NULL)"
    ),
    "execution_receipts": (
        "CREATE TABLE execution_receipts ("
        "id VARCHAR(36) PRIMARY KEY, "
        "machine_id VARCHAR(36) NOT NULL, "
        "use_id VARCHAR(36) NOT NULL, "
        "grant_id VARCHAR(36) NOT NULL, "
        "authorization_event_id VARCHAR(36) NOT NULL, "
        "action_type VARCHAR NOT NULL, "
        "resource VARCHAR NOT NULL, "
        "outcome VARCHAR NOT NULL, "
        "result_digest VARCHAR(64) NOT NULL, "
        "occurred_at VARCHAR NOT NULL, "
        "previous_receipt_id VARCHAR(36), "
        "content_hash VARCHAR(64), "
        "chain_hash VARCHAR(64))"
    ),
}


def _ensure_unconstrained_table(client, table):
    """Recreate ``table`` without its inline UNIQUE constraints, once.

    A UNIQUE-constraint index (``origin = 'u'``) marks the constrained shape;
    while one exists the table is dropped and rebuilt from
    :data:`_UNCONSTRAINED_TABLE_DDL`, after which later inserts in the same
    test find no such index and keep the already-inserted rows.
    """
    with client.app.state.engine.begin() as conn:
        origins = [row._mapping["origin"] for row in conn.execute(
            text(f"PRAGMA index_list({table})")
        ).all()]
        if "u" not in origins:
            return
        conn.execute(text(f"DROP TABLE {table}"))
        conn.execute(text(_UNCONSTRAINED_TABLE_DDL[table]))


def _insert_grant(client, record_id, machine_id, event_id, issued_at):
    _ensure_unconstrained_table(client, "authorization_grants")
    execute_sql(
        client,
        "INSERT INTO authorization_grants (id, machine_id, event_id, "
        "issued_at, expires_at, status, consumed_at, revoked_at) VALUES "
        "(:id, :m, :e, :at, :at, 'active', NULL, NULL)",
        {"id": record_id, "m": machine_id, "e": event_id, "at": issued_at},
    )


def _insert_use(client, record_id, machine_id, event_id, consumed_at):
    _ensure_unconstrained_table(client, "authorization_grant_uses")
    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :ghost, :m, :e, :at)",
        {
            "id": record_id,
            "ghost": GHOST_ID,
            "m": machine_id,
            "e": event_id,
            "at": consumed_at,
        },
    )


def _insert_lifecycle(client, record_id, machine_id, event_id, occurred_at):
    execute_sql(
        client,
        "INSERT INTO authorization_grant_lifecycle_events (id, machine_id, "
        "grant_id, authorization_event_id, type, occurred_at, "
        "previous_event_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :ghost, :e, 'issued', :at, NULL, NULL, NULL)",
        {
            "id": record_id,
            "m": machine_id,
            "ghost": GHOST_ID,
            "e": event_id,
            "at": occurred_at,
        },
    )


def _insert_receipt(client, record_id, machine_id, event_id, occurred_at):
    _ensure_unconstrained_table(client, "execution_receipts")
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :ghost, :ghost, :e, 'read', 'res/x', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {
            "id": record_id,
            "m": machine_id,
            "ghost": GHOST_ID,
            "e": event_id,
            "digest": "f" * 64,
            "at": occurred_at,
        },
    )


@pytest.mark.parametrize(
    "insert,group,timestamp_field",
    [
        (_insert_grant, "grants", "issued_at"),
        (_insert_use, "grant_uses", "consumed_at"),
        (_insert_lifecycle, "lifecycle_events", "occurred_at"),
        (_insert_receipt, "execution_receipts", "occurred_at"),
    ],
)
def test_exact_second_before_fractional_and_damaged_stamp_last(
    client, insert, group, timestamp_field
):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    # Insert in a deliberately non-chronological order.
    insert(client, "33333333-3333-3333-3333-333333333333",
           machine_id, event["id"], "not-a-time")
    insert(client, "22222222-2222-2222-2222-222222222222",
           machine_id, event["id"], T0_HALF)
    insert(client, "11111111-1111-1111-1111-111111111111",
           machine_id, event["id"], T0)

    items = trace(client, machine_id, event["id"]).json()[group]
    assert [item["id"] for item in items] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]
    # The damaged stamp is emitted verbatim, not repaired.
    assert items[2][timestamp_field] == "not-a-time"


@pytest.mark.parametrize(
    "insert,group",
    [
        (_insert_grant, "grants"),
        (_insert_use, "grant_uses"),
        (_insert_lifecycle, "lifecycle_events"),
        (_insert_receipt, "execution_receipts"),
    ],
)
def test_same_instant_ties_break_by_id_in_every_group(client, insert, group):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    insert(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
           machine_id, event["id"], T0)
    insert(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
           machine_id, event["id"], T0)

    items = trace(client, machine_id, event["id"]).json()[group]
    assert [item["id"] for item in items] == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]


# --------------------------------------------------------------------------- #
# Serialization, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_a_single_newline(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)

    response = trace(client, machine_id, event["id"])
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    json.loads(response.content)


def test_group_field_order_is_fixed(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    raw = trace(client, machine_id, event["id"]).content.decode("utf-8")
    positions = [raw.index(f'"{name}"') for name in GROUP_ORDER]
    assert positions == sorted(positions)


def test_repeated_calls_are_byte_identical_and_read_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    def snapshot():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(text(f"SELECT * FROM {table}")).fetchall()
                for table in (
                    "authorization_decision_events",
                    "authorization_grants",
                    "authorization_grant_uses",
                    "authorization_grant_lifecycle_events",
                    "execution_receipts",
                )
            }

    before = snapshot()
    first = trace(client, machine_id, event["id"]).content
    second = trace(client, machine_id, event["id"]).content
    assert first == second
    assert snapshot() == before


def test_trace_survives_restart_byte_identical(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = record_event(first, machine_id, allow=True)
        grant = issue_grant(first, machine_id, event["id"])
        use = consume_grant(first, machine_id, grant["id"])
        register_receipt(first, machine_id, use)
        body_before = trace(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        body_after = trace(second, machine_id, event["id"]).content
        assert body_after == body_before
        assert list(json.loads(body_after).keys()) == list(GROUP_ORDER)


def test_existing_accountability_trace_is_unaffected(client):
    # The new entry shares the event but never changes the existing
    # accountability-trace response or its six groups.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/accountability-trace"
    )
    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "event_summary",
        "evidence",
        "incidents",
        "status_history",
        "responsibility_assignments",
        "causal_links",
    ]


def test_trace_body_has_no_floating_point_tokens(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    raw = trace(client, machine_id, event["id"]).content
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    assert b"-0.0" not in raw
