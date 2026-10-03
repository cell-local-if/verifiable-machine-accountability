"""Tests for the read-only single-event execution accountability trace.

Covers `GET /machines/{machine_id}/authorization-decision-events/{event_id}/
execution-trace`: GET-only 405 (including HEAD) without reading records,
``invalid_query`` 422 for any query parameter, a repeated parameter, or a
carried body before the machine/event lookup, ``not_found`` 404 for a missing
machine/event and for an event owned by another machine with no trace data,
``internal_error`` 500 on a real read fault with no event summary or
execution-chain arrays, the fixed five-group response (the event summary
object plus the grants, grant-uses, lifecycle-events, and execution-receipts
arrays), per-machine and per-event selection on each record's own columns,
child records kept when a parent dangles, complete stored fields emitted
verbatim (the stored grant ``status`` included, never a derived expiry view),
tolerant (instant, id) ordering on each group's own business timestamp with
damaged stamps last, empty-group preservation for grant-less and denied
events, machine isolation, compact single-newline JSON, read-only
byte-identical repeats, and persistence across restarts.
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
GHOST_GRANT_ID = "99999999-9999-9999-9999-999999999999"
GHOST_USE_ID = "88888888-8888-8888-8888-888888888888"

GROUP_ORDER = (
    "event_summary",
    "grants",
    "grant_uses",
    "lifecycle_events",
    "execution_receipts",
)

TRACE_TABLES = (
    "authorization_decision_events",
    "authorization_grants",
    "authorization_grant_uses",
    "authorization_grant_lifecycle_events",
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


def record_event(client, machine_id, resource="res/a"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201, response.text
    return response.json()


def allow_event(client, machine_id, resource="res/a"):
    """One committed ``allowed_by_policy`` allow event of the path machine."""
    declare(client, machine_id)
    create_rule(client)
    return record_event(client, machine_id, resource=resource)


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


def record_receipt(client, machine_id, use_id, resource="res/a",
                   outcome="succeeded"):
    response = client.post(
        f"/machines/{machine_id}/execution-receipts",
        json={
            "use_id": use_id,
            "action_type": "read",
            "resource": resource,
            "outcome": outcome,
            "result_digest": "a" * 64,
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


def insert_grant(client, record_id, machine_id, event_id, issued_at,
                 expires_at="2026-03-01T00:05:00Z", status="active",
                 consumed_at=None, revoked_at=None):
    execute_sql(
        client,
        "INSERT INTO authorization_grants (id, machine_id, event_id, "
        "issued_at, expires_at, status, consumed_at, revoked_at) VALUES "
        "(:id, :m, :e, :issued, :expires, :status, :consumed, :revoked)",
        {
            "id": record_id,
            "m": machine_id,
            "e": event_id,
            "issued": issued_at,
            "expires": expires_at,
            "status": status,
            "consumed": consumed_at,
            "revoked": revoked_at,
        },
    )


def insert_use(client, record_id, machine_id, event_id, consumed_at,
               grant_id=GHOST_GRANT_ID):
    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :g, :m, :e, :at)",
        {"id": record_id, "g": grant_id, "m": machine_id, "e": event_id,
         "at": consumed_at},
    )


def insert_lifecycle(client, record_id, machine_id, event_id, occurred_at,
                     grant_id=GHOST_GRANT_ID, event_type="issued"):
    execute_sql(
        client,
        "INSERT INTO authorization_grant_lifecycle_events (id, machine_id, "
        "grant_id, authorization_event_id, type, occurred_at, "
        "previous_event_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :g, :e, :type, :at, NULL, NULL, NULL)",
        {"id": record_id, "m": machine_id, "g": grant_id, "e": event_id,
         "type": event_type, "at": occurred_at},
    )


def insert_receipt(client, record_id, machine_id, event_id, occurred_at,
                   use_id=None, grant_id=GHOST_GRANT_ID):
    # ``use_id`` is unique per receipt row; default it to the record id so
    # repeated direct inserts never collide on the constraint.
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :u, :g, :e, 'read', 'res/a', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {"id": record_id, "m": machine_id, "u": use_id or record_id,
         "g": grant_id, "e": event_id, "digest": "b" * 64, "at": occurred_at},
    )


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    response = getattr(client, method)(trace_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"event_summary" not in response.content


def test_method_routing_does_not_read_records(client):
    # With every table the trace could read dropped, only routing is in play.
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        for table in TRACE_TABLES:
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
    event = allow_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_repeated_parameter_is_invalid_query_even_with_a_value(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + "?x=1&x=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
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
    event = allow_event(client, machine_id)
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
    event = allow_event(client, machine_id)
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
    event = allow_event(client, machine_id)
    response = client.get(trace_url(other, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content


def test_event_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
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
    event = allow_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"lifecycle_events" not in response.content


# --------------------------------------------------------------------------- #
# Success shape and the five fixed groups
# --------------------------------------------------------------------------- #


def test_event_with_no_grant_has_summary_and_four_empty_arrays(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    body = trace(client, machine_id, event["id"]).json()
    assert list(body.keys()) == list(GROUP_ORDER)
    assert isinstance(body["event_summary"], dict)
    for group in GROUP_ORDER[1:]:
        assert body[group] == []


def test_denied_event_returns_an_empty_execution_chain(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    assert event["allowed"] is False

    body = trace(client, machine_id, event["id"]).json()
    assert body["event_summary"]["allowed"] is False
    assert body["grants"] == []
    assert body["grant_uses"] == []
    assert body["lifecycle_events"] == []
    assert body["execution_receipts"] == []


def test_event_summary_carries_result_reason_moment_and_chain_fields(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

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
    # Identifying/context fields are intentionally not part of the summary.
    assert "id" not in summary
    assert "machine_id" not in summary
    assert "action_type" not in summary
    assert "resource" not in summary


# --------------------------------------------------------------------------- #
# The full execution chain: issue, consume, receipt, and revocation
# --------------------------------------------------------------------------- #


def test_full_flow_collects_grant_use_lifecycle_and_receipt(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = record_receipt(client, machine_id, use["use_id"])

    body = trace(client, machine_id, event["id"]).json()

    assert [item["id"] for item in body["grants"]] == [grant["id"]]
    grant_item = body["grants"][0]
    assert list(grant_item.keys()) == [
        "id",
        "machine_id",
        "event_id",
        "issued_at",
        "expires_at",
        "status",
        "consumed_at",
        "revoked_at",
    ]
    assert grant_item == {
        "id": grant["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "issued_at": grant["issued_at"],
        "expires_at": grant["expires_at"],
        "status": "consumed",
        "consumed_at": use["consumed_at"],
        "revoked_at": None,
    }

    assert [item["id"] for item in body["grant_uses"]] == [use["use_id"]]
    use_item = body["grant_uses"][0]
    assert list(use_item.keys()) == [
        "id",
        "machine_id",
        "grant_id",
        "event_id",
        "consumed_at",
    ]
    assert use_item == {
        "id": use["use_id"],
        "machine_id": machine_id,
        "grant_id": grant["id"],
        "event_id": event["id"],
        "consumed_at": use["consumed_at"],
    }

    # The complete flow surfaces the lifecycle in business order:
    # issued first, then the single consumption.
    assert [item["type"] for item in body["lifecycle_events"]] == [
        "issued",
        "consumed",
    ]
    for item in body["lifecycle_events"]:
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
    assert body["lifecycle_events"][0]["occurred_at"] == grant["issued_at"]
    assert body["lifecycle_events"][1]["occurred_at"] == use["consumed_at"]

    assert [item["id"] for item in body["execution_receipts"]] == [
        receipt["id"]
    ]
    receipt_item = body["execution_receipts"][0]
    assert list(receipt_item.keys()) == [
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
    assert receipt_item["machine_id"] == machine_id
    assert receipt_item["use_id"] == use["use_id"]
    assert receipt_item["grant_id"] == grant["id"]
    assert receipt_item["authorization_event_id"] == event["id"]
    assert receipt_item["action_type"] == "read"
    assert receipt_item["resource"] == "res/a"
    assert receipt_item["outcome"] == "succeeded"
    assert receipt_item["result_digest"] == "a" * 64
    assert receipt_item["occurred_at"] == receipt["occurred_at"]
    assert receipt_item["previous_receipt_id"] == receipt["previous_receipt_id"]
    assert receipt_item["content_hash"] == receipt["content_hash"]
    assert receipt_item["chain_hash"] == receipt["chain_hash"]


def test_revoked_grant_flow_traces_issue_and_revocation(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    revocation = revoke_grant(client, machine_id, grant["id"])

    body = trace(client, machine_id, event["id"]).json()
    grant_item = body["grants"][0]
    # The stored terminal state is emitted verbatim.
    assert grant_item["status"] == "revoked"
    assert grant_item["revoked_at"] == revocation["revoked_at"]
    assert grant_item["consumed_at"] is None
    assert [item["type"] for item in body["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]
    assert body["grant_uses"] == []
    assert body["execution_receipts"] == []


def test_issued_but_untouched_grant_traces_only_its_issue(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["grants"]] == [grant["id"]]
    assert body["grants"][0]["status"] == "active"
    assert body["grants"][0]["consumed_at"] is None
    assert body["grants"][0]["revoked_at"] is None
    assert [item["type"] for item in body["lifecycle_events"]] == ["issued"]
    assert body["grant_uses"] == []
    assert body["execution_receipts"] == []


def test_grant_status_is_stored_verbatim_never_derived_from_now(client):
    # A grant whose TTL has already passed keeps its stored ``active``
    # status: the read-only trace never derives an ``expired`` view from the
    # current instant and never rewrites the stored row.
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    insert_grant(
        client,
        "11111111-1111-1111-1111-111111111111",
        machine_id,
        event["id"],
        issued_at="2020-01-01T00:00:00Z",
        expires_at="2020-01-01T00:05:00Z",
        status="active",
    )

    grant_item = trace(client, machine_id, event["id"]).json()["grants"][0]
    assert grant_item["status"] == "active"
    assert grant_item["issued_at"] == "2020-01-01T00:00:00Z"
    assert grant_item["expires_at"] == "2020-01-01T00:05:00Z"


# --------------------------------------------------------------------------- #
# Membership, dangling parents, machine isolation
# --------------------------------------------------------------------------- #


def test_dangling_parent_never_filters_child_records(client):
    # Directly inserted rows are selected by their own (machine_id, event)
    # columns even when their stored grant/use parents do not exist.
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    insert_use(client, "11111111-1111-1111-1111-111111111111",
               machine_id, event["id"], T1)
    insert_lifecycle(client, "22222222-2222-2222-2222-222222222222",
                     machine_id, event["id"], T2, event_type="consumed")
    insert_receipt(client, "33333333-3333-3333-3333-333333333333",
                   machine_id, event["id"], T3, use_id=GHOST_USE_ID)

    body = trace(client, machine_id, event["id"]).json()
    assert body["grants"] == []
    assert [r["id"] for r in body["grant_uses"]] == [
        "11111111-1111-1111-1111-111111111111"
    ]
    assert body["grant_uses"][0]["grant_id"] == GHOST_GRANT_ID
    assert [r["id"] for r in body["lifecycle_events"]] == [
        "22222222-2222-2222-2222-222222222222"
    ]
    assert body["lifecycle_events"][0]["grant_id"] == GHOST_GRANT_ID
    assert [r["id"] for r in body["execution_receipts"]] == [
        "33333333-3333-3333-3333-333333333333"
    ]
    assert body["execution_receipts"][0]["use_id"] == GHOST_USE_ID
    assert body["execution_receipts"][0]["grant_id"] == GHOST_GRANT_ID


def test_records_of_another_event_and_another_machine_never_enter(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    event = allow_event(client, machine_id)
    sibling = record_event(client, machine_id, resource="res/b")
    declare(client, other)
    foreign = record_event(client, other, resource="res/c")

    # A full foreign flow on the other machine and a sibling-event flow on
    # this machine: none of it may enter the selected event's trace.
    foreign_grant = issue_grant(client, other, foreign["id"])
    foreign_use = consume_grant(client, other, foreign_grant["id"])
    record_receipt(client, other, foreign_use["use_id"], resource="res/c")
    sibling_grant = issue_grant(client, machine_id, sibling["id"])
    sibling_use = consume_grant(client, machine_id, sibling_grant["id"])
    record_receipt(client, machine_id, sibling_use["use_id"], resource="res/b")

    # Other-machine rows that happen to reference the selected event id are
    # still other-machine data and must never enter. (The grant table
    # enforces one grant per event globally, so the cross-machine reference
    # is only possible on the use, lifecycle, and receipt rows.)
    insert_use(client, "11111111-1111-1111-1111-111111111111",
               other, event["id"], T1)
    insert_lifecycle(client, "22222222-2222-2222-2222-222222222222",
                     other, event["id"], T2)
    insert_receipt(client, "33333333-3333-3333-3333-333333333333",
                   other, event["id"], T3)

    body = trace(client, machine_id, event["id"]).json()
    assert body["grants"] == []
    assert body["grant_uses"] == []
    assert body["lifecycle_events"] == []
    assert body["execution_receipts"] == []


def test_tracing_one_event_does_not_disturb_an_unrelated_event_trace(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    sibling = record_event(client, machine_id, resource="res/b")
    grant = issue_grant(client, machine_id, event["id"])
    sibling_grant = issue_grant(client, machine_id, sibling["id"])

    first = trace(client, machine_id, event["id"]).json()
    second = trace(client, machine_id, sibling["id"]).json()
    assert [item["id"] for item in first["grants"]] == [grant["id"]]
    assert [item["id"] for item in second["grants"]] == [sibling_grant["id"]]


# --------------------------------------------------------------------------- #
# Ordering on each group's own business timestamp
# --------------------------------------------------------------------------- #


def test_exact_second_sorts_before_fractional_and_damaged_stamp_sorts_last(
    client,
):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    # Insert in a deliberately non-chronological order.
    insert_lifecycle(client, "33333333-3333-3333-3333-333333333333",
                     machine_id, event["id"], "not-a-time")
    insert_lifecycle(client, "22222222-2222-2222-2222-222222222222",
                     machine_id, event["id"], T0_HALF)
    insert_lifecycle(client, "11111111-1111-1111-1111-111111111111",
                     machine_id, event["id"], T0)

    items = trace(client, machine_id, event["id"]).json()["lifecycle_events"]
    assert [item["id"] for item in items] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]
    # The damaged stamp is emitted verbatim, not repaired.
    assert items[2]["occurred_at"] == "not-a-time"


def test_damaged_stamps_sort_among_themselves_by_id(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    insert_receipt(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                   machine_id, event["id"], "not-a-time")
    insert_receipt(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                   machine_id, event["id"], "also-not-a-time")
    insert_receipt(client, "cccccccc-cccc-cccc-cccc-cccccccccccc",
                   machine_id, event["id"], T0)

    items = trace(client, machine_id, event["id"]).json()["execution_receipts"]
    assert [item["id"] for item in items] == [
        "cccccccc-cccc-cccc-cccc-cccccccccccc",
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]


def test_same_instant_ties_break_by_id_in_every_array_group(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    insert_use(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
               machine_id, event["id"], T0, grant_id=GHOST_GRANT_ID)
    insert_use(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
               machine_id, event["id"], T0,
               grant_id="77777777-7777-7777-7777-777777777777")
    insert_lifecycle(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                     machine_id, event["id"], T0)
    insert_lifecycle(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                     machine_id, event["id"], T0)
    insert_receipt(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                   machine_id, event["id"], T0)
    insert_receipt(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                   machine_id, event["id"], T0)

    body = trace(client, machine_id, event["id"]).json()
    for group in ("grant_uses", "lifecycle_events", "execution_receipts"):
        assert [item["id"] for item in body[group]] == [
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        ], group


def test_each_group_orders_by_its_own_business_timestamp(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    # Uses sort by consumed_at, receipts by occurred_at — not by insertion
    # order or by any other column.
    insert_use(client, "11111111-1111-1111-1111-111111111111",
               machine_id, event["id"], T2, grant_id=GHOST_GRANT_ID)
    insert_use(client, "22222222-2222-2222-2222-222222222222",
               machine_id, event["id"], T1,
               grant_id="77777777-7777-7777-7777-777777777777")
    insert_receipt(client, "33333333-3333-3333-3333-333333333333",
                   machine_id, event["id"], T2)
    insert_receipt(client, "44444444-4444-4444-4444-444444444444",
                   machine_id, event["id"], T1, use_id="66666666-6666-6666-6666-666666666666")

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["grant_uses"]] == [
        "22222222-2222-2222-2222-222222222222",
        "11111111-1111-1111-1111-111111111111",
    ]
    assert [item["id"] for item in body["execution_receipts"]] == [
        "44444444-4444-4444-4444-444444444444",
        "33333333-3333-3333-3333-333333333333",
    ]


# --------------------------------------------------------------------------- #
# Serialization, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_a_single_newline(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)

    response = trace(client, machine_id, event["id"])
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    json.loads(response.content)


def test_group_field_order_is_fixed(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    record_receipt(client, machine_id, use["use_id"])

    raw = trace(client, machine_id, event["id"]).content.decode("utf-8")
    positions = [raw.index(f'"{name}"') for name in GROUP_ORDER]
    assert positions == sorted(positions)


def test_repeated_calls_are_byte_identical_and_read_only(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    record_receipt(client, machine_id, use["use_id"])

    def snapshot():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(text(f"SELECT * FROM {table}")).fetchall()
                for table in TRACE_TABLES
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
        event = allow_event(first, machine_id)
        grant = issue_grant(first, machine_id, event["id"])
        use = consume_grant(first, machine_id, grant["id"])
        record_receipt(first, machine_id, use["use_id"])
        body_before = trace(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        body_after = trace(second, machine_id, event["id"]).content
        assert body_after == body_before
        assert list(json.loads(body_after).keys()) == list(GROUP_ORDER)


def test_trace_body_has_no_floating_point_tokens(client):
    machine_id = create_machine(client)
    event = allow_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    record_receipt(client, machine_id, use["use_id"])

    raw = trace(client, machine_id, event["id"]).content
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    assert b"-0.0" not in raw
