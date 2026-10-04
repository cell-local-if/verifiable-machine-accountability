"""Tests for the read-only single-grant execution accountability trace.

Covers `GET /machines/{machine_id}/authorization-grants/{grant_id}/
grant-trace`: GET-only 405 (including HEAD) without reading records,
``invalid_query`` 422 for any query parameter, a repeated parameter, or a
carried body before the machine/grant lookup, ``not_found`` 404 for a
missing machine/grant and for a grant owned by another machine with no
trace data, ``internal_error`` 500 on a real read fault with no grant
object or record arrays, the fixed four-group response (the grant object
plus the grant-uses, lifecycle-events, and execution-receipts arrays), the
grant object's audit-list fields and derived-status semantics (``expired``
never persisted), per-machine selection on each array's stored ``grant_id``
equal to the path grant, complete stored fields emitted verbatim without
repair or fabrication, tolerant (instant, id) ordering per array's own
business timestamp with damaged stamps last, empty-array preservation,
machine isolation, compact single-newline JSON, read-only byte-identical
repeats, and persistence across restarts.
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
GHOST_ID = "99999999-9999-9999-9999-999999999999"

GROUP_ORDER = (
    "grant",
    "grant_uses",
    "lifecycle_events",
    "execution_receipts",
)

GRANT_FIELDS = [
    "id",
    "machine_id",
    "event_id",
    "issued_at",
    "expires_at",
    "status",
    "consumed_at",
    "revoked_at",
    "use_id",
    "use_at",
]


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


def record_event(client, machine_id, resource="res/x", allow=True):
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


def trace_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/"
        f"{grant_id}/grant-trace"
    )


def trace(client, machine_id, grant_id):
    response = client.get(trace_url(machine_id, grant_id))
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
    grant = issue_grant(client, machine_id, event["id"])
    response = getattr(client, method)(trace_url(machine_id, grant["id"]))
    assert response.status_code == 405
    assert b"grant_uses" not in response.content


def test_method_routing_does_not_read_records(client):
    # With every table the trace could read dropped, only routing is in play.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    with client.app.state.engine.begin() as conn:
        for table in (
            "execution_receipts",
            "authorization_grant_lifecycle_events",
            "authorization_grant_uses",
            "authorization_grants",
            "machines",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(
            trace_url(machine_id, grant["id"])
        ).status_code == 405


@pytest.mark.parametrize(
    "query",
    ["?x=1", "?limit=10", "?x=1&x=2", "?=", "?foo"],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(machine_id, grant["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_repeated_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(machine_id, grant["id"]) + "?x=1&x=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.request(
        "GET",
        trace_url(machine_id, grant["id"]),
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
    grant = issue_grant(client, machine_id, event["id"])
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grants"))
        conn.execute(text("DROP TABLE machines"))
    assert client.get(
        trace_url(machine_id, grant["id"]) + "?x=1"
    ).status_code == 422
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(
            trace_url(machine_id, grant["id"])
        ).status_code == 405


def test_missing_machine_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(MISSING_ID, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content
    assert b"execution_receipts" not in response.content


def test_missing_grant_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    response = client.get(trace_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content
    assert b"lifecycle_events" not in response.content


def test_grant_owned_by_another_machine_is_404(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(other, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content


def test_grant_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grants"))
    response = client.get(trace_url(machine_id, grant["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"grant_uses" not in response.content
    assert b"execution_receipts" not in response.content


@pytest.mark.parametrize(
    "table",
    [
        "authorization_grant_uses",
        "authorization_grant_lifecycle_events",
        "execution_receipts",
    ],
)
def test_associated_read_failure_is_500_with_no_trace_data(client, table):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    response = client.get(trace_url(machine_id, grant["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"grant_uses" not in response.content
    assert b"lifecycle_events" not in response.content


# --------------------------------------------------------------------------- #
# Success shape and the four fixed groups
# --------------------------------------------------------------------------- #


def test_fresh_grant_has_the_grant_object_and_three_empty_arrays(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])

    response = trace(client, machine_id, grant["id"])
    body = response.json()
    assert list(body.keys()) == list(GROUP_ORDER)
    assert isinstance(body["grant"], dict)
    assert body["grant_uses"] == []
    assert body["lifecycle_events"][0]["type"] == "issued"
    assert body["execution_receipts"] == []


def test_grant_object_uses_the_audit_list_fields_and_shape(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])

    item = trace(client, machine_id, grant["id"]).json()["grant"]
    assert list(item.keys()) == GRANT_FIELDS
    assert item["id"] == grant["id"]
    assert item["machine_id"] == machine_id
    assert item["event_id"] == event["id"]
    assert item["issued_at"] == grant["issued_at"]
    assert item["expires_at"] == grant["expires_at"]
    assert item["status"] == "active"
    assert item["consumed_at"] is None
    assert item["revoked_at"] is None
    assert item["use_id"] is None
    assert item["use_at"] is None


def test_consumed_grant_presents_terminal_status_and_use_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])

    item = trace(client, machine_id, grant["id"]).json()["grant"]
    assert item["status"] == "consumed"
    assert item["consumed_at"] == use["consumed_at"]
    assert item["revoked_at"] is None
    assert item["use_id"] == use["use_id"]
    assert item["use_at"] == use["consumed_at"]


def test_revoked_grant_presents_terminal_status_without_use_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    revocation = revoke_grant(client, machine_id, grant["id"])

    item = trace(client, machine_id, grant["id"]).json()["grant"]
    assert item["status"] == "revoked"
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["consumed_at"] is None
    assert item["use_id"] is None
    assert item["use_at"] is None


def test_expired_status_is_derived_and_never_persisted(client):
    # A non-terminal grant past its expires_at presents ``expired`` at read
    # time; the stored status is never rewritten.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :id",
        {"past": "2000-01-01T00:00:00Z", "id": grant["id"]},
    )

    item = trace(client, machine_id, grant["id"]).json()["grant"]
    assert item["status"] == "expired"
    assert item["expires_at"] == "2000-01-01T00:00:00Z"
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id"),
            {"id": grant["id"]},
        ).scalar()
    assert stored == "active"


def test_terminal_status_is_kept_even_after_the_ttl_elapses(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    consume_grant(client, machine_id, grant["id"])
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :id",
        {"past": "2000-01-01T00:00:00Z", "id": grant["id"]},
    )

    item = trace(client, machine_id, grant["id"]).json()["grant"]
    assert item["status"] == "consumed"


def test_full_consumed_flow_collects_use_lifecycle_and_receipt(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = register_receipt(client, machine_id, use)

    body = trace(client, machine_id, grant["id"]).json()
    assert [item["id"] for item in body["grant_uses"]] == [use["use_id"]]
    lifecycle = body["lifecycle_events"]
    assert [item["type"] for item in lifecycle] == ["issued", "consumed"]
    assert {item["grant_id"] for item in lifecycle} == {grant["id"]}
    assert [item["id"] for item in body["execution_receipts"]] == [
        receipt["id"]
    ]


def test_revoked_flow_collects_revocation_lifecycle_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    revoke_grant(client, machine_id, grant["id"])

    body = trace(client, machine_id, grant["id"]).json()
    assert body["grant_uses"] == []
    assert [item["type"] for item in body["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]
    assert body["execution_receipts"] == []


# --------------------------------------------------------------------------- #
# Complete stored fields, verbatim
# --------------------------------------------------------------------------- #


def test_grant_use_emits_complete_stored_fields_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])

    item = trace(client, machine_id, grant["id"]).json()["grant_uses"][0]
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
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    consume_grant(client, machine_id, grant["id"])

    items = trace(client, machine_id, grant["id"]).json()["lifecycle_events"]
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
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = register_receipt(
        client, machine_id, use, outcome="failed", digest="b" * 64
    )

    item = trace(client, machine_id, grant["id"]).json()[
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
# Dangling references and machine isolation
# --------------------------------------------------------------------------- #


def test_child_records_with_dangling_references_are_kept_verbatim(client):
    # A use, lifecycle event, or receipt whose stored event/use reference
    # names nothing is still traced when its stored grant_id matches the
    # path grant: a dangling or damaged value is never repaired or filtered.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])

    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :grant, :m, :e, :at)",
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "grant": grant["id"],
            "m": machine_id,
            "e": GHOST_ID,
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
            "grant": grant["id"],
            "e": GHOST_ID,
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
            "grant": grant["id"],
            "e": GHOST_ID,
            "digest": "c" * 64,
            "at": T3,
        },
    )

    body = trace(client, machine_id, grant["id"]).json()
    uses = body["grant_uses"]
    assert "11111111-1111-1111-1111-111111111111" in [r["id"] for r in uses]
    dangling_use = next(
        r for r in uses if r["id"] == "11111111-1111-1111-1111-111111111111"
    )
    assert dangling_use["event_id"] == GHOST_ID
    lifecycle_ids = [r["id"] for r in body["lifecycle_events"]]
    assert "22222222-2222-2222-2222-222222222222" in lifecycle_ids
    dangling_event = next(
        r
        for r in body["lifecycle_events"]
        if r["id"] == "22222222-2222-2222-2222-222222222222"
    )
    assert dangling_event["authorization_event_id"] == GHOST_ID
    receipts = body["execution_receipts"]
    assert [r["id"] for r in receipts] == [
        "33333333-3333-3333-3333-333333333333"
    ]
    assert receipts[0]["use_id"] == GHOST_ID
    assert receipts[0]["authorization_event_id"] == GHOST_ID


def test_records_of_another_grant_and_another_machine_never_enter(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    # One global allow rule; each machine's own enabled declaration is what
    # its events need, so no rule is created a second time.
    create_rule(client)
    declare(client, machine_id)
    declare(client, other)
    event = record_event(client, machine_id, allow=False)
    sibling_event = record_event(client, machine_id, resource="res/y",
                                 allow=False)
    foreign_event = record_event(client, other, allow=False)

    grant = issue_grant(client, machine_id, event["id"])
    # Real records on a sibling grant of the same machine and on a foreign
    # machine's grant.
    sibling_grant = issue_grant(client, machine_id, sibling_event["id"])
    consume_grant(client, machine_id, sibling_grant["id"])
    foreign_grant = issue_grant(client, other, foreign_event["id"])
    foreign_use = consume_grant(client, other, foreign_grant["id"])
    register_receipt(client, other, foreign_use, digest="d" * 64)
    # A foreign-machine row that happens to name the path grant id is still
    # other-machine data and must never enter any array.
    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :grant, :m, :e, :at)",
        {
            "id": "55555555-5555-5555-5555-555555555555",
            "grant": grant["id"],
            "m": other,
            "e": foreign_event["id"],
            "at": T2,
        },
    )
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :ghost, :grant, :e, 'read', 'res/x', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {
            "id": "66666666-6666-6666-6666-666666666666",
            "m": other,
            "ghost": GHOST_ID,
            "grant": grant["id"],
            "e": foreign_event["id"],
            "digest": "e" * 64,
            "at": T3,
        },
    )

    body = trace(client, machine_id, grant["id"]).json()
    assert body["grant"]["id"] == grant["id"]
    # Only the path grant's own ``issued`` lifecycle event is present.
    assert [item["type"] for item in body["lifecycle_events"]] == ["issued"]
    assert body["grant_uses"] == []
    assert body["execution_receipts"] == []


# --------------------------------------------------------------------------- #
# Per-array ordering on each array's own business timestamp
# --------------------------------------------------------------------------- #


# Ordering tests insert several rows of one array for the same grant, which
# the business UNIQUE constraints (one use per grant, one receipt per use)
# forbid. SQLite will not drop the index attached to an inline table UNIQUE
# constraint, so the table is recreated once with only its primary key and
# the exact columns the trace reads; foreign-key enforcement is off for the
# test databases, and each test gets a fresh database, so no other feature
# sees the unconstrained shape.
_UNCONSTRAINED_TABLE_DDL = {
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


def _insert_use(client, record_id, machine_id, grant_id, consumed_at):
    _ensure_unconstrained_table(client, "authorization_grant_uses")
    execute_sql(
        client,
        "INSERT INTO authorization_grant_uses (id, grant_id, machine_id, "
        "event_id, consumed_at) VALUES (:id, :grant, :m, :e, :at)",
        {
            "id": record_id,
            "grant": grant_id,
            "m": machine_id,
            "e": GHOST_ID,
            "at": consumed_at,
        },
    )


def _insert_lifecycle(client, record_id, machine_id, grant_id, occurred_at):
    execute_sql(
        client,
        "INSERT INTO authorization_grant_lifecycle_events (id, machine_id, "
        "grant_id, authorization_event_id, type, occurred_at, "
        "previous_event_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :grant, :e, 'issued', :at, NULL, NULL, NULL)",
        {
            "id": record_id,
            "m": machine_id,
            "grant": grant_id,
            "e": GHOST_ID,
            "at": occurred_at,
        },
    )


def _insert_receipt(client, record_id, machine_id, grant_id, occurred_at):
    _ensure_unconstrained_table(client, "execution_receipts")
    execute_sql(
        client,
        "INSERT INTO execution_receipts (id, machine_id, use_id, grant_id, "
        "authorization_event_id, action_type, resource, outcome, "
        "result_digest, occurred_at, previous_receipt_id, content_hash, "
        "chain_hash) VALUES (:id, :m, :ghost, :grant, :e, 'read', 'res/x', "
        "'succeeded', :digest, :at, NULL, NULL, NULL)",
        {
            "id": record_id,
            "m": machine_id,
            "ghost": GHOST_ID,
            "grant": grant_id,
            "e": GHOST_ID,
            "digest": "f" * 64,
            "at": occurred_at,
        },
    )


@pytest.mark.parametrize(
    "insert,group,timestamp_field",
    [
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
    grant = issue_grant(client, machine_id, event["id"])

    # Insert in a deliberately non-chronological order.
    insert(client, "33333333-3333-3333-3333-333333333333",
           machine_id, grant["id"], "not-a-time")
    insert(client, "22222222-2222-2222-2222-222222222222",
           machine_id, grant["id"], T0_HALF)
    insert(client, "11111111-1111-1111-1111-111111111111",
           machine_id, grant["id"], T0)

    items = [
        item
        for item in trace(client, machine_id, grant["id"]).json()[group]
        if item["id"] in {
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        }
    ]
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
        (_insert_use, "grant_uses"),
        (_insert_lifecycle, "lifecycle_events"),
        (_insert_receipt, "execution_receipts"),
    ],
)
def test_same_instant_ties_break_by_id_in_every_array(client, insert, group):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    insert(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
           machine_id, grant["id"], T0)
    insert(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
           machine_id, grant["id"], T0)

    ids = [
        item["id"]
        for item in trace(client, machine_id, grant["id"]).json()[group]
        if item["id"] in {
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        }
    ]
    assert ids == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]


# --------------------------------------------------------------------------- #
# Serialization, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_a_single_newline(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])

    response = trace(client, machine_id, grant["id"])
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    json.loads(response.content)


def test_group_field_order_is_fixed(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    raw = trace(client, machine_id, grant["id"]).content.decode("utf-8")
    positions = [raw.index(f'"{name}"') for name in GROUP_ORDER]
    assert positions == sorted(positions)


def test_repeated_calls_are_byte_identical_and_read_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    def snapshot():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(text(f"SELECT * FROM {table}")).fetchall()
                for table in (
                    "authorization_grants",
                    "authorization_grant_uses",
                    "authorization_grant_lifecycle_events",
                    "execution_receipts",
                )
            }

    before = snapshot()
    first = trace(client, machine_id, grant["id"]).content
    second = trace(client, machine_id, grant["id"]).content
    assert first == second
    assert snapshot() == before


def test_trace_survives_restart_byte_identical(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = record_event(first, machine_id)
        grant = issue_grant(first, machine_id, event["id"])
        use = consume_grant(first, machine_id, grant["id"])
        register_receipt(first, machine_id, use)
        body_before = trace(first, machine_id, grant["id"]).content

    with TestClient(app) as second:
        body_after = trace(second, machine_id, grant["id"]).content
        assert body_after == body_before
        assert list(json.loads(body_after).keys()) == list(GROUP_ORDER)


def test_existing_grant_views_are_unaffected(client):
    # The new entry shares the grant but never changes the existing audit
    # list, the lifecycle events, or the reconciliation response.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)
    trace(client, machine_id, grant["id"])

    listing = client.get(f"/machines/{machine_id}/authorization-grants")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["items"]] == [grant["id"]]

    reconciliation = client.get(
        f"/machines/{machine_id}/authorization-grants/reconciliation"
    )
    assert reconciliation.status_code == 200
    assert reconciliation.json()["valid"] is True


def test_trace_body_has_no_floating_point_tokens(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)

    raw = trace(client, machine_id, grant["id"]).content
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    assert b"-0.0" not in raw
