"""Tests for the read-only single-grant execution accountability trace.

Covers `GET /machines/{machine_id}/authorization-grants/{grant_id}/
grant-trace`: GET-only 405 (including HEAD) without reading records,
``invalid_query`` 422 for any query parameter, a repeated parameter, or a
carried body before the machine/grant lookup, ``not_found`` 404 for a
missing machine, a missing grant, and a grant owned by another machine,
``internal_error`` 500 on a real read fault with no grant object or record
arrays, the fixed four-group response (the grant audit object plus the
grant-uses, lifecycle-events, and execution-receipt arrays), per-machine
selection on each group's own ``grant_id``, complete stored fields emitted
verbatim with dangling/cross-machine references kept, the audit-list
derived status semantics (``expired`` never persisted), tolerant
(instant, id) ordering per group's own business timestamp with damaged
stamps last, empty-group preservation, machine isolation, compact
single-newline JSON, read-only byte-identical repeats, and persistence
across restarts.
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

GROUP_ORDER = ("grant", "grant_uses", "lifecycle_events", "execution_receipts")

GRANT_FIELDS = (
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
)

USE_FIELDS = ("id", "grant_id", "machine_id", "event_id", "consumed_at")

LIFECYCLE_FIELDS = (
    "id",
    "machine_id",
    "grant_id",
    "authorization_event_id",
    "type",
    "occurred_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
)

RECEIPT_FIELDS = (
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
    # Policy rules are global: a second machine's setup re-uses the rule.
    assert response.status_code in (201, 409)


def record_event(client, machine_id, resource="res/x"):
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


def insert_lifecycle(client, row_id, machine_id, grant_id, event_id,
                     type_, occurred_at):
    execute_sql(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        "occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :mid, :gid, :eid, :type, :at, :prev, :ch, :hh)",
        {
            "id": row_id,
            "mid": machine_id,
            "gid": grant_id,
            "eid": event_id,
            "type": type_,
            "at": occurred_at,
            "prev": f"prev-{row_id}",
            "ch": f"content-{row_id}",
            "hh": f"chain-{row_id}",
        },
    )


def insert_receipt(client, row_id, machine_id, use_id, grant_id, event_id,
                   occurred_at):
    execute_sql(
        client,
        "INSERT INTO execution_receipts "
        "(id, machine_id, use_id, grant_id, authorization_event_id, "
        "action_type, resource, outcome, result_digest, occurred_at, "
        "previous_receipt_id, content_hash, chain_hash) "
        "VALUES (:id, :mid, :uid, :gid, :eid, 'read', 'res/x', "
        "'succeeded', :digest, :at, :prev, :ch, :hh)",
        {
            "id": row_id,
            "mid": machine_id,
            "uid": use_id,
            "gid": grant_id,
            "eid": event_id,
            "digest": "b" * 64,
            "at": occurred_at,
            "prev": f"prev-{row_id}",
            "ch": f"content-{row_id}",
            "hh": f"chain-{row_id}",
        },
    )


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


# --------------------------------------------------------------------------- #
# Lookup outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(MISSING_ID, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content


def test_missing_grant_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    response = client.get(trace_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content


def test_grant_owned_by_another_machine_is_404(client):
    machine_id = create_machine(client)
    other_id = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    response = client.get(trace_url(other_id, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"grant_uses" not in response.content


def test_machine_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(client, "DROP TABLE machines")
    response = client.get(trace_url(machine_id, grant["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"grant_uses" not in response.content


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
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(client, f"DROP TABLE {table}")
    response = client.get(trace_url(machine_id, grant["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"grant_uses" not in response.content


# --------------------------------------------------------------------------- #
# Response shape and the grant object
# --------------------------------------------------------------------------- #


def test_fresh_grant_has_the_grant_object_and_three_arrays(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    payload = trace(client, machine_id, grant["id"]).json()
    assert list(payload) == list(GROUP_ORDER)
    assert list(payload["grant"]) == list(GRANT_FIELDS)
    assert payload["grant_uses"] == []
    # Issuing the grant already appended its ``issued`` lifecycle event.
    assert [row["type"] for row in payload["lifecycle_events"]] == ["issued"]
    assert payload["execution_receipts"] == []


def test_missing_history_leaves_empty_arrays(client):
    # A grant whose lifecycle history predates the feature keeps empty
    # arrays; no historical record is fabricated.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(client, "DELETE FROM authorization_grant_lifecycle_events")
    payload = trace(client, machine_id, grant["id"]).json()
    assert payload["lifecycle_events"] == []
    assert payload["grant_uses"] == []
    assert payload["execution_receipts"] == []


def test_grant_object_mirrors_the_audit_list_view(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    payload = trace(client, machine_id, grant["id"]).json()
    view = payload["grant"]
    assert view["id"] == grant["id"]
    assert view["machine_id"] == machine_id
    assert view["event_id"] == event["id"]
    assert view["issued_at"] == grant["issued_at"]
    assert view["expires_at"] == grant["expires_at"]
    assert view["status"] == "active"
    assert view["consumed_at"] is None
    assert view["revoked_at"] is None
    assert view["use_id"] is None
    assert view["use_at"] is None


def test_expired_grant_presents_expired_without_a_write_back(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"], ttl_seconds=1)
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :gid",
        {"past": T0, "gid": grant["id"]},
    )
    payload = trace(client, machine_id, grant["id"]).json()
    assert payload["grant"]["status"] == "expired"
    assert payload["grant"]["expires_at"] == T0
    # The derived status is never persisted: the stored row keeps ``active``.
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :gid"),
            {"gid": grant["id"]},
        ).scalar_one()
    assert stored == "active"
    # And a repeat read still derives ``expired`` from the stored stamps.
    assert trace(client, machine_id, grant["id"]).json()["grant"][
        "status"
    ] == "expired"


def test_consumed_grant_keeps_its_terminal_status_and_use_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"], ttl_seconds=1)
    use = consume_grant(client, machine_id, grant["id"])
    # Expiry must never downgrade the terminal consumed state.
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :gid",
        {"past": T0, "gid": grant["id"]},
    )
    view = trace(client, machine_id, grant["id"]).json()["grant"]
    assert view["status"] == "consumed"
    assert view["consumed_at"] == use["consumed_at"]
    assert view["use_id"] == use["use_id"]
    assert view["use_at"] == use["consumed_at"]
    assert view["revoked_at"] is None


def test_revoked_grant_keeps_its_terminal_status(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"], ttl_seconds=1)
    revocation = revoke_grant(client, machine_id, grant["id"])
    execute_sql(
        client,
        "UPDATE authorization_grants SET expires_at = :past WHERE id = :gid",
        {"past": T0, "gid": grant["id"]},
    )
    view = trace(client, machine_id, grant["id"]).json()["grant"]
    assert view["status"] == "revoked"
    assert view["revoked_at"] == revocation["revoked_at"]
    assert view["consumed_at"] is None
    assert view["use_id"] is None
    assert view["use_at"] is None


# --------------------------------------------------------------------------- #
# The three record arrays
# --------------------------------------------------------------------------- #


def test_full_consumed_flow_collects_use_lifecycle_and_receipt(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    receipt = register_receipt(client, machine_id, use)
    payload = trace(client, machine_id, grant["id"]).json()

    assert [list(row) for row in payload["grant_uses"]] == [list(USE_FIELDS)]
    use_row = payload["grant_uses"][0]
    assert use_row == {
        "id": use["use_id"],
        "grant_id": grant["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "consumed_at": use["consumed_at"],
    }

    assert [row["type"] for row in payload["lifecycle_events"]] == [
        "issued",
        "consumed",
    ]
    for row in payload["lifecycle_events"]:
        assert list(row) == list(LIFECYCLE_FIELDS)
        assert row["machine_id"] == machine_id
        assert row["grant_id"] == grant["id"]
        assert row["authorization_event_id"] == event["id"]

    assert [list(row) for row in payload["execution_receipts"]] == [
        list(RECEIPT_FIELDS)
    ]
    receipt_row = payload["execution_receipts"][0]
    assert receipt_row["id"] == receipt["id"]
    assert receipt_row["use_id"] == use["use_id"]
    assert receipt_row["grant_id"] == grant["id"]
    assert receipt_row["authorization_event_id"] == event["id"]
    assert receipt_row["action_type"] == "read"
    assert receipt_row["resource"] == "res/x"
    assert receipt_row["outcome"] == "succeeded"
    assert receipt_row["result_digest"] == "a" * 64
    assert receipt_row["occurred_at"] == receipt["occurred_at"]


def test_revoked_flow_collects_the_revocation_lifecycle_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    revoke_grant(client, machine_id, grant["id"])
    payload = trace(client, machine_id, grant["id"]).json()
    assert [row["type"] for row in payload["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]
    assert payload["grant_uses"] == []
    assert payload["execution_receipts"] == []


def test_lifecycle_events_emit_chain_fields_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    insert_lifecycle(
        client, "lc-1", machine_id, grant["id"], GHOST_ID, "issued", T1
    )
    payload = trace(client, machine_id, grant["id"]).json()
    row = next(r for r in payload["lifecycle_events"] if r["id"] == "lc-1")
    # A dangling event reference and the stored chain fields survive verbatim.
    assert row["authorization_event_id"] == GHOST_ID
    assert row["previous_event_id"] == "prev-lc-1"
    assert row["content_hash"] == "content-lc-1"
    assert row["chain_hash"] == "chain-lc-1"
    assert row["occurred_at"] == T1


def test_records_of_another_grant_and_another_machine_never_enter(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id)
    other_event = record_event(client, other_machine)
    grant = issue_grant(client, machine_id, event["id"])
    other_grant = issue_grant(client, other_machine, other_event["id"])
    consume_grant(client, machine_id, grant["id"])
    consume_grant(client, other_machine, other_grant["id"])

    payload = trace(client, machine_id, grant["id"]).json()
    assert [row["grant_id"] for row in payload["grant_uses"]] == [grant["id"]]
    assert {row["grant_id"] for row in payload["lifecycle_events"]} == {
        grant["id"]
    }
    for group in ("grant_uses", "lifecycle_events", "execution_receipts"):
        assert all(
            row["machine_id"] == machine_id for row in payload[group]
        )

    # A row of the path machine naming another grant never enters either.
    insert_lifecycle(
        client, "lc-x", machine_id, other_grant["id"], event["id"],
        "issued", T1,
    )
    insert_receipt(
        client, "rc-x", machine_id, "use-x", other_grant["id"], event["id"],
        T1,
    )
    payload = trace(client, machine_id, grant["id"]).json()
    assert "lc-x" not in {row["id"] for row in payload["lifecycle_events"]}
    assert "rc-x" not in {row["id"] for row in payload["execution_receipts"]}
    # But the same rows do enter the other grant's own trace on this machine
    # only if owned by it: lc-x/rc-x belong to the path machine yet name a
    # foreign grant, so the other machine's trace never sees them.
    other_payload = trace(client, other_machine, other_grant["id"]).json()
    assert "lc-x" not in {
        row["id"] for row in other_payload["lifecycle_events"]
    }
    assert "rc-x" not in {
        row["id"] for row in other_payload["execution_receipts"]
    }


def test_child_records_are_kept_when_their_event_reference_is_cross_machine(
    client,
):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id)
    other_event = record_event(client, other_machine)
    grant = issue_grant(client, machine_id, event["id"])
    # A stored cross-machine association value is emitted verbatim, never
    # repaired, normalized, or filtered out.
    insert_lifecycle(
        client, "lc-1", machine_id, grant["id"], other_event["id"],
        "issued", T1,
    )
    insert_receipt(
        client, "rc-1", machine_id, GHOST_ID, grant["id"], other_event["id"],
        T2,
    )
    payload = trace(client, machine_id, grant["id"]).json()
    assert payload["lifecycle_events"][0]["authorization_event_id"] == (
        other_event["id"]
    )
    receipt_row = payload["execution_receipts"][0]
    assert receipt_row["authorization_event_id"] == other_event["id"]
    assert receipt_row["use_id"] == GHOST_ID


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_lifecycle_events_order_by_instant_then_id_with_damaged_last(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    execute_sql(client, "DELETE FROM authorization_grant_lifecycle_events")
    insert_lifecycle(client, "lc-b", machine_id, grant["id"], event["id"],
                     "issued", T1)
    insert_lifecycle(client, "lc-a", machine_id, grant["id"], event["id"],
                     "issued", T1)
    insert_lifecycle(client, "lc-c", machine_id, grant["id"], event["id"],
                     "issued", T0_HALF)
    insert_lifecycle(client, "lc-d", machine_id, grant["id"], event["id"],
                     "issued", T0)
    insert_lifecycle(client, "lc-z", machine_id, grant["id"], event["id"],
                     "issued", "not-a-timestamp")
    payload = trace(client, machine_id, grant["id"]).json()
    assert [row["id"] for row in payload["lifecycle_events"]] == [
        "lc-d",  # exact second before the fractional second of T0
        "lc-c",
        "lc-a",  # same instant, id ascending
        "lc-b",
        "lc-z",  # unparseable stamp kept verbatim, sorted last
    ]
    assert payload["lifecycle_events"][-1]["occurred_at"] == "not-a-timestamp"


def test_receipts_order_by_instant_then_id_with_damaged_last(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    insert_receipt(client, "rc-b", machine_id, "use-b", grant["id"],
                   event["id"], T1)
    insert_receipt(client, "rc-a", machine_id, "use-a", grant["id"],
                   event["id"], T1)
    insert_receipt(client, "rc-c", machine_id, "use-c", grant["id"],
                   event["id"], T0)
    insert_receipt(client, "rc-z", machine_id, "use-z", grant["id"],
                   event["id"], "garbage")
    payload = trace(client, machine_id, grant["id"]).json()
    assert [row["id"] for row in payload["execution_receipts"]] == [
        "rc-c",
        "rc-a",
        "rc-b",
        "rc-z",
    ]
    assert payload["execution_receipts"][-1]["occurred_at"] == "garbage"


def test_grant_use_emits_complete_stored_fields_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    # Damage the stored consumption moment: it is kept verbatim in both the
    # use row and the grant view, and never repaired or dropped.
    execute_sql(
        client,
        "UPDATE authorization_grant_uses SET consumed_at = :bad "
        "WHERE id = :uid",
        {"bad": "not-a-timestamp", "uid": use["use_id"]},
    )
    payload = trace(client, machine_id, grant["id"]).json()
    assert payload["grant_uses"][0]["consumed_at"] == "not-a-timestamp"
    assert payload["grant"]["use_at"] == "not-a-timestamp"


# --------------------------------------------------------------------------- #
# Stability, isolation, and read-only behavior
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_a_single_newline(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    consume_grant(client, machine_id, grant["id"])
    raw = trace(client, machine_id, grant["id"]).content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b" " not in raw.split(b"\n")[0].replace(b'": "', b"")
    assert json.loads(raw.decode("utf-8"))["grant"]["id"] == grant["id"]


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
    first = trace(client, machine_id, grant["id"]).content
    second = trace(client, machine_id, grant["id"]).content
    assert first == second
    with client.app.state.engine.connect() as conn:
        counts = {
            table: conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar_one()
            for table in (
                "authorization_grants",
                "authorization_grant_uses",
                "authorization_grant_lifecycle_events",
                "execution_receipts",
            )
        }
    assert counts == {
        "authorization_grants": 1,
        "authorization_grant_uses": 1,
        "authorization_grant_lifecycle_events": 2,
        "execution_receipts": 1,
    }


def test_trace_survives_restart_byte_identical(tmp_path, monkeypatch):
    database = tmp_path / "restart.db"
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{database}"
    )
    with TestClient(app) as first_client:
        machine_id = create_machine(first_client)
        event = record_event(first_client, machine_id)
        grant = issue_grant(first_client, machine_id, event["id"])
        use = consume_grant(first_client, machine_id, grant["id"])
        register_receipt(first_client, machine_id, use)
        first_body = trace(first_client, machine_id, grant["id"]).content
    with TestClient(app) as second_client:
        assert trace(second_client, machine_id, grant["id"]).content == (
            first_body
        )


def test_trace_body_has_no_floating_point_tokens(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)
    raw = trace(client, machine_id, grant["id"]).content
    for token in (b"NaN", b"Infinity", b"-0.0"):
        assert token not in raw


def test_existing_execution_trace_is_unaffected(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"])
    use = consume_grant(client, machine_id, grant["id"])
    register_receipt(client, machine_id, use)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/execution-trace"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_summary"]["allowed"] is True
    assert [row["id"] for row in payload["grants"]] == [grant["id"]]
    assert [row["id"] for row in payload["grant_uses"]] == [use["use_id"]]
