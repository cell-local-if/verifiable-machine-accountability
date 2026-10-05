"""Tests for the read-only grant/use/lifecycle/receipt execution reconciliation.

    GET /machines/{machine_id}/authorization-grants/execution-reconciliation

The endpoint cross-checks, strictly read-only and only for the path
machine, the full life of every non-historical grant: the grant itself,
its immutable lifecycle events, its single use record, and its single
execution receipt. A ``consumed`` grant must carry exactly one use and
exactly one receipt bound to the same use, grant, and source allow
decision with a verbatim action/resource and well-formed outcome,
result digest, and content hash; every other state carries neither a use
nor a receipt. Grants without any lifecycle event are historical
(old-database compatibility): counted, never judged, never given
fabricated events or receipts.

These tests cover the sound flows (issued, consumed with receipt,
revoked, historical, empty), every anomaly category (timestamp, grant
binding, grant state, use count, receipt missing, receipt binding,
receipt content), the completed-count semantics, the first-broken-grant
ordering, orphan records, machine isolation, request-shape validation
(405 for non-GET, 422 before 404), 404, and the read-only guarantee
across repeated calls and restarts.
"""
import uuid

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


def consume(client, machine_id, grant_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )
    assert response.status_code == 200
    return response.json()


def receipt(client, machine_id, use_id, action_type="read", resource="res/x",
            outcome="succeeded"):
    response = client.post(
        f"/machines/{machine_id}/execution-receipts",
        json={
            "use_id": use_id,
            "action_type": action_type,
            "resource": resource,
            "outcome": outcome,
            "result_digest": "a" * 64,
        },
    )
    assert response.status_code == 201
    return response.json()


def reconciliation_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-grants/execution-reconciliation"
    )


@pytest.fixture
def allowed_event(client):
    """A machine + enabled declaration + allow rule + allowed event."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    assert event["reason"] == "allowed_by_policy"
    return machine_id, event


@pytest.fixture
def issued_grant(allowed_event, client):
    """A machine with one freshly issued (active) grant."""
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    return machine_id, event, grant


@pytest.fixture
def consumed_grant(issued_grant, client):
    """A machine with one consumed grant carrying its receipt."""
    machine_id, event, grant = issued_grant
    use = consume(client, machine_id, grant["id"])
    receipt_record = receipt(client, machine_id, use["use_id"])
    return machine_id, event, grant, use, receipt_record


def reconcile(client, machine_id):
    response = client.get(reconciliation_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_execute(client, statement, **params):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement).bindparams(**params))


def insert_historical_grant(client, machine_id, event_id,
                            issued_at="2026-01-01T00:00:00Z",
                            status="active"):
    """A grant row with no lifecycle event, as old databases carry them."""
    grant_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO authorization_grants "
        "(id, machine_id, event_id, issued_at, expires_at, status, "
        " consumed_at, revoked_at) "
        "VALUES (:id, :machine_id, :event_id, :issued_at, :expires_at, "
        "        :status, NULL, NULL)",
        id=grant_id,
        machine_id=machine_id,
        event_id=event_id,
        issued_at=issued_at,
        expires_at="2026-01-02T00:00:00Z",
        status=status,
    )
    return grant_id


def grant_row(client, grant_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT id, machine_id, event_id, issued_at, expires_at, "
                "status, consumed_at, revoked_at "
                "FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant_id)
        ).mappings().one()


# --------------------------------------------------------------------------- #
# Sound flows reconcile clean
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid(client):
    machine_id = create_machine(client)
    assert reconcile(client, machine_id) == {
        "valid": True,
        "checked_grant_count": 0,
        "historical_grant_count": 0,
        "completed_count": 0,
        "broken_grant_id": None,
        "broken_record_id": None,
        "anomaly": None,
    }


def test_issued_grant_is_valid(issued_grant, client):
    machine_id, _, _ = issued_grant
    assert reconcile(client, machine_id) == {
        "valid": True,
        "checked_grant_count": 1,
        "historical_grant_count": 0,
        "completed_count": 0,
        "broken_grant_id": None,
        "broken_record_id": None,
        "anomaly": None,
    }


def test_consumed_grant_with_receipt_completes(consumed_grant, client):
    machine_id, _, _, _, _ = consumed_grant
    body = reconcile(client, machine_id)
    assert body["valid"] is True
    assert body["checked_grant_count"] == 1
    assert body["completed_count"] == 1


def test_revoked_grant_is_valid(issued_grant, client):
    machine_id, _, grant = issued_grant
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/revoke"
    )
    assert response.status_code == 200
    body = reconcile(client, machine_id)
    assert body["valid"] is True
    assert body["completed_count"] == 0


def test_historical_grant_is_counted_not_judged(allowed_event, client):
    machine_id, event = allowed_event
    # A consumed grant row without any lifecycle event, use, or receipt, as
    # databases that predate the features carry them: counted as
    # historical, never judged, and never completed.
    insert_historical_grant(client, machine_id, event["id"], status="consumed")
    body = reconcile(client, machine_id)
    assert body == {
        "valid": True,
        "checked_grant_count": 1,
        "historical_grant_count": 1,
        "completed_count": 0,
        "broken_grant_id": None,
        "broken_record_id": None,
        "anomaly": None,
    }


def test_mixed_modern_and_historical_grants(consumed_grant, client):
    machine_id, _, _, _, _ = consumed_grant
    second_event = record_event(client, machine_id).json()
    insert_historical_grant(client, machine_id, second_event["id"])
    body = reconcile(client, machine_id)
    assert body["valid"] is True
    assert body["checked_grant_count"] == 2
    assert body["historical_grant_count"] == 1
    assert body["completed_count"] == 1


# --------------------------------------------------------------------------- #
# Request shape, machine lookup, and method validation
# --------------------------------------------------------------------------- #


def test_query_string_is_rejected_before_lookup(client):
    response = client.get(reconciliation_url("missing"), params={"x": "1"})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_query_param_is_rejected(issued_grant, client):
    machine_id, _, _ = issued_grant
    response = client.get(reconciliation_url(machine_id) + "?a=1&a=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_is_rejected_before_lookup(client):
    response = client.request(
        "GET", reconciliation_url("missing"), content=b"{}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_404(client):
    response = client.get(reconciliation_url("missing-machine"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_non_get_methods_are_405(issued_grant, client):
    machine_id, _, _ = issued_grant
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(reconciliation_url(machine_id))
        assert response.status_code == 405


def test_head_is_405(issued_grant, client):
    machine_id, _, _ = issued_grant
    response = client.head(reconciliation_url(machine_id))
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Anomaly: timestamps
# --------------------------------------------------------------------------- #


def test_unparseable_issued_at_is_timestamp_anomaly(issued_grant, client):
    machine_id, _, grant = issued_grant
    db_execute(
        client,
        "UPDATE authorization_grants SET issued_at = 'not-a-time' "
        "WHERE id = :id",
        id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == grant["id"]
    assert body["anomaly"] == "timestamp_unparseable"


def test_unparseable_event_moment_names_the_event(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    event_row_id = None
    with client.app.state.engine.connect() as conn:
        event_row_id = conn.execute(
            text(
                "SELECT id FROM authorization_grant_lifecycle_events "
                "WHERE grant_id = :grant_id AND type = 'consumed'"
            ).bindparams(grant_id=grant["id"])
        ).scalar_one()
    db_execute(
        client,
        "UPDATE authorization_grant_lifecycle_events "
        "SET occurred_at = 'garbage' WHERE id = :id",
        id=event_row_id,
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == event_row_id
    assert body["anomaly"] == "timestamp_unparseable"
    assert body["completed_count"] == 0


def test_unparseable_time_is_reported_first(allowed_event, client):
    machine_id, event = allowed_event
    grant_a = issue(client, machine_id, event["id"]).json()
    second = record_event(client, machine_id).json()
    grant_b = issue(client, machine_id, second["id"]).json()
    # Grant B (issued later) has a stray use; grant A's issued_at is
    # unparseable. The unparseable time is reported first.
    db_execute(
        client,
        "UPDATE authorization_grants SET issued_at = 'garbage' "
        "WHERE id = :id",
        id=grant_a["id"],
    )
    db_execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) "
        "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
        id=str(uuid.uuid4()),
        grant_id=grant_b["id"],
        machine_id=machine_id,
        event_id=second["id"],
        consumed_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["broken_grant_id"] == grant_a["id"]
    assert body["anomaly"] == "timestamp_unparseable"


# --------------------------------------------------------------------------- #
# Anomaly: lifecycle binding and state
# --------------------------------------------------------------------------- #


def test_event_decision_reference_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    db_execute(
        client,
        "UPDATE authorization_grant_lifecycle_events "
        "SET authorization_event_id = 'other-event' "
        "WHERE grant_id = :grant_id",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "grant_binding_mismatch"


def test_event_terminal_moment_mismatch(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    db_execute(
        client,
        "UPDATE authorization_grant_lifecycle_events "
        "SET occurred_at = '2030-01-01T00:00:00Z' "
        "WHERE grant_id = :grant_id AND type = 'consumed'",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "grant_binding_mismatch"


def test_orphan_event_is_located_by_grant_id(issued_grant, client):
    machine_id, event, _ = issued_grant
    orphan_grant_id = str(uuid.uuid4())
    orphan_event_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :grant_id, :event_id, 'issued', "
        "        :occurred_at, NULL, NULL, NULL)",
        id=orphan_event_id,
        machine_id=machine_id,
        grant_id=orphan_grant_id,
        event_id=event["id"],
        occurred_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == orphan_grant_id
    assert body["broken_record_id"] == orphan_event_id
    assert body["anomaly"] == "grant_binding_mismatch"


def test_missing_terminal_event_is_state_mismatch(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    db_execute(
        client,
        "DELETE FROM authorization_grant_lifecycle_events "
        "WHERE grant_id = :grant_id AND type = 'consumed'",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == grant["id"]
    assert body["anomaly"] == "grant_state_mismatch"


def test_terminal_event_on_active_grant_is_state_mismatch(issued_grant,
                                                          client):
    machine_id, _, grant = issued_grant
    row = grant_row(client, grant["id"])
    db_execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :grant_id, :event_id, 'revoked', "
        "        :occurred_at, NULL, NULL, NULL)",
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        grant_id=grant["id"],
        event_id=row["event_id"],
        occurred_at=row["issued_at"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "grant_state_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: use count
# --------------------------------------------------------------------------- #


def test_consumed_grant_without_use_is_use_mismatch(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    db_execute(
        client,
        "DELETE FROM authorization_grant_uses WHERE grant_id = :grant_id",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_on_active_grant_is_use_mismatch(issued_grant, client):
    machine_id, event, grant = issued_grant
    use_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) "
        "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
        id=use_id,
        grant_id=grant["id"],
        machine_id=machine_id,
        event_id=event["id"],
        consumed_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == use_id
    assert body["anomaly"] == "use_mismatch"


def test_orphan_use_is_located_by_grant_id(issued_grant, client):
    machine_id, event, _ = issued_grant
    orphan_grant_id = str(uuid.uuid4())
    orphan_use_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) "
        "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
        id=orphan_use_id,
        grant_id=orphan_grant_id,
        machine_id=machine_id,
        event_id=event["id"],
        consumed_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == orphan_grant_id
    assert body["broken_record_id"] == orphan_use_id
    assert body["anomaly"] == "use_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: receipts
# --------------------------------------------------------------------------- #


def test_consumed_grant_without_receipt_is_receipt_missing(issued_grant,
                                                           client):
    machine_id, _, grant = issued_grant
    use = consume(client, machine_id, grant["id"])
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == use["use_id"]
    assert body["anomaly"] == "receipt_missing"
    assert body["completed_count"] == 0


def test_receipt_scope_mismatch_is_receipt_mismatch(consumed_grant, client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    db_execute(
        client,
        "UPDATE execution_receipts SET action_type = 'write' "
        "WHERE id = :id",
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_mismatch"


def test_receipt_grant_binding_mismatch(consumed_grant, client):
    machine_id, _, grant, use, receipt_record = consumed_grant
    # The receipt still names the use but its grant reference dangles:
    # cross-wired, it is judged with the use's grant as a binding break.
    db_execute(
        client,
        "UPDATE execution_receipts SET grant_id = :dangling "
        "WHERE id = :id",
        dangling=str(uuid.uuid4()),
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_mismatch"


def test_receipt_event_binding_mismatch(consumed_grant, client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    db_execute(
        client,
        "UPDATE execution_receipts "
        "SET authorization_event_id = 'other-event' WHERE id = :id",
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_mismatch"


def test_receipt_on_active_grant_is_receipt_mismatch(issued_grant, client):
    machine_id, event, grant = issued_grant
    receipt_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO execution_receipts "
        "(id, machine_id, use_id, grant_id, authorization_event_id, "
        " action_type, resource, outcome, result_digest, occurred_at, "
        " previous_receipt_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :use_id, :grant_id, :event_id, "
        "        'read', 'res/x', 'succeeded', :digest, "
        "        '2026-01-01T00:00:00Z', NULL, NULL, NULL)",
        id=receipt_id,
        machine_id=machine_id,
        use_id=str(uuid.uuid4()),
        grant_id=grant["id"],
        event_id=event["id"],
        digest="b" * 64,
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_id
    assert body["anomaly"] == "receipt_mismatch"


def test_orphan_receipt_is_located_by_grant_id(issued_grant, client):
    machine_id, event, _ = issued_grant
    orphan_grant_id = str(uuid.uuid4())
    orphan_receipt_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO execution_receipts "
        "(id, machine_id, use_id, grant_id, authorization_event_id, "
        " action_type, resource, outcome, result_digest, occurred_at, "
        " previous_receipt_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :use_id, :grant_id, :event_id, "
        "        'read', 'res/x', 'succeeded', :digest, "
        "        '2026-01-01T00:00:00Z', NULL, NULL, NULL)",
        id=orphan_receipt_id,
        machine_id=machine_id,
        use_id=str(uuid.uuid4()),
        grant_id=orphan_grant_id,
        event_id=event["id"],
        digest="b" * 64,
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == orphan_grant_id
    assert body["broken_record_id"] == orphan_receipt_id
    assert body["anomaly"] == "receipt_mismatch"


def test_damaged_outcome_is_receipt_content_invalid(consumed_grant, client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    db_execute(
        client,
        "UPDATE execution_receipts SET outcome = 'unknown' WHERE id = :id",
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_content_invalid"


def test_damaged_result_digest_is_receipt_content_invalid(consumed_grant,
                                                          client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    db_execute(
        client,
        "UPDATE execution_receipts SET result_digest = 'not-hex' "
        "WHERE id = :id",
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_content_invalid"


def test_stale_content_hash_is_receipt_content_invalid(consumed_grant,
                                                       client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    # A well-formed digest that no longer covers the stored content.
    db_execute(
        client,
        "UPDATE execution_receipts SET result_digest = :digest "
        "WHERE id = :id",
        digest="c" * 64,
        id=receipt_record["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["broken_record_id"] == receipt_record["id"]
    assert body["anomaly"] == "receipt_content_invalid"


# --------------------------------------------------------------------------- #
# Completed count, first-broken ordering, isolation, read-only guarantee
# --------------------------------------------------------------------------- #


def test_completed_count_counts_only_clean_consumed_grants(allowed_event,
                                                           client):
    machine_id, event = allowed_event
    # One fully reconciled consumed grant...
    grant_a = issue(client, machine_id, event["id"]).json()
    use_a = consume(client, machine_id, grant_a["id"])
    receipt(client, machine_id, use_a["use_id"])
    # ...and one consumed grant still awaiting its receipt.
    second = record_event(client, machine_id).json()
    grant_b = issue(client, machine_id, second["id"]).json()
    consume(client, machine_id, grant_b["id"])
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["checked_grant_count"] == 2
    assert body["completed_count"] == 1
    assert body["broken_grant_id"] == grant_b["id"]
    assert body["anomaly"] == "receipt_missing"


def test_first_problem_grant_by_issued_at(allowed_event, client):
    machine_id, event = allowed_event
    first_grant = issue(client, machine_id, event["id"]).json()
    second_event = record_event(client, machine_id).json()
    second_grant = issue(client, machine_id, second_event["id"]).json()
    # Both grants are consumed without a receipt; the earlier-issued one
    # is reported.
    consume(client, machine_id, first_grant["id"])
    consume(client, machine_id, second_grant["id"])
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["checked_grant_count"] == 2
    assert body["completed_count"] == 0
    assert body["broken_grant_id"] == first_grant["id"]
    assert body["anomaly"] == "receipt_missing"


def test_other_machine_damage_does_not_leak(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    other_id = create_machine(client, external_id="machine-2")
    db_execute(
        client,
        "UPDATE authorization_grants SET issued_at = 'garbage' "
        "WHERE id = :id",
        id=grant["id"],
    )
    assert reconcile(client, other_id) == {
        "valid": True,
        "checked_grant_count": 0,
        "historical_grant_count": 0,
        "completed_count": 0,
        "broken_grant_id": None,
        "broken_record_id": None,
        "anomaly": None,
    }
    assert reconcile(client, machine_id)["valid"] is False


def test_reconciliation_is_read_only_and_stable(consumed_grant, client):
    machine_id, _, grant, _, receipt_record = consumed_grant
    first = client.get(reconciliation_url(machine_id))
    second = client.get(reconciliation_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content
    # Nothing was created, repaired, recomputed, or rewritten.
    row = grant_row(client, grant["id"])
    assert row["status"] == "consumed"
    with client.app.state.engine.connect() as conn:
        counts = conn.execute(
            text(
                "SELECT "
                "(SELECT COUNT(*) FROM authorization_grant_lifecycle_events "
                " WHERE machine_id = :machine_id), "
                "(SELECT COUNT(*) FROM execution_receipts "
                " WHERE machine_id = :machine_id)"
            ).bindparams(machine_id=machine_id)
        ).one()
    assert tuple(counts) == (2, 1)
    with client.app.state.engine.connect() as conn:
        stored_hash = conn.execute(
            text(
                "SELECT content_hash FROM execution_receipts WHERE id = :id"
            ).bindparams(id=receipt_record["id"])
        ).scalar_one()
    assert stored_hash == receipt_record["content_hash"]


def test_conclusion_survives_restart(consumed_grant, client, tmp_path,
                                     monkeypatch):
    machine_id, _, grant, _, _ = consumed_grant
    db_execute(
        client,
        "UPDATE authorization_grants SET issued_at = 'garbage' "
        "WHERE id = :id",
        id=grant["id"],
    )
    before = reconcile(client, machine_id)
    assert before["valid"] is False

    # A fresh app instance over the same database file reaches the same
    # conclusion; the read-only query never rewrote anything.
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as second_client:
        assert reconcile(second_client, machine_id) == before


def test_broken_conclusion_keeps_counts(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    second_event = record_event(client, machine_id).json()
    insert_historical_grant(client, machine_id, second_event["id"])
    # A stray terminal event on the active grant breaks the sequence.
    row = grant_row(client, grant["id"])
    db_execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :grant_id, :event_id, 'consumed', "
        "        :occurred_at, NULL, NULL, NULL)",
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        grant_id=grant["id"],
        event_id=row["event_id"],
        occurred_at=row["issued_at"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["checked_grant_count"] == 2
    assert body["historical_grant_count"] == 1
    assert body["completed_count"] == 0
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "grant_state_mismatch"
