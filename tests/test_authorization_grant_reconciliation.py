"""Tests for the read-only grant/use/lifecycle reconciliation endpoint.

    GET /machines/{machine_id}/authorization-grants/reconciliation

The endpoint cross-checks, strictly read-only and only for the path
machine, the three record families of a grant's life: the grant itself,
its single use record, and its immutable lifecycle events. A grant with
lifecycle events must begin with a unique first ``issued`` event; an
``active`` grant has no terminal event and no use; a ``consumed`` grant
has exactly one ``consumed`` event and one use of the same ownership and
moment; a ``revoked`` grant has exactly one ``revoked`` event and no use;
and every event's ``authorization_event_id``, ``grant_id``, and terminal
moment must agree with its grant. Grants without any lifecycle event are
historical (old-database compatibility): counted, never judged, never
given fabricated events.

These tests cover the sound flows (issued, consumed, revoked, historical,
empty), every anomaly category (timestamp, reference binding, sequence or
state, use), the first-broken-grant ordering, orphan records, machine
isolation, request-shape validation (422 before 404), 404, 405, and the
read-only guarantee across repeated calls and restarts.
"""
import sqlite3
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


def reconciliation_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/reconciliation"


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
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_issued_grant_is_valid(issued_grant, client):
    machine_id, _, _ = issued_grant
    assert reconcile(client, machine_id) == {
        "valid": True,
        "checked_grant_count": 1,
        "historical_grant_count": 0,
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_consumed_grant_is_valid(issued_grant, client):
    machine_id, _, grant = issued_grant
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
    assert response.status_code == 200
    assert reconcile(client, machine_id)["valid"] is True
    assert reconcile(client, machine_id)["checked_grant_count"] == 1


def test_revoked_grant_is_valid(issued_grant, client):
    machine_id, _, grant = issued_grant
    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/revoke"
    )
    assert response.status_code == 200
    assert reconcile(client, machine_id)["valid"] is True


def test_historical_grant_is_counted_not_judged(allowed_event, client):
    machine_id, event = allowed_event
    # A grant row without any lifecycle event, as databases that predate
    # the lifecycle feature carry them: counted as historical, not judged.
    insert_historical_grant(client, machine_id, event["id"], status="consumed")
    body = reconcile(client, machine_id)
    assert body == {
        "valid": True,
        "checked_grant_count": 1,
        "historical_grant_count": 1,
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_mixed_modern_and_historical_grants(allowed_event, client):
    machine_id, event = allowed_event
    issue(client, machine_id, event["id"])
    second_event = record_event(client, machine_id).json()
    insert_historical_grant(client, machine_id, second_event["id"])
    body = reconcile(client, machine_id)
    assert body["valid"] is True
    assert body["checked_grant_count"] == 2
    assert body["historical_grant_count"] == 1


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
    assert body["anomaly"] == "timestamp_unparseable"


def test_unparseable_time_is_reported_first(allowed_event, client):
    machine_id, event = allowed_event
    grant_a = issue(client, machine_id, event["id"]).json()
    second = record_event(client, machine_id).json()
    grant_b = issue(client, machine_id, second["id"]).json()
    # Grant B (issued later) has a broken use; grant A's issued_at is
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
# Anomaly: reference binding
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
    assert body["anomaly"] == "reference_mismatch"


def test_event_terminal_moment_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
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
    assert body["anomaly"] == "reference_mismatch"


def test_orphan_event_is_located_by_grant_id(issued_grant, client):
    machine_id, event, _ = issued_grant
    orphan_grant_id = str(uuid.uuid4())
    db_execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :grant_id, :event_id, 'issued', "
        "        :occurred_at, NULL, NULL, NULL)",
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        grant_id=orphan_grant_id,
        event_id=event["id"],
        occurred_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == orphan_grant_id
    assert body["anomaly"] == "reference_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: event sequence or state
# --------------------------------------------------------------------------- #


def test_missing_terminal_event_is_sequence_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
    db_execute(
        client,
        "DELETE FROM authorization_grant_lifecycle_events "
        "WHERE grant_id = :grant_id AND type = 'consumed'",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


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
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_duplicate_issued_event_is_sequence_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    row = grant_row(client, grant["id"])
    db_execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :grant_id, :event_id, 'issued', "
        "        :occurred_at, NULL, NULL, NULL)",
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        grant_id=grant["id"],
        event_id=row["event_id"],
        occurred_at=row["issued_at"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_flipped_status_is_state_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/revoke"
    )
    db_execute(
        client,
        "UPDATE authorization_grants SET status = 'active' WHERE id = :id",
        id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: use record
# --------------------------------------------------------------------------- #


def test_consumed_grant_without_use_is_use_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
    db_execute(
        client,
        "DELETE FROM authorization_grant_uses WHERE grant_id = :grant_id",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_moment_mismatch(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
    db_execute(
        client,
        "UPDATE authorization_grant_uses "
        "SET consumed_at = '2030-01-01T00:00:00Z' "
        "WHERE grant_id = :grant_id",
        grant_id=grant["id"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_on_active_grant_is_use_mismatch(issued_grant, client):
    machine_id, event, grant = issued_grant
    db_execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) "
        "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
        id=str(uuid.uuid4()),
        grant_id=grant["id"],
        machine_id=machine_id,
        event_id=event["id"],
        consumed_at="2026-01-01T00:00:00Z",
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_on_revoked_grant_is_use_mismatch(issued_grant, client):
    machine_id, event, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/revoke"
    )
    row = grant_row(client, grant["id"])
    db_execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) "
        "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
        id=str(uuid.uuid4()),
        grant_id=grant["id"],
        machine_id=machine_id,
        event_id=event["id"],
        consumed_at=row["revoked_at"],
    )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


# --------------------------------------------------------------------------- #
# First-broken ordering, isolation, and the read-only guarantee
# --------------------------------------------------------------------------- #


def test_first_problem_grant_by_issued_at(allowed_event, client):
    machine_id, event = allowed_event
    first_grant = issue(client, machine_id, event["id"]).json()
    second_event = record_event(client, machine_id).json()
    second_grant = issue(client, machine_id, second_event["id"]).json()
    # Break both grants the same way; the earlier-issued one is reported.
    for grant in (first_grant, second_grant):
        db_execute(
            client,
            "INSERT INTO authorization_grant_uses "
            "(id, grant_id, machine_id, event_id, consumed_at) "
            "VALUES (:id, :grant_id, :machine_id, :event_id, :consumed_at)",
            id=str(uuid.uuid4()),
            grant_id=grant["id"],
            machine_id=machine_id,
            event_id=grant["event_id"],
            consumed_at="2026-01-01T00:00:00Z",
        )
    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["checked_grant_count"] == 2
    assert body["broken_grant_id"] == first_grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_other_machine_damage_does_not_leak(issued_grant, client):
    machine_id, _, grant = issued_grant
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
        "broken_grant_id": None,
        "anomaly": None,
    }
    assert reconcile(client, machine_id)["valid"] is False


def test_reconciliation_is_read_only_and_stable(issued_grant, client):
    machine_id, _, grant = issued_grant
    client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )
    first = client.get(reconciliation_url(machine_id))
    second = client.get(reconciliation_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content
    # Nothing was created, repaired, recomputed, or rewritten.
    row = grant_row(client, grant["id"])
    assert row["status"] == "consumed"
    with client.app.state.engine.connect() as conn:
        event_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_lifecycle_events "
                "WHERE machine_id = :machine_id"
            ).bindparams(machine_id=machine_id)
        ).scalar_one()
    assert event_count == 2


def test_conclusion_survives_restart(issued_grant, client, tmp_path,
                                     monkeypatch):
    machine_id, _, grant = issued_grant
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
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"
