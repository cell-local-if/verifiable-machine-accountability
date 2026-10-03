"""Tests for the authorization grant reconciliation query.

``GET /machines/{machine_id}/authorization-grants/reconciliation`` is the
read-only cross-check of one machine's grants against their single use
record and their immutable lifecycle events. A grant with lifecycle events
must open with a unique ``issued`` event; an ``active`` grant carries no
terminal event and no use, a ``consumed`` grant exactly one ``consumed``
event plus one use of the same ownership and moment, and a ``revoked``
grant exactly one ``revoked`` event and no use; every event's
``authorization_event_id``, ``grant_id``, and terminal moment must agree
with the grant. Grants with no lifecycle event are historical rows:
counted, never judged, never backfilled.

These tests cover the success shape and counts, every anomaly category
(``timestamp_unparseable``, ``reference_mismatch``,
``sequence_or_state_mismatch``, ``use_mismatch``), the first-problem-grant
ordering, the 422/404/405 outcomes, machine isolation, the strictly
read-only behavior, and restart persistence.
"""

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


def issue(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


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


def issue_grant(client, machine_id):
    """One more allowed event signed into a fresh grant."""
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    response = issue(client, machine_id, event["id"])
    assert response.status_code == 201
    return response.json()


def reconcile(client, machine_id):
    response = client.get(reconciliation_url(machine_id))
    assert response.status_code == 200
    return response.json()


def grant_row(client, grant_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT id, machine_id, event_id, issued_at, expires_at, "
                "status, consumed_at, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant_id)
        ).first()._mapping


def execute(client, statement, **params):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement).bindparams(**params))


# --------------------------------------------------------------------------- #
# Sound data: shape, counts, historical grants
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid(client):
    machine_id = create_machine(client)
    response = client.get(reconciliation_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "valid",
        "checked_grant_count",
        "historical_grant_count",
        "broken_grant_id",
        "anomaly",
    ]
    assert body == {
        "valid": True,
        "checked_grant_count": 0,
        "historical_grant_count": 0,
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_full_lifecycle_is_valid(allowed_event, client):
    machine_id, _ = allowed_event
    active = issue_grant(client, machine_id)
    consumed = issue_grant(client, machine_id)
    assert client.post(consume_url(machine_id, consumed["id"])).status_code == 200
    revoked = issue_grant(client, machine_id)
    assert client.post(revoke_url(machine_id, revoked["id"])).status_code == 200

    assert reconcile(client, machine_id) == {
        "valid": True,
        "checked_grant_count": 3,
        "historical_grant_count": 0,
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_historical_grants_are_counted_not_judged(allowed_event, client):
    """A grant with no lifecycle event is historical: counted, not judged."""
    machine_id, event = allowed_event
    modern = issue(client, machine_id, event["id"]).json()

    # A historical grant row from before the lifecycle feature: no events,
    # and even a terminal state with a use record is not judged.
    historical_event = record_event(client, machine_id).json()
    execute(
        client,
        "INSERT INTO authorization_grants "
        "(id, machine_id, event_id, issued_at, expires_at, status, "
        " consumed_at, revoked_at) VALUES "
        "('historical-grant', :machine_id, :event_id, "
        " '2020-01-01T00:00:00Z', '2020-01-01T00:01:00Z', 'consumed', "
        " '2020-01-01T00:00:30Z', NULL)",
        machine_id=machine_id,
        event_id=historical_event["id"],
    )
    execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) VALUES "
        "('historical-use', 'historical-grant', :machine_id, :event_id, "
        " '2020-01-01T00:00:30Z')",
        machine_id=machine_id,
        event_id=historical_event["id"],
    )

    assert reconcile(client, machine_id) == {
        "valid": True,
        "checked_grant_count": 2,
        "historical_grant_count": 1,
        "broken_grant_id": None,
        "anomaly": None,
    }
    # The historical rows were not rewritten and no events were fabricated.
    assert grant_row(client, "historical-grant")["status"] == "consumed"
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_lifecycle_events "
                "WHERE grant_id = 'historical-grant'"
            )
        ).scalar()
    assert count == 0
    assert modern["id"] != "historical-grant"


# --------------------------------------------------------------------------- #
# Request validation and routing
# --------------------------------------------------------------------------- #


def test_rejects_query_params_and_body(allowed_event, client):
    machine_id, _ = allowed_event
    response = client.get(
        reconciliation_url(machine_id), params={"limit": 1}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        reconciliation_url(machine_id) + "?x=1&x=2"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        reconciliation_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # The query check wins over the machine lookup.
    response = client.get(
        reconciliation_url("missing-machine"), params={"x": "1"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_404(client):
    response = client.get(reconciliation_url("missing-machine"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_non_get_methods_are_405(allowed_event, client):
    machine_id, _ = allowed_event
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(reconciliation_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Anomaly categories
# --------------------------------------------------------------------------- #


def test_unparseable_issued_at_is_timestamp_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    execute(
        client,
        "UPDATE authorization_grants SET issued_at = 'not-a-time' "
        "WHERE id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["checked_grant_count"] == 1
    assert body["historical_grant_count"] == 0
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "timestamp_unparseable"


def test_unparseable_terminal_stamp_is_timestamp_anomaly(
    allowed_event, client
):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "UPDATE authorization_grants SET consumed_at = NULL WHERE id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "timestamp_unparseable"


def test_event_binding_mismatch_is_reference_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    execute(
        client,
        "UPDATE authorization_grant_lifecycle_events "
        "SET authorization_event_id = 'some-other-event' "
        "WHERE grant_id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "reference_mismatch"


def test_orphan_event_is_located_by_its_grant_id(allowed_event, client):
    machine_id, event = allowed_event
    issue(client, machine_id, event["id"])
    execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) VALUES "
        "('orphan-event', :machine_id, 'ghost-grant', :event_id, 'issued', "
        " '2026-01-01T00:00:00Z', NULL, NULL, NULL)",
        machine_id=machine_id,
        event_id=event["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == "ghost-grant"
    assert body["anomaly"] == "reference_mismatch"


def test_terminal_event_on_active_grant_is_sequence_anomaly(
    allowed_event, client
):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    execute(
        client,
        "INSERT INTO authorization_grant_lifecycle_events "
        "(id, machine_id, grant_id, authorization_event_id, type, "
        " occurred_at, previous_event_id, content_hash, chain_hash) VALUES "
        "('extra-terminal', :machine_id, :grant_id, :event_id, 'consumed', "
        " '2026-01-01T00:00:00Z', NULL, NULL, NULL)",
        machine_id=machine_id,
        grant_id=grant["id"],
        event_id=grant["event_id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_missing_issued_event_is_sequence_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "DELETE FROM authorization_grant_lifecycle_events "
        "WHERE grant_id = :id AND type = 'issued'",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_terminal_moment_mismatch_is_sequence_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    # The grant's terminal moment no longer agrees with its consumed event.
    execute(
        client,
        "UPDATE authorization_grants "
        "SET consumed_at = '2000-01-01T00:00:00Z' WHERE id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_missing_use_is_use_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "DELETE FROM authorization_grant_uses WHERE grant_id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_moment_mismatch_is_use_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "UPDATE authorization_grant_uses "
        "SET consumed_at = '2000-01-01T00:00:00Z' WHERE grant_id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


def test_use_on_revoked_grant_is_use_anomaly(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(revoke_url(machine_id, grant["id"]))
    execute(
        client,
        "INSERT INTO authorization_grant_uses "
        "(id, grant_id, machine_id, event_id, consumed_at) VALUES "
        "('stray-use', :grant_id, :machine_id, :event_id, "
        " '2026-01-01T00:00:00Z')",
        grant_id=grant["id"],
        machine_id=machine_id,
        event_id=grant["event_id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"


# --------------------------------------------------------------------------- #
# Ordering, isolation, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_first_problem_grant_is_ordered_by_issued_at(allowed_event, client):
    machine_id, _ = allowed_event
    later = issue_grant(client, machine_id)
    earlier = issue_grant(client, machine_id)
    # Make ``earlier`` sort before ``later`` by a valid stored stamp, keeping
    # its issued event in agreement so only the shared defect decides.
    execute(
        client,
        "UPDATE authorization_grants "
        "SET issued_at = '2020-01-01T00:00:00Z' WHERE id = :id",
        id=earlier["id"],
    )
    execute(
        client,
        "UPDATE authorization_grant_lifecycle_events "
        "SET occurred_at = '2020-01-01T00:00:00Z' "
        "WHERE grant_id = :id AND type = 'issued'",
        id=earlier["id"],
    )
    # Both grants carry the same defect: a terminal event while active.
    for grant in (later, earlier):
        execute(
            client,
            "INSERT INTO authorization_grant_lifecycle_events "
            "(id, machine_id, grant_id, authorization_event_id, type, "
            " occurred_at, previous_event_id, content_hash, chain_hash) "
            "VALUES "
            "('terminal-' || :suffix, :machine_id, :grant_id, :event_id, "
            " 'revoked', '2026-01-01T00:00:00Z', NULL, NULL, NULL)",
            suffix=grant["id"][:8],
            machine_id=machine_id,
            grant_id=grant["id"],
            event_id=grant["event_id"],
        )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == earlier["id"]
    assert body["anomaly"] == "sequence_or_state_mismatch"


def test_reconciliation_isolated_per_machine(allowed_event, client):
    machine_id, _ = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "DELETE FROM authorization_grant_uses WHERE grant_id = :id",
        id=grant["id"],
    )

    body = reconcile(client, machine_id)
    assert body["valid"] is False
    assert body["broken_grant_id"] == grant["id"]
    assert body["anomaly"] == "use_mismatch"
    # The other machine's empty reconciliation is unaffected by the damage.
    assert reconcile(client, other_id) == {
        "valid": True,
        "checked_grant_count": 0,
        "historical_grant_count": 0,
        "broken_grant_id": None,
        "anomaly": None,
    }


def test_reconciliation_is_strictly_read_only(allowed_event, client):
    machine_id, _ = allowed_event
    grant = issue_grant(client, machine_id)
    client.post(consume_url(machine_id, grant["id"]))
    execute(
        client,
        "DELETE FROM authorization_grant_uses WHERE grant_id = :id",
        id=grant["id"],
    )

    first = client.get(reconciliation_url(machine_id))
    second = client.get(reconciliation_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content
    assert first.json()["valid"] is False

    # The read repaired nothing: the broken state is still exactly as left.
    with client.app.state.engine.connect() as conn:
        uses = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar()
        events = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_lifecycle_events "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar()
    assert uses == 0
    assert events == 2


def test_reconciliation_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        grant = issue_grant(first, machine_id)
        first.post(consume_url(machine_id, grant["id"]))
        before = first.get(reconciliation_url(machine_id)).json()

    with TestClient(app) as second:
        assert second.get(reconciliation_url(machine_id)).json() == before
