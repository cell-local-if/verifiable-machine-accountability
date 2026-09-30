"""Tests for one-time, short-lived authorization grants and consumption.

Covers request/response shapes and every documented failure code, the
one-grant-per-event and exactly-once-consumption concurrency guarantees,
expiry semantics, the no-half-record rule, persistence across restarts, and
non-interference with the existing event / decision-basis reads.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
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


def allow_read_setup(client, machine_id, action_type="read", pattern="res/*"):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": pattern,
            "enabled": True,
        },
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": pattern,
            "effect": "allow",
            "priority": 0,
        },
    )


def record_event(
    client, machine_id, action_type="read", resource="res/x"
):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def allowed_event(client, machine_id, resource="res/x"):
    allow_read_setup(client, machine_id)
    response = record_event(client, machine_id, resource=resource)
    assert response.status_code == 201
    body = response.json()
    assert body["allowed"] is True
    assert body["reason"] == "allowed_by_policy"
    return body


def issue_grant(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume(client, machine_id, grant_id):
    return client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def grant_count(client):
    engine = client.app.state.engine
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM authorization_grants")).scalar_one()


def use_count(client):
    engine = client.app.state.engine
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM authorization_grant_uses")).scalar_one()


def delete_basis(client, event_id):
    engine = client.app.state.engine
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM authorization_decision_basis WHERE event_id = :e"),
            {"e": event_id},
        )


def corrupt_basis(client, event_id):
    """Rewrite the stored basis document so the integrity audit must fail."""
    engine = client.app.state.engine
    with engine.begin() as conn:
        document = conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :e"
            ),
            {"e": event_id},
        ).scalar_one()
        parsed = json.loads(document)
        # A stored final decision contradicting the committed event is a
        # malformed decision; the audit reports valid=false.
        parsed["decision"]["reason"] = "denied_by_policy"
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :d "
                "WHERE event_id = :e"
            ),
            {"d": json.dumps(parsed, separators=(",", ":")), "e": event_id},
        )


def expire_grant(client, grant_id):
    engine = client.app.state.engine
    past = (
        (datetime.now(timezone.utc) - timedelta(seconds=10))
        .isoformat()
        .replace("+00:00", "Z")
    )
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE authorization_grants SET expires_at = :t WHERE id = :g"),
            {"t": past, "g": grant_id},
        )


def parse_z(value):
    return datetime.fromisoformat(value[:-1] + "+00:00")


# --- issuance success --------------------------------------------------------


def test_issue_grant_success_shape(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)

    response = issue_grant(client, machine_id, event["id"], ttl_seconds=60)

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "issued_at",
        "expires_at",
        "status",
    }
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event["id"]
    assert body["status"] == "active"
    assert RFC3339_Z_RE.match(body["issued_at"])
    assert RFC3339_Z_RE.match(body["expires_at"])
    issued = parse_z(body["issued_at"])
    expires = parse_z(body["expires_at"])
    assert expires - issued == timedelta(seconds=60)


@pytest.mark.parametrize("ttl", [1, 300])
def test_issue_grant_ttl_boundaries_accepted(client, ttl):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)

    response = issue_grant(client, machine_id, event["id"], ttl_seconds=ttl)

    assert response.status_code == 201
    body = response.json()
    assert parse_z(body["expires_at"]) - parse_z(body["issued_at"]) == timedelta(
        seconds=ttl
    )


def test_issue_grant_strips_event_id_whitespace(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": f"  {event['id']}  ", "ttl_seconds": 5},
    )

    assert response.status_code == 201
    assert response.json()["event_id"] == event["id"]


# --- issuance request validation ---------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"event_id": "e"},
        {"ttl_seconds": 60},
        {"event_id": "e", "ttl_seconds": 60, "extra": 1},
        {"event_id": 123, "ttl_seconds": 60},
        {"event_id": True, "ttl_seconds": 60},
        {"event_id": None, "ttl_seconds": 60},
        {"event_id": "e", "ttl_seconds": True},
        {"event_id": "e", "ttl_seconds": False},
        {"event_id": "e", "ttl_seconds": 60.0},
        {"event_id": "e", "ttl_seconds": "60"},
        {"event_id": "e", "ttl_seconds": None},
        {"event_id": "e", "ttl_seconds": 0},
        {"event_id": "e", "ttl_seconds": -1},
        {"event_id": "e", "ttl_seconds": 301},
        {"event_id": "   ", "ttl_seconds": 60},
        {"event_id": "", "ttl_seconds": 60},
        ["e", 60],
        "not-an-object",
    ],
)
def test_issue_grant_invalid_body_is_422_invalid_grant_request(client, payload):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-grants", json=payload
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}
    assert grant_count(client) == 0


def test_issue_grant_non_json_body_is_422(client):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-grants",
        content="not json at all",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_issue_grant_extra_query_is_422_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/authorization-grants?x=1",
        json={"event_id": event["id"], "ttl_seconds": 60},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    assert grant_count(client) == 0


def test_issue_grant_invalid_query_wins_over_invalid_body(client):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/authorization-grants?x=1",
        json={"bogus": True},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --- issuance lookups and event/basis state ----------------------------------


def test_issue_grant_missing_machine_is_404(client):
    response = issue_grant(
        client, "00000000-0000-0000-0000-000000000000", "some-event"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert grant_count(client) == 0


def test_issue_grant_missing_event_is_404(client):
    machine_id = create_machine(client)

    response = issue_grant(
        client, machine_id, "00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_issue_grant_cross_machine_event_is_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event = allowed_event(client, machine_two, resource="res/two")

    response = issue_grant(client, machine_one, event["id"])

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert grant_count(client) == 0


def test_issue_grant_denied_event_is_409_event_not_allowed(client):
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/x",
            "effect": "deny",
            "priority": 0,
        },
    )
    event = record_event(client, machine_id).json()
    assert event["allowed"] is False

    response = issue_grant(client, machine_id, event["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}
    assert grant_count(client) == 0


def test_issue_grant_suspended_event_is_409_event_not_allowed(client):
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    assert (
        client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        ).status_code
        == 200
    )
    event = record_event(client, machine_id).json()
    assert event["reason"] == "machine_suspended"

    response = issue_grant(client, machine_id, event["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}


def test_issue_grant_without_basis_is_409_decision_basis_unavailable(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    delete_basis(client, event["id"])

    response = issue_grant(client, machine_id, event["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_unavailable"}}
    assert grant_count(client) == 0


def test_issue_grant_with_invalid_basis_is_409_decision_basis_invalid(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    corrupt_basis(client, event["id"])
    # The public audit reaches the same conclusion independently.
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis/integrity"
    ).json()
    assert integrity["valid"] is False

    response = issue_grant(client, machine_id, event["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_invalid"}}
    assert grant_count(client) == 0


def test_issue_grant_twice_second_is_409_grant_already_exists(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)

    first = issue_grant(client, machine_id, event["id"])
    assert first.status_code == 201
    second = issue_grant(client, machine_id, event["id"])

    assert second.status_code == 409
    assert second.json() == {"error": {"code": "grant_already_exists"}}
    assert grant_count(client) == 1


def test_concurrent_issue_per_event_succeeds_exactly_once(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    gate = threading.Event()

    def hit(_):
        gate.wait()
        return issue_grant(client, machine_id, event["id"])

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit, i) for i in range(20)]
        gate.set()
        responses = [f.result() for f in futures]

    assert sum(r.status_code == 201 for r in responses) == 1
    losers = [r for r in responses if r.status_code != 201]
    assert {r.status_code for r in losers} == {409}
    assert all(r.json() == {"error": {"code": "grant_already_exists"}} for r in losers)
    assert grant_count(client) == 1


def test_distinct_events_get_distinct_grants(client):
    machine_id = create_machine(client)
    allow_read_setup(client, machine_id)
    events = [
        record_event(client, machine_id, resource=f"res/{i}").json()
        for i in range(3)
    ]

    responses = [issue_grant(client, machine_id, e["id"]) for e in events]

    assert [r.status_code for r in responses] == [201, 201, 201]
    assert len({r.json()["id"] for r in responses}) == 3
    assert grant_count(client) == 3


# --- consumption --------------------------------------------------------------


def test_consume_active_grant_success(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()

    response = consume(client, machine_id, grant["id"])

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"grant_id", "use_id", "consumed_at"}
    assert body["grant_id"] == grant["id"]
    assert UUID_RE.match(body["use_id"])
    assert RFC3339_Z_RE.match(body["consumed_at"])
    assert use_count(client) == 1


def test_consume_twice_second_is_409_grant_consumed(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()

    first = consume(client, machine_id, grant["id"])
    second = consume(client, machine_id, grant["id"])

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "grant_consumed"}}
    assert use_count(client) == 1


def test_concurrent_consume_succeeds_exactly_once(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()
    gate = threading.Event()

    def hit(_):
        gate.wait()
        return consume(client, machine_id, grant["id"])

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit, i) for i in range(20)]
        gate.set()
        responses = [f.result() for f in futures]

    winners = [r for r in responses if r.status_code == 200]
    losers = [r for r in responses if r.status_code != 200]
    assert len(winners) == 1
    assert {r.status_code for r in losers} == {409}
    assert all(r.json() == {"error": {"code": "grant_consumed"}} for r in losers)
    assert len({r.json()["use_id"] for r in winners}) == 1
    assert use_count(client) == 1


def test_consume_expired_grant_is_409_grant_expired_and_writes_nothing(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"], ttl_seconds=1).json()
    expire_grant(client, grant["id"])

    response = consume(client, machine_id, grant["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}
    assert use_count(client) == 0

    # Repeated attempts keep reporting expiry and leave the grant active.
    again = consume(client, machine_id, grant["id"])
    assert again.status_code == 409
    assert again.json() == {"error": {"code": "grant_expired"}}
    assert use_count(client) == 0
    with client.app.state.engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :g"),
            {"g": grant["id"]},
        ).scalar_one()
    assert status == "active"


def test_consumed_grant_reports_consumed_even_after_expiry(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()
    assert consume(client, machine_id, grant["id"]).status_code == 200
    expire_grant(client, grant["id"])

    response = consume(client, machine_id, grant["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}
    assert use_count(client) == 1


def test_consume_missing_machine_is_404(client):
    response = consume(
        client, "00000000-0000-0000-0000-000000000000", "some-grant"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_consume_missing_grant_is_404(client):
    machine_id = create_machine(client)

    response = consume(
        client, machine_id, "00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_consume_cross_machine_grant_is_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event = allowed_event(client, machine_two, resource="res/two")
    grant = issue_grant(client, machine_two, event["id"]).json()

    response = consume(client, machine_one, grant["id"])

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert use_count(client) == 0


def test_consume_extra_query_is_422_invalid_query(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()

    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
        f"?x=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    # A rejected consumption flips nothing.
    assert consume(client, machine_id, grant["id"]).status_code == 200


def test_consume_empty_query_string_is_accepted(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()

    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    )

    assert response.status_code == 200


def test_consume_with_body_is_422_invalid_query(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    grant = issue_grant(client, machine_id, event["id"]).json()

    response = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume",
        json={},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    assert use_count(client) == 0


def test_consume_other_method_not_allowed(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/authorization-grants"
    )

    assert response.status_code == 405


# --- persistence and non-interference ----------------------------------------


def test_grants_and_uses_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)
    machine_id = None
    event_id = None
    grant_id = None

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = allowed_event(first, machine_id, resource="res/persist")
        event_id = event["id"]
        grant_id = issue_grant(first, machine_id, event_id).json()["id"]

    with TestClient(app) as second:
        # Re-issuance after a restart still observes the existing grant.
        reissue = issue_grant(second, machine_id, event_id)
        assert reissue.status_code == 409
        assert reissue.json() == {"error": {"code": "grant_already_exists"}}

        # The unconsumed grant survives and can still be consumed once.
        response = consume(second, machine_id, grant_id)
        assert response.status_code == 200
        use = response.json()
        assert use["grant_id"] == grant_id

        second_use = consume(second, machine_id, grant_id)
        assert second_use.status_code == 409
        assert second_use.json() == {"error": {"code": "grant_consumed"}}

        with second.app.state.engine.connect() as conn:
            stored = conn.execute(
                text(
                    "SELECT status, use_count FROM authorization_grants g "
                    "JOIN (SELECT grant_id, COUNT(*) AS use_count "
                    "FROM authorization_grant_uses GROUP BY grant_id) u "
                    "ON u.grant_id = g.id WHERE g.id = :g"
                ),
                {"g": grant_id},
            ).one()
        assert stored.status == "consumed"
        assert stored.use_count == 1


def test_grants_do_not_rewrite_events_or_basis(client):
    machine_id = create_machine(client)
    event = allowed_event(client, machine_id)
    basis_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis"
    ).content
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()

    grant = issue_grant(client, machine_id, event["id"]).json()
    consume(client, machine_id, grant["id"])
    # A losing second issuance also changes nothing.
    assert issue_grant(client, machine_id, event["id"]).status_code == 409

    events_after = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    basis_after = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis"
    ).content
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis/integrity"
    ).json()
    chain = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    assert events_after == events_before
    assert basis_after == basis_before
    assert integrity["valid"] is True
    assert chain["valid"] is True
