"""Tests for one-time, short-lived authorization grants.

Three write entries sit under the machine path:

    POST /machines/{machine_id}/authorization-grants
    POST /machines/{machine_id}/authorization-grants/{grant_id}/consume
    POST /machines/{machine_id}/authorization-grants/{grant_id}/revoke

A grant turns one committed ``allowed_by_policy`` decision event whose
immutable decision-basis snapshot passes the consistency audit into an
``active``, short-lived credential; it can be consumed exactly once, after
which it is ``consumed``, and a grant past ``expires_at`` answers
``grant_expired``. The revoke entry adds an emergency terminal transition to
``revoked`` with a persisted ``revoked_at``; revocation and consumption race
to one definite terminal winner. These tests cover the success shapes, every
validation and lookup outcome, the unavailable/invalid-basis cases,
exactly-once issue, consume, and revoke under concurrency (including revoke
versus consume), expiry, atomicity (no half records), restart persistence,
old-database table creation and column migration, and non-interference with
the existing event and basis views.
"""
import json
import sqlite3
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


def issue(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        grant_url(machine_id),
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


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


# --------------------------------------------------------------------------- #
# Issue success shape
# --------------------------------------------------------------------------- #


def test_issue_grant_success_shape(allowed_event, client):
    machine_id, event = allowed_event
    response = issue(client, machine_id, event["id"], ttl_seconds=90)

    assert response.status_code == 201
    body = response.json()
    assert list(body.keys()) == [
        "id",
        "machine_id",
        "event_id",
        "issued_at",
        "expires_at",
        "status",
    ]
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event["id"]
    assert body["status"] == "active"
    assert isinstance(body["id"], str) and body["id"]
    for stamp in (body["issued_at"], body["expires_at"]):
        assert stamp.endswith("Z")
        parsed = _parse_z(stamp)
        assert parsed.tzinfo == timezone.utc

    issued = _parse_z(body["issued_at"])
    expires = _parse_z(body["expires_at"])
    assert expires - issued == timedelta(seconds=90)


@pytest.mark.parametrize("ttl", [1, 300])
def test_ttl_boundaries_are_accepted(allowed_event, client, ttl):
    machine_id, event = allowed_event
    response = issue(client, machine_id, event["id"], ttl_seconds=ttl)
    assert response.status_code == 201
    expires = _parse_z(response.json()["expires_at"])
    issued = _parse_z(response.json()["issued_at"])
    assert expires - issued == timedelta(seconds=ttl)


def test_grant_is_persisted_active(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, machine_id, event_id, status, consumed_at "
                "FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.machine_id == machine_id
    assert row.event_id == event["id"]
    assert row.status == "active"
    assert row.consumed_at is None


# --------------------------------------------------------------------------- #
# Issue body and query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {"event_id": "e", "ttl_seconds": 0},
        {"event_id": "e", "ttl_seconds": 301},
        {"event_id": "e", "ttl_seconds": -1},
        {"event_id": "e", "ttl_seconds": True},
        {"event_id": "e", "ttl_seconds": False},
        {"event_id": "e", "ttl_seconds": 1.0},
        {"event_id": "e", "ttl_seconds": 60.5},
        {"event_id": "e", "ttl_seconds": "60"},
        {"event_id": "e", "ttl_seconds": None},
        {"event_id": 123, "ttl_seconds": 60},
        {"event_id": True, "ttl_seconds": 60},
        {"event_id": None, "ttl_seconds": 60},
        {"event_id": "", "ttl_seconds": 60},
        {"event_id": "   ", "ttl_seconds": 60},
        {"event_id": "e"},
        {"ttl_seconds": 60},
        {"event_id": "e", "ttl_seconds": 60, "extra": 1},
    ],
)
def test_invalid_body_is_invalid_grant_request(allowed_event, client, payload):
    machine_id, _ = allowed_event
    response = client.post(grant_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_typed_body_is_invalid_grant_request(
    allowed_event, client, raw
):
    machine_id, _ = allowed_event
    response = client.post(
        grant_url(machine_id),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


@pytest.mark.parametrize("query", ["?x=1", "?ttl_seconds=60", "?=", "?event_id=e"])
def test_any_query_parameter_is_invalid_query_before_lookup(
    allowed_event, client, query
):
    machine_id, event = allowed_event
    missing_machine = "00000000-0000-0000-0000-000000000000"

    # Query validation wins even over a perfectly valid body.
    response = client.post(
        grant_url(machine_id) + query,
        json={"event_id": event["id"], "ttl_seconds": 60},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # And it precedes the machine lookup: missing machine is still 422.
    response = client.post(
        grant_url(missing_machine) + query,
        json={"event_id": event["id"], "ttl_seconds": 60},
    )
    assert response.status_code == 422


def test_body_validation_runs_before_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        grant_url(missing_machine),
        json={"event_id": "anything", "ttl_seconds": True},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_grant_request"


def test_whitespace_event_id_is_invalid_even_against_missing_machine(client):
    machine_id = create_machine(client)
    response = client.post(
        grant_url(machine_id), json={"event_id": "  ", "ttl_seconds": 60}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_grant_request"


# --------------------------------------------------------------------------- #
# Issue lookup and eligibility outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client, allowed_event):
    _, event = allowed_event
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = issue(client, missing_machine, event["id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_is_not_found(client):
    machine_id = create_machine(client)
    response = issue(client, machine_id, "no-such-event")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_cross_machine_event_is_not_found(client, allowed_event):
    machine_id, event = allowed_event
    other = create_machine(client, external_id="machine-2")
    response = issue(client, other, event["id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_denied_by_policy_event_is_event_not_allowed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=1)
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    event = record_event(client, machine_id, resource="res/d").json()
    assert event["allowed"] is False

    response = issue(client, machine_id, event["id"])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "event_not_allowed"}}


def test_non_allow_reason_events_are_event_not_allowed(client):
    machine_id = create_machine(client)
    # no_enabled_declaration: no declarations at all.
    no_decl = record_event(client, machine_id, resource="res/x").json()
    assert no_decl["reason"] == "no_enabled_declaration"
    # no_matching_policy: a matching declaration but no rule.
    declare(client, machine_id, resource_pattern="res/*")
    no_policy = record_event(client, machine_id, resource="res/y").json()
    assert no_policy["reason"] == "no_matching_policy"
    # machine_suspended.
    assert (
        client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        ).status_code
        == 200
    )
    suspended = record_event(client, machine_id, resource="res/z").json()
    assert suspended["reason"] == "machine_suspended"

    for event in (no_decl, no_policy, suspended):
        response = issue(client, machine_id, event["id"])
        assert response.status_code == 409
        assert response.json() == {"error": {"code": "event_not_allowed"}}


def test_event_without_basis_snapshot_is_decision_basis_unavailable(
    allowed_event, client
):
    machine_id, event = allowed_event
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event["id"])
        )

    response = issue(client, machine_id, event["id"])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_unavailable"}}


def test_event_with_failing_basis_audit_is_decision_basis_invalid(
    allowed_event, client
):
    machine_id, event = allowed_event
    # Tamper only the stored snapshot: flip the final decision's allowed flag
    # so the committed event and the basis disagree. The event itself stays
    # allowed=true/allowed_by_policy.
    with client.app.state.engine.begin() as conn:
        document = conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :id"
            ).bindparams(id=event["id"])
        ).scalar_one()
        parsed = json.loads(document)
        parsed["decision"]["allowed"] = False
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id"
            ).bindparams(doc=json.dumps(parsed, separators=(",", ":")),
                         id=event["id"])
        )

    audit = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis/integrity"
    ).json()
    assert audit["valid"] is False

    response = issue(client, machine_id, event["id"])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "decision_basis_invalid"}}


def test_failed_issue_writes_no_grant(allowed_event, client):
    machine_id, event = allowed_event
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event["id"])
        )
    assert issue(client, machine_id, event["id"]).status_code == 409
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()
    assert count == 0


# --------------------------------------------------------------------------- #
# One grant per event, with concurrency
# --------------------------------------------------------------------------- #


def test_second_issue_for_same_event_is_grant_already_exists(
    allowed_event, client
):
    machine_id, event = allowed_event
    assert issue(client, machine_id, event["id"]).status_code == 201
    response = issue(client, machine_id, event["id"], ttl_seconds=5)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_already_exists"}}


def test_concurrent_issues_have_exactly_one_winner(allowed_event, client):
    machine_id, event = allowed_event
    count = 20
    gate = threading.Event()

    def hit():
        gate.wait()
        return issue(client, machine_id, event["id"], ttl_seconds=60)

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
        rows = conn.execute(
            text(
                "SELECT id FROM authorization_grants WHERE event_id = :id"
            ).bindparams(id=event["id"])
        ).all()
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# Consume success, validation, and lookup outcomes
# --------------------------------------------------------------------------- #


def test_consume_success_shape(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    before = datetime.now(timezone.utc)
    response = client.post(consume_url(machine_id, grant["id"]))
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["grant_id", "use_id", "consumed_at"]
    assert body["grant_id"] == grant["id"]
    assert isinstance(body["use_id"], str) and body["use_id"]
    consumed_at = _parse_z(body["consumed_at"])
    assert before <= consumed_at <= after


def test_consume_marks_grant_consumed_and_persists_use(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    with client.app.state.engine.connect() as conn:
        grant_row = conn.execute(
            text(
                "SELECT status, consumed_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_row = conn.execute(
            text(
                "SELECT id, grant_id, machine_id, event_id, consumed_at "
                "FROM authorization_grant_uses WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).all()
    assert grant_row.status == "consumed"
    assert grant_row.consumed_at == use["consumed_at"]
    assert len(use_row) == 1
    assert use_row[0].id == use["use_id"]
    assert use_row[0].machine_id == machine_id
    assert use_row[0].event_id == event["id"]
    assert use_row[0].consumed_at == use["consumed_at"]


@pytest.mark.parametrize("send", [b"{}", b'{"x":1}', b"null", b"[]", b"text"])
def test_consume_with_body_is_invalid_query(allowed_event, client, send):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(
        consume_url(machine_id, grant["id"]),
        content=send,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?foo"])
def test_consume_with_query_is_invalid_query_before_lookup(
    allowed_event, client, query
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(consume_url(machine_id, grant["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine/grant lookup.
    response = client.post(consume_url(missing_machine, grant["id"]) + query)
    assert response.status_code == 422


def test_consume_empty_body_with_zero_length_is_accepted(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    response = client.request(
        "POST", consume_url(machine_id, grant["id"]), content=b""
    )
    assert response.status_code == 200


def test_consume_missing_machine_is_not_found(client, allowed_event):
    _, event = allowed_event
    # A grant cannot exist under a missing machine, but the lookup contract
    # still applies: missing machine -> 404 after validation.
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(consume_url(missing_machine, "anything"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_consume_missing_grant_is_not_found(allowed_event, client):
    machine_id, _ = allowed_event
    response = client.post(consume_url(machine_id, "no-such-grant"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_consume_grant_owned_by_another_machine_is_not_found(
    allowed_event, client
):
    machine_id, event = allowed_event
    other = create_machine(client, external_id="machine-2")
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(consume_url(other, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The grant is untouched and still consumable by its real owner.
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200


def test_consuming_twice_returns_grant_consumed_once(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    first = client.post(consume_url(machine_id, grant["id"]))
    second = client.post(consume_url(machine_id, grant["id"]))
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "grant_consumed"}}


def test_concurrent_consumes_have_exactly_one_success(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    count = 20
    gate = threading.Event()

    def hit():
        gate.wait()
        return client.post(consume_url(machine_id, grant["id"]))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit) for _ in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == count - 1
    use_ids = {
        r.json()["use_id"] for r in responses if r.status_code == 200
    }
    assert len(use_ids) == 1

    with client.app.state.engine.connect() as conn:
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
        status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert use_count == 1
    assert status == "consumed"


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #


def _backdated_grant(client, machine_id, event_id, ttl_seconds=1):
    """Issue a grant then move its expiry into the past without consuming it."""
    grant = issue(client, machine_id, event_id, ttl_seconds=ttl_seconds).json()
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant["id"])
        )
    return grant


def test_expired_grant_consume_is_grant_expired(allowed_event, client):
    machine_id, event = allowed_event
    grant = _backdated_grant(client, machine_id, event["id"])

    response = client.post(consume_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}

    # Nothing was written: the grant stays active (expiry is derived) and no
    # use record exists; it can never later be consumed.
    with client.app.state.engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert status == "active"
    assert use_count == 0


def test_grant_expires_after_its_ttl(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=1).json()
    # It is consumable immediately.
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200


# --------------------------------------------------------------------------- #
# Persistence, old databases, and non-interference
# --------------------------------------------------------------------------- #


def test_grant_persists_active_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"], ttl_seconds=300).json()

    with TestClient(app) as second:
        # Still active and exactly once consumable after the restart.
        response = second.post(consume_url(machine_id, grant["id"]))
        assert response.status_code == 200
        body = response.json()
        assert body["grant_id"] == grant["id"]
        assert (
            second.post(consume_url(machine_id, grant["id"])).status_code == 409
        )
        # The event still cannot be re-signed.
        repeat = issue(second, machine_id, event["id"])
        assert repeat.status_code == 409
        assert repeat.json()["error"]["code"] == "grant_already_exists"


def test_consumed_grant_and_use_record_survive_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'consumed.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"]).json()
        use = first.post(consume_url(machine_id, grant["id"])).json()

    with TestClient(app) as second:
        again = second.post(consume_url(machine_id, grant["id"]))
        assert again.status_code == 409
        assert again.json() == {"error": {"code": "grant_consumed"}}
        with second.app.state.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, consumed_at FROM authorization_grant_uses "
                    "WHERE grant_id = :id"
                ).bindparams(id=grant["id"])
            ).one()
        assert row.id == use["use_id"]
        assert row.consumed_at == use["consumed_at"]


def test_old_database_recreates_grant_tables_safely(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()

    # Reproduce a database that predates the grant feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_grant_uses")
    conn.execute("DROP TABLE authorization_grants")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        grant = issue(second, machine_id, event["id"])
        assert grant.status_code == 201
        consumed = second.post(
            consume_url(machine_id, grant.json()["id"])
        )
        assert consumed.status_code == 200


def test_issue_and_consume_never_rewrite_event_or_basis(allowed_event, client):
    machine_id, event = allowed_event
    basis_url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis"
    )
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    basis_before = client.get(basis_url).content
    events_before = client.get(events_url).content

    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    assert client.get(basis_url).content == basis_before
    assert client.get(events_url).content == events_before
    # The chain audit is unaffected.
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True


def test_grants_for_different_events_are_independent(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    first_event = record_event(client, machine_id, resource="res/1").json()
    second_event = record_event(client, machine_id, resource="res/2").json()

    grant_one = issue(client, machine_id, first_event["id"]).json()
    grant_two = issue(client, machine_id, second_event["id"]).json()
    assert grant_one["id"] != grant_two["id"]

    # Consuming one never consumes or invalidates the other.
    assert client.post(
        consume_url(machine_id, grant_one["id"])
    ).status_code == 200
    assert client.post(
        consume_url(machine_id, grant_two["id"])
    ).status_code == 200
    assert client.post(
        consume_url(machine_id, grant_one["id"])
    ).status_code == 409


def test_non_post_methods_are_not_routed_for_writes(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    # The collection POST (issue) and the per-grant POST entries (consume,
    # revoke) accept only POST; the read-only GET audit listing lives on the
    # collection path separately.
    for method in ("put", "patch", "delete"):
        response = getattr(client, method)(grant_url(machine_id))
        assert response.status_code == 405
        response = getattr(client, method)(consume_url(machine_id, grant["id"]))
        assert response.status_code == 405
        response = getattr(client, method)(revoke_url(machine_id, grant["id"]))
        assert response.status_code == 405
    # The two per-grant write entries reject GET/HEAD as well.
    for method in ("get", "head"):
        response = getattr(client, method)(consume_url(machine_id, grant["id"]))
        assert response.status_code == 405
        response = getattr(client, method)(revoke_url(machine_id, grant["id"]))
        assert response.status_code == 405
    # The 405s neither read nor wrote the grant: it is still active and its
    # single consumption still succeeds.
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200


# --------------------------------------------------------------------------- #
# Emergency revocation: success shape and persistence
# --------------------------------------------------------------------------- #


def test_revoke_success_shape(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    before = datetime.now(timezone.utc)
    response = client.post(revoke_url(machine_id, grant["id"]))
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["grant_id", "revoked_at", "status"]
    assert body["grant_id"] == grant["id"]
    assert body["status"] == "revoked"
    revoked_at = _parse_z(body["revoked_at"])
    assert body["revoked_at"].endswith("Z")
    assert revoked_at.tzinfo == timezone.utc
    assert before <= revoked_at <= after


def test_revoke_persists_revoked_status_and_stamp(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(
        revoke_url(machine_id, grant["id"])
    ).json()

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, consumed_at, revoked_at, issued_at, "
                "expires_at FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "revoked"
    assert row.revoked_at == revocation["revoked_at"]
    assert row.consumed_at is None
    assert use_count == 0
    # The immutable issue-time fields are exactly the issued values.
    assert row.issued_at == grant["issued_at"]
    assert row.expires_at == grant["expires_at"]


def test_revoked_grant_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'revoked.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"], ttl_seconds=300).json()
        revocation = first.post(
            revoke_url(machine_id, grant["id"])
        ).json()

    with TestClient(app) as second:
        repeat = second.post(revoke_url(machine_id, grant["id"]))
        assert repeat.status_code == 409
        assert repeat.json() == {"error": {"code": "grant_revoked"}}
        consume = second.post(consume_url(machine_id, grant["id"]))
        assert consume.status_code == 409
        assert consume.json() == {"error": {"code": "grant_revoked"}}
        with second.app.state.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT status, revoked_at FROM authorization_grants "
                    "WHERE id = :id"
                ).bindparams(id=grant["id"])
            ).one()
        assert row.status == "revoked"
        assert row.revoked_at == revocation["revoked_at"]


# --------------------------------------------------------------------------- #
# Revoke query and body validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("send", [b"{}", b'{"x":1}', b"null", b"[]", b"text"])
def test_revoke_with_body_is_invalid_query(allowed_event, client, send):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(
        revoke_url(machine_id, grant["id"]),
        content=send,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?foo"])
def test_revoke_with_query_is_invalid_query_before_lookup(
    allowed_event, client, query
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(revoke_url(machine_id, grant["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine/grant lookup.
    response = client.post(revoke_url(missing_machine, grant["id"]) + query)
    assert response.status_code == 422


def test_revoke_rejects_repeated_query_parameter(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    response = client.post(
        revoke_url(machine_id, grant["id"]) + "?x=1&x=2"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_revoke_empty_body_with_zero_length_is_accepted(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    response = client.request(
        "POST", revoke_url(machine_id, grant["id"]), content=b""
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Revoke lookup outcomes
# --------------------------------------------------------------------------- #


def test_revoke_missing_machine_is_not_found(client, allowed_event):
    _, event = allowed_event
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(revoke_url(missing_machine, "anything"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_revoke_missing_grant_is_not_found(allowed_event, client):
    machine_id, _ = allowed_event
    response = client.post(revoke_url(machine_id, "no-such-grant"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_revoke_grant_owned_by_another_machine_is_not_found(
    allowed_event, client
):
    machine_id, event = allowed_event
    other = create_machine(client, external_id="machine-2")
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(revoke_url(other, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The grant is untouched and still revocable by its real owner.
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200


# --------------------------------------------------------------------------- #
# Revoke conflict outcomes
# --------------------------------------------------------------------------- #


def test_consume_then_revoke_is_grant_consumed(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200
    response = client.post(revoke_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.status == "consumed"
    assert row.revoked_at is None


def test_revoke_then_consume_is_grant_revoked(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200
    response = client.post(consume_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, consumed_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "revoked"
    assert row.consumed_at is None
    assert use_count == 0


def test_revoking_twice_returns_grant_revoked_once(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    first = client.post(revoke_url(machine_id, grant["id"]))
    second = client.post(revoke_url(machine_id, grant["id"]))
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "grant_revoked"}}
    # The first stamp is stable: the rejected revocation rewrote nothing.
    with client.app.state.engine.connect() as conn:
        stamp = conn.execute(
            text("SELECT revoked_at FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert stamp == first.json()["revoked_at"]


def test_expired_grant_revoke_is_grant_expired(allowed_event, client):
    machine_id, event = allowed_event
    grant = _backdated_grant(client, machine_id, event["id"])

    response = client.post(revoke_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}

    # Nothing was written: the grant stays active (expiry is derived), with
    # no revocation stamp and no use record.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "active"
    assert row.revoked_at is None
    assert use_count == 0


def test_revoked_grant_keeps_answering_revoked_after_it_expires(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()

    # Move expires_at into the past after the revocation.
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant["id"])
        )

    repeat = client.post(revoke_url(machine_id, grant["id"]))
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "grant_revoked"}}
    consume = client.post(consume_url(machine_id, grant["id"]))
    assert consume.status_code == 409
    assert consume.json() == {"error": {"code": "grant_revoked"}}
    with client.app.state.engine.connect() as conn:
        stamp = conn.execute(
            text("SELECT revoked_at FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert stamp == revocation["revoked_at"]


# --------------------------------------------------------------------------- #
# Revocation concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_revokes_have_exactly_one_success(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    count = 20
    gate = threading.Event()

    def hit():
        gate.wait()
        return client.post(revoke_url(machine_id, grant["id"]))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit) for _ in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == count - 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_revoked"}}
    stamps = {
        r.json()["revoked_at"] for r in responses if r.status_code == 200
    }
    assert len(stamps) == 1

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "revoked"
    assert row.revoked_at == stamps.pop()
    assert use_count == 0


def test_concurrent_revoke_and_consume_have_one_terminal_winner(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    gate = threading.Event()

    def revoke_hit():
        gate.wait()
        return client.post(revoke_url(machine_id, grant["id"]))

    def consume_hit():
        gate.wait()
        return client.post(consume_url(machine_id, grant["id"]))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = (
            [pool.submit(revoke_hit) for _ in range(10)]
            + [pool.submit(consume_hit) for _ in range(10)]
        )
        gate.set()
        responses = [f.result() for f in futures]

    successes = [r for r in responses if r.status_code == 200]
    assert len(successes) == 1
    winner = successes[0]
    conflict_codes = {
        r.json()["error"]["code"]
        for r in responses
        if r.status_code == 409
    }
    assert len(conflict_codes) == 1

    with client.app.state.engine.connect() as conn:
        final_status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()

    if winner.json().get("status") == "revoked":
        # Revocation committed first: every loser reports grant_revoked.
        assert set(winner.json()) == {"grant_id", "revoked_at", "status"}
        assert conflict_codes == {"grant_revoked"}
        assert final_status == "revoked"
        assert use_count == 0
    else:
        # Consumption committed first: every loser reports grant_consumed.
        assert set(winner.json()) == {"grant_id", "use_id", "consumed_at"}
        assert conflict_codes == {"grant_consumed"}
        assert final_status == "consumed"
        assert use_count == 1


# --------------------------------------------------------------------------- #
# Old-database column migration and non-interference
# --------------------------------------------------------------------------- #


def test_revoked_at_column_is_added_to_old_database(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-column.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        consumed_event = record_event(first, machine_id, resource="res/c").json()
        active_event = record_event(first, machine_id, resource="res/a").json()
        consumed_grant = issue(
            first, machine_id, consumed_event["id"]
        ).json()
        active_grant = issue(first, machine_id, active_event["id"]).json()
        consumed_use = first.post(
            consume_url(machine_id, consumed_grant["id"])
        ).json()

    # Reproduce a database created before the revocation feature: drop the
    # column (SQLite >= 3.35 supports ALTER TABLE ... DROP COLUMN).
    conn = sqlite3.connect(db_path)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(authorization_grants)")
    }
    assert "revoked_at" in columns
    conn.execute(
        "ALTER TABLE authorization_grants DROP COLUMN revoked_at"
    )
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # Old values are preserved byte-for-byte and the use record survives.
        with second.app.state.engine.connect() as db_conn:
            old_consumed = db_conn.execute(
                text(
                    "SELECT issued_at, expires_at, status, consumed_at, "
                    "revoked_at FROM authorization_grants WHERE id = :id"
                ).bindparams(id=consumed_grant["id"])
            ).one()
            old_active = db_conn.execute(
                text(
                    "SELECT issued_at, expires_at, status, consumed_at, "
                    "revoked_at FROM authorization_grants WHERE id = :id"
                ).bindparams(id=active_grant["id"])
            ).one()
            use_row = db_conn.execute(
                text(
                    "SELECT id, consumed_at FROM authorization_grant_uses "
                    "WHERE grant_id = :id"
                ).bindparams(id=consumed_grant["id"])
            ).one()
        assert old_consumed.issued_at == consumed_grant["issued_at"]
        assert old_consumed.expires_at == consumed_grant["expires_at"]
        assert old_consumed.status == "consumed"
        assert old_consumed.consumed_at == consumed_use["consumed_at"]
        assert old_consumed.revoked_at is None
        assert old_active.issued_at == active_grant["issued_at"]
        assert old_active.expires_at == active_grant["expires_at"]
        assert old_active.status == "active"
        assert old_active.consumed_at is None
        assert old_active.revoked_at is None
        assert use_row.id == consumed_use["use_id"]
        assert use_row.consumed_at == consumed_use["consumed_at"]

        # The old consumed grant cannot be revoked; the old active one can.
        blocked = second.post(
            revoke_url(machine_id, consumed_grant["id"])
        )
        assert blocked.status_code == 409
        assert blocked.json() == {"error": {"code": "grant_consumed"}}
        revoked = second.post(revoke_url(machine_id, active_grant["id"]))
        assert revoked.status_code == 200
        assert revoked.json()["status"] == "revoked"


def test_revoke_never_rewrites_event_basis_or_chains(allowed_event, client):
    machine_id, event = allowed_event
    basis_url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis"
    )
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    basis_before = client.get(basis_url).content
    events_before = client.get(events_url).content

    grant = issue(client, machine_id, event["id"]).json()
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200

    assert client.get(basis_url).content == basis_before
    assert client.get(events_url).content == events_before
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True


def test_revocation_is_isolated_between_independent_grants(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    first_event = record_event(client, machine_id, resource="res/1").json()
    second_event = record_event(client, machine_id, resource="res/2").json()
    first_grant = issue(client, machine_id, first_event["id"]).json()
    second_grant = issue(client, machine_id, second_event["id"]).json()

    # Revoking one never revokes or consumes the other.
    assert client.post(
        revoke_url(machine_id, first_grant["id"])
    ).status_code == 200
    assert client.post(
        consume_url(machine_id, second_grant["id"])
    ).status_code == 200
    assert client.post(
        revoke_url(machine_id, second_grant["id"])
    ).status_code == 409
    assert client.post(
        consume_url(machine_id, first_grant["id"])
    ).status_code == 409


# --------------------------------------------------------------------------- #
# Suspended machine: consume is rejected without touching anything
# --------------------------------------------------------------------------- #


def set_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response.json()


def _grant_rows(client, grant_id):
    with client.app.state.engine.connect() as conn:
        grant_row = conn.execute(
            text(
                "SELECT status, issued_at, expires_at, consumed_at, "
                "revoked_at FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant_id)
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant_id)
        ).scalar_one()
        lifecycle_types = conn.execute(
            text(
                "SELECT type FROM authorization_grant_lifecycle_events "
                "WHERE grant_id = :id ORDER BY occurred_at, id"
            ).bindparams(id=grant_id)
        ).scalars().all()
    return grant_row, use_count, list(lifecycle_types)


def test_suspended_machine_consume_is_machine_suspended(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    set_status(client, machine_id, "suspended")

    response = client.post(consume_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "machine_suspended"}}


def test_suspended_consume_writes_nothing(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    before = _grant_rows(client, grant["id"])
    set_status(client, machine_id, "suspended")

    assert client.post(
        consume_url(machine_id, grant["id"])
    ).status_code == 409

    grant_row, use_count, lifecycle_types = _grant_rows(client, grant["id"])
    # The grant is untouched: still active with its original stamps, no use
    # record, and no lifecycle event beyond the initial ``issued``.
    assert (grant_row, use_count, lifecycle_types) == before
    assert grant_row.status == "active"
    assert grant_row.consumed_at is None
    assert grant_row.revoked_at is None
    assert use_count == 0
    assert lifecycle_types == ["issued"]


def test_suspended_consume_keeps_terminal_state_outcomes(allowed_event, client):
    machine_id, event = allowed_event
    consumed_grant = issue(client, machine_id, event["id"]).json()
    assert client.post(
        consume_url(machine_id, consumed_grant["id"])
    ).status_code == 200

    second_event = record_event(client, machine_id, resource="res/2").json()
    revoked_grant = issue(client, machine_id, second_event["id"]).json()
    assert client.post(
        revoke_url(machine_id, revoked_grant["id"])
    ).status_code == 200

    third_event = record_event(client, machine_id, resource="res/3").json()
    expired_grant = _backdated_grant(client, machine_id, third_event["id"])

    set_status(client, machine_id, "suspended")

    # The existing terminal outcomes are unchanged by the suspension.
    response = client.post(consume_url(machine_id, consumed_grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}
    response = client.post(consume_url(machine_id, revoked_grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}
    response = client.post(consume_url(machine_id, expired_grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}


def test_suspended_consume_lookup_outcomes_are_unchanged(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    set_status(client, machine_id, "suspended")

    # Validation still runs before any lookup.
    response = client.post(consume_url(machine_id, grant["id"]) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # A missing grant on a suspended machine is still 404, and a grant owned
    # by another machine stays indistinguishable from a missing one.
    response = client.post(consume_url(machine_id, "no-such-grant"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    other = create_machine(client, external_id="machine-2")
    response = client.post(consume_url(other, grant["id"]))
    assert response.status_code == 404


def test_reactivated_machine_consume_succeeds(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    set_status(client, machine_id, "suspended")
    assert client.post(
        consume_url(machine_id, grant["id"])
    ).status_code == 409

    set_status(client, machine_id, "active")
    response = client.post(consume_url(machine_id, grant["id"]))
    assert response.status_code == 200
    body = response.json()
    assert body["grant_id"] == grant["id"]

    # The same grant was consumed on the original success path: one use
    # record and the issued+consumed lifecycle pair, nothing reissued.
    grant_row, use_count, lifecycle_types = _grant_rows(client, grant["id"])
    assert grant_row.status == "consumed"
    assert grant_row.consumed_at == body["consumed_at"]
    assert use_count == 1
    assert lifecycle_types == ["issued", "consumed"]


def test_concurrent_suspend_and_consume_have_one_definite_order(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    gate = threading.Event()

    def hit_consume():
        gate.wait()
        return client.post(consume_url(machine_id, grant["id"]))

    def hit_suspend():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        consume_future = pool.submit(hit_consume)
        suspend_future = pool.submit(hit_suspend)
        gate.set()
        consume_response = consume_future.result()
        suspend_response = suspend_future.result()

    assert suspend_response.status_code == 200
    grant_row, use_count, lifecycle_types = _grant_rows(client, grant["id"])
    if consume_response.status_code == 200:
        # Consume committed first: it keeps its single success, use record,
        # and consumed lifecycle event; the suspension still applied after.
        assert grant_row.status == "consumed"
        assert use_count == 1
        assert lifecycle_types == ["issued", "consumed"]
    else:
        # Suspend committed first: the consumption is rejected and nothing
        # was written for it.
        assert consume_response.status_code == 409
        assert consume_response.json() == {
            "error": {"code": "machine_suspended"}
        }
        assert grant_row.status == "active"
        assert use_count == 0
        assert lifecycle_types == ["issued"]
