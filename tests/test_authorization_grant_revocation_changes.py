"""Tests for emergency revocation of one-time authorization grants.

The new write entry sits beside consume under the machine path:

    POST /machines/{machine_id}/authorization-grants/{grant_id}/revoke

Only an unconsumed, unrevoked grant whose current instant precedes
``expires_at`` can be revoked; success answers
``{grant_id, revoked_at, status: "revoked"}`` and persists. These tests
cover the success shape, every validation and lookup outcome, the three
conflict outcomes (consumed / revoked / expired) with nothing written on
rejection, exactly-once revoke under concurrency, the revoke/consume race
with exactly one terminal-state winner, machine isolation, restart
persistence, the safe column addition on databases created before the
feature, non-interference with issue/consume and the event/basis views, and
405 for every non-POST method.
"""
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
def active_grant(client):
    """A machine + allowed event + freshly issued active grant."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    assert event["reason"] == "allowed_by_policy"
    grant = issue(client, machine_id, event["id"]).json()
    assert grant["status"] == "active"
    return machine_id, event, grant


# --------------------------------------------------------------------------- #
# Revoke success shape and persistence
# --------------------------------------------------------------------------- #


def test_revoke_success_shape(active_grant, client):
    machine_id, event, grant = active_grant
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


def test_revoke_persists_status_and_revoked_at(active_grant, client):
    machine_id, event, grant = active_grant
    response = client.post(revoke_url(machine_id, grant["id"]))
    assert response.status_code == 200
    revoked_at = response.json()["revoked_at"]

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at, issued_at, expires_at, "
                "consumed_at FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()

    assert row.status == "revoked"
    assert row.revoked_at == revoked_at
    # The immutable issue/expiry stamps and the absent consumption stay.
    assert row.issued_at == grant["issued_at"]
    assert row.expires_at == grant["expires_at"]
    assert row.consumed_at is None
    assert use_count == 0


def test_revoked_grant_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'revoked.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"], ttl_seconds=300).json()
        revoked = first.post(revoke_url(machine_id, grant["id"])).json()

    with TestClient(app) as second:
        # The revoked state and stamp survive; re-revoke and consume both
        # lose to the committed terminal state.
        again = second.post(revoke_url(machine_id, grant["id"]))
        assert again.status_code == 409
        assert again.json() == {"error": {"code": "grant_revoked"}}
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
        assert row.revoked_at == revoked["revoked_at"]


# --------------------------------------------------------------------------- #
# Empty query and body validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("send", [b"{}", b'{"x":1}', b"null", b"[]", b"text"])
def test_revoke_with_body_is_invalid_query(active_grant, client, send):
    machine_id, _, grant = active_grant
    response = client.post(
        revoke_url(machine_id, grant["id"]),
        content=send,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?foo", "?x=1&x=2"])
def test_revoke_with_query_is_invalid_query_before_lookup(
    active_grant, client, query
):
    machine_id, _, grant = active_grant
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(revoke_url(machine_id, grant["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine/grant lookup and even the body check.
    response = client.post(
        revoke_url(missing_machine, grant["id"]) + query,
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_revoke_empty_body_with_zero_length_is_accepted(active_grant, client):
    machine_id, _, grant = active_grant
    response = client.request(
        "POST", revoke_url(machine_id, grant["id"]), content=b""
    )
    assert response.status_code == 200


def test_invalid_revoke_writes_nothing(active_grant, client):
    machine_id, _, grant = active_grant
    client.post(
        revoke_url(machine_id, grant["id"]) + "?x=1",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.status == "active"
    assert row.revoked_at is None


# --------------------------------------------------------------------------- #
# Lookup outcomes and machine isolation
# --------------------------------------------------------------------------- #


def test_revoke_missing_machine_is_not_found(client, active_grant):
    _, _, grant = active_grant
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(revoke_url(missing_machine, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_revoke_missing_grant_is_not_found(active_grant, client):
    machine_id, _, _ = active_grant
    response = client.post(revoke_url(machine_id, "no-such-grant"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_revoke_grant_owned_by_another_machine_is_not_found(
    active_grant, client
):
    machine_id, _, grant = active_grant
    other = create_machine(client, external_id="machine-2")

    response = client.post(revoke_url(other, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The grant is untouched and still revocable by its real owner.
    assert (
        client.post(revoke_url(machine_id, grant["id"])).status_code == 200
    )


# --------------------------------------------------------------------------- #
# Conflict outcomes: consumed / already revoked / expired
# --------------------------------------------------------------------------- #


def test_revoke_consumed_grant_is_grant_consumed(active_grant, client):
    machine_id, _, grant = active_grant
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200

    response = client.post(revoke_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, consumed_at, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "consumed"
    assert row.revoked_at is None
    assert use_count == 1


def test_revoking_twice_returns_grant_revoked_once(active_grant, client):
    machine_id, _, grant = active_grant
    first = client.post(revoke_url(machine_id, grant["id"]))
    second = client.post(revoke_url(machine_id, grant["id"]))
    assert first.status_code == 200
    first_stamp = first.json()["revoked_at"]
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "grant_revoked"}}

    # The first revocation's stamp is the one that persists.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.status == "revoked"
    assert row.revoked_at == first_stamp


def test_revoked_grant_consume_is_grant_revoked(active_grant, client):
    machine_id, _, grant = active_grant
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200

    response = client.post(consume_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}

    with client.app.state.engine.connect() as conn:
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert use_count == 0


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


def test_expired_grant_revoke_is_grant_expired(active_grant, client):
    machine_id, _, _ = active_grant
    # Each event can back exactly one grant, so the expired grant is minted
    # from a fresh event.
    event = record_event(client, machine_id, resource="res/expired").json()
    grant = _backdated_grant(client, machine_id, event["id"])

    response = client.post(revoke_url(machine_id, grant["id"]))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}

    # Nothing was written: the grant stays active (expiry is derived), with
    # no revocation stamp and no use record.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at, consumed_at FROM authorization_grants "
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
    assert row.consumed_at is None
    assert use_count == 0


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_revokes_have_exactly_one_winner(active_grant, client):
    machine_id, _, grant = active_grant
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
    revoked_stamps = {
        r.json()["revoked_at"] for r in responses if r.status_code == 200
    }
    assert len(revoked_stamps) == 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_revoked"}}

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
    assert row.revoked_at == revoked_stamps.pop()
    assert use_count == 0


def test_concurrent_revoke_and_consume_have_one_terminal_winner(
    active_grant, client
):
    machine_id, _, grant = active_grant
    # One contender of each kind, repeated across independent grants is
    # unnecessary: hammer one grant with a mixed burst and require exactly
    # one terminal transition to ever commit.
    count = 20
    gate = threading.Event()

    def hit(kind):
        gate.wait()
        if kind == "revoke":
            return ("revoke", client.post(revoke_url(machine_id, grant["id"])))
        return ("consume", client.post(consume_url(machine_id, grant["id"])))

    kinds = ["revoke", "consume"] * (count // 2)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit, kind) for kind in kinds]
        gate.set()
        outcomes = [f.result() for f in futures]

    successes = [
        (kind, response)
        for kind, response in outcomes
        if response.status_code == 200
    ]
    assert len(successes) == 1
    winner_kind, winner = successes[0]

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, revoked_at, consumed_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()

    if winner_kind == "consume":
        assert row.status == "consumed"
        assert row.revoked_at is None
        assert use_count == 1
        # Every revoke lost with grant_consumed; losing consumes were 409 too.
        for kind, response in outcomes:
            if response is winner:
                continue
            assert response.status_code == 409
            if kind == "revoke":
                assert response.json() == {"error": {"code": "grant_consumed"}}
    else:
        assert row.status == "revoked"
        assert row.consumed_at is None
        assert row.revoked_at == winner.json()["revoked_at"]
        assert use_count == 0
        for kind, response in outcomes:
            if response is winner:
                continue
            assert response.status_code == 409
            if kind == "consume":
                assert response.json() == {"error": {"code": "grant_revoked"}}


# --------------------------------------------------------------------------- #
# Non-interference, legacy databases, and routing
# --------------------------------------------------------------------------- #


def test_revoke_never_rewrites_event_or_basis(active_grant, client):
    machine_id, event, grant = active_grant
    basis_url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/decision-basis"
    )
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    basis_before = client.get(basis_url).content
    events_before = client.get(events_url).content

    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200

    assert client.get(basis_url).content == basis_before
    assert client.get(events_url).content == events_before
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity["valid"] is True


def test_old_database_adds_revoked_at_safely(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"], ttl_seconds=300).json()

    # Reproduce a database that predates the revocation column, keeping the
    # issued grant rows and their original values.
    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute(
        "PRAGMA table_info(authorization_grants)"
    )}
    assert "revoked_at" in columns
    conn.execute(
        "ALTER TABLE authorization_grants RENAME TO authorization_grants_old"
    )
    keep = [
        "id", "machine_id", "event_id", "issued_at", "expires_at",
        "status", "consumed_at",
    ]
    conn.execute(
        "CREATE TABLE authorization_grants ("
        "id VARCHAR(36) NOT NULL PRIMARY KEY, "
        "machine_id VARCHAR(36) NOT NULL, "
        "event_id VARCHAR(36) NOT NULL UNIQUE, "
        "issued_at VARCHAR NOT NULL, "
        "expires_at VARCHAR NOT NULL, "
        "status VARCHAR NOT NULL, "
        "consumed_at VARCHAR)"
    )
    conn.execute(
        f"INSERT INTO authorization_grants ({', '.join(keep)}) "
        f"SELECT {', '.join(keep)} FROM authorization_grants_old"
    )
    conn.execute("DROP TABLE authorization_grants_old")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # The old grant's original values are intact and it is revocable.
        revoked = second.post(revoke_url(machine_id, grant["id"]))
        assert revoked.status_code == 200
        assert revoked.json()["status"] == "revoked"
        with second.app.state.engine.connect() as db:
            row = db.execute(
                text(
                    "SELECT issued_at, expires_at, status, consumed_at, "
                    "revoked_at FROM authorization_grants WHERE id = :id"
                ).bindparams(id=grant["id"])
            ).one()
        assert row.issued_at == grant["issued_at"]
        assert row.expires_at == grant["expires_at"]
        assert row.consumed_at is None
        assert row.status == "revoked"
        assert row.revoked_at is not None

        # A freshly issued grant on the migrated database works end to end.
        fresh_event = record_event(
            second, machine_id, resource="res/fresh"
        ).json()
        fresh_grant = issue(
            second, machine_id, fresh_event["id"]
        ).json()
        assert (
            second.post(consume_url(machine_id, fresh_grant["id"])).status_code
            == 200
        )


def test_issue_consume_semantics_unchanged_after_revocation_feature(
    active_grant, client
):
    machine_id, _, revoked = active_grant
    assert client.post(revoke_url(machine_id, revoked["id"])).status_code == 200

    # A second event still issues an active grant and consumes exactly once.
    fresh_event = record_event(client, machine_id, resource="res/other").json()
    grant = issue(client, machine_id, fresh_event["id"])
    assert grant.status_code == 201
    assert grant.json()["status"] == "active"
    assert client.post(
        consume_url(machine_id, grant.json()["id"])
    ).status_code == 200
    repeat = client.post(consume_url(machine_id, grant.json()["id"]))
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "grant_consumed"}}


def test_non_post_methods_on_revoke_are_not_routed(active_grant, client):
    machine_id, _, grant = active_grant
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(revoke_url(machine_id, grant["id"]))
        assert response.status_code == 405

    # And the rejected methods left the grant readable and revocable.
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200
