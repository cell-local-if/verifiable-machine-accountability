"""Tests for the authorization-grant lifecycle audit chain.

Two read-only entries sit under the machine path:

    GET /machines/{machine_id}/authorization-grant-lifecycle-events/integrity
    GET /machines/{machine_id}/authorization-grant-lifecycle-events/changes

The existing grant write entries (issue / consume / revoke) are unchanged on
the wire, but every successful action now appends one lifecycle audit event
(``issued`` / ``consumed`` / ``revoked``) in the same locked transaction,
timestamped with the success response moment and linked into a per-machine
tamper-evident hash chain. These tests cover:

* one audit event per successful action, the nine stored fields, the reused
  response timestamps, and the chain links/hashes recomputing exactly;
* no event written by a failed, duplicate, expired, or concurrency-losing
  action, and exactly one terminal (consumed or revoked) event per grant;
* the integrity endpoint: empty and sound chains valid, the first broken
  event reported for any tampered field/link/hash, strict query/body
  validation, 404 for a missing machine, 405 for non-GET methods, 500 on a
  read failure, strict read-only behavior;
* the changes endpoint: bad_limit / invalid_cursor / invalid_query
  validation before lookup, the fixed envelope, nine-field records,
  (occurred_at, id) ordering with exact-before-fractional seconds,
  exclusive non-repeating cursors, cross-machine cursor rejection, machine
  isolation, 404/405/500 contracts, compact newline JSON, and restart
  persistence;
* startup migration of an old database: the table is created and old grant
  data is never rewritten.
"""
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability import grant_lifecycle_chain


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


def integrity_path(machine_id):
    return (
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/"
        "integrity"
    )


def changes_path(machine_id):
    return (
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + cursor
    return f"{changes_path(machine_id)}?{query}"


def fetch_events(client, machine_id):
    with client.app.state.engine.connect() as conn:
        return [
            dict(row._mapping)
            for row in conn.execute(
                text(
                    "SELECT * FROM authorization_grant_lifecycle_events "
                    "ORDER BY occurred_at, id"
                )
            )
        ]


@pytest.fixture
def allowed_event(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    assert event["reason"] == "allowed_by_policy"
    return machine_id, event


NINE_KEYS = [
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

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


# --------------------------------------------------------------------------- #
# Lifecycle events written by successful actions
# --------------------------------------------------------------------------- #


def test_issue_appends_one_issued_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    rows = fetch_events(client, machine_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "issued"
    assert row["machine_id"] == machine_id
    assert row["grant_id"] == grant["id"]
    assert row["authorization_event_id"] == event["id"]
    # occurred_at reuses the success response's own moment.
    assert row["occurred_at"] == grant["issued_at"]
    assert row["previous_event_id"] is None
    assert isinstance(row["id"], str) and row["id"]


def test_consume_appends_consumed_event_with_response_moment(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == ["issued", "consumed"]
    consumed = rows[1]
    assert consumed["grant_id"] == grant["id"]
    assert consumed["authorization_event_id"] == event["id"]
    assert consumed["occurred_at"] == use["consumed_at"]
    assert consumed["previous_event_id"] == rows[0]["id"]


def test_revoke_appends_revoked_event_with_response_moment(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == ["issued", "revoked"]
    revoked = rows[1]
    assert revoked["grant_id"] == grant["id"]
    assert revoked["authorization_event_id"] == event["id"]
    assert revoked["occurred_at"] == revocation["revoked_at"]
    assert revoked["previous_event_id"] == rows[0]["id"]


def test_lifecycle_chain_hashes_recompute_exactly(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(revoke_url(machine_id, grant["id"]))

    rows = fetch_events(client, machine_id)
    previous_chain = ""
    previous_id = None
    for row in rows:
        content = grant_lifecycle_chain.compute_content_hash(
            id=row["id"],
            machine_id=row["machine_id"],
            grant_id=row["grant_id"],
            authorization_event_id=row["authorization_event_id"],
            type=row["type"],
            occurred_at=row["occurred_at"],
        )
        assert row["content_hash"] == content
        assert row["previous_event_id"] == previous_id
        chained = grant_lifecycle_chain.compute_chain_hash(previous_chain, content)
        assert row["chain_hash"] == chained
        previous_chain = chained
        previous_id = row["id"]


def test_events_for_distinct_grants_share_one_machine_chain(allowed_event, client):
    machine_id, first_event = allowed_event
    second_event = record_event(client, machine_id, resource="res/y").json()
    g1 = issue(client, machine_id, first_event["id"]).json()
    g2 = issue(client, machine_id, second_event["id"]).json()
    client.post(consume_url(machine_id, g1["id"]))
    client.post(revoke_url(machine_id, g2["id"]))

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == [
        "issued",
        "issued",
        "consumed",
        "revoked",
    ]
    # A single unbroken chain across all of the machine's grants.
    for index, row in enumerate(rows):
        assert row["previous_event_id"] == (
            rows[index - 1]["id"] if index else None
        )


# --------------------------------------------------------------------------- #
# Failed and duplicate actions write no event
# --------------------------------------------------------------------------- #


def test_duplicate_consume_writes_no_second_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 409
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 409

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == ["issued", "consumed"]


def test_duplicate_revoke_writes_no_second_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 409
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 409

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == ["issued", "revoked"]


def test_failed_issue_writes_no_event(client):
    # A denied event can never be signed; no lifecycle event may appear.
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=1)
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()
    assert denied["allowed"] is False

    assert issue(client, machine_id, denied["id"]).status_code == 409
    assert fetch_events(client, machine_id) == []


def test_expired_action_writes_no_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=1).json()
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at WHERE id = :id"
            ).bindparams(at="2000-01-01T00:00:00Z", id=grant["id"])
        )
    assert client.post(consume_url(machine_id, grant["id"])).status_code == 409
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 409

    rows = fetch_events(client, machine_id)
    assert [r["type"] for r in rows] == ["issued"]


def test_concurrent_terminal_actions_leave_one_terminal_event(
    allowed_event, client
):
    machine_id, event = allowed_event
    gate = threading.Event()

    def make_grant():
        return issue(client, machine_id, event["id"]).json()

    # Two grants: one contended by consume bursts, one by revoke/consume mix.
    grant_a = make_grant()
    event_b = record_event(client, machine_id, resource="res/z").json()
    grant_b = issue(client, machine_id, event_b["id"]).json()

    def hit(grant, path):
        gate.wait()
        return client.post(path(machine_id, grant["id"]))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [
            pool.submit(hit, grant_a, consume_url) for _ in range(12)
        ] + [
            pool.submit(hit, grant_b, revoke_url) for _ in range(6)
        ] + [
            pool.submit(hit, grant_b, consume_url) for _ in range(6)
        ]
        gate.set()
        responses = [f.result() for f in futures]

    terminal_ok = [r for r in responses if r.status_code == 200]
    assert len(terminal_ok) == 2

    rows = fetch_events(client, machine_id)
    by_grant = {}
    for row in rows:
        by_grant.setdefault(row["grant_id"], []).append(row["type"])
    assert by_grant[grant_a["id"]] == ["issued", "consumed"]
    # grant_b has exactly one terminal event, consumed or revoked, and the
    # grant's persisted terminal status agrees with the recorded event.
    b_events = by_grant[grant_b["id"]]
    assert len(b_events) == 2 and b_events[0] == "issued"
    assert b_events[1] in ("consumed", "revoked")
    with client.app.state.engine.connect() as conn:
        b_status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id"),
            {"id": grant_b["id"]},
        ).scalar_one()
    assert b_status == b_events[1]


# --------------------------------------------------------------------------- #
# Integrity endpoint
# --------------------------------------------------------------------------- #


def test_integrity_empty_chain_is_valid(client):
    machine_id = create_machine(client)
    response = client.get(integrity_path(machine_id))
    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "valid",
        "checked_count",
        "broken_event_id",
    ]
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_integrity_sound_chain_after_full_lifecycle(allowed_event, client):
    machine_id, event = allowed_event
    g1 = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, g1["id"]))
    event2 = record_event(client, machine_id, resource="res/2").json()
    g2 = issue(client, machine_id, event2["id"]).json()
    client.post(revoke_url(machine_id, g2["id"]))

    body = client.get(integrity_path(machine_id)).json()
    assert body == {
        "valid": True,
        "checked_count": 4,
        "broken_event_id": None,
    }


@pytest.mark.parametrize(
    "column",
    [
        "type",
        "grant_id",
        "authorization_event_id",
        "previous_event_id",
        "content_hash",
        "chain_hash",
    ],
)
def test_integrity_reports_first_broken_event(allowed_event, client, column):
    machine_id, event = allowed_event
    g1 = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, g1["id"]))
    event2 = record_event(client, machine_id, resource="res/2").json()
    g2 = issue(client, machine_id, event2["id"]).json()
    client.post(revoke_url(machine_id, g2["id"]))

    rows = fetch_events(client, machine_id)
    target = rows[2]  # the third event is the first broken one
    with client.app.state.engine.begin() as conn:
        if column in ("content_hash", "chain_hash"):
            conn.execute(
                text(
                    f"UPDATE authorization_grant_lifecycle_events "
                    f"SET {column} = :v WHERE id = :id"
                ).bindparams(v="f" * 64, id=target["id"])
            )
        elif column == "previous_event_id":
            conn.execute(
                text(
                    "UPDATE authorization_grant_lifecycle_events "
                    "SET previous_event_id = :v WHERE id = :id"
                ).bindparams(v=rows[0]["id"], id=target["id"])
            )
        else:
            conn.execute(
                text(
                    f"UPDATE authorization_grant_lifecycle_events "
                    f"SET {column} = :v WHERE id = :id"
                ).bindparams(v="tampered-value", id=target["id"])
            )

    body = client.get(integrity_path(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 4
    assert body["broken_event_id"] == target["id"]


def test_integrity_detects_occurred_at_tamper_in_place(allowed_event, client):
    # Re-render the same instant with the offset spelling so the event keeps
    # its chain position; the changed stored text must break its content hash.
    machine_id, event = allowed_event
    g1 = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, g1["id"]))
    event2 = record_event(client, machine_id, resource="res/2").json()
    g2 = issue(client, machine_id, event2["id"]).json()
    client.post(revoke_url(machine_id, g2["id"]))

    rows = fetch_events(client, machine_id)
    target = rows[2]
    rewritten = (
        datetime.fromisoformat(target["occurred_at"].replace("Z", "+00:00"))
        .isoformat()
    )
    assert rewritten != target["occurred_at"]
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events "
                "SET occurred_at = :v WHERE id = :id"
            ).bindparams(v=rewritten, id=target["id"])
        )

    body = client.get(integrity_path(machine_id)).json()
    assert body["valid"] is False
    assert body["broken_event_id"] == target["id"]


def test_integrity_detects_machine_attribution_tamper(allowed_event, client):
    # Moving an event to another machine removes it from the path filter; the
    # successor's predecessor link can then no longer resolve, so the chain is
    # reported broken at the successor and the moved event stays in the other
    # machine's total.
    machine_id, event = allowed_event
    g1 = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, g1["id"]))
    event2 = record_event(client, machine_id, resource="res/2").json()
    g2 = issue(client, machine_id, event2["id"]).json()
    client.post(revoke_url(machine_id, g2["id"]))
    other = create_machine(client, external_id="machine-2")

    rows = fetch_events(client, machine_id)
    moved = rows[2]
    successor = rows[3]
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events "
                "SET machine_id = :v WHERE id = :id"
            ).bindparams(v=other, id=moved["id"])
        )

    body = client.get(integrity_path(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 3
    assert body["broken_event_id"] == successor["id"]


def test_integrity_is_machine_isolated(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    other = create_machine(client, external_id="machine-2")

    # Tampering the machine attribution breaks the owning machine's chain
    # only from that event; the other machine's empty chain stays valid.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events SET type='revoked' "
                "WHERE type='consumed'"
            )
        )
    assert client.get(integrity_path(machine_id)).json()["valid"] is False
    other_body = client.get(integrity_path(other)).json()
    assert other_body == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


@pytest.mark.parametrize("query", ["?x=1", "?=", "?limit=1", "?foo"])
def test_integrity_rejects_query_parameters(client, query):
    machine_id = create_machine(client)
    response = client.get(integrity_path(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    # Validation precedes the machine lookup.
    assert client.get(integrity_path(MISSING_MACHINE) + query).status_code == 422


def test_integrity_rejects_carried_body(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        integrity_path(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_integrity_missing_machine_is_404(client):
    response = client.get(integrity_path(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_integrity_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(integrity_path(machine_id))
    assert response.status_code == 405


def test_integrity_read_failure_is_500(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_lifecycle_events"))
    response = client.get(integrity_path(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_integrity_is_read_only(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(revoke_url(machine_id, grant["id"]))
    with client.app.state.engine.connect() as conn:
        before = list(
            conn.execute(
                text("SELECT * FROM authorization_grant_lifecycle_events")
            )
        )
    for _ in range(3):
        client.get(integrity_path(machine_id))
    with client.app.state.engine.connect() as conn:
        after = list(
            conn.execute(
                text("SELECT * FROM authorization_grant_lifecycle_events")
            )
        )
    assert before == after


# --------------------------------------------------------------------------- #
# Changes endpoint: validation and routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?limit=",
        "?limit=0",
        "?limit=101",
        "?limit=-1",
        "?limit=1.0",
        "?limit=1.5",
        "?limit=true",
        "?limit=false",
        "?limit=abc",
        "?limit= 1",
        "?limit=1 ",
        "?limit=0x1",
    ],
)
def test_changes_bad_limit_is_422(client, query):
    machine_id = create_machine(client)
    response = client.get(changes_path(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}


def test_changes_boundary_limits_accepted(client):
    machine_id = create_machine(client)
    for value in (1, 100):
        response = client.get(changes_url(machine_id, limit=value))
        assert response.status_code == 200
        assert response.json()["limit"] == value


@pytest.mark.parametrize(
    "cursor",
    ["", "not-a-cursor", "|", f"2026-01-01T00:00:00Z|", "|event-id"],
)
def test_changes_malformed_cursor_is_422(client, cursor):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_changes_unlocatable_cursor_is_422(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    rows = fetch_events(client, machine_id)
    real_stamp = rows[0]["occurred_at"]

    for cursor in (
        f"{real_stamp}|00000000-0000-0000-0000-000000000000",
        "garbage-stamp|00000000-0000-0000-0000-000000000000",
        f"{real_stamp}||x",
    ):
        response = client.get(changes_url(machine_id, limit=1, cursor=cursor))
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_changes_cursor_from_another_machine_is_invalid_cursor(
    allowed_event, client
):
    machine_id, event = allowed_event
    issue(client, machine_id, event["id"])
    other = create_machine(client, external_id="machine-2")

    # A cursor naming the first machine's stored (occurred_at, id) position
    # cannot be located in another machine's event set.
    body = client.get(changes_url(machine_id, limit=100)).json()
    assert body["records"]
    foreign_cursor = (
        f"{body['records'][0]['occurred_at']}|{body['records'][0]['id']}"
    )
    response = client.get(
        changes_url(other, limit=10, cursor=foreign_cursor)
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_changes_unknown_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_changes_repeated_parameter_names_are_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        changes_path(machine_id), params=[("limit", "1"), ("limit", "2")]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.get(
        changes_path(machine_id),
        params=[
            ("limit", "1"),
            ("cursor", "a|b"),
            ("cursor", "c|d"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_changes_carried_body_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        f"{changes_path(machine_id)}?limit=10",
        content=b'{"x": 1}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_changes_parameter_errors_take_priority_over_machine_lookup(client):
    assert client.get(changes_path(MISSING_MACHINE)).status_code == 422
    assert client.get(
        f"{changes_path(MISSING_MACHINE)}?limit=0"
    ).json() == {"error": {"code": "bad_limit"}}
    assert client.get(
        f"{changes_path(MISSING_MACHINE)}?limit=1&cursor=garbage"
    ).json() == {"error": {"code": "invalid_cursor"}}
    assert client.get(
        f"{changes_path(MISSING_MACHINE)}?limit=1&x=1"
    ).json() == {"error": {"code": "invalid_query"}}


def test_changes_missing_machine_is_404(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_changes_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
    assert response.status_code == 405


def test_changes_read_failure_is_500(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_lifecycle_events"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Changes endpoint: envelope, ordering, paging, isolation
# --------------------------------------------------------------------------- #


def _insert_event(
    client,
    machine_id,
    n,
    *,
    grant_id="grant-0",
    authorization_event_id="decision-0",
    type="issued",
    occurred_at=None,
):
    """Insert a chain-completed lifecycle row directly at a fixed position."""
    event_id = f"00000000-0000-0000-0000-{n:012d}"
    rows = fetch_events(client, machine_id)
    previous_event_id = rows[-1]["id"] if rows else None
    previous_chain = rows[-1]["chain_hash"] if rows else ""
    content_hash = grant_lifecycle_chain.compute_content_hash(
        id=event_id,
        machine_id=machine_id,
        grant_id=grant_id,
        authorization_event_id=authorization_event_id,
        type=type,
        occurred_at=occurred_at,
    )
    chain_hash = grant_lifecycle_chain.compute_chain_hash(
        previous_chain, content_hash
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_grant_lifecycle_events "
                "(id, machine_id, grant_id, authorization_event_id, type, "
                "occurred_at, previous_event_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :grant_id, :authorization_event_id, "
                ":type, :occurred_at, :previous_event_id, :content_hash, "
                ":chain_hash)"
            ),
            {
                "id": event_id,
                "machine_id": machine_id,
                "grant_id": grant_id,
                "authorization_event_id": authorization_event_id,
                "type": type,
                "occurred_at": occurred_at,
                "previous_event_id": previous_event_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        )
    return event_id


T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T0_FRAC = "2026-03-01T00:00:00.500000Z"


def _eid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def test_changes_empty_page_envelope(client):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=25))
    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "machine_id",
        "limit",
        "records",
        "next_cursor",
        "has_more",
    ]
    assert response.json() == {
        "machine_id": machine_id,
        "limit": 25,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_changes_records_have_exactly_nine_fields(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    record = client.get(changes_url(machine_id, limit=10)).json()["records"][0]
    assert list(record.keys()) == NINE_KEYS
    assert record["machine_id"] == machine_id
    assert record["grant_id"] == grant["id"]
    assert record["authorization_event_id"] == event["id"]
    assert record["type"] == "issued"
    assert record["occurred_at"] == grant["issued_at"]
    assert record["previous_event_id"] is None
    assert len(record["content_hash"]) == 64
    assert len(record["chain_hash"]) == 64


def test_changes_orders_by_instant_then_id(client):
    machine_id = create_machine(client)
    # Insert in scrambled instant order; ids are fixed ascending.
    _insert_event(client, machine_id, 1, occurred_at=T2)
    _insert_event(client, machine_id, 2, occurred_at=T0)
    _insert_event(client, machine_id, 3, occurred_at=T1)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [_eid(2), _eid(3), _eid(1)]


def test_changes_exact_second_before_fractional(client):
    machine_id = create_machine(client)
    _insert_event(client, machine_id, 2, occurred_at=T0_FRAC)
    _insert_event(client, machine_id, 1, occurred_at=T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [_eid(1), _eid(2)]


def test_changes_same_instant_tie_breaks_by_id(client):
    machine_id = create_machine(client)
    # Direct rows with independent predecessor links are built in insertion
    # order; same-instant ids sort ascending regardless of insert sequence.
    _insert_event_same_instant(client, machine_id, "b", T1)
    _insert_event_same_instant(client, machine_id, "a", T1)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == ["a", "b"]


def _insert_event_same_instant(client, machine_id, suffix, stamp):
    rows = fetch_events(client, machine_id)
    previous_event_id = rows[-1]["id"] if rows else None
    previous_chain = rows[-1]["chain_hash"] if rows else ""
    content_hash = grant_lifecycle_chain.compute_content_hash(
        id=suffix,
        machine_id=machine_id,
        grant_id="g",
        authorization_event_id="e",
        type="issued",
        occurred_at=stamp,
    )
    chain_hash = grant_lifecycle_chain.compute_chain_hash(
        previous_chain, content_hash
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_grant_lifecycle_events "
                "(id, machine_id, grant_id, authorization_event_id, type, "
                "occurred_at, previous_event_id, content_hash, chain_hash) "
                "VALUES (:id, :m, :g, :e, :t, :o, :p, :c, :h)"
            ),
            {
                "id": suffix,
                "m": machine_id,
                "g": "g",
                "e": "e",
                "t": "issued",
                "o": stamp,
                "p": previous_event_id,
                "c": content_hash,
                "h": chain_hash,
            },
        )


def test_changes_exclusive_cursor_pagination_without_repeats(client):
    machine_id = create_machine(client)
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        _insert_event(client, machine_id, n, occurred_at=stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["id"] for r in first["records"]] == [_eid(1), _eid(2)]
    assert first["has_more"] is True
    assert first["next_cursor"] == f"{T1}|{_eid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [_eid(3), _eid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # Repeating the cursor returns the byte-identical page.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    )
    assert repeated.content == client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).content


def test_changes_exact_page_size_has_no_more(client):
    machine_id = create_machine(client)
    for n, stamp in ((1, T0), (2, T1)):
        _insert_event(client, machine_id, n, occurred_at=stamp)
    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_changes_is_machine_isolated(client):
    machine_id = create_machine(client)
    other = create_machine(client, external_id="machine-2")
    _insert_event(client, machine_id, 1, occurred_at=T1)
    _insert_event(client, other, 2, occurred_at=T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [_eid(1)]
    other_body = client.get(changes_url(other, limit=10)).json()
    assert [r["id"] for r in other_body["records"]] == [_eid(2)]


def test_changes_unparseable_occurred_at_sorts_last_and_stays_pageable(client):
    from urllib.parse import quote

    machine_id = create_machine(client)
    _insert_event(client, machine_id, 1, occurred_at=T0)
    _insert_event(client, machine_id, 2, occurred_at="broken|stamp")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [_eid(1)]
    second = client.get(
        f"{changes_path(machine_id)}?limit=1&cursor="
        + quote(first["next_cursor"], safe="")
    ).json()
    assert [r["id"] for r in second["records"]] == [_eid(2)]
    assert second["records"][0]["occurred_at"] == "broken|stamp"
    assert second["next_cursor"] is None
    assert second["has_more"] is False


def test_changes_body_is_compact_newline_terminated_json(client):
    machine_id = create_machine(client)
    _insert_event(client, machine_id, 1, occurred_at=T0)
    raw = client.get(changes_url(machine_id, limit=10)).content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw
    json.loads(raw)


def test_changes_does_not_leak_sensitive_fields(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(revoke_url(machine_id, grant["id"]))

    raw = client.get(changes_url(machine_id, limit=10)).content
    text_body = raw.decode("utf-8")
    # No public keys, policy text, action/resource strings, or decision basis.
    assert "key-1" not in text_body
    assert "res/x" not in text_body
    assert "read" != json.loads(text_body)["records"][0].get("action_type", None)
    assert "public_key" not in text_body
    assert "document" not in text_body
    assert "expires_at" not in text_body


def test_changes_persists_across_restart_with_real_actions(
    tmp_path, monkeypatch
):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        ev1 = record_event(first, machine_id, resource="res/1").json()
        ev2 = record_event(first, machine_id, resource="res/2").json()
        g1 = issue(first, machine_id, ev1["id"]).json()
        first.post(consume_url(machine_id, g1["id"]))
        g2 = issue(first, machine_id, ev2["id"]).json()
        first.post(revoke_url(machine_id, g2["id"]))
        page = first.get(changes_url(machine_id, limit=2)).json()
        cursor = page["next_cursor"]
        expected = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))
        assert response.status_code == 200
        assert response.content == expected
        assert [r["type"] for r in response.json()["records"]] == [
            "issued",
            "revoked",
        ]
        integrity = second.get(integrity_path(machine_id)).json()
        assert integrity["valid"] is True
        assert integrity["checked_count"] == 4


# --------------------------------------------------------------------------- #
# Startup migration of an old database
# --------------------------------------------------------------------------- #


def test_old_database_creates_lifecycle_table_and_keeps_old_data(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"]).json()
        use = first.post(consume_url(machine_id, grant["id"])).json()

    # Reproduce a database that predates the lifecycle feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_grant_lifecycle_events")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # Old grant and use rows survive byte-for-byte.
        with second.app.state.engine.connect() as db_conn:
            grant_row = db_conn.execute(
                text(
                    "SELECT status, consumed_at, issued_at, expires_at "
                    "FROM authorization_grants WHERE id = :id"
                ).bindparams(id=grant["id"])
            ).one()
            use_row = db_conn.execute(
                text(
                    "SELECT id, consumed_at FROM authorization_grant_uses "
                    "WHERE grant_id = :id"
                ).bindparams(id=grant["id"])
            ).one()
        assert grant_row.status == "consumed"
        assert grant_row.consumed_at == use["consumed_at"]
        assert use_row.id == use["use_id"]

        # The empty chain is valid; the old data was not backfilled with
        # synthetic events.
        integrity = second.get(integrity_path(machine_id)).json()
        assert integrity == {
            "valid": True,
            "checked_count": 0,
            "broken_event_id": None,
        }

        # A new action on another event starts a fresh chain soundly.
        event2 = record_event(second, machine_id, resource="res/2").json()
        g2 = issue(second, machine_id, event2["id"]).json()
        second.post(revoke_url(machine_id, g2["id"]))
        body = second.get(integrity_path(machine_id)).json()
        assert body["valid"] is True
        assert body["checked_count"] == 2


def test_integrity_and_changes_do_not_modify_other_chains(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    decision_integrity_url = (
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )
    before = client.get(decision_integrity_url).content
    client.get(integrity_path(machine_id))
    client.get(changes_url(machine_id, limit=1))
    assert client.get(decision_integrity_url).content == before
