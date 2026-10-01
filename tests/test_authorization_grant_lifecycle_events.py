"""Tests for the authorization grant lifecycle audit chain.

Every successful grant action — issue, the single consumption, the one
emergency revocation — appends exactly one immutable lifecycle event to the
machine's per-machine hash chain inside the same locked write transaction;
failed actions append nothing. Two read-only entries sit under the machine
path:

    GET /machines/{machine_id}/authorization-grant-lifecycle-events/integrity
    GET /machines/{machine_id}/authorization-grant-lifecycle-events/changes

These tests cover the event fields and chaining (including that
``occurred_at`` reuses the success response's own moment), the exactly-one
terminal winner under repeated consume/revoke, the integrity audit (sound,
empty, and tampered chains), the incremental keyset-paginated query (paging,
cursors, and every validation outcome), machine isolation, restart
persistence, and old-database migration that never rewrites old data.
"""
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

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


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/integrity"
    )


def changes_url(machine_id):
    return (
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes"
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


def get_changes(client, machine_id, **params):
    return client.get(changes_url(machine_id), params=params)


def all_events(client, machine_id):
    """Every lifecycle event of the machine, via one big page."""
    response = get_changes(client, machine_id, limit=100)
    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    return body["records"]


def content_hash_of(record):
    document = json.dumps(
        {
            "id": record["id"],
            "machine_id": record["machine_id"],
            "grant_id": record["grant_id"],
            "authorization_event_id": record["authorization_event_id"],
            "type": record["type"],
            "occurred_at": record["occurred_at"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash_of(previous_chain_hash, content_hash):
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


EVENT_FIELDS = [
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


# --------------------------------------------------------------------------- #
# Successful actions append linked events
# --------------------------------------------------------------------------- #


def test_issue_appends_issued_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=90).json()

    records = all_events(client, machine_id)
    assert len(records) == 1
    record = records[0]
    assert list(record.keys()) == EVENT_FIELDS
    assert record["machine_id"] == machine_id
    assert record["grant_id"] == grant["id"]
    assert record["authorization_event_id"] == event["id"]
    assert record["type"] == "issued"
    # occurred_at reuses the success response's own moment.
    assert record["occurred_at"] == grant["issued_at"]
    assert record["previous_event_id"] is None
    assert record["content_hash"] == content_hash_of(record)
    assert record["chain_hash"] == chain_hash_of("", record["content_hash"])


def test_consume_appends_chained_consumed_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    records = all_events(client, machine_id)
    assert [record["type"] for record in records] == ["issued", "consumed"]
    consumed = records[1]
    assert consumed["grant_id"] == grant["id"]
    assert consumed["authorization_event_id"] == event["id"]
    assert consumed["occurred_at"] == use["consumed_at"]
    assert consumed["previous_event_id"] == records[0]["id"]
    assert consumed["content_hash"] == content_hash_of(consumed)
    assert consumed["chain_hash"] == chain_hash_of(
        records[0]["chain_hash"], consumed["content_hash"]
    )


def test_revoke_appends_chained_revoked_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()

    records = all_events(client, machine_id)
    assert [record["type"] for record in records] == ["issued", "revoked"]
    revoked = records[1]
    assert revoked["grant_id"] == grant["id"]
    assert revoked["authorization_event_id"] == event["id"]
    assert revoked["occurred_at"] == revocation["revoked_at"]
    assert revoked["previous_event_id"] == records[0]["id"]
    assert revoked["content_hash"] == content_hash_of(revoked)
    assert revoked["chain_hash"] == chain_hash_of(
        records[0]["chain_hash"], revoked["content_hash"]
    )


def test_full_lifecycle_chain_across_grants(allowed_event, client):
    machine_id, event = allowed_event
    # A second allowed event of the same machine gets its own grant; both
    # grants share the one per-machine chain.
    second_event = record_event(client, machine_id, resource="res/y").json()
    grant_one = issue(client, machine_id, event["id"]).json()
    grant_two = issue(client, machine_id, second_event["id"]).json()
    client.post(consume_url(machine_id, grant_one["id"]))
    client.post(revoke_url(machine_id, grant_two["id"]))

    records = all_events(client, machine_id)
    assert [record["type"] for record in records] == [
        "issued",
        "issued",
        "consumed",
        "revoked",
    ]
    previous_id = None
    previous_chain_hash = ""
    for record in records:
        assert record["previous_event_id"] == previous_id
        assert record["content_hash"] == content_hash_of(record)
        assert record["chain_hash"] == chain_hash_of(
            previous_chain_hash, record["content_hash"]
        )
        previous_id = record["id"]
        previous_chain_hash = record["chain_hash"]


# --------------------------------------------------------------------------- #
# Failed actions append nothing; one terminal winner
# --------------------------------------------------------------------------- #


def test_failed_actions_append_no_events(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    # Lookups and terminal-state conflicts write nothing.
    assert client.post(
        consume_url(machine_id, "missing-grant")
    ).status_code == 404
    assert client.post(
        revoke_url("missing-machine", grant["id"])
    ).status_code == 404
    assert issue(client, machine_id, event["id"]).status_code == 409

    assert client.post(consume_url(machine_id, grant["id"])).status_code == 200
    assert (
        client.post(consume_url(machine_id, grant["id"])).status_code == 409
    )
    assert (
        client.post(revoke_url(machine_id, grant["id"])).status_code == 409
    )

    records = all_events(client, machine_id)
    assert [record["type"] for record in records] == ["issued", "consumed"]


def test_concurrent_consume_has_one_terminal_event(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    barrier = threading.Barrier(8)

    def consume():
        barrier.wait()
        return client.post(consume_url(machine_id, grant["id"])).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: consume(), range(8)))
    assert outcomes.count(200) == 1
    assert outcomes.count(409) == 7

    records = all_events(client, machine_id)
    assert [record["type"] for record in records] == ["issued", "consumed"]


def test_concurrent_consume_and_revoke_have_one_terminal_event(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()

    barrier = threading.Barrier(2)

    def call(url):
        barrier.wait()
        return client.post(url).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                call,
                [
                    consume_url(machine_id, grant["id"]),
                    revoke_url(machine_id, grant["id"]),
                ],
            )
        )
    assert sorted(outcomes) == [200, 409]

    records = all_events(client, machine_id)
    # Exactly one terminal event exists, matching the winning action.
    assert len(records) == 2
    assert records[0]["type"] == "issued"
    assert records[1]["type"] in ("consumed", "revoked")


# --------------------------------------------------------------------------- #
# Integrity endpoint
# --------------------------------------------------------------------------- #


def test_integrity_empty_chain(allowed_event, client):
    machine_id, _ = allowed_event
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_integrity_sound_chain(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["valid", "checked_count", "broken_event_id"]
    assert body == {"valid": True, "checked_count": 2, "broken_event_id": None}


def test_integrity_reports_tampered_record(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    first, second = all_events(client, machine_id)

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events "
                "SET type = 'revoked' WHERE id = :id"
            ).bindparams(id=first["id"])
        )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    # The first record's stored content no longer matches its digest.
    assert body["broken_event_id"] == first["id"]
    assert body["broken_event_id"] != second["id"]


def test_integrity_reports_broken_link(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    _, second = all_events(client, machine_id)

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events "
                "SET previous_event_id = 'bogus' WHERE id = :id"
            ).bindparams(id=second["id"])
        )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["broken_event_id"] == second["id"]


def test_integrity_isolated_per_machine(allowed_event, client):
    machine_id, event = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    grant = issue(client, machine_id, event["id"]).json()
    first = all_events(client, machine_id)[0]

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_lifecycle_events "
                "SET content_hash = '0' * 64 WHERE id = :id"
            ).bindparams(id=first["id"])
        )

    # The other machine's empty chain is unaffected by the damage.
    assert client.get(integrity_url(other_id)).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }
    assert client.get(integrity_url(machine_id)).json()["valid"] is False
    # The damage wrote nothing new: still a single event.
    assert len(all_events(client, machine_id)) == 1
    assert grant["id"] == first["grant_id"]


def test_integrity_rejects_query_and_body(allowed_event, client):
    machine_id, _ = allowed_event
    response = client.get(integrity_url(machine_id), params={"limit": 1})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.request(
        "GET",
        integrity_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    # The query check wins over the machine lookup.
    response = client.get(integrity_url("missing-machine"), params={"x": "1"})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_integrity_missing_machine_is_404(client):
    response = client.get(integrity_url("missing-machine"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_non_get_methods_are_405(allowed_event, client):
    machine_id, _ = allowed_event
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Changes endpoint: paging and cursors
# --------------------------------------------------------------------------- #


def test_changes_empty_machine(client):
    machine_id = create_machine(client)
    response = get_changes(client, machine_id, limit=10)
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "limit": 10,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_changes_response_shape_and_order(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=7).json()
    client.post(revoke_url(machine_id, grant["id"]))

    response = get_changes(client, machine_id, limit=100)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "machine_id",
        "limit",
        "records",
        "next_cursor",
        "has_more",
    ]
    assert body["machine_id"] == machine_id
    assert body["limit"] == 100
    assert [record["type"] for record in body["records"]] == [
        "issued",
        "revoked",
    ]
    for record in body["records"]:
        assert list(record.keys()) == EVENT_FIELDS
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_changes_paginates_without_repeats(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    second_event = record_event(client, machine_id, resource="res/y").json()
    grant_two = issue(client, machine_id, second_event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    client.post(revoke_url(machine_id, grant_two["id"]))

    seen = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        body = get_changes(client, machine_id, **params).json()
        assert body["limit"] == 2
        seen.extend(record["id"] for record in body["records"])
        pages += 1
        if not body["has_more"]:
            assert body["next_cursor"] is None
            break
        assert body["next_cursor"] is not None
        cursor = body["next_cursor"]

    assert pages == 2
    assert len(seen) == 4
    assert len(set(seen)) == 4
    # The paged order equals the single-page order.
    assert seen == [record["id"] for record in all_events(client, machine_id)]


def test_changes_cursor_points_after_last_record(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    first_page = get_changes(client, machine_id, limit=1).json()
    assert first_page["has_more"] is True
    first = first_page["records"][0]
    assert first_page["next_cursor"] == (
        f"{first['occurred_at']}|{first['id']}"
    )

    second_response = get_changes(
        client, machine_id, limit=1, cursor=first_page["next_cursor"]
    )
    second_page = second_response.json()
    assert [record["id"] for record in second_page["records"]] != [first["id"]]
    assert second_page["has_more"] is False
    assert second_page["next_cursor"] is None

    # Repeating the same cursor against unchanged data is byte-identical.
    repeat = get_changes(
        client, machine_id, limit=1, cursor=first_page["next_cursor"]
    )
    assert repeat.content == second_response.content


def test_changes_isolated_per_machine(allowed_event, client):
    machine_id, event = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    issue(client, machine_id, event["id"])

    assert len(all_events(client, machine_id)) == 1
    assert all_events(client, other_id) == []


def test_changes_cursor_from_other_machine_is_invalid(allowed_event, client):
    machine_id, event = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    declare(client, other_id)
    grant = issue(client, machine_id, event["id"]).json()
    assert grant["machine_id"] == machine_id

    foreign = all_events(client, machine_id)[0]
    cursor = f"{foreign['occurred_at']}|{foreign['id']}"
    response = get_changes(client, other_id, limit=10, cursor=cursor)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_changes_missing_machine_is_404(client):
    response = get_changes(client, "missing-machine", limit=10)
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_changes_limit_validation(allowed_event, client):
    machine_id, _ = allowed_event
    for raw in ("0", "101", "-1", "1.5", "abc", "true", ""):
        response = client.get(
            changes_url(machine_id), params={"limit": raw}
        )
        assert response.status_code == 422, raw
        assert response.json() == {"error": {"code": "bad_limit"}}
    # A missing limit is bad_limit too.
    response = client.get(changes_url(machine_id))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}
    # Boundary values are accepted.
    assert get_changes(client, machine_id, limit=1).status_code == 200
    assert get_changes(client, machine_id, limit=100).status_code == 200


def test_changes_cursor_shape_validation(allowed_event, client):
    machine_id, _ = allowed_event
    for raw in ("", "no-separator", "|only-id", "only-stamp|"):
        response = client.get(
            changes_url(machine_id), params={"limit": 10, "cursor": raw}
        )
        assert response.status_code == 422, raw
        assert response.json() == {"error": {"code": "invalid_cursor"}}
    # A well-shaped cursor naming no stored record is invalid_cursor.
    response = client.get(
        changes_url(machine_id),
        params={"limit": 10, "cursor": "2026-01-01T00:00:00Z|missing"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_changes_rejects_unknown_and_repeated_params_and_body(
    allowed_event, client
):
    machine_id, _ = allowed_event
    response = client.get(
        changes_url(machine_id), params={"limit": 10, "type": "issued"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        changes_url(machine_id) + "?limit=1&limit=2"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        changes_url(machine_id),
        params={"limit": 10},
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Parameter errors win over the machine lookup.
    response = client.get(changes_url("missing-machine"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}


def test_changes_non_get_methods_are_405(allowed_event, client):
    machine_id, _ = allowed_event
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(changes_url(machine_id))
        assert response.status_code == 405


def test_changes_never_leaks_sensitive_fields(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    body = get_changes(client, machine_id, limit=100).json()
    for record in body["records"]:
        assert set(record.keys()) == set(EVENT_FIELDS)
    raw = json.dumps(body)
    assert "public_key" not in raw
    assert "key-1" not in raw


# --------------------------------------------------------------------------- #
# Persistence and migration
# --------------------------------------------------------------------------- #


def test_lifecycle_events_survive_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"]).json()
        first.post(consume_url(machine_id, grant["id"]))
        before = all_events(first, machine_id)

    with TestClient(app) as second:
        assert all_events(second, machine_id) == before
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_event_id": None,
        }


def test_old_database_gets_table_without_touching_old_data(
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

    # Reproduce a database that predates the lifecycle feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_grant_lifecycle_events")
    grant_row = conn.execute(
        "SELECT id, machine_id, event_id, issued_at, expires_at, status, "
        "consumed_at, revoked_at FROM authorization_grants"
    ).fetchall()
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # The table is recreated empty: no events are fabricated for the
        # historical grant, and the old grant row is untouched.
        assert all_events(second, machine_id) == []
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 0,
            "broken_event_id": None,
        }
        with second.app.state.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, machine_id, event_id, issued_at, expires_at, "
                    "status, consumed_at, revoked_at FROM authorization_grants"
                )
            ).all()
        assert [tuple(row) for row in rows] == grant_row

        # New actions on the old database chain normally.
        second_event = record_event(second, machine_id, resource="res/y")
        new_grant = issue(second, machine_id, second_event.json()["id"])
        assert new_grant.status_code == 201
        records = all_events(second, machine_id)
        assert [record["type"] for record in records] == ["issued"]
        assert second.get(integrity_url(machine_id)).json()["valid"] is True
