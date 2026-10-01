"""Tests for the read-only authorization grant audit listing.

One read-only entry sits under the machine grant collection:

    GET /machines/{machine_id}/authorization-grants

It returns the machine's grants as a keyset-paginated audit list, ordered by
the actual UTC instant of ``issued_at`` and then by grant id, with each
grant's effective status derived read-only (``active`` / ``expired``) while
the two terminal states (``consumed`` / ``revoked``) stay sticky. These
tests cover the response shape, the four status derivations, the unique use
record fields, ordering and keyset pagination without repeats or misses, the
opaque cursor, every validation outcome and its precedence over the machine
lookup, strict machine isolation, the default page size, read-only/no-write
guarantees (including that expiry never rewrites storage and no lifecycle
event is fabricated for an old grant), verbatim field reads, and stability
across application restarts.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


ITEM_FIELDS = [
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


def list_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants"


def get_list(client, machine_id, **params):
    return client.get(list_url(machine_id), params=params or None)


def all_items(client, machine_id):
    return get_list(client, machine_id, limit=100).json()["items"]


def backdate_expiry(client, grant_id, seconds_ago=10):
    """Move an existing grant's expiry into the past without any state flip."""
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant_id)
        )


def set_issued_at(client, grant_id, stamp):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET issued_at = :at "
                "WHERE id = :id"
            ).bindparams(at=stamp, id=grant_id)
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


# --------------------------------------------------------------------------- #
# Shape and empty/missing cases
# --------------------------------------------------------------------------- #


def test_list_empty_machine(client):
    machine_id = create_machine(client)
    response = get_list(client, machine_id)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    # Compact UTF-8 JSON terminated by a single newline.
    assert response.content == b'{"items":[],"next_cursor":null}\n'


def test_list_missing_machine_is_404(client):
    response = get_list(client, "missing-machine")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_list_top_level_shape(allowed_event, client):
    machine_id, event = allowed_event
    issue(client, machine_id, event["id"])

    response = get_list(client, machine_id, limit=10)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["items", "next_cursor"]
    assert len(body["items"]) == 1
    assert list(body["items"][0].keys()) == ITEM_FIELDS


# --------------------------------------------------------------------------- #
# Status derivation and use-record fields
# --------------------------------------------------------------------------- #


def test_active_grant_item_shape(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=300).json()

    item = all_items(client, machine_id)[0]
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


def test_expired_status_is_derived_without_rewriting_storage(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=1).json()
    backdate_expiry(client, grant["id"])

    item = all_items(client, machine_id)[0]
    # The derived status is expired; the single consumption moment stays
    # absent and the stored status column was never flipped.
    assert item["status"] == "expired"
    assert item["consumed_at"] is None
    assert item["revoked_at"] is None
    assert item["use_id"] is None
    assert item["use_at"] is None
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert stored == "active"

    # The derived value is stable on a repeat read.
    assert all_items(client, machine_id)[0]["status"] == "expired"


def test_consumed_grant_keeps_terminal_status_and_use_record(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=300).json()
    use = client.post(
        consume_url(machine_id, grant["id"])
    ).json()

    item = all_items(client, machine_id)[0]
    assert item["status"] == "consumed"
    assert item["use_id"] == use["use_id"]
    assert item["use_at"] == use["consumed_at"]
    assert item["consumed_at"] == use["consumed_at"]
    assert item["revoked_at"] is None

    # A terminal consumed grant stays consumed once its TTL has elapsed.
    backdate_expiry(client, grant["id"])
    item = all_items(client, machine_id)[0]
    assert item["status"] == "consumed"
    assert item["use_id"] == use["use_id"]
    assert item["use_at"] == use["consumed_at"]


def test_revoked_grant_keeps_terminal_status(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=300).json()
    revocation = client.post(
        revoke_url(machine_id, grant["id"])
    ).json()

    item = all_items(client, machine_id)[0]
    assert item["status"] == "revoked"
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["consumed_at"] is None
    assert item["use_id"] is None
    assert item["use_at"] is None

    # A terminal revoked grant stays revoked once its TTL has elapsed.
    backdate_expiry(client, grant["id"])
    item = all_items(client, machine_id)[0]
    assert item["status"] == "revoked"
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["use_id"] is None
    assert item["use_at"] is None


def test_item_fields_match_stored_rows_verbatim(allowed_event, client):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=300).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    with client.app.state.engine.connect() as conn:
        grant_row = conn.execute(
            text(
                "SELECT id, machine_id, event_id, issued_at, expires_at, "
                "status, consumed_at, revoked_at FROM authorization_grants "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_row = conn.execute(
            text(
                "SELECT id, consumed_at FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).one()

    item = all_items(client, machine_id)[0]
    assert item["id"] == grant_row.id
    assert item["machine_id"] == grant_row.machine_id
    assert item["event_id"] == grant_row.event_id
    assert item["issued_at"] == grant_row.issued_at
    assert item["expires_at"] == grant_row.expires_at
    assert item["consumed_at"] == grant_row.consumed_at
    assert item["revoked_at"] is None
    assert grant_row.revoked_at is None
    assert item["use_id"] == use_row.id == use["use_id"]
    assert item["use_at"] == use_row.consumed_at == use["consumed_at"]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _issue_n_grants(client, machine_id, n, resource_prefix="res"):
    grants = []
    for index in range(n):
        event = record_event(
            client, machine_id, resource=f"{resource_prefix}/{index}"
        ).json()
        grants.append(issue(client, machine_id, event["id"]).json()["id"])
    return grants


def test_ordering_by_issued_at_instant_then_id(allowed_event, client):
    machine_id, _ = allowed_event
    ids = _issue_n_grants(client, machine_id, 3)
    # An exact-second stamp must sort BEFORE a fractional stamp of the same
    # second, even though lexicographically ``.`` precedes ``Z``.
    set_issued_at(client, ids[0], "2025-06-01T00:00:00.500000Z")
    set_issued_at(client, ids[1], "2025-06-01T00:00:00Z")
    set_issued_at(client, ids[2], "2025-06-01T00:00:01Z")

    ordered = [item["id"] for item in all_items(client, machine_id)]
    assert ordered == [ids[1], ids[0], ids[2]]


def test_ordering_tie_breaks_by_id(allowed_event, client):
    machine_id, _ = allowed_event
    ids = _issue_n_grants(client, machine_id, 2)
    stamp = "2025-06-02T00:00:00Z"
    set_issued_at(client, ids[0], stamp)
    set_issued_at(client, ids[1], stamp)

    ordered = [item["id"] for item in all_items(client, machine_id)]
    assert ordered == sorted(ids)


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


def test_pagination_walks_every_grant_once(allowed_event, client):
    machine_id, _ = allowed_event
    ids = _issue_n_grants(client, machine_id, 5)
    stamps = [
        "2025-07-01T00:00:00Z",
        "2025-07-01T00:00:01.250000Z",
        "2025-07-01T00:00:01Z",
        "2025-07-01T00:00:02Z",
        "2025-07-02T00:00:00Z",
    ]
    for grant_id, stamp in zip(ids, stamps):
        set_issued_at(client, grant_id, stamp)
    expected = [ids[0], ids[2], ids[1], ids[3], ids[4]]

    seen = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": 2}
        if cursor is not None:
            params["cursor"] = cursor
        body = get_list(client, machine_id, **params).json()
        assert len(body["items"]) <= 2
        seen.extend(item["id"] for item in body["items"])
        pages += 1
        if body["next_cursor"] is None:
            break
        cursor = body["next_cursor"]

    assert pages == 3
    assert seen == expected
    assert len(seen) == len(set(seen)) == 5


def test_next_cursor_points_at_next_page_first_item(allowed_event, client):
    machine_id, _ = allowed_event
    ids = _issue_n_grants(client, machine_id, 3)
    for index, grant_id in enumerate(ids):
        set_issued_at(client, grant_id, f"2025-08-01T00:00:0{index}Z")

    first = get_list(client, machine_id, limit=1).json()
    assert first["next_cursor"] == (
        f"2025-08-01T00:00:00Z|{ids[0]}"
    )
    second = get_list(
        client, machine_id, limit=1, cursor=first["next_cursor"]
    )
    second_body = second.json()
    assert [item["id"] for item in second_body["items"]] == [ids[1]]

    # Repeating the same cursor against unchanged data is byte-identical.
    repeat = get_list(
        client, machine_id, limit=1, cursor=first["next_cursor"]
    )
    assert repeat.content == second.content

    last = get_list(
        client, machine_id, limit=1, cursor=second_body["next_cursor"]
    ).json()
    assert [item["id"] for item in last["items"]] == [ids[2]]
    assert last["next_cursor"] is None


def test_next_cursor_null_when_limit_covers_all(allowed_event, client):
    machine_id, _ = allowed_event
    ids = set(_issue_n_grants(client, machine_id, 2))
    body = get_list(client, machine_id, limit=100).json()
    assert {item["id"] for item in body["items"]} == ids
    assert body["next_cursor"] is None


def test_default_limit_is_fifty(allowed_event, client):
    machine_id, _ = allowed_event
    # Bulk-insert 51 grants directly: one grant per distinct event, far-future
    # expiry so status derivation is irrelevant to the page-size check.
    event_ids = [str(uuid.uuid4()) for _ in range(51)]
    grant_ids = [str(uuid.uuid4()) for _ in range(51)]
    with client.app.state.engine.begin() as conn:
        for index, (event_id, grant_id) in enumerate(
            zip(event_ids, grant_ids)
        ):
            conn.execute(
                text(
                    "INSERT INTO authorization_decision_events "
                    "(id, machine_id, action_type, resource, allowed, "
                    "reason, created_at) VALUES (:id, :machine_id, 'read', "
                    ":resource, 1, 'allowed_by_policy', :created_at)"
                ).bindparams(
                    id=event_id,
                    machine_id=machine_id,
                    resource=f"bulk/{index}",
                    created_at=f"2025-09-01T00:00:{index:02d}Z",
                )
            )
            conn.execute(
                text(
                    "INSERT INTO authorization_grants "
                    "(id, machine_id, event_id, issued_at, expires_at, "
                    "status, consumed_at, revoked_at) VALUES (:id, "
                    ":machine_id, :event_id, :issued_at, :expires_at, "
                    "'active', NULL, NULL)"
                ).bindparams(
                    id=grant_id,
                    machine_id=machine_id,
                    event_id=event_id,
                    issued_at=f"2025-09-02T00:00:{index:02d}Z",
                    expires_at="2099-01-01T00:00:00Z",
                )
            )

    first = get_list(client, machine_id)
    first_body = first.json()
    assert len(first_body["items"]) == 50
    assert first_body["next_cursor"] is not None
    second = get_list(client, machine_id, cursor=first_body["next_cursor"])
    second_body = second.json()
    assert len(second_body["items"]) == 1
    assert second_body["next_cursor"] is None
    walked = {item["id"] for item in first_body["items"]} | {
        item["id"] for item in second_body["items"]
    }
    assert walked == set(grant_ids)


def test_limit_boundaries_accepted(allowed_event, client):
    machine_id, _ = allowed_event
    assert get_list(client, machine_id, limit=1).status_code == 200
    assert get_list(client, machine_id, limit=100).status_code == 200


# --------------------------------------------------------------------------- #
# Query validation
# --------------------------------------------------------------------------- #


def test_limit_validation(allowed_event, client):
    machine_id, _ = allowed_event
    for raw in ("0", "101", "-1", "1.5", "abc", "true", "false", "", "  "):
        response = client.get(list_url(machine_id), params={"limit": raw})
        assert response.status_code == 422, raw
        assert response.json() == {"error": {"code": "invalid_query"}}, raw
    # ``1.0`` and an explicit boolean are non-integers.
    for raw in ("1.0",):
        response = client.get(list_url(machine_id), params={"limit": raw})
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_and_repeated_params_and_body_are_invalid_query(
    allowed_event, client
):
    machine_id, _ = allowed_event

    response = get_list(client, machine_id, status="active")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(list_url(machine_id) + "?limit=1&limit=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(list_url(machine_id) + "?cursor=a&cursor=b")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        list_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    # A carried body on a GET is rejected as an invalid query shape.
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_cursor_shape_validation(allowed_event, client):
    machine_id, event = allowed_event
    issue(client, machine_id, event["id"])
    for raw in ("", "no-separator", "|only-id", "only-stamp|"):
        response = get_list(client, machine_id, limit=10, cursor=raw)
        assert response.status_code == 422, raw
        assert response.json() == {"error": {"code": "invalid_query"}}
    # A well-shaped cursor naming no stored grant is invalid_query too; an
    # extra separator in the id segment passes the split shape but still
    # names no stored position.
    for raw in (
        "2025-01-01T00:00:00Z|missing-grant",
        "2025-01-01T00:00:00Z|some-id|extra",
    ):
        response = get_list(client, machine_id, limit=10, cursor=raw)
        assert response.status_code == 422, raw
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_precedes_machine_lookup(client):
    # Every malformed query against a non-existent machine is 422, not 404.
    assert get_list(client, "missing-machine", limit="abc").status_code == 422
    assert (
        get_list(
            client, "missing-machine", cursor="2025-01-01T00:00:00Z|nope"
        ).status_code
        == 422
    )
    response = client.get(
        "/machines/missing-machine/authorization-grants?unknown=1"
    )
    assert response.status_code == 422
    response = client.request(
        "GET",
        "/machines/missing-machine/authorization-grants",
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    # A valid query against the same missing machine is the lookup 404.
    assert get_list(client, "missing-machine", limit=10).status_code == 404


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_listing_isolated_per_machine(allowed_event, client):
    machine_id, event = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    # Policy rules are global, so the first allow rule already governs the
    # second machine's matching declaration.
    declare(client, other_id, resource_pattern="res/*")
    other_event = record_event(client, other_id).json()

    grant = issue(client, machine_id, event["id"]).json()
    other_grant = issue(client, other_id, other_event["id"]).json()

    assert [item["id"] for item in all_items(client, machine_id)] == [
        grant["id"]
    ]
    assert [item["id"] for item in all_items(client, other_id)] == [
        other_grant["id"]
    ]
    assert all(
        item["machine_id"] == machine_id
        for item in all_items(client, machine_id)
    )


def test_cursor_from_other_machine_is_invalid(allowed_event, client):
    machine_id, event = allowed_event
    other_id = create_machine(client, external_id="machine-2")
    grant = issue(client, machine_id, event["id"]).json()
    item = all_items(client, machine_id)[0]
    cursor = f"{item['issued_at']}|{grant['id']}"

    response = get_list(client, other_id, limit=10, cursor=cursor)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    # The foreign cursor attempt returned no grants from either machine.
    assert all_items(client, other_id) == []


# --------------------------------------------------------------------------- #
# Read-only guarantees
# --------------------------------------------------------------------------- #


def test_listing_writes_nothing_and_fabricates_no_lifecycle_event(
    allowed_event, client
):
    machine_id, event = allowed_event
    grant = issue(client, machine_id, event["id"], ttl_seconds=1).json()
    backdate_expiry(client, grant["id"])

    def dump():
        with client.app.state.engine.connect() as conn:
            grants_table = conn.execute(
                text(
                    "SELECT id, machine_id, event_id, issued_at, expires_at, "
                    "status, consumed_at, revoked_at FROM authorization_grants "
                    "ORDER BY id"
                )
            ).all()
            uses_table = conn.execute(
                text(
                    "SELECT id, grant_id, machine_id, event_id, consumed_at "
                    "FROM authorization_grant_uses ORDER BY id"
                )
            ).all()
            lifecycle_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM authorization_grant_lifecycle_events"
                )
            ).scalar_one()
        return grants_table, uses_table, lifecycle_count

    before = dump()
    integrity_url = (
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/integrity"
    )
    integrity_before = client.get(integrity_url).content

    for _ in range(3):
        response = get_list(client, machine_id, limit=100)
        assert response.status_code == 200
        # Walk a second time with tiny pages; still no writes.
        cursor = None
        while True:
            params = {"limit": 1}
            if cursor is not None:
                params["cursor"] = cursor
            body = get_list(client, machine_id, **params).json()
            if body["next_cursor"] is None:
                break
            cursor = body["next_cursor"]

    assert dump() == before
    assert client.get(integrity_url).content == integrity_before
    # The expired grant's derived status did not fabricate a terminal event.
    assert dump()[2] == 1  # only the original ``issued`` event


def test_non_get_methods_on_collection_are_405(allowed_event, client):
    machine_id, _ = allowed_event
    for method in ("put", "patch", "delete"):
        response = getattr(client, method)(list_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Restart stability
# --------------------------------------------------------------------------- #


def test_listing_is_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        events = [
            record_event(first, machine_id, resource=f"res/{i}").json()
            for i in range(4)
        ]
        active = issue(first, machine_id, events[0]["id"], ttl_seconds=300)
        assert active.status_code == 201
        consumed_grant = issue(first, machine_id, events[1]["id"]).json()
        first.post(consume_url(machine_id, consumed_grant["id"]))
        revoked_grant = issue(first, machine_id, events[2]["id"]).json()
        first.post(revoke_url(machine_id, revoked_grant["id"]))
        expired_grant = issue(first, machine_id, events[3]["id"]).json()
        backdate_expiry(first, expired_grant["id"])

        first_page = get_list(first, machine_id, limit=2)
        first_full = get_list(first, machine_id, limit=100)
        assert first_page.status_code == first_full.status_code == 200
        first_cursor = first_page.json()["next_cursor"]

    with TestClient(app) as second:
        # The same first page, cursor continuation, and full list are
        # byte-identical after the restart, statuses included.
        assert get_list(second, machine_id, limit=2).content == first_page.content
        second_full = get_list(second, machine_id, limit=100)
        assert second_full.content == first_full.content

        continued = get_list(
            second, machine_id, limit=2, cursor=first_cursor
        ).json()
        statuses = {
            item["id"]: item["status"] for item in second_full.json()["items"]
        }
        assert statuses[consumed_grant["id"]] == "consumed"
        assert statuses[revoked_grant["id"]] == "revoked"
        assert statuses[expired_grant["id"]] == "expired"
        assert statuses[active.json()["id"]] == "active"
        # The continuation after restart still pages without repeat or miss.
        seen = get_list(second, machine_id, limit=2).json()["items"]
        seen += continued["items"]
        cursor = continued["next_cursor"]
        while cursor is not None:
            body = get_list(
                second, machine_id, limit=2, cursor=cursor
            ).json()
            seen += body["items"]
            cursor = body["next_cursor"]
        full_ids = [
            item["id"] for item in second_full.json()["items"]
        ]
        assert [item["id"] for item in seen] == full_ids
        assert len({item["id"] for item in seen}) == len(seen) == 4
