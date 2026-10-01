"""Tests for the read-only grant audit listing.

The audit entry is::

    GET /machines/{machine_id}/authorization-grants

It is strictly read-only: it never creates, repairs, updates, or deletes a
grant, its consumption record, a lifecycle event, or a chain hash. These tests
cover its fixed ten-field item shape, the derived ``active``/``expired``
status (with consumed/revoked terminals kept), use-record mapping, ordering by
the actual UTC instant of ``issued_at`` then id, keyset pagination that never
repeats or omits an item, strict machine isolation (including cursors), every
422 ``invalid_query`` validation case and its priority over the machine
lookup, 404 for a missing machine, method routing, restart persistence,
byte-identical repeats, non-interference with issue/consume/revoke and the
lifecycle chain, and coexistence of the GET listing with the POST issue entry
on the same path.
"""
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


def list_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants"


def issue(client, machine_id, event_id, ttl_seconds=300):
    return client.post(
        list_url(machine_id),
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


def allowed_machine(client, external_id="machine-1"):
    """A machine with an enabled declaration and an allow rule."""
    machine_id = create_machine(client, external_id=external_id)
    declare(client, machine_id)
    create_rule(client)
    return machine_id


def issue_grants(client, machine_id, count, ttl_seconds=300):
    grants = []
    for index in range(count):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        response = issue(client, machine_id, event["id"], ttl_seconds)
        assert response.status_code == 201
        grants.append(response.json())
    return grants


def set_issued_at(client, grant_id, issued_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET issued_at = :at WHERE id = :id"
            ).bindparams(at=issued_at, id=grant_id)
        )


def backdate_expires_at(client, grant_id, seconds=10):
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant_id)
        )
    return past


def fetch_all_pages(client, machine_id, limit=1):
    """Walk the listing to its end, returning the collected items."""
    collected = []
    cursor = None
    seen_cursors = set()
    while True:
        query = f"?limit={limit}"
        if cursor is not None:
            query += f"&cursor={cursor}"
        response = client.get(list_url(machine_id) + query)
        assert response.status_code == 200
        body = response.json()
        collected.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert cursor not in seen_cursors
        seen_cursors.add(cursor)
    return collected


MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"

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


# --------------------------------------------------------------------------- #
# Empty listing and basic shape
# --------------------------------------------------------------------------- #


def test_machine_without_grants_lists_empty_page(client):
    machine_id = allowed_machine(client)
    response = client.get(list_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}
    assert response.content.endswith(b"\n")


def test_missing_machine_is_404(client):
    response = client.get(list_url(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_active_grant_item_shape(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()

    response = client.get(list_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["items", "next_cursor"]
    assert body["next_cursor"] is None
    item = body["items"][0]
    assert list(item.keys()) == ITEM_FIELDS
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


# --------------------------------------------------------------------------- #
# Terminal states and the unique use record
# --------------------------------------------------------------------------- #


def test_consumed_grant_carries_terminal_status_and_use_record(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["status"] == "consumed"
    assert item["use_id"] == use["use_id"]
    assert item["use_at"] == use["consumed_at"]
    assert item["consumed_at"] == use["consumed_at"]
    assert item["revoked_at"] is None
    # The use record really is the unique stored consumption row.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, consumed_at FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.id == use["use_id"]
    assert row.consumed_at == use["consumed_at"]


def test_revoked_grant_carries_terminal_status_and_stamp(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()

    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["status"] == "revoked"
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["consumed_at"] is None
    assert item["use_id"] is None
    assert item["use_at"] is None


def test_expired_unterminal_grant_presents_expired_without_storage_change(
    client,
):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    backdate_expires_at(client, grant["id"])

    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["status"] == "expired"
    assert item["use_id"] is None
    assert item["use_at"] is None
    # Derived only: storage still says active and nothing was written.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, consumed_at, revoked_at FROM "
                "authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
        use_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_uses "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert row.status == "active"
    assert row.consumed_at is None
    assert row.revoked_at is None
    assert use_count == 0


def test_consumed_grant_keeps_consumed_after_expiry(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()
    backdate_expires_at(client, grant["id"])

    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["status"] == "consumed"
    assert item["use_id"] == use["use_id"]
    assert item["use_at"] == use["consumed_at"]


def test_revoked_grant_keeps_revoked_after_expiry(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()
    backdate_expires_at(client, grant["id"])

    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["status"] == "revoked"
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["use_id"] is None
    assert item["use_at"] is None


def test_stored_timestamps_are_emitted_verbatim(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT issued_at, expires_at, consumed_at, revoked_at "
                "FROM authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    item = client.get(list_url(machine_id)).json()["items"][0]
    assert item["issued_at"] == row.issued_at
    assert item["expires_at"] == row.expires_at
    assert item["consumed_at"] == row.consumed_at
    assert item["revoked_at"] is row.revoked_at


# --------------------------------------------------------------------------- #
# Ordering: issued_at UTC instant, then id
# --------------------------------------------------------------------------- #


def test_ordering_uses_utc_instant_then_id(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    ids = sorted(grant["id"] for grant in grants)

    # Same second: the grant with the larger id gets the exact-second stamp
    # and the one with the smaller id gets a later fractional stamp. Naive
    # text ordering would reverse both; instant ordering must win, then id.
    set_issued_at(client, ids[2], "2026-01-01T00:00:00Z")
    set_issued_at(client, ids[1], "2026-01-01T00:00:00.5Z")
    set_issued_at(client, ids[0], "2026-01-01T00:00:01Z")

    items = client.get(list_url(machine_id) + "?limit=10").json()["items"]
    assert [item["id"] for item in items] == [ids[2], ids[1], ids[0]]
    # Stamps are emitted verbatim.
    assert [item["issued_at"] for item in items] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00.5Z",
        "2026-01-01T00:00:01Z",
    ]


def test_same_instant_ties_break_by_id_ascending(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    stamp = "2026-02-02T08:09:10.25Z"
    for grant in grants:
        set_issued_at(client, grant["id"], stamp)

    items = client.get(list_url(machine_id) + "?limit=10").json()["items"]
    ordered_ids = sorted(grant["id"] for grant in grants)
    assert [item["id"] for item in items] == ordered_ids
    assert all(item["issued_at"] == stamp for item in items)


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


def test_default_limit_is_50(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 51)
    # Distinct issued_at moments keep insertion-time ordering readable.
    ordered_ids = [
        item["id"]
        for item in client.get(list_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    ]
    assert len(ordered_ids) == 51

    first = client.get(list_url(machine_id)).json()
    assert len(first["items"]) == 50
    assert first["next_cursor"] is not None
    assert [item["id"] for item in first["items"]] == ordered_ids[:50]

    second = client.get(
        list_url(machine_id) + f"?cursor={first['next_cursor']}"
    ).json()
    assert [item["id"] for item in second["items"]] == ordered_ids[50:]
    assert second["next_cursor"] is None


@pytest.mark.parametrize("page_limit", [1, 2, 3, 7])
def test_pagination_never_repeats_or_omits(client, page_limit):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 7)
    expected = [
        item["id"]
        for item in client.get(list_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    ]

    collected = fetch_all_pages(client, machine_id, limit=page_limit)
    assert [item["id"] for item in collected] == expected
    assert len({item["id"] for item in collected}) == len(expected)


def test_cursor_points_to_next_page_start(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 4)
    full = [
        item["id"]
        for item in client.get(list_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    ]

    first = client.get(list_url(machine_id) + "?limit=2").json()
    assert [item["id"] for item in first["items"]] == full[:2]
    second = client.get(
        list_url(machine_id) + f"?limit=2&cursor={first['next_cursor']}"
    ).json()
    assert [item["id"] for item in second["items"]] == full[2:4]
    assert second["next_cursor"] is None


def test_limit_boundaries_are_accepted(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 2)
    for value in (1, 100):
        response = client.get(list_url(machine_id) + f"?limit={value}")
        assert response.status_code == 200


def test_repeating_cursor_returns_byte_identical_page(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 3)
    first = client.get(list_url(machine_id) + "?limit=1")
    cursor = first.json()["next_cursor"]
    url = list_url(machine_id) + f"?limit=1&cursor={cursor}"
    assert client.get(url).content == client.get(url).content
    # Repeating the first cursor never brings the first item back.
    repeated = client.get(url).json()
    assert repeated["items"][0]["id"] != first.json()["items"][0]["id"]


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_listing_is_strictly_isolated_by_machine(client):
    first = allowed_machine(client, external_id="machine-1")
    second = create_machine(client, external_id="machine-2")
    declare(client, second)
    first_grants = issue_grants(client, first, 2)
    second_grants = issue_grants(client, second, 1)

    first_items = client.get(list_url(first) + "?limit=100").json()["items"]
    second_items = client.get(list_url(second) + "?limit=100").json()["items"]
    assert {item["id"] for item in first_items} == {
        grant["id"] for grant in first_grants
    }
    assert {item["id"] for item in second_items} == {
        grant["id"] for grant in second_grants
    }
    assert all(item["machine_id"] == first for item in first_items)
    assert all(item["machine_id"] == second for item in second_items)


def test_cursor_from_one_machine_is_invalid_for_another(client):
    first = allowed_machine(client, external_id="machine-1")
    second = create_machine(client, external_id="machine-2")
    declare(client, second)
    issue_grants(client, first, 2)
    issue_grants(client, second, 2)

    cursor = client.get(list_url(first) + "?limit=1").json()["next_cursor"]
    response = client.get(list_url(second) + f"?limit=1&cursor={cursor}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Query validation — all 422 invalid_query, before machine or grant reads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?x=1",
        "?foo",
        "?=",
        "?limit=1&x=2",
        "?limit=1&limit=2",
        "?cursor=a&cursor=b",
        "?limit",
        "?limit=",
        "?limit=0",
        "?limit=101",
        "?limit=-1",
        "?limit=1.0",
        "?limit=0.5",
        "?limit=true",
        "?limit=false",
        "?limit=abc",
        "?limit=0x1",
        "?limit=%201",
        "?limit=%2B1",
        "?cursor=garbage",
        "?cursor=2026-01-01T00:00:00Z",
        "?cursor=2026-01-01T00:00:00Z|",
        "?cursor=|grant-id",
        "?cursor=not-a-stamp|grant-id",
        "?cursor=2026-01-01T00:00:00Z|id|extra",
        "?cursor=2026-01-01 00:00:00|id",
        "?limit=1&cursor=2026-01-01T00:00:00Z|missing-id",
    ],
)
def test_invalid_query_shape_is_422(client, query):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(list_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "query",
    [
        "?x=1",
        "?limit=1&limit=2",
        "?limit=0",
        "?limit=101",
        "?limit=1.5",
        "?limit=true",
        "?limit=",
        "?cursor=garbage",
        "?cursor=2026-01-01T00:00:00Z|",
        "?cursor=2026-01-01T00:00:00Z|anything",
    ],
)
def test_validation_runs_before_machine_lookup(client, query):
    # A malformed query against a non-existent machine is still 422, never
    # 404: validation precedes the machine and grant reads.
    response = client.get(list_url(MISSING_MACHINE) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_carried_body_is_invalid_query_before_lookup(client):
    machine_id = allowed_machine(client)
    response = client.request(
        "GET",
        list_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        list_url(MISSING_MACHINE),
        content=b"anything",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_well_shaped_cursor_naming_no_grant_is_invalid_query(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(
        list_url(machine_id)
        + "?limit=1&cursor=2026-01-01T00:00:00.123456Z|no-such-grant"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Read-only guarantees, routing, coexistence, persistence
# --------------------------------------------------------------------------- #


def test_listing_writes_nothing_and_keeps_chain_valid(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    client.post(consume_url(machine_id, grants[0]["id"]))
    client.post(revoke_url(machine_id, grants[1]["id"]))
    backdate_expires_at(client, grants[2]["id"])

    integrity_url = (
        f"/machines/{machine_id}/"
        "authorization-grant-lifecycle-events/integrity"
    )
    before = client.get(integrity_url).content
    snapshot = {}
    with client.app.state.engine.connect() as conn:
        for table in (
            "authorization_grants",
            "authorization_grant_uses",
            "authorization_grant_lifecycle_events",
        ):
            snapshot[table] = conn.execute(
                text(f"SELECT * FROM {table} ORDER BY id")
            ).all()

    for _ in range(3):
        fetch_all_pages(client, machine_id, limit=2)

    after = client.get(integrity_url).content
    assert after == before
    assert client.get(integrity_url).json()["valid"] is True
    with client.app.state.engine.connect() as conn:
        for table, rows in snapshot.items():
            current = conn.execute(
                text(f"SELECT * FROM {table} ORDER BY id")
            ).all()
            assert current == rows


def test_non_get_methods_are_405_on_collection(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    for method in ("head", "put", "patch", "delete"):
        response = getattr(client, method)(list_url(machine_id))
        assert response.status_code == 405
        response = getattr(client, method)(list_url(MISSING_MACHINE))
        assert response.status_code == 405


def test_get_listing_and_post_issue_coexist_on_same_path(client):
    machine_id = allowed_machine(client)
    assert client.get(list_url(machine_id)).json() == {
        "items": [],
        "next_cursor": None,
    }
    event = record_event(client, machine_id).json()
    created = issue(client, machine_id, event["id"])
    assert created.status_code == 201
    listing = client.get(list_url(machine_id)).json()
    assert [item["id"] for item in listing["items"]] == [created.json()["id"]]


def test_listing_persists_consistently_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = allowed_machine(first)
        grants = issue_grants(first, machine_id, 3)
        first.post(consume_url(machine_id, grants[0]["id"]))
        first.post(revoke_url(machine_id, grants[1]["id"]))
        backdate_expires_at(first, grants[2]["id"])
        expected_pages = []
        cursor = None
        while True:
            query = "?limit=1"
            if cursor is not None:
                query += f"&cursor={cursor}"
            body = first.get(list_url(machine_id) + query).content
            expected_pages.append(body)
            cursor = first.get(list_url(machine_id) + query).json()[
                "next_cursor"
            ]
            if cursor is None:
                break

    with TestClient(app) as second:
        cursor = None
        index = 0
        while True:
            query = "?limit=1"
            if cursor is not None:
                query += f"&cursor={cursor}"
            response = second.get(list_url(machine_id) + query)
            assert response.content == expected_pages[index]
            index += 1
            cursor = response.json()["next_cursor"]
            if cursor is None:
                break
        assert index == len(expected_pages)
        statuses = {
            item["id"]: item["status"]
            for item in second.get(list_url(machine_id) + "?limit=100").json()[
                "items"
            ]
        }
        assert set(statuses.values()) == {"consumed", "revoked", "expired"}
