"""Tests for the read-only grant compliance window export.

The export entry is::

    GET /machines/{machine_id}/authorization-grants/compliance-export

It is strictly read-only: it never creates, repairs, updates, revokes, or
deletes a grant, its consumption record, a lifecycle event, a decision, a
decision basis, or a chain hash, and it adds no renewal, transfer, or
deletion. These tests cover the fixed envelope and item shapes, the derived
``active``/``expired``/``consumed``/``revoked`` booleans, the unique ``use``
record (or ``null``), the lifecycle events with their chain fields, the
half-open ``[start_at, end_at)`` period, ordering by the actual UTC instant
of ``issued_at`` then id, keyset pagination that never repeats or omits an
item, tolerant handling of damaged stored data (emitted verbatim, never
repaired), strict machine isolation (including cursors), the 422
``invalid_query`` / ``invalid_range`` / ``invalid_cursor`` validation cases
and their priority over the machine lookup, 404 for a missing machine, 405
for non-GET methods without reads, 500 with no partial result on a read
failure, restart persistence, byte-identical repeats, and non-interference
with issue/consume/revoke, the listing, and the lifecycle chain.
"""
import hashlib
import json as jsonlib
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


def grants_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants"


def export_url(machine_id):
    return grants_url(machine_id) + "/compliance-export"


def issue(client, machine_id, event_id, ttl_seconds=300):
    return client.post(
        grants_url(machine_id),
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume_url(machine_id, grant_id):
    return f"{grants_url(machine_id)}/{grant_id}/consume"


def revoke_url(machine_id, grant_id):
    return f"{grants_url(machine_id)}/{grant_id}/revoke"


def allowed_machine(client, external_id="machine-1"):
    """A machine with an enabled declaration and an allow rule."""
    machine_id = create_machine(client, external_id=external_id)
    declare(client, machine_id)
    create_rule(client)
    return machine_id


def issue_grants(client, machine_id, count, ttl_seconds=300):
    granted = []
    for index in range(count):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        response = issue(client, machine_id, event["id"], ttl_seconds)
        assert response.status_code == 201
        granted.append(response.json())
    return granted


def set_column(client, table, grant_id, column, value):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE {table} SET {column} = :value WHERE id = :id"
            ).bindparams(value=value, id=grant_id)
        )


def set_issued_at(client, grant_id, issued_at):
    set_column(client, "authorization_grants", grant_id, "issued_at", issued_at)


def backdate_expires_at(client, grant_id, seconds=10):
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")
    set_column(client, "authorization_grants", grant_id, "expires_at", past)
    return past


def fetch_all_pages(client, machine_id, window_query="", limit=1):
    """Walk the export to its end, collecting the returned items.

    ``window_query`` carries only period bounds (or is empty); the page
    ``limit`` and ``cursor`` are managed here so they never repeat.
    """
    collected = []
    cursor = None
    seen = set()
    while True:
        suffix = f"limit={limit}"
        if cursor is not None:
            suffix += f"&cursor={cursor}"
        url = (
            export_url(machine_id)
            + window_query
            + ("&" if window_query else "?")
            + suffix
        )
        response = client.get(url)
        assert response.status_code == 200
        body = response.json()
        collected.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            assert body["has_more"] is False
            break
        assert body["has_more"] is True
        assert cursor not in seen
        seen.add(cursor)
    return collected


MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-02T00:00:00Z"
T2 = "2026-01-03T00:00:00Z"

ENVELOPE_FIELDS = ["machine_id", "start_at", "end_at", "items",
                   "next_cursor", "has_more"]

ITEM_FIELDS = [
    "id",
    "machine_id",
    "event_id",
    "issued_at",
    "expires_at",
    "status",
    "consumed_at",
    "revoked_at",
    "active",
    "expired",
    "consumed",
    "revoked",
    "use",
    "lifecycle_events",
]

USE_FIELDS = ["id", "grant_id", "machine_id", "event_id", "consumed_at"]

LIFECYCLE_FIELDS = [
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
# Empty result, basic shape, 404
# --------------------------------------------------------------------------- #


def test_machine_without_grants_exports_empty_page(client):
    machine_id = allowed_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert list(response.json().keys()) == ENVELOPE_FIELDS
    assert response.json() == {
        "machine_id": machine_id,
        "start_at": None,
        "end_at": None,
        "items": [],
        "next_cursor": None,
        "has_more": False,
    }
    assert response.content.endswith(b"\n")


def test_missing_machine_is_404(client):
    response = client.get(export_url(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_bounds_are_echoed_verbatim(client):
    machine_id = allowed_machine(client)
    response = client.get(
        export_url(machine_id) + f"?start_at={T0}&end_at={T2}"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["start_at"] == T0
    assert body["end_at"] == T2
    # A single bound echoes the other as null.
    response = client.get(export_url(machine_id) + f"?start_at={T0}")
    body = response.json()
    assert body["start_at"] == T0
    assert body["end_at"] is None
    response = client.get(export_url(machine_id) + f"?end_at={T2}")
    body = response.json()
    assert body["start_at"] is None
    assert body["end_at"] == T2


# --------------------------------------------------------------------------- #
# Item shape and derived state
# --------------------------------------------------------------------------- #


def test_active_grant_item_shape(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert list(item.keys()) == ITEM_FIELDS
    assert item["id"] == grant["id"]
    assert item["machine_id"] == machine_id
    assert item["event_id"] == event["id"]
    assert item["issued_at"] == grant["issued_at"]
    assert item["expires_at"] == grant["expires_at"]
    assert item["status"] == "active"
    assert item["consumed_at"] is None
    assert item["revoked_at"] is None
    assert item["active"] is True
    assert item["expired"] is False
    assert item["consumed"] is False
    assert item["revoked"] is False
    assert item["use"] is None
    events = item["lifecycle_events"]
    assert [event_record["type"] for event_record in events] == ["issued"]
    assert list(events[0].keys()) == LIFECYCLE_FIELDS
    assert events[0]["grant_id"] == grant["id"]
    assert events[0]["authorization_event_id"] == event["id"]
    assert events[0]["occurred_at"] == grant["issued_at"]


def test_consumed_grant_carries_flags_use_and_events(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    use = client.post(consume_url(machine_id, grant["id"])).json()

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert item["status"] == "consumed"
    assert item["active"] is False
    assert item["expired"] is False
    assert item["consumed"] is True
    assert item["revoked"] is False
    assert item["consumed_at"] == use["consumed_at"]
    assert item["revoked_at"] is None
    assert item["use"] == {
        "id": use["use_id"],
        "grant_id": grant["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "consumed_at": use["consumed_at"],
    }
    assert list(item["use"].keys()) == USE_FIELDS
    assert [record["type"] for record in item["lifecycle_events"]] == [
        "issued",
        "consumed",
    ]
    consumed_event = item["lifecycle_events"][1]
    assert consumed_event["occurred_at"] == use["consumed_at"]
    # The chain fields are present and the issued event is the predecessor.
    assert consumed_event["previous_event_id"] == item["lifecycle_events"][0]["id"]


def test_revoked_grant_carries_flags_stamp_and_events(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    revocation = client.post(revoke_url(machine_id, grant["id"])).json()

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert item["status"] == "revoked"
    assert item["active"] is False
    assert item["expired"] is False
    assert item["consumed"] is False
    assert item["revoked"] is True
    assert item["revoked_at"] == revocation["revoked_at"]
    assert item["use"] is None
    assert [record["type"] for record in item["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]


def test_expired_unterminal_grant_presents_expired_without_storage_change(
    client,
):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    backdate_expires_at(client, grant["id"])

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert item["status"] == "active"
    assert item["active"] is False
    assert item["expired"] is True
    assert item["consumed"] is False
    assert item["revoked"] is False
    assert item["use"] is None
    # Derived only: storage still says active and nothing was written.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, consumed_at, revoked_at FROM "
                "authorization_grants WHERE id = :id"
            ).bindparams(id=grant["id"])
        ).one()
    assert row.status == "active"
    assert row.consumed_at is None
    assert row.revoked_at is None


def test_terminal_grants_keep_terminal_flags_after_expiry(client):
    machine_id = allowed_machine(client)
    consumed_event = record_event(client, machine_id, resource="res/c").json()
    revoked_event = record_event(client, machine_id, resource="res/r").json()
    consumed = issue(client, machine_id, consumed_event["id"]).json()
    revoked = issue(client, machine_id, revoked_event["id"]).json()
    client.post(consume_url(machine_id, consumed["id"]))
    client.post(revoke_url(machine_id, revoked["id"]))
    backdate_expires_at(client, consumed["id"])
    backdate_expires_at(client, revoked["id"])

    by_id = {
        item["id"]: item
        for item in client.get(export_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    }
    consumed_item = by_id[consumed["id"]]
    revoked_item = by_id[revoked["id"]]
    assert consumed_item["consumed"] is True
    assert consumed_item["expired"] is False
    assert consumed_item["active"] is False
    assert revoked_item["revoked"] is True
    assert revoked_item["expired"] is False
    assert revoked_item["active"] is False


# --------------------------------------------------------------------------- #
# Half-open window
# --------------------------------------------------------------------------- #


def test_window_is_half_open_on_issued_at(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    set_issued_at(client, grants[0]["id"], T0)
    set_issued_at(client, grants[1]["id"], T1)
    set_issued_at(client, grants[2]["id"], T2)

    # [T0, T2) includes exactly the T0 and T1 grants.
    body = client.get(
        export_url(machine_id) + f"?start_at={T0}&end_at={T2}"
    ).json()
    assert [item["id"] for item in body["items"]] == [
        grants[0]["id"],
        grants[1]["id"],
    ]

    # An empty (equal-bound) window is rejected as invalid_range rather than
    # returning an empty page.
    response = client.get(
        export_url(machine_id) + f"?start_at={T1}&end_at={T1}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_range"}}


def test_single_bounds_filter_half_open(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    set_issued_at(client, grants[0]["id"], T0)
    set_issued_at(client, grants[1]["id"], T1)
    set_issued_at(client, grants[2]["id"], T2)

    body = client.get(export_url(machine_id) + f"?start_at={T1}").json()
    assert [item["id"] for item in body["items"]] == [
        grants[1]["id"],
        grants[2]["id"],
    ]
    # End bound is exclusive: T2 itself is out.
    body = client.get(export_url(machine_id) + f"?end_at={T2}").json()
    assert [item["id"] for item in body["items"]] == [
        grants[0]["id"],
        grants[1]["id"],
    ]


def test_unbounded_export_covers_every_grant(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    body = client.get(export_url(machine_id) + "?limit=100").json()
    assert {item["id"] for item in body["items"]} == {
        grant["id"] for grant in grants
    }


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_ordering_uses_utc_instant_then_id(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    ids = sorted(grant["id"] for grant in grants)
    set_issued_at(client, ids[2], "2026-01-01T00:00:00Z")
    set_issued_at(client, ids[1], "2026-01-01T00:00:00.5Z")
    set_issued_at(client, ids[0], "2026-01-01T00:00:01Z")

    items = client.get(export_url(machine_id) + "?limit=10").json()["items"]
    assert [item["id"] for item in items] == [ids[2], ids[1], ids[0]]
    assert [item["issued_at"] for item in items] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00.5Z",
        "2026-01-01T00:00:01Z",
    ]


def test_same_instant_ties_break_by_id_ascending(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    for grant in grants:
        set_issued_at(client, grant["id"], "2026-02-02T08:09:10.25Z")
    items = client.get(export_url(machine_id) + "?limit=10").json()["items"]
    assert [item["id"] for item in items] == sorted(
        grant["id"] for grant in grants
    )


# --------------------------------------------------------------------------- #
# Damaged data: emitted verbatim, never repaired
# --------------------------------------------------------------------------- #


def test_unparseable_issued_at_excluded_from_window_but_in_unbounded_tail(
    client,
):
    machine_id = allowed_machine(client)
    good = issue_grants(client, machine_id, 1)[0]
    damaged = issue_grants(client, machine_id, 1)[0]
    set_issued_at(client, damaged["id"], "not-a-timestamp")

    bounded = client.get(
        export_url(machine_id) + f"?start_at=2000-01-01T00:00:00Z"
    ).json()
    assert [item["id"] for item in bounded["items"]] == [good["id"]]

    unbounded = client.get(export_url(machine_id) + "?limit=100").json()
    ids = [item["id"] for item in unbounded["items"]]
    assert ids[-1] == damaged["id"]
    assert unbounded["items"][-1]["issued_at"] == "not-a-timestamp"


def test_damaged_row_is_pageable_through_a_cursor(client):
    machine_id = allowed_machine(client)
    good = issue_grants(client, machine_id, 1)[0]
    damaged = issue_grants(client, machine_id, 1)[0]
    set_issued_at(client, damaged["id"], "garbage|with|pipes")

    first = client.get(export_url(machine_id) + "?limit=1").json()
    assert first["items"][0]["id"] == good["id"]
    second = client.get(
        export_url(machine_id) + f"?limit=1&cursor={first['next_cursor']}"
    ).json()
    assert second["items"][0]["id"] == damaged["id"]
    assert second["next_cursor"] is None
    assert second["has_more"] is False


def test_consumed_status_without_use_row_emits_null_use(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))
    # Damage: delete the unique use row; the stored status/stamps stay.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM authorization_grant_uses WHERE grant_id = :id")
            .bindparams(id=grant["id"])
        )

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert item["status"] == "consumed"
    assert item["consumed"] is True
    assert item["use"] is None
    assert [record["type"] for record in item["lifecycle_events"]] == [
        "issued",
        "consumed",
    ]


def test_contradictory_stored_status_is_not_corrected(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    # Damage: an unknown status word with a consumed_at stamp set.
    set_column(
        client,
        "authorization_grants",
        grant["id"],
        "status",
        "mysterious",
    )
    set_column(
        client,
        "authorization_grants",
        grant["id"],
        "consumed_at",
        "2026-03-03T03:03:03Z",
    )

    item = client.get(export_url(machine_id)).json()["items"][0]
    assert item["status"] == "mysterious"
    assert item["consumed_at"] == "2026-03-03T03:03:03Z"
    # The derived flags treat only the stored status word: not a recognized
    # terminal, so it presents as active (expires_at is still in the
    # future); the contradiction is surfaced, never reconciled.
    assert item["consumed"] is False
    assert item["revoked"] is False
    assert item["active"] is True
    assert item["use"] is None


def test_lifecycle_events_are_chain_ordered_with_hashes_intact(client):
    machine_id = allowed_machine(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    client.post(consume_url(machine_id, grant["id"]))

    item = client.get(export_url(machine_id)).json()["items"][0]
    records = item["lifecycle_events"]
    previous_id = None
    previous_chain = ""
    for index, record in enumerate(records):
        assert record["previous_event_id"] == previous_id
        assert isinstance(record["content_hash"], str)
        assert len(record["content_hash"]) == 64
        assert isinstance(record["chain_hash"], str)
        assert len(record["chain_hash"]) == 64
        document = jsonlib.dumps(
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
        content_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
        assert record["content_hash"] == content_hash
        chain_hash = hashlib.sha256(
            f"{previous_chain}:{content_hash}".encode("utf-8")
        ).hexdigest()
        assert record["chain_hash"] == chain_hash
        previous_id = record["id"]
        previous_chain = chain_hash
        assert index == 0 or record["previous_event_id"] is not None


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


def test_default_limit_is_50(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 51)
    full = [
        item["id"]
        for item in client.get(export_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    ]

    first = client.get(export_url(machine_id)).json()
    assert len(first["items"]) == 50
    assert first["has_more"] is True
    assert [item["id"] for item in first["items"]] == full[:50]

    second = client.get(
        export_url(machine_id) + f"?cursor={first['next_cursor']}"
    ).json()
    assert [item["id"] for item in second["items"]] == full[50:]
    assert second["next_cursor"] is None
    assert second["has_more"] is False


@pytest.mark.parametrize("page_limit", [1, 2, 3, 7])
def test_pagination_never_repeats_or_omits(client, page_limit):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 7)
    expected = [
        item["id"]
        for item in client.get(export_url(machine_id) + "?limit=100").json()[
            "items"
        ]
    ]
    collected = fetch_all_pages(
        client,
        machine_id,
        window_query="?start_at=2000-01-01T00:00:00Z",
        limit=page_limit,
    )
    assert [item["id"] for item in collected] == expected
    assert len({item["id"] for item in collected}) == len(expected)


def test_limit_boundaries_are_accepted(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 2)
    for value in (1, 100):
        response = client.get(export_url(machine_id) + f"?limit={value}")
        assert response.status_code == 200


def test_repeating_cursor_returns_byte_identical_page(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 3)
    first = client.get(export_url(machine_id) + "?limit=1").json()
    url = export_url(machine_id) + f"?limit=1&cursor={first['next_cursor']}"
    assert client.get(url).content == client.get(url).content
    assert client.get(url).json()["items"][0]["id"] != first["items"][0]["id"]


def test_cursor_naming_grant_outside_window_is_invalid_cursor(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 2)
    set_issued_at(client, grants[0]["id"], T0)
    set_issued_at(client, grants[1]["id"], T2)
    # A cursor pointing at the T2 grant, fetched in the [T0, T1) window.
    cursor = f"{T2}|{grants[1]['id']}"
    response = client.get(
        export_url(machine_id)
        + f"?start_at={T0}&end_at={T1}&limit=1&cursor={cursor}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_export_is_strictly_isolated_by_machine(client):
    first = allowed_machine(client, external_id="machine-1")
    second = create_machine(client, external_id="machine-2")
    declare(client, second)
    first_grants = issue_grants(client, first, 2)
    second_grants = issue_grants(client, second, 1)

    first_items = client.get(
        export_url(first) + "?limit=100"
    ).json()["items"]
    second_items = client.get(
        export_url(second) + "?limit=100"
    ).json()["items"]
    assert {item["id"] for item in first_items} == {
        grant["id"] for grant in first_grants
    }
    assert {item["id"] for item in second_items} == {
        grant["id"] for grant in second_grants
    }
    for item in first_items:
        assert item["machine_id"] == first
        assert all(
            record["machine_id"] == first
            for record in item["lifecycle_events"]
        )


def test_cursor_from_one_machine_is_invalid_for_another(client):
    first = allowed_machine(client, external_id="machine-1")
    second = create_machine(client, external_id="machine-2")
    declare(client, second)
    issue_grants(client, first, 1)
    issue_grants(client, second, 1)

    cursor = client.get(export_url(first) + "?limit=1").json()["next_cursor"]
    response = client.get(export_url(second) + f"?limit=1&cursor={cursor}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


# --------------------------------------------------------------------------- #
# Query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?x=1",
        "?foo",
        "?=",
        "?start_at=1&x=2",
        "?start_at=2026-01-01T00:00:00Z&start_at=2026-01-02T00:00:00Z",
        "?end_at=2026-01-01T00:00:00Z&end_at=2026-01-02T00:00:00Z",
        "?limit=1&limit=2",
        "?cursor=a&cursor=b",
        "?start_at",
        "?start_at=",
        "?start_at=2026-01-01",
        "?start_at=2026-01-01T00:00:00+00:00",
        "?start_at=2026-01-01T00:00:00",
        "?start_at=2026-13-01T00:00:00Z",
        "?start_at=2026-01-01T25:00:00Z",
        "?end_at=not-a-time",
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
    ],
)
def test_invalid_query_shape_is_422(client, query):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(export_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "query",
    [
        "?x=1",
        "?limit=0",
        "?limit=101",
        "?limit=1.5",
        "?limit=true",
        "?start_at=not-a-time",
        "?end_at=2026-01-01T00:00:00",
        "?start_at=2026-01-01T00:00:00Z&start_at=2026-01-02T00:00:00Z",
        "?limit=1&limit=2",
    ],
)
def test_invalid_query_runs_before_machine_lookup(client, query):
    response = client.get(export_url(MISSING_MACHINE) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "query",
    [
        f"?start_at={T1}&end_at={T0}",
        f"?start_at={T0}&end_at={T0}",
    ],
)
def test_invalid_range_is_422_before_lookup(client, query):
    # Existing machine with data: still invalid_range.
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(export_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_range"}}
    # And against a missing machine, range validation also precedes 404.
    response = client.get(export_url(MISSING_MACHINE) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_range"}}


def test_malformed_bound_takes_invalid_query_over_invalid_range(client):
    # An unparseable bound is a malformed parameter, not a range error.
    response = client.get(
        export_url(MISSING_MACHINE)
        + "?start_at=garbage&end_at=2026-01-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "cursor",
    ["", "not-a-cursor", "|", f"{T0}|", f"|grant-id"],
)
def test_malformed_cursor_is_422_invalid_cursor(client, cursor):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(
        export_url(machine_id) + f"?limit=1&cursor={cursor}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unlocatable_cursor_is_422_before_machine_lookup(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    for cursor in (
        f"{T0}|no-such-grant",
        "garbage-stamp|no-such-grant",
    ):
        response = client.get(
            export_url(machine_id) + f"?limit=1&cursor={cursor}"
        )
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_cursor"}}
    # A well-shaped cursor against a missing machine cannot locate either.
    response = client.get(
        export_url(MISSING_MACHINE) + f"?limit=1&cursor={T0}|anything"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_carried_body_is_invalid_query_before_lookup(client):
    machine_id = allowed_machine(client)
    response = client.request(
        "GET",
        export_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.request(
        "GET",
        export_url(MISSING_MACHINE),
        content=b"anything",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Routing, read-only guarantees, failures, persistence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method", ["head", "post", "put", "patch", "delete"]
)
def test_non_get_methods_are_405_without_reading(client, method):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = getattr(client, method)(export_url(machine_id))
    assert response.status_code == 405
    response = getattr(client, method)(export_url(MISSING_MACHINE))
    assert response.status_code == 405


def test_export_writes_nothing_and_keeps_chain_valid(client):
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

    for window_query, page_limit in (
        ("", 1),
        ("?start_at=2000-01-01T00:00:00Z&end_at=2030-01-01T00:00:00Z", 2),
        ("?start_at=2020-01-01T00:00:00Z", 3),
    ):
        fetch_all_pages(
            client, machine_id, window_query=window_query, limit=page_limit
        )

    assert client.get(integrity_url).content == before
    assert client.get(integrity_url).json()["valid"] is True
    with client.app.state.engine.connect() as conn:
        for table, rows in snapshot.items():
            current = conn.execute(
                text(f"SELECT * FROM {table} ORDER BY id")
            ).all()
            assert current == rows


def test_read_failure_is_500_with_no_partial_result(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    # Validation must still win with the table gone; a valid query then
    # fails while reading and answers 500, never a partial page.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grants"))

    assert client.get(export_url(machine_id) + "?limit=0").status_code == 422
    response = client.get(export_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_export_persists_consistently_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = allowed_machine(first)
        grants = issue_grants(first, machine_id, 3)
        first.post(consume_url(machine_id, grants[0]["id"]))
        first.post(revoke_url(machine_id, grants[1]["id"]))
        backdate_expires_at(first, grants[2]["id"])
        queries = [
            "",
            f"?start_at=2000-01-01T00:00:00Z&end_at=2030-01-01T00:00:00Z",
            "?limit=1",
        ]
        expected = [first.get(export_url(machine_id) + q).content for q in queries]

    with TestClient(app) as second:
        for query, content in zip(queries, expected):
            response = second.get(export_url(machine_id) + query)
            assert response.content == content

        flags = {
            item["id"]: (item["consumed"], item["revoked"], item["expired"])
            for item in second.get(
                export_url(machine_id) + "?limit=100"
            ).json()["items"]
        }
        assert set(flags.values()) == {
            (True, False, False),
            (False, True, False),
            (False, False, True),
        }
