"""Tests for the stable, read-only incremental decision-basis snapshot query.

Covers `GET /machines/{machine_id}/authorization-decision-events/
decision-basis/changes`:

- validation before any machine or snapshot is read: ``bad_limit`` for a
  missing, blank, fractional, boolean, non-decimal, or out-of-range
  ``limit``; ``invalid_cursor`` for an empty, non-string, shape-mismatching,
  or unlocatable ``cursor``; ``invalid_query`` for unknown parameters,
  repeated names, or a carried request body — all 422 and taking priority
  over the machine lookup;
- GET-only ``405`` without reading snapshots, ``404 not_found`` for a
  missing machine carrying no records, and ``500 internal_error`` with no
  partial page when the snapshots or the machine cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where each record is exactly ``{group, record}`` tagged
  ``decision_basis`` and carrying the same five groups the single-snapshot
  query exposes, and only the path machine's snapshots;
- only events that already have a snapshot enter the page: historical
  events without a snapshot are not fabricated, not returned, and never
  enter the cursor;
- keyset pagination in (actual UTC created_at instant, event id) order, an
  exact-second record before a fractional-second record of the same
  second, exclusive cursors, and ``next_cursor``/``has_more`` on full,
  partial, empty, and past-the-end pages;
- a snapshot with an unparseable ``created_at`` kept verbatim, sorted
  last, and still pageable (including a damaged stamp text containing
  ``|``), while a JSON-valid but damaged document body is re-emitted
  exactly as stored;
- byte-identical repeat pages, no resurfacing after earlier inserts,
  strict read-only behavior, compact newline-terminated JSON with no
  non-finite tokens, and persistence across a restart.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"
T0_FRAC = "2026-03-01T00:00:00.500000Z"

HASH_A = "a" * 64
HASH_B = "b" * 64

GROUP_KEYS = [
    "event_summary",
    "status_basis",
    "declaration_basis",
    "policy_candidates",
    "decision",
]
ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def machine_id(client):
    created = client.post(
        "/machines",
        json={
            "external_id": "machine-1",
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    ).json()
    return created["id"]


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def changes_path(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "decision-basis/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + cursor
    return f"{changes_path(machine_id)}?{query}"


def basis_document(
    n,
    machine_id,
    created_at,
    *,
    action_type="read",
    resource=None,
    allowed=True,
    reason="allowed_by_policy",
    status="active",
    previous_event_id=None,
):
    """Build a structurally complete five-group basis document for one event."""
    event_id = rid(n)
    resource = resource if resource is not None else f"res/{n}"
    active = status == "active"
    return {
        "event_summary": {
            "id": event_id,
            "machine_id": machine_id,
            "action_type": action_type,
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
            "created_at": created_at,
            "previous_event_id": previous_event_id,
            "content_hash": HASH_A,
            "chain_hash": HASH_B,
        },
        "status_basis": {
            "machine_id": machine_id,
            "status": status,
            "captured_at": created_at,
            "declarations_read": active,
            "policies_read": False,
        },
        "declaration_basis": {"read": active, "declarations": []},
        "policy_candidates": {
            "read": False,
            "candidates": [],
            "winners": [],
            "conflicts": [],
        },
        "decision": {"allowed": allowed, "reason": reason},
    }


def insert_event(client, machine_id, n, created_at, **kwargs):
    """Insert an event row directly with a fixed id and timestamp.

    The chain columns are non-null so the startup backfill leaves the row as
    given; the changes query only walks the snapshot table, but the snapshot
    foreign key needs this parent event row.
    """
    doc = basis_document(n, machine_id, created_at, **kwargs)
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "action_type": doc["event_summary"]["action_type"],
        "resource": doc["event_summary"]["resource"],
        "allowed": doc["event_summary"]["allowed"],
        "reason": doc["event_summary"]["reason"],
        "created_at": created_at,
        "previous_event_id": doc["event_summary"]["previous_event_id"],
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at, previous_event_id, content_hash, chain_hash) "
                "VALUES "
                "(:id, :machine_id, :action_type, :resource, :allowed, "
                ":reason, :created_at, :previous_event_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
        )
    return values


def insert_snapshot(client, machine_id, n, created_at, *, document=None, **kwargs):
    """Insert an event plus its snapshot row with a fixed id and timestamp."""
    insert_event(client, machine_id, n, created_at, **kwargs)
    if document is None:
        document = basis_document(n, machine_id, created_at, **kwargs)
    document_text = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_basis "
                "(event_id, machine_id, created_at, document) "
                "VALUES (:event_id, :machine_id, :created_at, :document)"
            ),
            {
                "event_id": rid(n),
                "machine_id": machine_id,
                "created_at": created_at,
                "document": document_text,
            },
        )
    return document


def write_snapshot_text(client, event_id, machine_id, created_at, raw):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id AND machine_id = :mid"
            ).bindparams(doc=raw, id=event_id, mid=machine_id)
        )


def fetch_all_pages(client, machine_id, limit):
    """Walk the cursor chain from the start and return every record."""
    seen = []
    cursor = None
    pages = 0
    while True:
        response = client.get(changes_url(machine_id, limit=limit, cursor=cursor))
        assert response.status_code == 200
        body = response.json()
        seen.extend(body["records"])
        pages += 1
        if body["next_cursor"] is None:
            assert body["has_more"] is False
            break
        assert body["has_more"] is True
        cursor = body["next_cursor"]
        assert pages < 1000
    return seen, pages


def _quote(value):
    from urllib.parse import quote

    return quote(value, safe="")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",                  # limit missing
        "?limit=",           # blank
        "?limit=0",          # below range
        "?limit=101",        # above range
        "?limit=-1",
        "?limit=1.0",        # decimal forms
        "?limit=1.5",
        "?limit=true",       # booleans are not integers
        "?limit=false",
        "?limit=abc",
        "?limit= 1",
        "?limit=1 ",
        "?limit=0x1",
    ],
)
def test_bad_limit_is_422(client, machine_id, query):
    response = client.get(f"{changes_path(machine_id)}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}


def test_boundary_limits_are_accepted(client, machine_id):
    for value in (1, 100):
        response = client.get(changes_url(machine_id, limit=value))
        assert response.status_code == 200
        assert response.json()["limit"] == value


@pytest.mark.parametrize(
    "cursor",
    [
        "",                # empty value
        "not-a-cursor",    # no separator
        "|",               # both segments empty
        f"{T0}|",          # missing event id
        f"|{rid(1)}",      # missing created-at text
    ],
)
def test_malformed_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


@pytest.mark.parametrize(
    "cursor",
    [
        # Well-shaped positions that name no stored snapshot; the timestamp
        # segment is taken as original text, so a non-RFC text still passes
        # the shape check and is rejected when it cannot be located.
        f"{T0}|{rid(999)}",
        "garbage-stamp|some-event",
        f"{T0}|not-a-uuid-but-nonempty",
        f"2026-13-01T00:00:00Z|{rid(999)}",
        f"{T0}||{rid(1)}",
    ],
)
def test_unlocatable_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_snapshot_is_accepted(client, machine_id):
    # A cursor locates by the exact stored (created_at text, event id) pair.
    insert_snapshot(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}")
    )
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_naming_an_event_without_a_snapshot_is_422(client, machine_id):
    # The event exists but has no snapshot row: the position cannot be
    # located in the snapshot set, so it is invalid rather than an empty page.
    insert_event(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(2)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_naming_another_machines_snapshot_is_422(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_snapshot(client, other["id"], 1, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client, machine_id):
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client, machine_id):
    response = client.get(
        changes_path(machine_id), params=[("limit", "1"), ("limit", "2")]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        changes_path(machine_id),
        params=[("limit", "1"), ("cursor", f"{T0}|{rid(1)}"),
                ("cursor", f"{T1}|{rid(2)}")],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_reads(client, machine_id):
    response = client.request(
        "GET", f"{changes_path(machine_id)}?limit=10",
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_parameter_errors_take_priority_over_machine_lookup(client):
    # No machine exists at all; every parameter error is still its 422.
    assert client.get(changes_path(MISSING_MACHINE)).status_code == 422
    assert client.get(f"{changes_path(MISSING_MACHINE)}?limit=0").json() == {
        "error": {"code": "bad_limit"}
    }
    assert client.get(
        f"{changes_path(MISSING_MACHINE)}?limit=1&cursor=garbage"
    ).json() == {"error": {"code": "invalid_cursor"}}
    assert client.get(f"{changes_path(MISSING_MACHINE)}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    # A well-shaped cursor names no snapshot of a missing machine, so it is
    # an unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(MISSING_MACHINE, limit=1, cursor=f"{T0}|{rid(1)}")
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_snapshots(client, machine_id):
    # With the table dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))

    assert client.get(changes_path(machine_id)).status_code == 422
    assert client.get(f"{changes_path(machine_id)}?limit=0").json() == {
        "error": {"code": "bad_limit"}
    }
    assert client.get(
        f"{changes_path(machine_id)}?limit=1&cursor=garbage"
    ).json() == {"error": {"code": "invalid_cursor"}}
    assert client.get(f"{changes_path(machine_id)}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }


def test_only_get_is_accepted_without_reading_snapshots(client, machine_id):
    # Drop the table so any snapshot read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
        assert response.status_code == 405


def test_missing_machine_is_404_with_no_records(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content


def test_read_failure_is_500_with_no_partial_page(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_page(client, machine_id):
    # The machine lookup happens after the snapshots are read; a failure
    # there is still a read-layer fault with the internal_error envelope,
    # never the framework default body and never a partial page.
    insert_snapshot(client, machine_id, 1, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_undecodable_snapshot_document_is_500_with_no_partial_page(
    client, machine_id
):
    # The rows read fine, but one captured document is no longer JSON: it
    # cannot be rendered as the promised five groups without fabricating or
    # repairing it, and it must not be silently skipped. The whole page
    # fails with no partial output.
    insert_snapshot(client, machine_id, 1, T0)
    insert_snapshot(client, machine_id, 2, T1)
    write_snapshot_text(
        client, rid(1), machine_id, T0, "this is not json"
    )
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, ordering, paging
# --------------------------------------------------------------------------- #


def test_empty_database_returns_full_empty_page(client, machine_id):
    response = client.get(changes_url(machine_id, limit=25))
    assert response.status_code == 200
    assert list(response.json().keys()) == ENVELOPE_KEYS
    assert response.json() == {
        "machine_id": machine_id,
        "limit": 25,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_events_without_snapshots_are_not_fabricated(client, machine_id):
    # Two historical events with no snapshot rows; the page stays empty and
    # no basis is reconstructed from current data.
    insert_event(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=25)).json()
    assert body["records"] == []
    assert body["has_more"] is False and body["next_cursor"] is None
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT COUNT(*) FROM authorization_decision_basis")
        ).scalar_one() == 0


def test_snapshots_and_snapshotless_events_interleave_in_page(client, machine_id):
    # Events 1 and 3 have snapshots; event 2 (between them in time) does not.
    # Only snapshots enter the page and the cursor.
    insert_snapshot(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, T1)
    insert_snapshot(client, machine_id, 3, T2)

    records, pages = fetch_all_pages(client, machine_id, limit=1)
    assert pages == 2
    assert [r["record"]["event_summary"]["id"] for r in records] == [
        rid(1), rid(3)
    ]


def test_single_page_smaller_than_limit(client, machine_id):
    insert_snapshot(client, machine_id, 1, T1)
    insert_snapshot(client, machine_id, 2, T3)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [
        r["record"]["event_summary"]["id"] for r in body["records"]
    ] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    insert_snapshot(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_snapshot_in_order(client, machine_id):
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        insert_snapshot(client, machine_id, n, stamp)

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [r["record"]["event_summary"]["id"] for r in records] == [
        rid(2), rid(4), rid(3), rid(5), rid(1)
    ]
    assert [
        r["record"]["event_summary"]["created_at"] for r in records
    ] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_snapshot(client, machine_id, n, stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["record"]["event_summary"]["id"] for r in first["records"]] == [
        rid(1), rid(2)
    ]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text verbatim, after the row.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["record"]["event_summary"]["id"] for r in second["records"]] == [
        rid(3), rid(4)
    ]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_is_unlocatable_422(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)

    # A non-null cursor always names a page's last stored snapshot, so a
    # position beyond the table cannot be located and is invalid rather
    # than an empty page.
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|{rid(999)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_snapshot(client, machine_id, 2, T0_FRAC)
    insert_snapshot(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["record"]["event_summary"]["id"] for r in body["records"]] == [
        rid(1), rid(2)
    ]
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_event_id_and_cursor(client, machine_id):
    insert_snapshot(client, machine_id, 30, T2)
    insert_snapshot(client, machine_id, 20, T2)
    insert_snapshot(client, machine_id, 10, T3)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["record"]["event_summary"]["id"] for r in first["records"]] == [
        rid(20)
    ]
    assert first["next_cursor"] == f"{T2}|{rid(20)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [r["record"]["event_summary"]["id"] for r in second["records"]] == [
        rid(30)
    ]

    third = client.get(
        changes_url(machine_id, limit=1, cursor=second["next_cursor"])
    ).json()
    assert [r["record"]["event_summary"]["id"] for r in third["records"]] == [
        rid(10)
    ]
    assert third["next_cursor"] is None


def test_records_are_tagged_and_carry_the_five_groups(client, machine_id):
    document = insert_snapshot(
        client, machine_id, 7, "2026-03-01T00:00:00.250Z",
        action_type="write", resource="res/secret", allowed=False,
        reason="denied_by_policy",
    )
    item = client.get(changes_url(machine_id, limit=10)).json()["records"][0]
    assert list(item.keys()) == ["group", "record"]
    assert item["group"] == "decision_basis"
    assert list(item["record"].keys()) == GROUP_KEYS
    assert item["record"] == document


def test_changes_record_matches_the_single_snapshot_query(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    changes_item = client.get(
        changes_url(machine_id, limit=10)
    ).json()["records"][0]["record"]
    single = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{rid(1)}/decision-basis"
    )
    assert single.status_code == 200
    assert changes_item == json.loads(single.content)


def test_only_the_path_machines_snapshots_are_returned(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_snapshot(client, machine_id, 1, T0)
    insert_snapshot(client, other["id"], 2, T1)
    insert_snapshot(client, machine_id, 3, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["record"]["event_summary"]["id"] for r in body["records"]] == [
        rid(1), rid(3)
    ]
    assert all(
        r["record"]["event_summary"]["machine_id"] == machine_id
        for r in body["records"]
    )

    other_body = client.get(changes_url(other["id"], limit=10)).json()
    assert [
        r["record"]["event_summary"]["id"] for r in other_body["records"]
    ] == [rid(2)]


def test_real_registered_events_are_pageable_with_their_basis(client, machine_id):
    # Events registered through the real write path capture a genuine
    # snapshot; the incremental walk returns every one with its five groups.
    for resource in ("res/a", "res/b", "res/c"):
        created = client.post(
            f"/machines/{machine_id}/authorization-decision-events",
            json={"action_type": "read", "resource": resource},
        )
        assert created.status_code == 201

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 2
    assert [r["record"]["event_summary"]["resource"] for r in records] == [
        "res/a", "res/b", "res/c"
    ]
    for item in records:
        assert item["group"] == "decision_basis"
        assert list(item["record"].keys()) == GROUP_KEYS
        summary = item["record"]["event_summary"]
        assert item["record"]["decision"] == {
            "allowed": summary["allowed"],
            "reason": summary["reason"],
        }


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client, machine_id):
    insert_snapshot(client, machine_id, 1, T1)
    insert_snapshot(client, machine_id, 2, "not-a-timestamp")
    insert_snapshot(client, machine_id, 3, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["record"]["event_summary"]["id"] for r in rows] == [
        rid(3), rid(1), rid(2)
    ]
    assert rows[-1]["record"]["event_summary"]["created_at"] == "not-a-timestamp"


def test_out_of_range_calendar_stamp_sorts_last_verbatim(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    insert_snapshot(client, machine_id, 2, "2026-13-40T99:99:99Z")

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["record"]["event_summary"]["id"] for r in rows] == [
        rid(1), rid(2)
    ]
    assert rows[1]["record"]["event_summary"]["created_at"] == (
        "2026-13-40T99:99:99Z"
    )


def test_multiple_unparseable_stamps_sort_by_id(client, machine_id):
    insert_snapshot(client, machine_id, 30, "zzz")
    insert_snapshot(client, machine_id, 10, "yyy")
    insert_snapshot(client, machine_id, 20, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["record"]["event_summary"]["id"] for r in rows] == [
        rid(20), rid(10), rid(30)
    ]


def test_damaged_json_valid_document_is_emitted_unmodified(client, machine_id):
    # The changes query does not audit or repair: a JSON-valid document with
    # a wrong field type is re-emitted exactly as stored and still pages.
    insert_snapshot(client, machine_id, 1, T1)
    document = json.loads(_load_document(client, rid(1)))
    document["status_basis"]["declarations_read"] = "damaged"
    document["decision"]["allowed"] = "not-a-bool"
    write_snapshot_text(
        client, rid(1), machine_id, T1,
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
    )

    (item,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert item["record"] == document


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(
    client, machine_id
):
    insert_snapshot(client, machine_id, 1, T0)
    # The snapshot's own created_at carries a pipe; the event parent keeps a
    # sound stamp. Repoint the snapshot row only.
    insert_snapshot(client, machine_id, 2, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET created_at = :at "
                "WHERE event_id = :id"
            ).bindparams(at="not-a-timestamp|weird", id=rid(2))
        )

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["record"]["event_summary"]["id"] for r in first["records"]] == [
        rid(1)
    ]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{rid(1)}"

    second = client.request(
        "GET",
        f"{changes_path(machine_id)}?limit=1&cursor=" + _quote(cursor),
    )
    assert second.status_code == 200
    body = second.json()
    assert [r["record"]["event_summary"]["id"] for r in body["records"]] == [
        rid(2)
    ]
    assert body["records"][0]["record"]["status_basis"]["captured_at"] == T1
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def _load_document(client, event_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :id"
            ).bindparams(id=event_id)
        ).scalar_one()


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_snapshot(client, machine_id, n, stamp)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    page_one = client.get(
        changes_url(machine_id, limit=2, cursor=cursor)
    ).content
    page_two = client.get(
        changes_url(machine_id, limit=2, cursor=cursor)
    ).content
    assert page_one == page_two


def test_new_earlier_snapshots_do_not_revisit_returned_pages(client, machine_id):
    insert_snapshot(client, machine_id, 1, T2)
    insert_snapshot(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["record"]["event_summary"]["id"] for r in first["records"]] == [
        rid(1)
    ]
    old_cursor = first["next_cursor"]

    # Insert snapshots sorting before the cursor position and one between.
    insert_snapshot(client, machine_id, 3, T1)
    insert_snapshot(client, machine_id, 4, T3)

    second = client.get(
        changes_url(machine_id, limit=10, cursor=old_cursor)
    ).json()
    assert [r["record"]["event_summary"]["id"] for r in second["records"]] == [
        rid(4), rid(2)
    ]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [r["record"]["event_summary"]["id"] for r in records] == [
        rid(3), rid(1), rid(4), rid(2)
    ]


def test_has_more_reflects_snapshots_after_position_only(client, machine_id):
    insert_snapshot(client, machine_id, 1, T0)
    insert_snapshot(client, machine_id, 2, T1)
    insert_snapshot(client, machine_id, 3, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{rid(2)}")
    ).json()
    assert [r["record"]["event_summary"]["id"] for r in body["records"]] == [
        rid(3)
    ]
    assert body["has_more"] is False


def test_query_is_read_only(client, machine_id):
    insert_snapshot(client, machine_id, 1, T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM authorization_decision_basis")))

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T5}|{rid(9)}"))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T1}|{rid(1)}"))
    after = table_state()
    assert before == after


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(
    client, machine_id
):
    document = insert_snapshot(client, machine_id, 1, T1)
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw

    expected = {
        "machine_id": machine_id,
        "limit": 10,
        "records": [{"group": "decision_basis", "record": document}],
        "next_cursor": None,
        "has_more": False,
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    # No NaN/Infinity-style non-finite tokens.
    json.loads(raw, parse_constant=_reject_non_finite)


def test_utf8_content_is_round_tripped_byte_stably(client, machine_id):
    # Stored document text is emitted verbatim and must survive as compact
    # UTF-8 JSON without ASCII escaping.
    insert_snapshot(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        document = json.loads(
            conn.execute(
                text(
                    "SELECT document FROM authorization_decision_basis "
                    "WHERE event_id = :id"
                ).bindparams(id=rid(1))
            ).scalar_one()
        )
        document["event_summary"]["resource"] = "res/対象-★"
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :value "
                "WHERE event_id = :id"
            ).bindparams(
                value=json.dumps(
                    document, ensure_ascii=False, separators=(",", ":")
                ),
                id=rid(1),
            )
        )

    first = client.get(changes_url(machine_id, limit=10)).content
    second = client.get(changes_url(machine_id, limit=10)).content
    assert first == second
    assert "対象".encode("utf-8") in first
    body = json.loads(first)
    assert (
        body["records"][0]["record"]["event_summary"]["resource"]
        == "res/対象-★"
    )


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        created = first.post(
            "/machines",
            json={
                "external_id": "machine-1",
                "display_name": "Machine One",
                "public_key": "key-1",
            },
        ).json()
        machine_id = created["id"]
        for n, stamp in ((1, T0), (2, T1), (3, T2)):
            insert_snapshot(first, machine_id, n, stamp)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [r["record"]["event_summary"]["id"] for r in response.json()["records"]] == [
        rid(3)
    ]
