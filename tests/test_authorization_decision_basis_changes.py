"""Tests for the stable, read-only incremental decision-basis snapshot query.

Covers `GET /machines/{machine_id}/authorization-decision-events/
decision-basis/changes`:

- validation before any machine or snapshot is read: ``bad_limit`` for a
  missing, blank, fractional, boolean, non-decimal, or out-of-range
  ``limit``; ``invalid_cursor`` for an empty, shape-mismatching, or
  unlocatable ``cursor``; ``invalid_query`` for unknown parameters,
  repeated names, or a carried request body — all 422 and taking priority
  over the machine lookup;
- GET-only ``405`` without reading snapshots, ``404 not_found`` for a
  missing machine carrying no records, and ``500 internal_error`` with no
  partial page when the snapshots or the machine cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where each record is tagged ``decision_basis`` and then
  carries exactly the five snapshot groups byte-for-byte as the single
  snapshot query emits them, and only the path machine's snapshots;
- only events that already have a snapshot enter the feed; a historical
  event without a snapshot is neither fabricated nor pageable;
- keyset pagination in (actual UTC created_at instant, event id) order,
  an exact-second record before a fractional-second record of the same
  second, exclusive cursors, ``next_cursor``/``has_more`` on full,
  partial, empty, and past-the-end pages;
- a snapshot with an unparseable ``created_at`` kept verbatim, sorted
  last, and still pageable (including a damaged stamp text containing
  ``|``); a damaged stored document is never repaired or skipped;
- byte-identical repeat pages, no resurfacing after earlier inserts,
  strict read-only behavior, compact newline-terminated JSON with no
  non-finite tokens, and persistence across a restart.
"""
import json
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T0_FRAC = "2026-03-01T00:00:00.500000Z"

ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]
GROUP_KEYS = [
    "event_summary",
    "status_basis",
    "declaration_basis",
    "policy_candidates",
    "decision",
]

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


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


def basis_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/decision-basis"
    )


def set_basis_created_at(client, event_id, stamp):
    """Retime one captured snapshot row directly, keeping its document."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET created_at = :at "
                "WHERE event_id = :id"
            ).bindparams(at=stamp, id=event_id)
        )


def delete_snapshot(client, event_id):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event_id)
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


def capture_event(client, machine_id, stamp=None, *, resource="res/x"):
    """Create an event+snapshot through the real write path, optional retime."""
    event = record_event(client, machine_id, resource=resource).json()
    if stamp is not None:
        set_basis_created_at(client, event["id"], stamp)
    return event


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
def test_bad_limit_is_422(client, query):
    machine_id = create_machine(client)
    response = client.get(f"{changes_path(machine_id)}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}


def test_boundary_limits_are_accepted(client):
    machine_id = create_machine(client)
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
        f"|event-id",      # missing created-at text
    ],
)
def test_malformed_cursor_is_422(client, cursor):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


@pytest.mark.parametrize(
    "cursor",
    [
        # Well-shaped positions that name no stored basis snapshot; the
        # timestamp segment is taken as original text, so a non-RFC text
        # still passes the shape check and is rejected when unlocatable.
        f"{T0}|no-such-event",
        "garbage-stamp|some-event",
        f"2026-13-01T00:00:00Z|some-event",
        f"{T0}||weird",
    ],
)
def test_unlocatable_cursor_is_422(client, cursor):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # A position naming the machine's own event that has *no snapshot* is
    # also unlocatable: historical snapshot-less events never enter.
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{event['id']}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}

    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_snapshot_is_accepted(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = capture_event(client, machine_id, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{event['id']}")
    )
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|missing")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_naming_another_machines_snapshot_is_422(client):
    one = create_machine(client, "machine-1")
    two = create_machine(client, "machine-2")
    for machine_id in (one, two):
        declare(client, machine_id)
    create_rule(client)
    event_two = capture_event(client, two, T0)

    response = client.get(
        changes_url(one, limit=10, cursor=f"{T0}|{event_two['id']}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client):
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
            ("cursor", f"{T0}|a"),
            ("cursor", f"{T1}|b"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_reads(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET", f"{changes_path(machine_id)}?limit=10",
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_zero_content_length_get_is_accepted(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET", changes_url(machine_id, limit=10), content=b""
    )
    assert response.status_code == 200


def test_parameter_errors_take_priority_over_machine_lookup(client):
    # No machine exists; every parameter error is still its 422.
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
    # A well-shaped cursor names no basis row of a missing machine, so it is
    # an unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(MISSING_MACHINE, limit=1, cursor=f"{T0}|some-event")
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_snapshots(client):
    machine_id = create_machine(client)
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


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted_without_reading_snapshots(client, method):
    machine_id = create_machine(client)
    # Drop every table the query could read; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
        conn.execute(text("DROP TABLE machines"))
    response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
    assert response.status_code == 405
    assert b"records" not in response.content


def test_missing_machine_is_404_with_no_records(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content


def test_snapshot_read_failure_is_500_with_no_partial_page(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_page(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_unlocatable_cursor_422_emits_no_page_and_precedes_404(client):
    # Cursor positioning reads the (empty) snapshot set first; even once the
    # machine no longer exists, an unlocatable cursor fails as 422 with no
    # page, never as 404 or 500.
    machine_id = create_machine(client, "machine-9")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM machines WHERE id = :id").bindparams(id=machine_id)
        )
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|ghost")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, contents, ordering, paging
# --------------------------------------------------------------------------- #


def test_empty_database_returns_full_empty_page(client):
    machine_id = create_machine(client)
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


def test_single_page_smaller_than_limit(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    first = capture_event(client, machine_id, T1)
    second = capture_event(client, machine_id, T3)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["event_summary"]["id"] for r in body["records"]] == [
        first["id"], second["id"]
    ]


def test_each_record_is_tagged_and_carries_the_five_groups(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = capture_event(client, machine_id)

    body = client.get(changes_url(machine_id, limit=10)).json()
    (record,) = body["records"]
    assert list(record.keys()) == ["kind"] + GROUP_KEYS
    assert record["kind"] == "decision_basis"
    assert record["event_summary"]["id"] == event["id"]


def test_embedded_groups_are_byte_identical_to_the_snapshot_query(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    event = record_event(client, machine_id, resource="res/abc").json()

    snapshot = client.get(basis_url(machine_id, event["id"])).content.strip()
    body = client.get(changes_url(machine_id, limit=10)).content
    # The five groups are the stored object verbatim; only the kind tag is
    # prepended and the envelope wrapped around the records array.
    embedded = b'{"kind":"decision_basis",' + snapshot[1:]
    assert embedded in body
    record = json.loads(body)["records"][0]
    assert json.dumps(
        {key: record[key] for key in GROUP_KEYS},
        ensure_ascii=False, separators=(",", ":"),
    ).encode() == snapshot


def test_only_events_with_snapshots_enter_the_feed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    first = capture_event(client, machine_id, T0)
    legacy = record_event(client, machine_id, resource="res/legacy").json()
    third = capture_event(client, machine_id, T2)

    # Simulate a pre-feature event: its event row exists, its snapshot does
    # not. It must never enter the feed or be fabricated.
    delete_snapshot(client, legacy["id"])

    body = client.get(changes_url(machine_id, limit=10)).json()
    ids = [r["event_summary"]["id"] for r in body["records"]]
    assert ids == [first["id"], third["id"]]
    assert legacy["id"] not in ids
    assert body["has_more"] is False

    # The snapshot-less event's own position is not a valid cursor.
    response = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{legacy['id']}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_page_size_has_no_more(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T0)
    capture_event(client, machine_id, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_snapshot_in_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    by_stamp = {}
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        by_stamp[stamp] = capture_event(
            client, machine_id, stamp, resource=f"res/{n}"
        )["id"]

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [r["event_summary"]["id"] for r in records] == [
        by_stamp[T0], by_stamp[T1], by_stamp[T2], by_stamp[T3], by_stamp[T4]
    ]


def test_page_cursors_are_exclusive(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    e0 = capture_event(client, machine_id, T0)
    e1 = capture_event(client, machine_id, T1)
    e2 = capture_event(client, machine_id, T2)
    e3 = capture_event(client, machine_id, T3)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["event_summary"]["id"] for r in first["records"]] == [
        e0["id"], e1["id"]
    ]
    assert first["has_more"] is True
    assert first["next_cursor"] == f"{T1}|{e1['id']}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["event_summary"]["id"] for r in second["records"]] == [
        e2["id"], e3["id"]
    ]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_is_unlocatable_422(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T4}|ghost-event")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    fractional = capture_event(client, machine_id, T0_FRAC, resource="res/f")
    exact = capture_event(client, machine_id, T0, resource="res/e")

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["event_summary"]["id"] for r in body["records"]] == [
        exact["id"], fractional["id"]
    ]


def test_same_instant_tie_breaks_by_event_id_and_cursor(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    # Real generated UUIDs; retime all three, then read the ordered ids back.
    capture_event(client, machine_id, T3, resource="res/later")
    capture_event(client, machine_id, T2, resource="res/a")
    capture_event(client, machine_id, T2, resource="res/b")

    first = client.get(changes_url(machine_id, limit=10)).json()
    tied = [r["event_summary"]["id"] for r in first["records"][:2]]
    later_id = first["records"][2]["event_summary"]["id"]
    assert tied == sorted(tied)

    page_one = client.get(changes_url(machine_id, limit=1)).json()
    assert page_one["records"][0]["event_summary"]["id"] == tied[0]
    assert page_one["next_cursor"] == f"{T2}|{tied[0]}"
    page_two = client.get(
        changes_url(machine_id, limit=1, cursor=page_one["next_cursor"])
    ).json()
    assert page_two["records"][0]["event_summary"]["id"] == tied[1]
    page_three = client.get(
        changes_url(machine_id, limit=1, cursor=page_two["next_cursor"])
    ).json()
    assert page_three["records"][0]["event_summary"]["id"] == later_id
    assert page_three["next_cursor"] is None


def test_only_the_path_machines_snapshots_are_returned(client):
    one = create_machine(client, "machine-1")
    two = create_machine(client, "machine-2")
    for machine_id in (one, two):
        declare(client, machine_id)
    create_rule(client)
    e1 = capture_event(client, one, T0, resource="a")
    e2 = capture_event(client, two, T1, resource="b")
    e3 = capture_event(client, one, T2, resource="c")

    body_one = client.get(changes_url(one, limit=10)).json()
    assert [r["event_summary"]["id"] for r in body_one["records"]] == [
        e1["id"], e3["id"]
    ]
    assert all(
        r["event_summary"]["machine_id"] == one for r in body_one["records"]
    )
    body_two = client.get(changes_url(two, limit=10)).json()
    assert [r["event_summary"]["id"] for r in body_two["records"]] == [e2["id"]]


def test_real_registered_events_are_pageable(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    for resource in ("res/a", "res/b", "res/c"):
        assert record_event(client, machine_id, resource=resource).status_code == 201

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 2
    assert [r["event_summary"]["resource"] for r in records] == [
        "res/a", "res/b", "res/c"
    ]
    assert all(r["kind"] == "decision_basis" for r in records)


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    middle = capture_event(client, machine_id, T1)
    damaged = record_event(client, machine_id, resource="res/d").json()
    set_basis_created_at(client, damaged["id"], "not-a-timestamp")
    first = capture_event(client, machine_id, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["event_summary"]["id"] for r in rows] == [
        first["id"], middle["id"], damaged["id"]
    ]


def test_multiple_unparseable_stamps_sort_by_event_id(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    zzz = record_event(client, machine_id, resource="res/z").json()
    set_basis_created_at(client, zzz["id"], "zzz")
    yyy = record_event(client, machine_id, resource="res/y").json()
    set_basis_created_at(client, yyy["id"], "yyy")
    good = capture_event(client, machine_id, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["event_summary"]["id"] for r in rows] == [
        good["id"],
        *sorted([yyy["id"], zzz["id"]]),
    ]


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    good = capture_event(client, machine_id, T0)
    weird = record_event(client, machine_id, resource="res/w").json()
    set_basis_created_at(client, weird["id"], "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["event_summary"]["id"] for r in first["records"]] == [good["id"]]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{good['id']}"

    second = client.get(
        changes_path(machine_id) + "?limit=1&cursor=" + quote(cursor, safe="")
    )
    assert second.status_code == 200
    body = second.json()
    assert [r["event_summary"]["id"] for r in body["records"]] == [weird["id"]]
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_damaged_document_is_emitted_verbatim_not_repaired_or_skipped(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = capture_event(client, machine_id, T0)

    # Tamper with the stored document only (still a JSON object): the
    # changes feed re-emits the stored five groups byte-for-byte, exactly
    # like the single snapshot query, and keeps the row in the page.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id"
            ).bindparams(
                doc='{"event_summary":{"extra":true}}', id=event["id"]
            )
        )
    raw = client.get(basis_url(machine_id, event["id"])).content.strip()

    body = client.get(changes_url(machine_id, limit=10)).json()
    (record,) = body["records"]
    assert record["kind"] == "decision_basis"
    # The damaged document carried only a misshapen event_summary; the feed
    # keeps exactly that group verbatim and neither repairs nor pads it.
    assert list(record) == ["kind", "event_summary"]
    assert record["event_summary"] == {"extra": True}
    embedded = json.dumps(
        {key: value for key, value in record.items() if key != "kind"},
        ensure_ascii=False, separators=(",", ":"),
    )
    assert embedded == raw.decode()


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    for stamp in (T0, T1, T2, T3):
        capture_event(client, machine_id, stamp)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    one = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    two = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    assert one == two


def test_new_earlier_snapshots_do_not_revisit_returned_pages(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T2, resource="res/1")
    later = capture_event(client, machine_id, T4, resource="res/2")

    first = client.get(changes_url(machine_id, limit=1)).json()
    old_cursor = first["next_cursor"]

    # Capture snapshots sorting before the cursor position and between.
    capture_event(client, machine_id, T1, resource="res/3")
    middle = capture_event(client, machine_id, T3, resource="res/4")

    second = client.get(
        changes_url(machine_id, limit=10, cursor=old_cursor)
    ).json()
    assert [r["event_summary"]["id"] for r in second["records"]] == [
        middle["id"], later["id"]
    ]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert len(records) == 4


def test_has_more_reflects_records_after_position_only(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    e0 = capture_event(client, machine_id, T0)
    e1 = capture_event(client, machine_id, T1)
    capture_event(client, machine_id, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{e1['id']}")
    ).json()
    assert len(body["records"]) == 1
    assert body["has_more"] is False

    body = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{e0['id']}")
    ).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False


def test_query_is_read_only(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = capture_event(client, machine_id, T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(
                conn.execute(
                    text("SELECT * FROM authorization_decision_basis")
                )
            )

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T4}|ghost"))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T1}|{event['id']}"))
    assert table_state() == before


def _reject_non_finite(marker):
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    capture_event(client, machine_id, T1)

    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw
    body = json.loads(raw, parse_constant=_reject_non_finite)
    assert list(body) == ENVELOPE_KEYS
    assert list(body["records"][0]) == ["kind"] + GROUP_KEYS
    assert all(
        isinstance(record["decision"]["allowed"], bool)
        for record in body["records"]
    )


def test_utf8_document_content_round_trips_byte_stably(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    event = record_event(
        client, machine_id, resource="res/対象-★"
    ).json()

    first = client.get(changes_url(machine_id, limit=10)).content
    second = client.get(changes_url(machine_id, limit=10)).content
    assert first == second
    assert "対象".encode("utf-8") in first
    body = json.loads(first)
    assert body["records"][0]["event_summary"]["resource"] == "res/対象-★"
    assert event["id"] == body["records"][0]["event_summary"]["id"]


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        capture_event(first, machine_id, T0, resource="res/0")
        e1 = capture_event(first, machine_id, T1, resource="res/1")
        e2 = capture_event(first, machine_id, T2, resource="res/2")
        cursor = first.get(changes_url(machine_id, limit=1)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second_client:
        response = second_client.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        )

    assert response.status_code == 200
    assert response.content == expected_second
    ids = [r["event_summary"]["id"] for r in response.json()["records"]]
    assert ids == [e1["id"], e2["id"]]
