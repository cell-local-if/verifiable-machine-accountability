"""Tests for the stable, read-only incremental incident status history query.

Covers `GET /machines/{machine_id}/incident-status-events/changes`:

- validation before any machine or status-history record is read:
  ``bad_limit`` for a missing, blank, fractional, boolean, non-decimal, or
  out-of-range ``limit``; ``invalid_cursor`` for an empty, non-string,
  shape-mismatching, or unlocatable ``cursor``; ``invalid_query`` for
  unknown parameters, repeated names, or a carried request body — all 422
  and taking priority over the machine lookup;
- GET-only ``405`` (including ``HEAD``) without reading records, ``404
  not_found`` for a missing machine carrying no records, and ``500
  internal_error`` with no partial page when the records cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where each item is ``{group: "status_history", record}`` with
  exactly the seven compliance-export fields of the stored status
  transition, and only the path machine's records;
- keyset pagination in (actual UTC created_at instant, category, record
  id) order, an exact-second record before a fractional-second record of
  the same second, exclusive cursors, ``next_cursor``/``has_more`` on
  full, partial, empty, and past-the-end pages;
- a record with an unparseable ``created_at`` kept verbatim, sorted last,
  and still pageable (including a damaged stamp text containing ``|``);
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

RECORD_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "from_status",
    "to_status",
    "created_at",
]
ITEM_KEYS = ["group", "record"]
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
    return f"/machines/{machine_id}/incident-status-events/changes"


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + cursor
    return f"{changes_path(machine_id)}?{query}"


def insert_status_event(
    client,
    machine_id,
    n,
    created_at,
    *,
    event_id=None,
    incident_id=None,
    from_status="open",
    to_status="acknowledged",
):
    """Insert a status-history row directly with a fixed id and timestamp.

    The chain columns are left null: the changes query emits only the
    seven stored transition fields and never recomputes anything.
    """
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": event_id if event_id is not None else rid(1000 + n),
        "incident_id": incident_id if incident_id is not None else rid(2000 + n),
        "from_status": from_status,
        "to_status": to_status,
        "created_at": created_at,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events "
                "(id, machine_id, event_id, incident_id, from_status, "
                "to_status, created_at) "
                "VALUES "
                "(:id, :machine_id, :event_id, :incident_id, :from_status, "
                ":to_status, :created_at)"
            ),
            values,
        )
    return values


def fetch_all_pages(client, machine_id, limit):
    """Walk the cursor chain from the start and return every item."""
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
    for limit in (1, 100):
        response = client.get(changes_url(machine_id, limit=limit))
        assert response.status_code == 200
        assert response.json()["limit"] == limit


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "no-separator",
        "|",
        f"|{rid(1)}",
        f"{T0}|",
        "||",
    ],
)
def test_malformed_cursor_is_422(client, machine_id, cursor):
    response = client.get(f"{changes_path(machine_id)}?limit=1&cursor={cursor}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


@pytest.mark.parametrize(
    "cursor",
    [
        f"{T0}|{rid(1)}",                 # nothing stored at all
        "2026-03-01T00:00:00Z|not-a-record",
    ],
)
def test_unlocatable_cursor_is_422(client, machine_id, cursor):
    insert_status_event(client, machine_id, 5, T2)
    response = client.get(
        f"{changes_path(machine_id)}?limit=1&cursor={_quote(cursor)}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_record_is_accepted(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    response = client.get(changes_url(machine_id, limit=1, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 200
    assert [item["record"]["id"] for item in response.json()["records"]] == [rid(2)]


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    response = client.get(changes_url(machine_id, limit=1, cursor=f"{T0}|{rid(2)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_naming_another_machines_record_is_422(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_status_event(client, other["id"], 1, T0)
    response = client.get(changes_url(machine_id, limit=1, cursor=f"{T0}|{rid(1)}"))
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
    assert response.json() == {"error": {"code": "bad_limit"}}

    response = client.get(
        changes_path(machine_id),
        params=[("limit", "1"), ("cursor", f"{T0}|{rid(1)}"),
                ("cursor", f"{T1}|{rid(2)}")],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


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
    # A well-shaped cursor names no record of a missing machine, so it is
    # an unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(MISSING_MACHINE, limit=1, cursor=f"{T0}|{rid(1)}")
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_records(client, machine_id):
    # With the table dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE incident_status_events"))

    assert client.get(changes_path(machine_id)).status_code == 422
    assert client.get(f"{changes_path(machine_id)}?limit=0").json() == {
        "error": {"code": "bad_limit"}
    }
    assert client.get(f"{changes_path(machine_id)}?limit=1&cursor=garbage").json() == {
        "error": {"code": "invalid_cursor"}
    }
    assert client.get(f"{changes_path(machine_id)}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }


def test_only_get_is_accepted_without_reading_records(client, machine_id):
    # Drop the table so any record read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE incident_status_events"))
    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
        assert response.status_code == 405


def test_missing_machine_is_404_with_no_records(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content


def test_read_failure_is_500_with_no_partial_page(client, machine_id):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE incident_status_events"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_page(client, machine_id):
    # The machine lookup happens after the records are read; a failure there
    # is still a read-layer fault reported with the internal_error envelope,
    # never the framework default body and never a partial page.
    insert_status_event(client, machine_id, 1, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
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


def test_single_page_smaller_than_limit(client, machine_id):
    insert_status_event(client, machine_id, 1, T1)
    insert_status_event(client, machine_id, 2, T3)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [item["record"]["id"] for item in body["records"]] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client, machine_id):
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        insert_status_event(client, machine_id, n, stamp)

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [item["record"]["id"] for item in records] == [
        rid(2), rid(4), rid(3), rid(5), rid(1),
    ]
    assert [item["record"]["created_at"] for item in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_status_event(client, machine_id, n, stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text verbatim, after the row.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [item["record"]["id"] for item in second["records"]] == [rid(3), rid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page; nothing is reread.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_at_the_last_record_returns_an_empty_last_page(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}")).json()
    assert body["records"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_cursor_past_the_end_is_unlocatable_422(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)

    # A non-null cursor always names a page's last stored record, so a
    # position beyond the table cannot be located and is invalid rather
    # than an empty page.
    response = client.get(changes_url(machine_id, limit=10, cursor=f"{T5}|{rid(999)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_status_event(client, machine_id, 2, T0_FRAC)
    insert_status_event(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(1), rid(2)]
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_record_id_and_cursor(client, machine_id):
    insert_status_event(client, machine_id, 30, T2)
    insert_status_event(client, machine_id, 20, T2)
    insert_status_event(client, machine_id, 10, T3)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(20)]
    assert first["next_cursor"] == f"{T2}|{rid(20)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [item["record"]["id"] for item in second["records"]] == [rid(30)]

    third = client.get(
        changes_url(machine_id, limit=1, cursor=second["next_cursor"])
    ).json()
    assert [item["record"]["id"] for item in third["records"]] == [rid(10)]
    assert third["next_cursor"] is None


def test_items_carry_the_group_tag_and_complete_record(client, machine_id):
    insert_status_event(
        client, machine_id, 7, "2026-03-01T00:00:00.250Z",
        event_id=rid(700), incident_id=rid(800),
        from_status="acknowledged", to_status="resolved",
    )
    (item,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert list(item.keys()) == ITEM_KEYS
    assert item["group"] == "status_history"
    assert list(item["record"].keys()) == RECORD_KEYS
    assert item["record"] == {
        "id": rid(7),
        "machine_id": machine_id,
        "event_id": rid(700),
        "incident_id": rid(800),
        "from_status": "acknowledged",
        "to_status": "resolved",
        "created_at": "2026-03-01T00:00:00.250Z",
    }


def test_only_the_path_machines_records_are_returned(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, other["id"], 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(1), rid(3)]
    assert all(
        item["record"]["machine_id"] == machine_id for item in body["records"]
    )

    other_body = client.get(changes_url(other["id"], limit=10)).json()
    assert [item["record"]["id"] for item in other_body["records"]] == [rid(2)]


def test_changes_pages_match_the_compliance_export_slice(client, machine_id):
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_status_event(client, machine_id, n, stamp)

    exported = client.get(
        f"/machines/{machine_id}/incident-status-events/compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z&to_created_at=2100-01-01T00:00:00Z"
    ).json()["status_history"]
    items, _ = fetch_all_pages(client, machine_id, limit=2)
    assert [item["record"] for item in items] == exported
    assert all(item["group"] == "status_history" for item in items)


def test_real_transitions_are_pageable(client, machine_id):
    event = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    incident = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents",
        json={"incident_type": "fraud", "summary": "suspicious decision"},
    ).json()
    for status in ("acknowledged", "resolved"):
        transitioned = client.post(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event['id']}/incidents/{incident['id']}/status",
            json={"status": status},
        )
        assert transitioned.status_code == 200

    items, _ = fetch_all_pages(client, machine_id, limit=1)
    assert len(items) == 2
    assert all(item["group"] == "status_history" for item in items)
    first, second = (item["record"] for item in items)
    assert (first["from_status"], first["to_status"]) == ("open", "acknowledged")
    assert (second["from_status"], second["to_status"]) == (
        "acknowledged", "resolved",
    )
    assert first["incident_id"] == incident["id"] == second["incident_id"]
    assert first["event_id"] == event["id"] == second["event_id"]


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client, machine_id):
    insert_status_event(client, machine_id, 1, T1)
    insert_status_event(client, machine_id, 2, "not-a-timestamp")
    insert_status_event(client, machine_id, 3, T0)

    items = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [item["record"]["id"] for item in items] == [rid(3), rid(1), rid(2)]
    assert items[-1]["record"]["created_at"] == "not-a-timestamp"


def test_out_of_range_calendar_stamp_sorts_last_verbatim(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, "2026-13-40T99:99:99Z")

    items = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [item["record"]["id"] for item in items] == [rid(1), rid(2)]
    assert items[1]["record"]["created_at"] == "2026-13-40T99:99:99Z"


def test_multiple_unparseable_stamps_sort_by_id(client, machine_id):
    insert_status_event(client, machine_id, 30, "zzz")
    insert_status_event(client, machine_id, 10, "yyy")
    insert_status_event(client, machine_id, 20, T0)

    items = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [item["record"]["id"] for item in items] == [rid(20), rid(10), rid(30)]


def test_damaged_fields_are_emitted_unmodified(client, machine_id):
    insert_status_event(
        client, machine_id, 1, T1,
        event_id="not-an-event", incident_id="",
        from_status="bogus", to_status="",
    )
    (item,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert item["record"]["event_id"] == "not-an-event"
    assert item["record"]["incident_id"] == ""
    assert item["record"]["from_status"] == "bogus"
    assert item["record"]["to_status"] == ""


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(
    client, machine_id
):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1)]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{rid(1)}"

    second = client.request(
        "GET",
        f"{changes_path(machine_id)}?limit=1&cursor=" + _quote(cursor),
    )
    assert second.status_code == 200
    body = second.json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(2)]
    assert body["records"][0]["record"]["created_at"] == "not-a-timestamp|weird"
    assert body["next_cursor"] is None
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_status_event(client, machine_id, n, stamp)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    page_one = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    page_two = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    assert page_one == page_two


def test_new_earlier_inserts_do_not_revisit_returned_pages(client, machine_id):
    insert_status_event(client, machine_id, 1, T2)
    insert_status_event(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert a record sorting before the cursor position and one between.
    insert_status_event(client, machine_id, 3, T1)
    insert_status_event(client, machine_id, 4, T3)

    second = client.get(changes_url(machine_id, limit=10, cursor=old_cursor)).json()
    assert [item["record"]["id"] for item in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    items, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [item["record"]["id"] for item in items] == [
        rid(3), rid(1), rid(4), rid(2),
    ]


def test_has_more_reflects_records_after_position_only(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{rid(2)}")
    ).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_query_is_read_only(client, machine_id):
    insert_status_event(client, machine_id, 1, T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM incident_status_events")))

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
    insert_status_event(
        client, machine_id, 1, T1,
        event_id=rid(100), incident_id=rid(200),
        from_status="open", to_status="acknowledged",
    )
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw

    expected_item = {
        "group": "status_history",
        "record": {
            "id": rid(1),
            "machine_id": machine_id,
            "event_id": rid(100),
            "incident_id": rid(200),
            "from_status": "open",
            "to_status": "acknowledged",
            "created_at": T1,
        },
    }
    expected = {
        "machine_id": machine_id,
        "limit": 10,
        "records": [expected_item],
        "next_cursor": None,
        "has_more": False,
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    # No NaN/Infinity/-0.0-style non-finite or float content.
    json.loads(raw, parse_constant=_reject_non_finite)


def test_utf8_content_is_round_tripped_byte_stably(client, machine_id):
    # Stored text is emitted verbatim and must survive as compact UTF-8
    # JSON without ASCII escaping.
    insert_status_event(client, machine_id, 1, T1, from_status="打开-★")
    first = client.get(changes_url(machine_id, limit=10)).content
    second = client.get(changes_url(machine_id, limit=10)).content
    assert first == second
    assert "打开".encode("utf-8") in first
    body = json.loads(first)
    assert body["records"][0]["record"]["from_status"] == "打开-★"


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
            insert_status_event(first, machine_id, n, stamp)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [item["record"]["id"] for item in response.json()["records"]] == [rid(3)]
