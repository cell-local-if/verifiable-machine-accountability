"""Tests for the stable, read-only incremental machine incident-view query.

Covers `GET /machines/{machine_id}/authorization-decision-events/incidents/changes`:

- validation before any machine or record is read: ``bad_limit`` for a
  missing, blank, fractional, boolean, non-decimal, or out-of-range
  ``limit``; ``invalid_cursor`` for an empty, non-string,
  shape-mismatching, or unlocatable ``cursor``; ``invalid_query`` for
  unknown parameters, repeated names, or a carried request body — all 422
  and taking priority over the machine lookup;
- GET-only ``405`` (including ``HEAD``) without reading records,
  ``404 not_found`` for a missing machine carrying no records, and
  ``500 internal_error`` with no partial page when the records cannot be
  read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where every item is ``{group, record}`` with the fixed group
  tag (``incidents``, ``status_history``, or
  ``responsibility_assignments``) and the complete record exactly as the
  corresponding endpoint emits it, and only the path machine's records;
- keyset pagination merging the three record groups in (actual UTC
  created_at instant, group tag, record id) order, an exact-second record
  before a fractional-second record of the same second, exclusive cursors,
  and ``next_cursor``/``has_more`` on full, partial, empty, and
  past-the-end pages;
- a record with an unparseable ``created_at`` kept verbatim, sorted last,
  and still pageable (including a damaged stamp text containing ``|``);
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
T5 = "2026-03-01T00:00:05Z"
T0_FRAC = "2026-03-01T00:00:00.500000Z"

HASH_A = "a" * 64
HASH_B = "b" * 64

INCIDENT_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_type",
    "summary",
    "status",
    "created_at",
]
STATUS_EVENT_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "from_status",
    "to_status",
    "created_at",
    "previous_status_event_id",
    "content_hash",
    "chain_hash",
]
ASSIGNMENT_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party",
    "role",
    "created_at",
    "previous_assignment_id",
    "content_hash",
    "chain_hash",
]
ITEM_KEYS = ["group", "record"]
ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]

INCIDENTS = "incidents"
STATUS_HISTORY = "status_history"
ASSIGNMENTS = "responsibility_assignments"
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
        f"/machines/{machine_id}/authorization-decision-events/incidents/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + quote(cursor, safe="")
    return f"{changes_path(machine_id)}?{query}"


def insert_incident(
    client,
    machine_id,
    n,
    created_at,
    *,
    event_id=None,
    incident_type="breach",
    summary="something happened",
    status="open",
):
    """Insert an incident row directly with a fixed id and timestamp.

    The default event id varies with ``n`` so several inserts in one test
    never collide on the ``(event_id, incident_type, summary)`` uniqueness
    constraint.
    """
    if event_id is None:
        event_id = rid(900 + n)
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": event_id,
        "incident_type": incident_type,
        "summary": summary,
        "status": status,
        "created_at": created_at,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_incidents "
                "(id, machine_id, event_id, incident_type, summary, status, "
                "created_at) "
                "VALUES "
                "(:id, :machine_id, :event_id, :incident_type, :summary, "
                ":status, :created_at)"
            ),
            values,
        )
    return values


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
    previous_status_event_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert a status-event row directly with a fixed id and timestamp.

    The chain columns are supplied explicitly and non-null so the startup
    backfill (which only rewrites when a chain hash is missing) leaves the
    row exactly as given; the changes query never recomputes them.
    """
    if event_id is None:
        event_id = rid(900)
    if incident_id is None:
        incident_id = rid(901)
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": event_id,
        "incident_id": incident_id,
        "from_status": from_status,
        "to_status": to_status,
        "created_at": created_at,
        "previous_status_event_id": previous_status_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events "
                "(id, machine_id, event_id, incident_id, from_status, "
                "to_status, created_at, previous_status_event_id, "
                "content_hash, chain_hash) "
                "VALUES "
                "(:id, :machine_id, :event_id, :incident_id, :from_status, "
                ":to_status, :created_at, :previous_status_event_id, "
                ":content_hash, :chain_hash)"
            ),
            values,
        )
    return values


def insert_assignment(
    client,
    machine_id,
    n,
    created_at,
    *,
    event_id=None,
    incident_id=None,
    party="team-a",
    role="owner",
    previous_assignment_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert a responsibility-assignment row directly with a fixed id.

    The chain columns are supplied explicitly and non-null so the startup
    backfill leaves the row exactly as given; the changes query never
    recomputes them. The default incident id varies with ``n`` so several
    inserts in one test never collide on the ``(incident_id, party, role)``
    uniqueness constraint.
    """
    if event_id is None:
        event_id = rid(900)
    if incident_id is None:
        incident_id = rid(700 + n)
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": event_id,
        "incident_id": incident_id,
        "party": party,
        "role": role,
        "created_at": created_at,
        "previous_assignment_id": previous_assignment_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments "
                "(id, machine_id, event_id, incident_id, party, role, "
                "created_at, previous_assignment_id, content_hash, "
                "chain_hash) "
                "VALUES "
                "(:id, :machine_id, :event_id, :incident_id, :party, :role, "
                ":created_at, :previous_assignment_id, :content_hash, "
                ":chain_hash)"
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


def create_real_records(client, machine_id):
    """Drive one incident through registration, transitions, and assignment."""
    event_id = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/x"},
    ).json()["id"]
    incident_id = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": "breach", "summary": "something happened"},
    ).json()["id"]
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status",
        json={"status": "acknowledged"},
    )
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status",
        json={"status": "resolved"},
    )
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments",
        json={"party": "team-a", "role": "owner"},
    )
    return event_id, incident_id


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
        "",                       # empty value
        "not-a-cursor",           # no separator
        f"{T0}|{INCIDENTS}",      # one separator only
        "||",                     # all segments empty
        f"{T0}||{rid(1)}",        # missing group
        f"{T0}|{INCIDENTS}|",     # missing record id
        f"|{INCIDENTS}|{rid(1)}",  # missing created-at text
    ],
)
def test_malformed_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


@pytest.mark.parametrize(
    "cursor",
    [
        # Well-shaped positions that name no stored record; the timestamp
        # segment is taken as original text, so a non-RFC text still passes
        # the shape check and is rejected when it cannot be located.
        f"{T0}|{INCIDENTS}|{rid(999)}",
        "garbage-stamp|incidents|nonempty-id",
        f"2026-13-01T00:00:00Z|{INCIDENTS}|{rid(999)}",
        # A group tag this query never emits cannot be located either.
        f"{T0}|events|{rid(1)}",
    ],
)
def test_unlocatable_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_record_is_accepted(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{INCIDENTS}|{rid(1)}")
    )
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{INCIDENTS}|{rid(2)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_with_the_wrong_group_is_422(client, machine_id):
    # The (created_at, id) pair exists, but under another group tag: the
    # position is located by the full triple, so it cannot be positioned.
    insert_incident(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{STATUS_HISTORY}|{rid(1)}")
    )
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
    ).json()["id"]
    insert_incident(client, other, 1, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|{INCIDENTS}|{rid(1)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client, machine_id):
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client, machine_id):
    # A repeated name is an unknown-shape query, so regardless of which name
    # is repeated it reports invalid_query (never silently one occurrence).
    response = client.get(
        changes_path(machine_id), params=[("limit", "1"), ("limit", "2")]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        changes_path(machine_id),
        params=[
            ("limit", "1"),
            ("cursor", f"{T0}|{INCIDENTS}|{rid(1)}"),
            ("cursor", f"{T1}|{INCIDENTS}|{rid(2)}"),
        ],
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
    # A well-shaped cursor names no record of a missing machine, so it is an
    # unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(MISSING_MACHINE, limit=1, cursor=f"{T0}|{INCIDENTS}|{rid(1)}")
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_records(client, machine_id):
    # With the tables dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))
        conn.execute(text("DROP TABLE incident_status_events"))
        conn.execute(text("DROP TABLE incident_responsibility_assignments"))

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
def test_only_get_is_accepted_without_reading_records(client, machine_id, method):
    # Drop the tables so any record read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))
        conn.execute(text("DROP TABLE incident_status_events"))
        conn.execute(text("DROP TABLE incident_responsibility_assignments"))
    response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
    assert response.status_code == 405


def test_missing_machine_is_404_with_no_records(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content
    assert b"has_more" not in response.content


@pytest.mark.parametrize(
    "table",
    [
        "authorization_decision_incidents",
        "incident_status_events",
        "incident_responsibility_assignments",
    ],
)
def test_read_failure_is_500_with_no_partial_page(client, machine_id, table):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_page(client, machine_id):
    # The machine lookup happens after the records are read; a failure there
    # is still a read-layer fault with the internal_error envelope, never the
    # framework default body and never a partial page.
    insert_incident(client, machine_id, 1, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, merging, ordering, paging
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


def test_the_three_groups_merge_in_instant_group_id_order(client, machine_id):
    insert_assignment(client, machine_id, 1, T2)
    insert_status_event(client, machine_id, 2, T0)
    insert_incident(client, machine_id, 3, T1)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [(item["group"], item["record"]["id"]) for item in body["records"]] == [
        (STATUS_HISTORY, rid(2)),
        (INCIDENTS, rid(3)),
        (ASSIGNMENTS, rid(1)),
    ]


def test_same_instant_tie_breaks_by_group_tag_then_id(client, machine_id):
    # One instant shared by all three groups: the fixed category tag orders
    # them (incidents < responsibility_assignments < status_history), and
    # the record id orders records inside one group.
    insert_status_event(client, machine_id, 1, T1)
    insert_assignment(client, machine_id, 3, T1)
    insert_incident(client, machine_id, 2, T1)
    insert_incident(client, machine_id, 4, T1)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [(item["group"], item["record"]["id"]) for item in body["records"]] == [
        (INCIDENTS, rid(2)),
        (INCIDENTS, rid(4)),
        (ASSIGNMENTS, rid(3)),
        (STATUS_HISTORY, rid(1)),
    ]


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_incident(client, machine_id, 2, T0_FRAC)
    insert_incident(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(1), rid(2)]
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client, machine_id):
    insert_incident(client, machine_id, 1, T4)
    insert_status_event(client, machine_id, 2, T0)
    insert_assignment(client, machine_id, 3, T2)
    insert_incident(client, machine_id, 4, T1)
    insert_status_event(client, machine_id, 5, T3)

    items, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [(item["group"], item["record"]["id"]) for item in items] == [
        (STATUS_HISTORY, rid(2)),
        (INCIDENTS, rid(4)),
        (ASSIGNMENTS, rid(3)),
        (STATUS_HISTORY, rid(5)),
        (INCIDENTS, rid(1)),
    ]


def test_page_cursors_are_exclusive(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    insert_assignment(client, machine_id, 3, T2)
    insert_incident(client, machine_id, 4, T3)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text and the group verbatim,
    # after the row.
    assert first["next_cursor"] == f"{T1}|{STATUS_HISTORY}|{rid(2)}"

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


def test_cursor_past_the_end_is_unlocatable_422(client, machine_id):
    insert_incident(client, machine_id, 1, T0)

    # A non-null cursor always names a page's last stored record, so a
    # position beyond the table cannot be located and is invalid rather than
    # an empty page.
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|{INCIDENTS}|{rid(999)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_page_size_has_no_more(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_every_item_carries_the_group_wrapper(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    insert_assignment(client, machine_id, 3, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [item["group"] for item in body["records"]] == [
        INCIDENTS,
        STATUS_HISTORY,
        ASSIGNMENTS,
    ]
    for item in body["records"]:
        assert list(item.keys()) == ITEM_KEYS


def test_records_carry_exactly_their_complete_endpoint_fields(
    client, machine_id
):
    insert_incident(
        client, machine_id, 1, T0,
        event_id=rid(5), incident_type="outage", summary="it broke",
        status="acknowledged",
    )
    insert_status_event(
        client, machine_id, 2, T1,
        event_id=rid(5), incident_id=rid(1),
        from_status="open", to_status="acknowledged",
        previous_status_event_id=None, content_hash=HASH_A, chain_hash=HASH_B,
    )
    insert_assignment(
        client, machine_id, 3, T2,
        event_id=rid(5), incident_id=rid(1), party="team-b", role="reviewer",
        previous_assignment_id=None, content_hash=HASH_B, chain_hash=HASH_A,
    )

    records = {
        item["group"]: item["record"]
        for item in client.get(changes_url(machine_id, limit=10)).json()["records"]
    }
    assert list(records[INCIDENTS].keys()) == INCIDENT_KEYS
    assert records[INCIDENTS] == {
        "id": rid(1),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_type": "outage",
        "summary": "it broke",
        "status": "acknowledged",
        "created_at": T0,
    }
    assert list(records[STATUS_HISTORY].keys()) == STATUS_EVENT_KEYS
    assert records[STATUS_HISTORY] == {
        "id": rid(2),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_id": rid(1),
        "from_status": "open",
        "to_status": "acknowledged",
        "created_at": T1,
        "previous_status_event_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    assert list(records[ASSIGNMENTS].keys()) == ASSIGNMENT_KEYS
    assert records[ASSIGNMENTS] == {
        "id": rid(3),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_id": rid(1),
        "party": "team-b",
        "role": "reviewer",
        "created_at": T2,
        "previous_assignment_id": None,
        "content_hash": HASH_B,
        "chain_hash": HASH_A,
    }


def test_only_the_path_machines_records_are_returned(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()["id"]
    insert_incident(client, machine_id, 1, T0)
    insert_incident(client, other, 2, T1)
    insert_status_event(client, other, 3, T1)
    insert_assignment(client, other, 4, T1)
    insert_incident(client, machine_id, 5, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(1), rid(5)]
    assert all(
        item["record"]["machine_id"] == machine_id for item in body["records"]
    )

    other_body = client.get(changes_url(other, limit=10)).json()
    assert [item["record"]["id"] for item in other_body["records"]] == [
        rid(2),
        rid(4),
        rid(3),
    ]


def test_real_records_are_pageable_and_match_the_source_endpoints(
    client, machine_id
):
    event_id, incident_id = create_real_records(client, machine_id)

    items, pages = fetch_all_pages(client, machine_id, limit=1)
    assert pages == 4
    assert [item["group"] for item in items] == [
        INCIDENTS,
        STATUS_HISTORY,
        STATUS_HISTORY,
        ASSIGNMENTS,
    ]

    listed_incidents = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"
    ).json()
    assert [items[0]["record"]] == listed_incidents

    listed_history = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status-history"
    ).json()
    assert [items[1]["record"], items[2]["record"]] == listed_history

    listed_assignments = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    ).json()
    assert [items[3]["record"]] == listed_assignments


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client, machine_id):
    insert_incident(client, machine_id, 1, T1)
    insert_incident(client, machine_id, 2, "not-a-timestamp")
    insert_incident(client, machine_id, 3, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [item["record"]["id"] for item in rows] == [rid(3), rid(1), rid(2)]
    assert rows[-1]["record"]["created_at"] == "not-a-timestamp"


def test_out_of_range_calendar_stamp_sorts_last_verbatim(client, machine_id):
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, "2026-13-40T99:99:99Z")

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [item["record"]["id"] for item in rows] == [rid(1), rid(2)]
    assert rows[1]["record"]["created_at"] == "2026-13-40T99:99:99Z"


def test_multiple_unparseable_stamps_sort_by_group_then_id(client, machine_id):
    insert_incident(client, machine_id, 30, "zzz")
    insert_assignment(client, machine_id, 10, "yyy")
    insert_incident(client, machine_id, 20, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [(item["group"], item["record"]["id"]) for item in rows] == [
        (INCIDENTS, rid(20)),
        (INCIDENTS, rid(30)),
        (ASSIGNMENTS, rid(10)),
    ]


def test_damaged_fields_are_emitted_unmodified(client, machine_id):
    dangling_event = "11111111-1111-1111-1111-111111111111"
    dangling_incident = "22222222-2222-2222-2222-222222222222"
    insert_incident(
        client, machine_id, 1, T0,
        event_id=dangling_event, incident_type="", summary="", status="weird",
    )
    insert_status_event(
        client, machine_id, 2, T1,
        event_id=dangling_event, incident_id=dangling_incident,
        from_status="resolved", to_status="open",
        previous_status_event_id=rid(7),
        content_hash="z" * 64, chain_hash="q" * 64,
    )
    insert_assignment(
        client, machine_id, 3, T2,
        event_id=dangling_event, incident_id=dangling_incident,
        party="", role="", previous_assignment_id=rid(8),
        content_hash="y" * 64, chain_hash="x" * 64,
    )

    records = {
        item["group"]: item["record"]
        for item in client.get(changes_url(machine_id, limit=10)).json()["records"]
    }
    assert records[INCIDENTS]["event_id"] == dangling_event
    assert records[INCIDENTS]["incident_type"] == ""
    assert records[INCIDENTS]["summary"] == ""
    assert records[INCIDENTS]["status"] == "weird"
    assert records[STATUS_HISTORY]["incident_id"] == dangling_incident
    assert records[STATUS_HISTORY]["from_status"] == "resolved"
    assert records[STATUS_HISTORY]["to_status"] == "open"
    assert records[STATUS_HISTORY]["previous_status_event_id"] == rid(7)
    assert records[STATUS_HISTORY]["content_hash"] == "z" * 64
    assert records[ASSIGNMENTS]["party"] == ""
    assert records[ASSIGNMENTS]["role"] == ""
    assert records[ASSIGNMENTS]["previous_assignment_id"] == rid(8)
    assert records[ASSIGNMENTS]["chain_hash"] == "x" * 64


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(
    client, machine_id
):
    insert_incident(client, machine_id, 1, T0)
    insert_incident(client, machine_id, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1)]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{INCIDENTS}|{rid(1)}"

    second = client.get(changes_url(machine_id, limit=1, cursor=cursor))
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
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    insert_assignment(client, machine_id, 3, T2)
    insert_incident(client, machine_id, 4, T3)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    page_one = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    page_two = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    assert page_one == page_two


def test_new_earlier_inserts_do_not_revisit_returned_pages(client, machine_id):
    insert_incident(client, machine_id, 1, T2)
    insert_incident(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [item["record"]["id"] for item in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert a record sorting before the cursor position and one between.
    insert_incident(client, machine_id, 3, T1)
    insert_incident(client, machine_id, 4, T3)

    second = client.get(changes_url(machine_id, limit=10, cursor=old_cursor)).json()
    assert [item["record"]["id"] for item in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [item["record"]["id"] for item in records] == [
        rid(3),
        rid(1),
        rid(4),
        rid(2),
    ]


def test_has_more_reflects_records_after_position_only(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_incident(client, machine_id, 2, T1)
    insert_incident(client, machine_id, 3, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{INCIDENTS}|{rid(2)}")
    ).json()
    assert [item["record"]["id"] for item in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_query_is_read_only(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T1)
    insert_assignment(client, machine_id, 3, T2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return [
                list(conn.execute(text(f"SELECT * FROM {table}")))
                for table in (
                    "authorization_decision_incidents",
                    "incident_status_events",
                    "incident_responsibility_assignments",
                )
            ]

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T5}|{INCIDENTS}|{rid(9)}"))
    client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{STATUS_HISTORY}|{rid(2)}")
    )
    after = table_state()
    assert before == after


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(
    client, machine_id
):
    insert_incident(client, machine_id, 1, T1, event_id=rid(5))
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
        "records": [
            {
                "group": INCIDENTS,
                "record": {
                    "id": rid(1),
                    "machine_id": machine_id,
                    "event_id": rid(5),
                    "incident_type": "breach",
                    "summary": "something happened",
                    "status": "open",
                    "created_at": T1,
                },
            }
        ],
        "next_cursor": None,
        "has_more": False,
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    # No NaN/Infinity/-0.0-style non-finite or float content.
    json.loads(raw, parse_constant=_reject_non_finite)


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
        insert_incident(first, machine_id, 1, T0)
        insert_status_event(first, machine_id, 2, T1)
        insert_assignment(first, machine_id, 3, T2)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [item["record"]["id"] for item in response.json()["records"]] == [rid(3)]
