"""Tests for the stable, read-only incremental closed-loop incident query.

Covers `GET /machines/{machine_id}/authorization-decision-events/incidents/changes`:

- validation before any machine or record is read: ``bad_limit`` for a
  missing, blank, fractional, boolean, non-decimal, or out-of-range
  ``limit``; ``invalid_cursor`` for an empty, non-string,
  shape-mismatching, or unlocatable ``cursor``; ``invalid_query`` for
  unknown parameters, repeated names, or a carried request body — all 422
  and taking priority over the machine lookup;
- GET-only ``405`` (including ``HEAD``) without reading records,
  ``404 not_found`` for a missing machine carrying no records, and
  ``500 internal_error`` with no partial page when the records or machine
  cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where every item is ``{group, record}`` with one of the fixed
  group tags ``incidents``, ``responsibility_assignments``,
  ``status_history`` and the complete record fields exactly as stored,
  machine-isolated;
- the merge of the three groups ordered by actual UTC created_at instant,
  then group tag, then record id, an exact-second record before a
  fractional-second record of the same second;
- keyset pagination with exclusive three-segment cursors,
  ``next_cursor``/``has_more`` on full, partial, empty, and past-the-end
  pages, and a damaged created_at (even one containing ``|``) kept
  verbatim, sorted last, and still pageable;
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
ITEM_KEYS = ["group", "record"]
ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]

GROUPS = ("incidents", "responsibility_assignments", "status_history")
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
        f"/machines/{machine_id}/authorization-decision-events"
        "/incidents/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + quote(cursor, safe="")
    return f"{changes_path(machine_id)}?{query}"


def _execute(client, sql, values):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(sql), values)


def insert_incident(
    client,
    machine_id,
    n,
    created_at,
    *,
    event_id=None,
    incident_type="fault",
    summary=None,
    status="open",
):
    """Insert an incident row directly with a fixed id and timestamp."""
    if event_id is None:
        event_id = rid(800 + n)
    if summary is None:
        summary = f"summary-{n}"
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": event_id,
        "incident_type": incident_type,
        "summary": summary,
        "status": status,
        "created_at": created_at,
    }
    _execute(
        client,
        "INSERT INTO authorization_decision_incidents "
        "(id, machine_id, event_id, incident_type, summary, status, created_at) "
        "VALUES (:id, :machine_id, :event_id, :incident_type, :summary, "
        ":status, :created_at)",
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
    """Insert a responsibility-assignment row directly with a fixed id."""
    if event_id is None:
        event_id = rid(900)
    if incident_id is None:
        incident_id = rid(901)
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
    _execute(
        client,
        "INSERT INTO incident_responsibility_assignments "
        "(id, machine_id, event_id, incident_id, party, role, created_at, "
        "previous_assignment_id, content_hash, chain_hash) VALUES "
        "(:id, :machine_id, :event_id, :incident_id, :party, :role, "
        ":created_at, :previous_assignment_id, :content_hash, :chain_hash)",
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
    """Insert a status-event row directly with a fixed id and timestamp."""
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
    _execute(
        client,
        "INSERT INTO incident_status_events "
        "(id, machine_id, event_id, incident_id, from_status, to_status, "
        "created_at, previous_status_event_id, content_hash, chain_hash) "
        "VALUES (:id, :machine_id, :event_id, :incident_id, :from_status, "
        ":to_status, :created_at, :previous_status_event_id, :content_hash, "
        ":chain_hash)",
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


def create_real_incident(client, machine_id):
    """Drive one incident through both transitions and an assignment."""
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
        "",                # empty value
        "not-a-cursor",    # no separator
        "a|b",             # only one separator
        "|",               # segments empty
        f"{T0}|incidents|",   # missing record id
        f"|incidents|{rid(1)}",  # missing created-at text
        f"{T0}||{rid(1)}",  # missing group
        "a|b|c|d",          # extra segments are still shape-checked
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
        f"{T0}|incidents|{rid(999)}",
        "garbage-stamp|incidents|nonempty-id",
        f"2026-13-01T00:00:00Z|status_history|{rid(999)}",
        # A syntactically fine position naming a group this merge never
        # carries (the events group belongs to other endpoints).
        f"{T0}|events|{rid(999)}",
    ],
)
def test_unlocatable_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_record_is_accepted(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|incidents|{rid(1)}")
    )
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|incidents|{rid(2)}")
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
        changes_url(machine_id, limit=10, cursor=f"{T0}|incidents|{rid(1)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client, machine_id):
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client, machine_id):
    # A repeated name is an unknown-shape query regardless of which name it
    # is: invalid_query, never silently one occurrence.
    response = client.get(
        changes_path(machine_id), params=[("limit", "1"), ("limit", "2")]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        changes_path(machine_id),
        params=[
            ("limit", "1"),
            ("cursor", f"{T0}|incidents|{rid(1)}"),
            ("cursor", f"{T1}|incidents|{rid(2)}"),
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
    # A well-shaped cursor names no record of a missing machine, so it is
    # an unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(
            MISSING_MACHINE, limit=1, cursor=f"{T0}|incidents|{rid(1)}"
        )
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_records(client, machine_id):
    # With every business table dropped, a read would fail; validation-phase
    # errors must still come back as their 422 codes, never a 500.
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
    # Drop the tables so any record read would 500; method routing wins.
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


def test_read_failure_is_500_with_no_partial_page(client, machine_id):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_page(client, machine_id):
    # The machine lookup happens after the records are read; a failure there
    # is still a read-layer fault with the internal_error envelope, never a
    # partial page.
    insert_incident(client, machine_id, 1, T0)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, merge ordering, paging
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
    insert_incident(client, machine_id, 1, T1)
    insert_status_event(client, machine_id, 2, T3)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [
        (item["group"], item["record"]["id"]) for item in body["records"]
    ] == [("incidents", rid(1)), ("status_history", rid(2))]


def test_exact_page_size_has_no_more(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_merge_order(client, machine_id):
    insert_status_event(client, machine_id, 1, T4)
    insert_incident(client, machine_id, 2, T0)
    insert_assignment(client, machine_id, 3, T2)
    insert_incident(client, machine_id, 4, T1)
    insert_status_event(client, machine_id, 5, T3)

    items, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [(i["group"], i["record"]["id"]) for i in items] == [
        ("incidents", rid(2)),
        ("incidents", rid(4)),
        ("responsibility_assignments", rid(3)),
        ("status_history", rid(5)),
        ("status_history", rid(1)),
    ]
    assert [i["record"]["created_at"] for i in items] == [
        T0, T1, T2, T3, T4
    ]


def test_same_instant_tie_breaks_by_group_then_id(client, machine_id):
    # One record per group at the same instant; the fixed group tag order is
    # incidents < responsibility_assignments < status_history.
    insert_status_event(client, machine_id, 30, T2)
    insert_assignment(client, machine_id, 20, T2)
    insert_incident(client, machine_id, 10, T2)
    # A later instant stays after the whole tie.
    insert_incident(client, machine_id, 11, T3)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert first["records"][0]["group"] == "incidents"
    assert first["records"][0]["record"]["id"] == rid(10)
    assert first["next_cursor"] == f"{T2}|incidents|{rid(10)}"

    seen, _ = fetch_all_pages(client, machine_id, limit=1)
    assert [(i["group"], i["record"]["id"]) for i in seen] == [
        ("incidents", rid(10)),
        ("responsibility_assignments", rid(20)),
        ("status_history", rid(30)),
        ("incidents", rid(11)),
    ]


def test_same_instant_same_group_tie_breaks_by_id(client, machine_id):
    insert_incident(client, machine_id, 30, T2)
    insert_incident(client, machine_id, 20, T2)
    insert_incident(client, machine_id, 10, T3)

    seen, _ = fetch_all_pages(client, machine_id, limit=1)
    assert [i["record"]["id"] for i in seen] == [rid(20), rid(30), rid(10)]


def test_page_cursors_are_exclusive_and_carry_group(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [
        (i["group"], i["record"]["id"]) for i in first["records"]
    ] == [("incidents", rid(1)), ("responsibility_assignments", rid(2))]
    assert first["has_more"] is True
    assert first["next_cursor"] == f"{T1}|responsibility_assignments|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [
        (i["group"], i["record"]["id"]) for i in second["records"]
    ] == [("status_history", rid(3))]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_is_unlocatable_422(client, machine_id):
    insert_incident(client, machine_id, 1, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|incidents|{rid(999)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_assignment(client, machine_id, 2, T0_FRAC)
    insert_incident(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [
        (i["group"], i["record"]["id"]) for i in body["records"]
    ] == [("incidents", rid(1)), ("responsibility_assignments", rid(2))]
    assert body["next_cursor"] is None


def test_every_item_carries_the_group_wrapper(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    for item in body["records"]:
        assert list(item.keys()) == ITEM_KEYS
        assert item["group"] in GROUPS


def test_each_group_carries_exactly_its_complete_fields(client, machine_id):
    insert_incident(
        client, machine_id, 1, T1,
        event_id=rid(5), incident_type="breach", summary="s-1", status="open",
    )
    insert_assignment(
        client, machine_id, 2, "2026-03-01T00:00:00.250Z",
        event_id=rid(5), incident_id=rid(6),
        party="team-b", role="lead",
        previous_assignment_id=rid(9),
        content_hash=HASH_A, chain_hash=HASH_B,
    )
    insert_status_event(
        client, machine_id, 3, "2026-03-01T00:00:00.750Z",
        event_id=rid(5), incident_id=rid(6),
        from_status="acknowledged", to_status="resolved",
        previous_status_event_id=rid(8),
        content_hash=HASH_A, chain_hash=HASH_B,
    )

    records = {
        item["group"]: item["record"]
        for item in client.get(changes_url(machine_id, limit=10)).json()["records"]
    }

    assert list(records["incidents"].keys()) == INCIDENT_KEYS
    assert records["incidents"] == {
        "id": rid(1),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_type": "breach",
        "summary": "s-1",
        "status": "open",
        "created_at": T1,
    }

    assert list(records["responsibility_assignments"].keys()) == ASSIGNMENT_KEYS
    assert records["responsibility_assignments"] == {
        "id": rid(2),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_id": rid(6),
        "party": "team-b",
        "role": "lead",
        "created_at": "2026-03-01T00:00:00.250Z",
        "previous_assignment_id": rid(9),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }

    assert list(records["status_history"].keys()) == STATUS_EVENT_KEYS
    assert records["status_history"] == {
        "id": rid(3),
        "machine_id": machine_id,
        "event_id": rid(5),
        "incident_id": rid(6),
        "from_status": "acknowledged",
        "to_status": "resolved",
        "created_at": "2026-03-01T00:00:00.750Z",
        "previous_status_event_id": rid(8),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
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
    insert_assignment(
        client, machine_id, 3, T2,
        incident_id=rid(901), party="team-a", role="owner",
    )
    insert_assignment(
        client, other, 4, T3,
        incident_id=rid(902), party="team-b", role="lead",
    )
    insert_status_event(client, machine_id, 5, T4)
    insert_status_event(client, other, 6, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [
        (i["group"], i["record"]["id"]) for i in body["records"]
    ] == [
        ("incidents", rid(1)),
        ("responsibility_assignments", rid(3)),
        ("status_history", rid(5)),
    ]
    assert all(
        i["record"]["machine_id"] == machine_id for i in body["records"]
    )

    other_body = client.get(changes_url(other, limit=10)).json()
    assert [
        (i["group"], i["record"]["id"]) for i in other_body["records"]
    ] == [
        ("status_history", rid(6)),
        ("incidents", rid(2)),
        ("responsibility_assignments", rid(4)),
    ]
    assert all(
        i["record"]["machine_id"] == other for i in other_body["records"]
    )


def test_real_incident_lifecycle_is_pageable_with_chain_fields(client, machine_id):
    event_id, incident_id = create_real_incident(client, machine_id)

    items, pages = fetch_all_pages(client, machine_id, limit=1)
    # One incident, two status transitions, one responsibility assignment.
    assert pages == 4
    records = [(i["group"], i["record"]) for i in items]
    groups = [group for group, _ in records]
    assert groups.count("incidents") == 1
    assert groups.count("status_history") == 2
    assert groups.count("responsibility_assignments") == 1

    incident = next(r for g, r in records if g == "incidents")
    assert incident["id"] == incident_id
    assert incident["event_id"] == event_id
    status_events = [r for g, r in records if g == "status_history"]
    assert [r["to_status"] for r in status_events] == [
        "acknowledged", "resolved"
    ]
    assert status_events[0]["previous_status_event_id"] is None
    assert status_events[1]["previous_status_event_id"] == status_events[0]["id"]
    assignment = next(r for g, r in records if g == "responsibility_assignments")
    assert assignment["incident_id"] == incident_id
    assert assignment["party"] == "team-a"


def test_changes_status_records_match_the_status_history_endpoint(
    client, machine_id
):
    event_id, incident_id = create_real_incident(client, machine_id)
    history_url = (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status-history"
    )
    listed = client.get(history_url).json()

    items, _ = fetch_all_pages(client, machine_id, limit=1)
    status_records = [
        i["record"] for i in items if i["group"] == "status_history"
    ]
    assert status_records == listed


def test_changes_assignment_records_match_the_assignment_endpoint(
    client, machine_id
):
    event_id, incident_id = create_real_incident(client, machine_id)
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    ).json()

    items, _ = fetch_all_pages(client, machine_id, limit=1)
    assignment_records = [
        i["record"] for i in items if i["group"] == "responsibility_assignments"
    ]
    assert assignment_records == listed


def test_changes_incident_records_match_the_incident_endpoint(
    client, machine_id
):
    event_id, incident_id = create_real_incident(client, machine_id)
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        "/incidents"
    ).json()

    items, _ = fetch_all_pages(client, machine_id, limit=1)
    incident_records = [
        i["record"] for i in items if i["group"] == "incidents"
    ]
    assert incident_records == listed
    assert incident_records[0]["id"] == incident_id


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client, machine_id):
    insert_assignment(client, machine_id, 1, T1)
    insert_incident(client, machine_id, 2, "not-a-timestamp")
    insert_status_event(client, machine_id, 3, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [
        (i["group"], i["record"]["id"]) for i in rows
    ] == [
        ("status_history", rid(3)),
        ("responsibility_assignments", rid(1)),
        ("incidents", rid(2)),
    ]
    assert rows[-1]["record"]["created_at"] == "not-a-timestamp"


def test_multiple_unparseable_stamps_still_order_by_group_and_id(
    client, machine_id
):
    insert_status_event(client, machine_id, 30, "zzz")
    insert_assignment(client, machine_id, 20, "yyy")
    insert_incident(client, machine_id, 10, "zzz")
    insert_incident(client, machine_id, 11, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    # Every unparseable stamp ties at the far-future instant, so the group
    # tag (then id) orders them regardless of their raw text.
    assert [
        (i["group"], i["record"]["id"]) for i in rows
    ] == [
        ("incidents", rid(11)),
        ("incidents", rid(10)),
        ("responsibility_assignments", rid(20)),
        ("status_history", rid(30)),
    ]


def test_damaged_fields_are_emitted_unmodified(client, machine_id):
    dangling_event = "11111111-1111-1111-1111-111111111111"
    dangling_incident = "22222222-2222-2222-2222-222222222222"
    insert_incident(
        client, machine_id, 1, T1,
        event_id=dangling_event, incident_type="weird", summary="s",
        status="not-a-real-status",
    )
    insert_assignment(
        client, machine_id, 2, T2,
        event_id=dangling_event, incident_id=dangling_incident,
        previous_assignment_id=rid(7),
        content_hash="z" * 64, chain_hash="q" * 64,
    )
    insert_status_event(
        client, machine_id, 3, T3,
        event_id=dangling_event, incident_id=dangling_incident,
        from_status="resolved", to_status="open",
        previous_status_event_id=rid(8),
        content_hash=None, chain_hash=None,
    )

    records = {
        item["group"]: item["record"]
        for item in client.get(changes_url(machine_id, limit=10)).json()["records"]
    }
    assert records["incidents"]["event_id"] == dangling_event
    assert records["incidents"]["status"] == "not-a-real-status"
    assert records["responsibility_assignments"]["previous_assignment_id"] == rid(7)
    assert records["responsibility_assignments"]["content_hash"] == "z" * 64
    assert records["status_history"]["previous_status_event_id"] == rid(8)
    assert records["status_history"]["content_hash"] is None
    assert records["status_history"]["chain_hash"] is None


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(
    client, machine_id
):
    insert_incident(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert first["records"][0]["group"] == "incidents"
    assert first["records"][0]["record"]["id"] == rid(1)
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|incidents|{rid(1)}"

    second = client.get(changes_url(machine_id, limit=1, cursor=cursor))
    assert second.status_code == 200
    body = second.json()
    assert body["records"][0]["group"] == "status_history"
    assert body["records"][0]["record"]["id"] == rid(2)
    assert body["records"][0]["record"]["created_at"] == "not-a-timestamp|weird"
    assert body["next_cursor"] is None
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    page_one = client.get(
        changes_url(machine_id, limit=2, cursor=cursor)
    ).content
    page_two = client.get(
        changes_url(machine_id, limit=2, cursor=cursor)
    ).content
    assert page_one == page_two


def test_new_earlier_inserts_do_not_revisit_returned_pages(client, machine_id):
    insert_incident(client, machine_id, 1, T2)
    insert_status_event(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert first["records"][0]["record"]["id"] == rid(1)
    old_cursor = first["next_cursor"]

    # Insert records sorting before the cursor position and between.
    insert_assignment(client, machine_id, 3, T1)
    insert_incident(client, machine_id, 4, T3)

    second = client.get(
        changes_url(machine_id, limit=10, cursor=old_cursor)
    ).json()
    assert [
        (i["group"], i["record"]["id"]) for i in second["records"]
    ] == [
        ("incidents", rid(4)),
        ("status_history", rid(2)),
    ]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [
        (i["group"], i["record"]["id"]) for i in records
    ] == [
        ("responsibility_assignments", rid(3)),
        ("incidents", rid(1)),
        ("incidents", rid(4)),
        ("status_history", rid(2)),
    ]


def test_has_more_reflects_records_after_position_only(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(
            machine_id, limit=1,
            cursor=f"{T1}|responsibility_assignments|{rid(2)}",
        )
    ).json()
    assert body["records"][0]["group"] == "status_history"
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_query_is_read_only(client, machine_id):
    insert_incident(client, machine_id, 1, T1)
    insert_assignment(client, machine_id, 2, T2)
    insert_status_event(client, machine_id, 3, T3)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (
                    "authorization_decision_incidents",
                    "incident_status_events",
                    "incident_responsibility_assignments",
                )
            }

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T5}|incidents|{rid(9)}"))
    client.get(
        changes_url(
            machine_id, limit=1,
            cursor=f"{T1}|incidents|{rid(1)}",
        )
    )
    client.post(f"{changes_path(machine_id)}?limit=1")
    after = table_state()
    assert before == after


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(
    client, machine_id
):
    insert_incident(client, machine_id, 1, T1)
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
                "group": "incidents",
                "record": {
                    "id": rid(1),
                    "machine_id": machine_id,
                    "event_id": rid(801),
                    "incident_type": "fault",
                    "summary": "summary-1",
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


def test_envelope_field_order_is_fixed(client, machine_id):
    insert_incident(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_status_event(client, machine_id, 3, T2)

    raw = client.get(changes_url(machine_id, limit=10)).content.decode("utf-8")
    assert (
        raw.index('"machine_id"')
        < raw.index('"limit"')
        < raw.index('"records"')
        < raw.index('"next_cursor"')
        < raw.index('"has_more"')
    )
    for item_match in ("incidents", "responsibility_assignments", "status_history"):
        part = raw[raw.index(f'"group":"{item_match}"'):]
        assert part.index('"group"') < part.index('"record"')


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
        insert_assignment(first, machine_id, 2, T1)
        insert_status_event(first, machine_id, 3, T2)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [
        (i["group"], i["record"]["id"])
        for i in response.json()["records"]
    ] == [("status_history", rid(3))]
