"""Tests for the stable, read-only incremental machine accountability query.

Covers `GET /machines/{machine_id}/accountability/compliance-export/changes`:

- validation before any record is read: ``bad_limit`` for a missing, blank,
  fractional, boolean, non-decimal, or out-of-range ``limit``;
  ``invalid_cursor`` for an empty, non-string, shape-mismatching, or
  unlocatable ``cursor``; ``invalid_query`` for unknown parameters,
  repeated names, or a carried request body — all 422 and identical
  against an empty database;
- GET-only ``405`` without reading records, ``404 not_found`` for a missing
  machine, and ``500 internal_error`` with no partial page when the records
  cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where each item is ``{group, record}`` with the fixed category
  tag and the complete record exactly as the machine accountability
  compliance export emits it for that group;
- keyset pagination merging the five groups in (actual UTC created_at
  instant, fixed category order, record id) order, an exact-second record
  before a fractional-second record of the same second, exclusive cursors,
  ``next_cursor``/``has_more`` on full, partial, empty, and past-the-end
  pages;
- a record with an unparseable ``created_at`` kept verbatim, sorted last,
  and still pageable (including a damaged stamp text containing ``|``);
- byte-identical repeat pages, no resurfacing after earlier inserts,
  machine isolation, strict read-only behavior, compact newline-terminated
  JSON with no non-finite tokens, and persistence across a restart.
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

GROUPS = [
    "events",
    "evidence",
    "incidents",
    "status_history",
    "responsibility_assignments",
]
ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]

EVENT_KEYS = [
    "id", "machine_id", "action_type", "resource", "allowed", "reason",
    "created_at", "previous_event_id", "content_hash", "chain_hash",
]
EVIDENCE_KEYS = [
    "id", "machine_id", "event_id", "evidence_type", "content_hash",
    "created_at",
]
INCIDENT_KEYS = [
    "id", "machine_id", "event_id", "incident_type", "summary", "status",
    "created_at",
]
STATUS_EVENT_KEYS = [
    "id", "machine_id", "event_id", "incident_id", "from_status",
    "to_status", "created_at",
]
ASSIGNMENT_KEYS = [
    "id", "machine_id", "event_id", "incident_id", "party", "role",
    "created_at", "previous_assignment_id", "content_hash", "chain_hash",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + quote(cursor, safe="")
    return (
        f"/machines/{machine_id}/accountability/compliance-export/changes"
        f"?{query}"
    )


def insert_event(client, machine_id, n, created_at, **overrides):
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "action_type": "read",
        "resource": f"res/{n}",
        "allowed": True,
        "reason": "allowed_by_policy",
        "created_at": created_at,
        "previous_event_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    values.update(overrides)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at, previous_event_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :action_type, :resource, "
                ":allowed, :reason, :created_at, :previous_event_id, "
                ":content_hash, :chain_hash)"
            ),
            values,
        )
    return values


def insert_evidence(client, machine_id, n, created_at, event_n=1, **overrides):
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": rid(event_n),
        "evidence_type": "log",
        "content_hash": HASH_A,
        "created_at": created_at,
        "previous_evidence_id": None,
        "content_digest": HASH_B,
        "chain_hash": HASH_B,
    }
    values.update(overrides)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_evidence "
                "(id, machine_id, event_id, evidence_type, content_hash, "
                "created_at, previous_evidence_id, content_digest, "
                "chain_hash) VALUES (:id, :machine_id, :event_id, "
                ":evidence_type, :content_hash, :created_at, "
                ":previous_evidence_id, :content_digest, :chain_hash)"
            ),
            values,
        )
    return values


def insert_incident(client, machine_id, n, created_at, event_n=1, **overrides):
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": rid(event_n),
        "incident_type": "fault",
        "summary": f"summary-{n}",
        "status": "open",
        "created_at": created_at,
    }
    values.update(overrides)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_incidents "
                "(id, machine_id, event_id, incident_type, summary, status, "
                "created_at) VALUES (:id, :machine_id, :event_id, "
                ":incident_type, :summary, :status, :created_at)"
            ),
            values,
        )
    return values


def insert_status_event(
    client, machine_id, n, created_at, event_n=1, incident_n=2, **overrides
):
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": rid(event_n),
        "incident_id": rid(incident_n),
        "from_status": "open",
        "to_status": "acknowledged",
        "created_at": created_at,
        "previous_status_event_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    values.update(overrides)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events "
                "(id, machine_id, event_id, incident_id, from_status, "
                "to_status, created_at, previous_status_event_id, "
                "content_hash, chain_hash) VALUES (:id, :machine_id, "
                ":event_id, :incident_id, :from_status, :to_status, "
                ":created_at, :previous_status_event_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
        )
    return values


def insert_assignment(
    client, machine_id, n, created_at, event_n=1, incident_n=2, **overrides
):
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "event_id": rid(event_n),
        "incident_id": rid(incident_n),
        "party": f"party-{n}",
        "role": "owner",
        "created_at": created_at,
        "previous_assignment_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    values.update(overrides)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments "
                "(id, machine_id, event_id, incident_id, party, role, "
                "created_at, previous_assignment_id, content_hash, "
                "chain_hash) VALUES (:id, :machine_id, :event_id, "
                ":incident_id, :party, :role, :created_at, "
                ":previous_assignment_id, :content_hash, :chain_hash)"
            ),
            values,
        )
    return values


def seed_all_groups(client, machine_id, stamps):
    """Insert one record per group; ``stamps`` maps group -> created_at."""
    insert_event(client, machine_id, 1, stamps["events"])
    insert_evidence(client, machine_id, 10, stamps["evidence"])
    insert_incident(client, machine_id, 2, stamps["incidents"])
    insert_status_event(client, machine_id, 11, stamps["status_history"])
    insert_assignment(
        client, machine_id, 12, stamps["responsibility_assignments"]
    )


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
    base = f"/machines/{machine_id}/accountability/compliance-export/changes"
    response = client.get(f"{base}{query}")
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
        "",                       # empty value
        "not-a-cursor",           # no separator
        "|",                      # every segment empty
        f"{T0}|",                 # missing group and record id
        f"{T0}|events|",          # missing record id
        f"{T0}||{rid(1)}",        # missing group
        f"|events|{rid(1)}",      # missing created-at text
        f"{T0}|unknown|{rid(1)}", # group is not a fixed category tag
        f"{T0}|{rid(1)}",         # only two segments
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
        # Well-shaped positions that name no stored record of the path
        # machine; the timestamp segment is taken as original text, so a
        # non-RFC text still passes the shape check and is rejected when it
        # cannot be located.
        f"{T0}|events|{rid(999)}",
        f"garbage-stamp|evidence|{rid(999)}",
        f"{T0}|incidents|not-a-uuid-but-nonempty",
        f"2026-13-01T00:00:00Z|status_history|{rid(999)}",
        f"{T0}|responsibility_assignments|{rid(999)}",
    ],
)
def test_unlocatable_cursor_is_422(client, cursor):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_naming_another_machines_record_is_422(client):
    machine_id = create_machine(client)
    other_id = create_machine(client, external_id="machine-2")
    insert_event(client, other_id, 1, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T0}|events|{rid(1)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    base = f"/machines/{machine_id}/accountability/compliance-export/changes"
    response = client.get(f"{base}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client):
    machine_id = create_machine(client)
    base = f"/machines/{machine_id}/accountability/compliance-export/changes"

    response = client.get(base, params=[("limit", "1"), ("limit", "2")])
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}

    response = client.get(
        base,
        params=[
            ("limit", "1"),
            ("cursor", f"{T0}|events|{rid(1)}"),
            ("cursor", f"{T1}|events|{rid(2)}"),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_get_with_a_body_is_invalid_query_before_reading_records(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        changes_url(machine_id, limit=10),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_records(client):
    # With a table dropped, a read would fail; validation-phase errors must
    # still come back as their 422 codes, never a 500.
    machine_id = create_machine(client)
    base = f"/machines/{machine_id}/accountability/compliance-export/changes"
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_events"))

    assert client.get(base).status_code == 422  # missing limit
    assert client.get(f"{base}?limit=0").json() == {
        "error": {"code": "bad_limit"}
    }
    assert client.get(f"{base}?limit=1&cursor=garbage").json() == {
        "error": {"code": "invalid_cursor"}
    }
    assert client.get(f"{base}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }


def test_only_get_is_accepted_without_reading_records(client):
    # Drop a table so any record read would 500; method routing must win.
    machine_id = create_machine(client)
    base = f"/machines/{machine_id}/accountability/compliance-export/changes"
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_events"))
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{base}?limit=10")
        assert response.status_code == 405


def test_read_failure_is_500_with_no_partial_page(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_events"))
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


def test_missing_machine_is_404(client):
    response = client.get(changes_url(rid(404), limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_validation_precedes_the_machine_lookup(client):
    base = f"/machines/{rid(404)}/accountability/compliance-export/changes"
    assert client.get(base).json() == {"error": {"code": "bad_limit"}}
    assert client.get(f"{base}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    assert client.get(f"{base}?limit=1&cursor=bad").json() == {
        "error": {"code": "invalid_cursor"}
    }


# --------------------------------------------------------------------------- #
# Envelope shape, merge order, paging
# --------------------------------------------------------------------------- #


def test_empty_machine_returns_full_empty_page(client):
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


def test_five_groups_merge_by_instant_then_category_then_id(client):
    machine_id = create_machine(client)
    # Same instant across all five groups: the fixed category order decides.
    seed_all_groups(
        client,
        machine_id,
        {
            "events": T2,
            "evidence": T2,
            "incidents": T2,
            "status_history": T2,
            "responsibility_assignments": T2,
        },
    )
    # A later event and an earlier one bracket the shared instant.
    insert_event(client, machine_id, 3, T4)
    insert_event(client, machine_id, 4, T1)

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    got = [(item["group"], item["record"]["id"]) for item in body["records"]]
    assert got == [
        ("events", rid(4)),
        ("events", rid(1)),
        ("evidence", rid(10)),
        ("incidents", rid(2)),
        ("status_history", rid(11)),
        ("responsibility_assignments", rid(12)),
        ("events", rid(3)),
    ]
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_same_group_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 30, T2)
    insert_event(client, machine_id, 20, T2)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [i["record"]["id"] for i in first["records"]] == [rid(20)]
    assert first["next_cursor"] == f"{T2}|events|{rid(20)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [i["record"]["id"] for i in second["records"]] == [rid(30)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    # Insert so lexicographic text order would put the fractional row first.
    insert_event(client, machine_id, 2, T0_FRAC)
    insert_event(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [i["record"]["id"] for i in body["records"]] == [rid(1), rid(2)]


def test_pagination_walks_every_record_in_order(client):
    machine_id = create_machine(client)
    seed_all_groups(
        client,
        machine_id,
        {
            "events": T4,
            "evidence": T0,
            "incidents": T2,
            "status_history": T1,
            "responsibility_assignments": T3,
        },
    )

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [item["group"] for item in records] == [
        "evidence",
        "status_history",
        "incidents",
        "responsibility_assignments",
        "events",
    ]


def test_page_cursors_are_exclusive_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, T1)
    insert_evidence(client, machine_id, 3, T2)
    insert_incident(client, machine_id, 4, T3)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [i["record"]["id"] for i in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text, the category tag, and
    # the record id, positioned after the page's last record.
    assert first["next_cursor"] == f"{T1}|events|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    )
    body = second.json()
    assert [i["record"]["id"] for i in body["records"]] == [rid(3), rid(4)]
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    # Repeating the same cursor returns the byte-identical page.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    )
    assert repeated.content == second.content


def test_exact_page_size_has_no_more(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_cursor_past_the_end_is_unlocatable_422(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T0)

    response = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|events|{rid(999)}")
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_records_carry_the_complete_export_fields_per_group(client):
    machine_id = create_machine(client)
    seed_all_groups(
        client,
        machine_id,
        {
            "events": T0,
            "evidence": T1,
            "incidents": T2,
            "status_history": T3,
            "responsibility_assignments": T4,
        },
    )

    body = client.get(changes_url(machine_id, limit=100)).json()
    expected_keys = {
        "events": EVENT_KEYS,
        "evidence": EVIDENCE_KEYS,
        "incidents": INCIDENT_KEYS,
        "status_history": STATUS_EVENT_KEYS,
        "responsibility_assignments": ASSIGNMENT_KEYS,
    }
    assert [item["group"] for item in body["records"]] == GROUPS
    for item in body["records"]:
        assert list(item.keys()) == ["group", "record"]
        assert list(item["record"].keys()) == expected_keys[item["group"]]
        assert item["record"]["machine_id"] == machine_id

    # Each record is exactly what the compliance export carries in its group.
    export = client.get(
        f"/machines/{machine_id}/accountability/compliance-export",
        params={
            "from_created_at": "2000-01-01T00:00:00Z",
            "to_created_at": "2100-01-01T00:00:00Z",
        },
    ).json()
    for item in body["records"]:
        assert item["record"] in export[item["group"]]


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)
    insert_event(client, machine_id, 2, "not-a-timestamp")
    insert_event(client, machine_id, 3, T0)

    items = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [i["record"]["id"] for i in items] == [rid(3), rid(1), rid(2)]
    assert items[-1]["record"]["created_at"] == "not-a-timestamp"


def test_multiple_unparseable_stamps_sort_by_category_then_id(client):
    machine_id = create_machine(client)
    insert_assignment(client, machine_id, 30, "zzz")
    insert_event(client, machine_id, 10, "yyy")
    insert_event(client, machine_id, 20, T0)

    items = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [(i["group"], i["record"]["id"]) for i in items] == [
        ("events", rid(20)),
        ("events", rid(10)),
        ("responsibility_assignments", rid(30)),
    ]
    assert items[1]["record"]["created_at"] == "yyy"
    assert items[2]["record"]["created_at"] == "zzz"


def test_unparseable_stamp_record_is_pageable_with_its_original_cursor(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [i["record"]["id"] for i in first["records"]] == [rid(1)]
    assert first["next_cursor"] == f"{T0}|events|{rid(1)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [i["record"]["id"] for i in second["records"]] == [rid(2)]
    assert second["records"][0]["record"]["created_at"] == "not-a-timestamp|weird"
    # The damaged stamp text containing ``|`` round-trips through a cursor.
    assert second["next_cursor"] is None
    assert second["has_more"] is False


def test_damaged_record_fields_are_emitted_unmodified(client):
    machine_id = create_machine(client)
    insert_event(
        client, machine_id, 1, T1,
        previous_event_id=rid(7), content_hash="z" * 64, chain_hash="q" * 64,
    )
    (item,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert item["record"]["previous_event_id"] == rid(7)
    assert item["record"]["content_hash"] == "z" * 64
    assert item["record"]["chain_hash"] == "q" * 64


# --------------------------------------------------------------------------- #
# Stability, inserts, isolation, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_new_earlier_inserts_do_not_revisit_returned_records(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T2)
    insert_event(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [i["record"]["id"] for i in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert records sorting before the cursor position and between.
    insert_event(client, machine_id, 3, T1)
    insert_evidence(client, machine_id, 4, T3)

    second = client.get(changes_url(machine_id, limit=10, cursor=old_cursor)).json()
    assert [(i["group"], i["record"]["id"]) for i in second["records"]] == [
        ("evidence", rid(4)),
        ("events", rid(2)),
    ]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [i["record"]["id"] for i in records] == [
        rid(3), rid(1), rid(4), rid(2)
    ]


def test_other_machines_records_never_enter_a_page(client):
    machine_id = create_machine(client)
    other_id = create_machine(client, external_id="machine-2")
    insert_event(client, machine_id, 1, T1)
    # Sound and damaged records under another machine.
    insert_event(client, other_id, 2, T0)
    insert_incident(client, other_id, 3, "broken-stamp")
    insert_assignment(client, other_id, 4, T2)

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert [(i["group"], i["record"]["id"]) for i in body["records"]] == [
        ("events", rid(1))
    ]
    assert body["has_more"] is False


def test_query_is_read_only(client):
    machine_id = create_machine(client)
    seed_all_groups(
        client,
        machine_id,
        {
            "events": T0,
            "evidence": T1,
            "incidents": T2,
            "status_history": T3,
            "responsibility_assignments": T4,
        },
    )
    tables = (
        "authorization_decision_events",
        "authorization_decision_evidence",
        "authorization_decision_incidents",
        "incident_status_events",
        "incident_responsibility_assignments",
    )

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                table: list(conn.execute(text(f"SELECT * FROM {table}")))
                for table in tables
            }

    before = table_state()
    client.get(changes_url(machine_id, limit=2))
    client.get(changes_url(machine_id, limit=2, cursor=f"{T5}|events|{rid(9)}"))
    client.get(changes_url(machine_id, limit=2, cursor=f"{T0}|events|{rid(1)}"))
    assert table_state() == before


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)

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
                "group": "events",
                "record": {
                    "id": rid(1),
                    "machine_id": machine_id,
                    "action_type": "read",
                    "resource": "res/1",
                    "allowed": True,
                    "reason": "allowed_by_policy",
                    "created_at": T1,
                    "previous_event_id": None,
                    "content_hash": HASH_A,
                    "chain_hash": HASH_B,
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


def test_utf8_content_is_round_tripped_byte_stably(client):
    machine_id = create_machine(client)
    insert_incident(client, machine_id, 1, T1, summary="故障 ★")

    first = client.get(changes_url(machine_id, limit=10)).content
    second = client.get(changes_url(machine_id, limit=10)).content
    assert first == second
    assert "故障".encode("utf-8") in first
    body = json.loads(first)
    assert body["records"][0]["record"]["summary"] == "故障 ★"


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_event(first, machine_id, 1, T0)
        insert_event(first, machine_id, 2, T1)
        insert_incident(first, machine_id, 3, T2)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [i["record"]["id"] for i in response.json()["records"]] == [rid(3)]
