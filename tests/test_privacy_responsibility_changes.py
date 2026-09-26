"""Tests for the stable, read-only incremental privacy responsibility query.

Covers `GET /machines/{machine_id}/authorization-decision-events/
privacy-responsibility/compliance-export/changes`:

- validation before any machine or responsibility record is read:
  ``bad_limit`` for a missing, blank, fractional, boolean, non-decimal, or
  out-of-range ``limit``; ``invalid_cursor`` for an empty, non-string,
  shape-mismatching, or unlocatable ``cursor``; ``invalid_query`` for
  unknown parameters, repeated names, or a carried request body — all 422
  and taking priority over the machine lookup;
- GET-only ``405`` without reading records, ``404 not_found`` for a missing
  machine carrying no responsibility records, and ``500 internal_error``
  with no partial page when the records cannot be read;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}``
  envelope, where each record carries exactly the desensitized privacy
  export fields (raw ``party``/``role`` never appear, only their
  ``party_ref``/``role_ref`` digests) exactly as stored, and only the path
  machine's records;
- keyset pagination in (actual UTC created_at instant, record id) order, an
  exact-second record before a fractional-second record of the same second,
  exclusive cursors, ``next_cursor``/``has_more`` on full, partial, empty,
  and past-the-end pages;
- an assignment with an unparseable ``created_at`` kept verbatim, sorted
  last, and still pageable (including a damaged stamp text containing
  ``|``);
- byte-identical repeat pages, no resurfacing after earlier inserts, strict
  read-only behavior, compact newline-terminated JSON with no non-finite
  tokens, and persistence across a restart.
"""
import hashlib
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

RECORD_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "created_at",
    "previous_assignment_id",
    "content_hash",
    "chain_hash",
    "party_ref",
    "role_ref",
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
        f"privacy-responsibility/compliance-export/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + cursor
    return f"{changes_path(machine_id)}?{query}"


def insert_assignment(
    client,
    machine_id,
    n,
    created_at,
    *,
    event_id=None,
    incident_id=None,
    party=None,
    role=None,
    previous_assignment_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert an assignment row directly with a fixed id and timestamp.

    The chain columns are supplied explicitly and non-null so the startup
    backfill (which only rewrites when a chain hash is missing) leaves the
    row exactly as given; the changes query never recomputes them.
    """
    if event_id is None:
        event_id = rid(1000 + n)
    if incident_id is None:
        incident_id = rid(2000 + n)
    if party is None:
        party = f"party-{n}"
    if role is None:
        role = f"role-{n}"
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
                "created_at, previous_assignment_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, :party, "
                ":role, :created_at, :previous_assignment_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
        )
    return values


def expected_ref(kind, machine_id, value):
    return hashlib.sha256(
        f"privacy:v1|{kind}{machine_id}{value}".encode("utf-8")
    ).hexdigest()


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
        f"{T0}|",          # missing assignment id
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
        # Well-shaped positions that name no stored assignment; the
        # timestamp segment is taken as original text, so a non-RFC text
        # still passes the shape check and is rejected when it cannot be
        # located.
        f"{T0}|{rid(999)}",
        f"garbage-stamp|{rid(999)}",
        f"{T0}|not-a-uuid-but-nonempty",
        f"2026-13-01T00:00:00Z|{rid(999)}",
        f"{T0}||{rid(1)}",
    ],
)
def test_unlocatable_cursor_is_422(client, machine_id, cursor):
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_assignment_is_accepted(client, machine_id):
    # A cursor locates by the exact stored (created_at text, id) pair.
    insert_assignment(client, machine_id, 1, T0)
    response = client.get(changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client, machine_id):
    insert_assignment(client, machine_id, 1, T0)
    response = client.get(changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(2)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_naming_another_machines_assignment_is_422(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_assignment(client, other["id"], 1, T0)

    response = client.get(changes_url(machine_id, limit=10, cursor=f"{T0}|{rid(1)}"))
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
    # A well-shaped cursor names no assignment of a missing machine, so it
    # is an unlocatable-position 422 rather than a 404.
    assert client.get(
        changes_url(MISSING_MACHINE, limit=1, cursor=f"{T0}|{rid(1)}")
    ).json() == {"error": {"code": "invalid_cursor"}}


def test_validation_errors_do_not_read_records(client, machine_id):
    # With the table dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE incident_responsibility_assignments"))

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
        conn.execute(text("DROP TABLE incident_responsibility_assignments"))
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{changes_path(machine_id)}?limit=10")
        assert response.status_code == 405


def test_missing_machine_is_404_with_no_responsibility_records(client):
    response = client.get(changes_url(MISSING_MACHINE, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"records" not in response.content


def test_read_failure_is_500_with_no_partial_page(client, machine_id):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE incident_responsibility_assignments"))
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
    insert_assignment(client, machine_id, 1, T1)
    insert_assignment(client, machine_id, 2, T3)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client, machine_id):
    insert_assignment(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client, machine_id):
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        insert_assignment(client, machine_id, n, stamp)

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [r["id"] for r in records] == [rid(2), rid(4), rid(3), rid(5), rid(1)]
    assert [r["created_at"] for r in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_assignment(client, machine_id, n, stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["id"] for r in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text verbatim, after the row.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [rid(3), rid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page; nothing is reread.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_is_unlocatable_422(client, machine_id):
    insert_assignment(client, machine_id, 1, T0)

    # A non-null cursor always names a page's last stored record, so a
    # position beyond the table cannot be located and is invalid rather
    # than an empty page.
    response = client.get(changes_url(machine_id, limit=10, cursor=f"{T5}|{rid(999)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_assignment(client, machine_id, 2, T0_FRAC)
    insert_assignment(client, machine_id, 1, T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_record_id_and_cursor(client, machine_id):
    insert_assignment(client, machine_id, 30, T2)
    insert_assignment(client, machine_id, 20, T2)
    insert_assignment(client, machine_id, 10, T3)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(20)]
    assert first["next_cursor"] == f"{T2}|{rid(20)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [rid(30)]

    third = client.get(
        changes_url(machine_id, limit=1, cursor=second["next_cursor"])
    ).json()
    assert [r["id"] for r in third["records"]] == [rid(10)]
    assert third["next_cursor"] is None


def test_records_carry_exactly_the_desensitized_fields(client, machine_id):
    insert_assignment(
        client, machine_id, 7, "2026-03-01T00:00:00.250Z",
        event_id=rid(107), incident_id=rid(207),
        party="ops-oncall", role="incident_commander",
        previous_assignment_id=rid(3),
        content_hash=HASH_A, chain_hash=HASH_B,
    )
    record = client.get(changes_url(machine_id, limit=10)).json()["records"][0]
    assert list(record.keys()) == RECORD_KEYS
    assert record == {
        "id": rid(7),
        "machine_id": machine_id,
        "event_id": rid(107),
        "incident_id": rid(207),
        "created_at": "2026-03-01T00:00:00.250Z",
        "previous_assignment_id": rid(3),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
        "party_ref": expected_ref("party", machine_id, "ops-oncall"),
        "role_ref": expected_ref("role", machine_id, "incident_commander"),
    }


def test_raw_party_and_role_never_appear(client, machine_id):
    insert_assignment(
        client, machine_id, 1, T1, party="alice@example.com", role="owner"
    )
    raw = client.get(changes_url(machine_id, limit=10)).content
    assert b"alice@example.com" not in raw
    assert b"owner" not in raw
    body = json.loads(raw)
    assert "party" not in body["records"][0]
    assert "role" not in body["records"][0]


def test_blank_party_and_role_digest_to_null(client, machine_id):
    insert_assignment(client, machine_id, 1, T1, party="   ", role="")
    (record,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert record["party_ref"] is None
    assert record["role_ref"] is None


def test_only_the_path_machines_records_are_returned(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_assignment(client, machine_id, 1, T0)
    insert_assignment(client, other["id"], 2, T1)
    insert_assignment(client, machine_id, 3, T2)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(1), rid(3)]
    assert all(r["machine_id"] == machine_id for r in body["records"])

    other_body = client.get(changes_url(other["id"], limit=10)).json()
    assert [r["id"] for r in other_body["records"]] == [rid(2)]


def test_changes_pages_match_the_privacy_export_slice(client, machine_id):
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_assignment(client, machine_id, n, stamp)

    exported = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        "privacy-responsibility/compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z"
        "&to_created_at=2100-01-01T00:00:00Z"
    ).json()["responsibility_assignments"]
    records, _ = fetch_all_pages(client, machine_id, limit=2)
    assert records == exported


def test_real_assignments_are_pageable(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )
    event = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/1"},
    ).json()
    incident = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents",
        json={"incident_type": "fault", "summary": "something failed"},
    ).json()
    created = []
    for n in range(3):
        assignment = client.post(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event['id']}/incidents/{incident['id']}/responsibility-assignments",
            json={"party": f"party-{n}", "role": f"role-{n}"},
        )
        assert assignment.status_code == 201
        created.append(assignment.json())

    records, _ = fetch_all_pages(client, machine_id, limit=2)
    assert [r["id"] for r in records] == [a["id"] for a in created]
    assert records[0]["previous_assignment_id"] is None
    assert records[1]["previous_assignment_id"] == records[0]["id"]
    assert records[2]["previous_assignment_id"] == records[1]["id"]
    for record, assignment in zip(records, created, strict=True):
        assert record["content_hash"] == assignment["content_hash"]
        assert record["chain_hash"] == assignment["chain_hash"]
        assert record["party_ref"] == expected_ref(
            "party", machine_id, assignment["party"]
        )
        assert record["role_ref"] == expected_ref(
            "role", machine_id, assignment["role"]
        )


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client, machine_id):
    insert_assignment(client, machine_id, 1, T1)
    insert_assignment(client, machine_id, 2, "not-a-timestamp")
    insert_assignment(client, machine_id, 3, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(3), rid(1), rid(2)]
    assert rows[-1]["created_at"] == "not-a-timestamp"


def test_out_of_range_calendar_stamp_sorts_last_verbatim(client, machine_id):
    insert_assignment(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, "2026-13-40T99:99:99Z")

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
    assert rows[1]["created_at"] == "2026-13-40T99:99:99Z"


def test_multiple_unparseable_stamps_sort_by_id(client, machine_id):
    insert_assignment(client, machine_id, 30, "zzz")
    insert_assignment(client, machine_id, 10, "yyy")
    insert_assignment(client, machine_id, 20, T0)

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(20), rid(10), rid(30)]


def test_damaged_chain_fields_are_emitted_unmodified(client, machine_id):
    insert_assignment(
        client, machine_id, 1, T1, previous_assignment_id=rid(7),
        content_hash="z" * 64, chain_hash="q" * 64,
    )
    (record,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert record["previous_assignment_id"] == rid(7)
    assert record["content_hash"] == "z" * 64
    assert record["chain_hash"] == "q" * 64


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(
    client, machine_id
):
    insert_assignment(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{rid(1)}"

    second = client.request(
        "GET",
        f"{changes_path(machine_id)}?limit=1&cursor=" + _quote(cursor),
    )
    assert second.status_code == 200
    body = second.json()
    assert [r["id"] for r in body["records"]] == [rid(2)]
    assert body["records"][0]["created_at"] == "not-a-timestamp|weird"
    assert body["next_cursor"] is None
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_assignment(client, machine_id, n, stamp)

    cursor = client.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
    page_one = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    page_two = client.get(changes_url(machine_id, limit=2, cursor=cursor)).content
    assert page_one == page_two


def test_new_earlier_inserts_do_not_revisit_returned_pages(client, machine_id):
    insert_assignment(client, machine_id, 1, T2)
    insert_assignment(client, machine_id, 2, T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert an assignment sorting before the cursor position and one between.
    insert_assignment(client, machine_id, 3, T1)
    insert_assignment(client, machine_id, 4, T3)

    second = client.get(changes_url(machine_id, limit=10, cursor=old_cursor)).json()
    assert [r["id"] for r in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [r["id"] for r in records] == [rid(3), rid(1), rid(4), rid(2)]


def test_has_more_reflects_records_after_position_only(client, machine_id):
    insert_assignment(client, machine_id, 1, T0)
    insert_assignment(client, machine_id, 2, T1)
    insert_assignment(client, machine_id, 3, T2)

    assert client.get(changes_url(machine_id, limit=2)).json()["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{rid(2)}")
    ).json()
    assert [r["id"] for r in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_query_is_read_only(client, machine_id):
    insert_assignment(client, machine_id, 1, T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(
                conn.execute(text("SELECT * FROM incident_responsibility_assignments"))
            )

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
    insert_assignment(
        client, machine_id, 1, T1,
        event_id=rid(101), incident_id=rid(201),
        party="ops", role="lead",
        previous_assignment_id=None, content_hash=HASH_A, chain_hash=HASH_B,
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
        "id": rid(1),
        "machine_id": machine_id,
        "event_id": rid(101),
        "incident_id": rid(201),
        "created_at": T1,
        "previous_assignment_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
        "party_ref": expected_ref("party", machine_id, "ops"),
        "role_ref": expected_ref("role", machine_id, "lead"),
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
    # Non-ASCII stored text influences only the digest; the response must
    # stay compact UTF-8 JSON without ASCII escaping and byte-stable.
    insert_assignment(client, machine_id, 1, T1, party="值班-★", role="负责人")

    first = client.get(changes_url(machine_id, limit=10)).content
    second = client.get(changes_url(machine_id, limit=10)).content
    assert first == second
    body = json.loads(first)
    assert body["records"][0]["party_ref"] == expected_ref(
        "party", machine_id, "值班-★"
    )
    assert body["records"][0]["role_ref"] == expected_ref(
        "role", machine_id, "负责人"
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
            insert_assignment(first, machine_id, n, stamp)
        cursor = first.get(changes_url(machine_id, limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [r["id"] for r in response.json()["records"]] == [rid(3)]
