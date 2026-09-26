"""Tests for the stable, read-only incremental global policy-rule query.

Covers `GET /policy-rules/changes`:

- validation before any rule is read: ``bad_limit`` for a missing, blank,
  fractional, boolean, non-decimal, or out-of-range ``limit``;
  ``invalid_cursor`` for an empty, non-string, shape-mismatching, or
  unlocatable ``cursor``; ``invalid_query`` for unknown parameters,
  repeated names, or a carried request body — all 422 and identical
  against an empty database;
- GET-only ``405`` without reading rules, and ``500 internal_error`` with
  no partial page when the rules cannot be read;
- the fixed ``{limit, records, next_cursor, has_more}`` envelope, where
  each record carries exactly the complete rule-chain fields (the seven
  visible fields plus ``previous_rule_id``/``content_hash``/``chain_hash``)
  exactly as stored;
- keyset pagination in (actual UTC created_at instant, rule id) order, an
  exact-second record before a fractional-second record of the same
  second, exclusive cursors, ``next_cursor``/``has_more`` on full,
  partial, empty, and past-the-end pages;
- a rule with an unparseable ``created_at`` kept verbatim, sorted last,
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

CHANGES_PATH = "/policy-rules/changes"
CHAIN_PATH = "/policy-rules/chain"

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
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
    "previous_rule_id",
    "content_hash",
    "chain_hash",
]
ENVELOPE_KEYS = ["limit", "records", "next_cursor", "has_more"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def changes_url(*, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += "&cursor=" + cursor
    return f"{CHANGES_PATH}?{query}"


def insert_rule(
    client,
    n,
    created_at,
    *,
    effect="allow",
    priority=None,
    previous_rule_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
    action_type="read",
):
    """Insert a policy-rule row directly with a fixed id and timestamp.

    The chain columns are supplied explicitly and non-null so the startup
    backfill (which only rewrites when a chain hash is missing) leaves the
    row exactly as given; the changes query never recomputes them.
    """
    if priority is None:
        priority = n
    values = {
        "id": rid(n),
        "action_type": action_type,
        "resource_pattern": f"res/{n}",
        "effect": effect,
        "priority": priority,
        "created_at": created_at,
        "updated_at": created_at,
        "previous_rule_id": previous_rule_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at, previous_rule_id, content_hash, "
                "chain_hash) VALUES "
                "(:id, :action_type, :resource_pattern, :effect, :priority, "
                ":created_at, :updated_at, :previous_rule_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
        )
    return values


def fetch_all_pages(client, limit):
    """Walk the cursor chain from the start and return every record."""
    seen = []
    cursor = None
    pages = 0
    while True:
        response = client.get(changes_url(limit=limit, cursor=cursor))
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
    response = client.get(f"{CHANGES_PATH}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}


def test_boundary_limits_are_accepted(client):
    for value in (1, 100):
        response = client.get(changes_url(limit=value))
        assert response.status_code == 200
        assert response.json()["limit"] == value


@pytest.mark.parametrize(
    "cursor",
    [
        "",                # empty value
        "not-a-cursor",    # no separator
        "|",               # both segments empty
        f"{T0}|",          # missing rule id
        f"|{rid(1)}",      # missing created-at text
    ],
)
def test_malformed_cursor_is_422(client, cursor):
    response = client.get(changes_url(limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


@pytest.mark.parametrize(
    "cursor",
    [
        # Well-shaped positions that name no stored rule; the timestamp
        # segment is taken as original text, so a non-RFC text still passes
        # the shape check and is rejected when it cannot be located.
        f"{T0}|{rid(999)}",
        f"garbage-stamp|{rid(999)}",
        f"{T0}|not-a-uuid-but-nonempty",
        f"2026-13-01T00:00:00Z|{rid(999)}",
        f"{T0}||{rid(1)}",
    ],
)
def test_unlocatable_cursor_is_422(client, cursor):
    response = client.get(changes_url(limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_pointing_at_an_actual_rule_is_accepted(client):
    # A cursor locates by the exact stored (created_at text, id) pair.
    insert_rule(client, 1, T0)
    response = client.get(changes_url(limit=10, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 200
    assert response.json()["records"] == []
    assert response.json()["has_more"] is False


def test_cursor_with_matching_timestamp_but_unknown_id_is_422(client):
    insert_rule(client, 1, T0)
    response = client.get(changes_url(limit=10, cursor=f"{T0}|{rid(2)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    response = client.get(f"{CHANGES_PATH}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_parameter_names_are_rejected(client):
    response = client.get(
        CHANGES_PATH, params=[("limit", "1"), ("limit", "2")]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_limit"}}

    response = client.get(
        CHANGES_PATH,
        params=[("limit", "1"), ("cursor", f"{T0}|{rid(1)}"),
                ("cursor", f"{T1}|{rid(2)}")],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_get_with_a_body_is_invalid_query_before_reading_rules(client):
    response = client.request(
        "GET", f"{CHANGES_PATH}?limit=10",
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_rules(client):
    # With the table dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    assert client.get(CHANGES_PATH).status_code == 422  # missing limit
    assert client.get(f"{CHANGES_PATH}?limit=0").json() == {
        "error": {"code": "bad_limit"}
    }
    assert client.get(f"{CHANGES_PATH}?limit=1&cursor=garbage").json() == {
        "error": {"code": "invalid_cursor"}
    }
    assert client.get(f"{CHANGES_PATH}?limit=1&x=1").json() == {
        "error": {"code": "invalid_query"}
    }


def test_only_get_is_accepted_without_reading_rules(client):
    # Drop the table so any rule read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(f"{CHANGES_PATH}?limit=10")
        assert response.status_code == 405


def test_read_failure_is_500_with_no_partial_page(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))
    response = client.get(changes_url(limit=10))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"records" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, ordering, paging
# --------------------------------------------------------------------------- #


def test_empty_database_returns_full_empty_page(client):
    response = client.get(changes_url(limit=25))
    assert response.status_code == 200
    assert list(response.json().keys()) == ENVELOPE_KEYS
    assert response.json() == {
        "limit": 25,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_single_page_smaller_than_limit(client):
    insert_rule(client, 1, T1)
    insert_rule(client, 2, T3)

    body = client.get(changes_url(limit=10)).json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client):
    insert_rule(client, 1, T0)
    insert_rule(client, 2, T1)

    body = client.get(changes_url(limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client):
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        insert_rule(client, n, stamp)

    records, pages = fetch_all_pages(client, limit=2)
    assert pages == 3
    assert [r["id"] for r in records] == [rid(2), rid(4), rid(3), rid(5), rid(1)]
    assert [r["created_at"] for r in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_rule(client, n, stamp)

    first = client.get(changes_url(limit=2)).json()
    assert [r["id"] for r in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor carries the stored created_at text verbatim, after the row.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(
        changes_url(limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [rid(3), rid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page; nothing is reread.
    repeated = client.get(
        changes_url(limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_is_unlocatable_422(client):
    insert_rule(client, 1, T0)

    # A non-null cursor always names a page's last stored record, so a
    # position beyond the table cannot be located and is invalid rather
    # than an empty page.
    response = client.get(changes_url(limit=10, cursor=f"{T5}|{rid(999)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_exact_second_sorts_before_fractional_same_second(client):
    # Insert so lexicographic text order would put the fractional row first.
    insert_rule(client, 2, T0_FRAC)
    insert_rule(client, 1, T0)

    body = client.get(changes_url(limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_rule_id_and_cursor(client):
    insert_rule(client, 30, T2)
    insert_rule(client, 20, T2)
    insert_rule(client, 10, T3)

    first = client.get(changes_url(limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(20)]
    assert first["next_cursor"] == f"{T2}|{rid(20)}"

    second = client.get(
        changes_url(limit=1, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [rid(30)]

    third = client.get(
        changes_url(limit=1, cursor=second["next_cursor"])
    ).json()
    assert [r["id"] for r in third["records"]] == [rid(10)]
    assert third["next_cursor"] is None


def test_records_carry_exactly_the_complete_chain_fields(client):
    stored = insert_rule(
        client, 7, "2026-03-01T00:00:00.250Z",
        effect="deny", previous_rule_id=rid(3),
        content_hash=HASH_A, chain_hash=HASH_B,
    )
    record = client.get(changes_url(limit=10)).json()["records"][0]
    assert list(record.keys()) == RECORD_KEYS
    assert record == {
        "id": rid(7),
        "action_type": "read",
        "resource_pattern": "res/7",
        "effect": "deny",
        "priority": 7,
        "created_at": "2026-03-01T00:00:00.250Z",
        "updated_at": "2026-03-01T00:00:00.250Z",
        "previous_rule_id": rid(3),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    assert stored["id"] == record["id"]


def test_changes_pages_match_the_chain_view_slice(client):
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_rule(client, n, stamp)

    chain = client.get(CHAIN_PATH).json()
    records, _ = fetch_all_pages(client, limit=2)
    assert records == chain


# --------------------------------------------------------------------------- #
# Damaged stored values are kept and stay pageable
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_kept_and_sorts_last(client):
    insert_rule(client, 1, T1)
    insert_rule(client, 2, "not-a-timestamp")
    insert_rule(client, 3, T0)

    rows = client.get(changes_url(limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(3), rid(1), rid(2)]
    assert rows[-1]["created_at"] == "not-a-timestamp"


def test_out_of_range_calendar_stamp_sorts_last_verbatim(client):
    insert_rule(client, 1, T0)
    insert_rule(client, 2, "2026-13-40T99:99:99Z")

    rows = client.get(changes_url(limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
    assert rows[1]["created_at"] == "2026-13-40T99:99:99Z"


def test_multiple_unparseable_stamps_sort_by_id(client):
    insert_rule(client, 30, "zzz")
    insert_rule(client, 10, "yyy")
    insert_rule(client, 20, T0)

    rows = client.get(changes_url(limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [rid(20), rid(10), rid(30)]


def test_damaged_chain_fields_are_emitted_unmodified(client):
    insert_rule(
        client, 1, T1, previous_rule_id=rid(7),
        content_hash="z" * 64, chain_hash="q" * 64,
    )
    (record,) = client.get(changes_url(limit=10)).json()["records"]
    assert record["previous_rule_id"] == rid(7)
    assert record["content_hash"] == "z" * 64
    assert record["chain_hash"] == "q" * 64


def test_unparseable_stamp_row_is_pageable_with_its_original_cursor(client):
    insert_rule(client, 1, T0)
    insert_rule(client, 2, "not-a-timestamp|weird")

    first = client.get(changes_url(limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    cursor = first["next_cursor"]
    assert cursor == f"{T0}|{rid(1)}"

    second = client.request(
        "GET", "/policy-rules/changes?limit=1&cursor=" + _quote(cursor)
    )
    assert second.status_code == 200
    body = second.json()
    assert [r["id"] for r in body["records"]] == [rid(2)]
    assert body["records"][0]["created_at"] == "not-a-timestamp|weird"
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def _quote(value):
    from urllib.parse import quote

    return quote(value, safe="")


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_rule(client, n, stamp)

    cursor = client.get(changes_url(limit=2)).json()["next_cursor"]
    page_one = client.get(changes_url(limit=2, cursor=cursor)).content
    page_two = client.get(changes_url(limit=2, cursor=cursor)).content
    assert page_one == page_two


def test_new_earlier_inserts_do_not_revisit_returned_pages(client):
    insert_rule(client, 1, T2)
    insert_rule(client, 2, T4)

    first = client.get(changes_url(limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert a rule sorting before the cursor position and one between.
    insert_rule(client, 3, T1)
    insert_rule(client, 4, T3)

    second = client.get(changes_url(limit=10, cursor=old_cursor)).json()
    assert [r["id"] for r in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the complete new stable order.
    records, _ = fetch_all_pages(client, limit=10)
    assert [r["id"] for r in records] == [rid(3), rid(1), rid(4), rid(2)]


def test_has_more_reflects_records_after_position_only(client):
    insert_rule(client, 1, T0)
    insert_rule(client, 2, T1)
    insert_rule(client, 3, T2)

    assert client.get(changes_url(limit=2)).json()["has_more"] is True

    body = client.get(changes_url(limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    body = client.get(
        changes_url(limit=1, cursor=f"{T1}|{rid(2)}")
    ).json()
    assert [r["id"] for r in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_query_is_read_only(client):
    insert_rule(client, 1, T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    client.get(changes_url(limit=1))
    client.get(changes_url(limit=1, cursor=f"{T5}|{rid(9)}"))
    client.get(changes_url(limit=1, cursor=f"{T1}|{rid(1)}"))
    after = table_state()
    assert before == after


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(client):
    insert_rule(
        client, 1, T1, effect="allow",
        previous_rule_id=None, content_hash=HASH_A, chain_hash=HASH_B,
    )
    response = client.get(changes_url(limit=10))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw

    expected_item = {
        "id": rid(1),
        "action_type": "read",
        "resource_pattern": "res/1",
        "effect": "allow",
        "priority": 1,
        "created_at": T1,
        "updated_at": T1,
        "previous_rule_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    expected = {
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
    assert all(
        isinstance(record["priority"], int)
        for record in json.loads(raw)["records"]
    )


def test_utf8_content_is_round_tripped_byte_stably(client):
    # action_type/resource_pattern are stored verbatim and must survive as
    # compact UTF-8 JSON without ASCII escaping.
    insert_rule(client, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE policy_rules SET resource_pattern = :pattern "
                "WHERE id = :id"
            ),
            {"pattern": "res/规则/★", "id": rid(1)},
        )

    first = client.get(changes_url(limit=10)).content
    second = client.get(changes_url(limit=10)).content
    assert first == second
    assert "规则".encode("utf-8") in first
    body = json.loads(first)
    assert body["records"][0]["resource_pattern"] == "res/规则/★"


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        for n, stamp in ((1, T0), (2, T1), (3, T2)):
            insert_rule(first, n, stamp)
        cursor = first.get(changes_url(limit=2)).json()["next_cursor"]
        expected_second = first.get(
            changes_url(limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [r["id"] for r in response.json()["records"]] == [rid(3)]
