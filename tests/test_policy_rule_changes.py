"""Tests for the stable incremental global policy-rule ``changes`` query.

Covers `GET /policy-rules/changes`:

- query validation before any rule is read: ``bad_limit`` for a missing,
  non-integer, boolean, or out-of-range ``limit``, ``invalid_cursor`` for a
  malformed/non-string/shape-mismatching or unlocatable ``cursor``,
  ``invalid_query`` for unknown parameters or a request body;
- GET-only ``405`` routing;
- keyset pagination over the global rules ordered by the actual UTC instant
  of ``created_at`` then rule id, exact-second before fractional-second
  within a second, unparseable stamps kept and sorted last;
- exclusive cursor semantics, ``next_cursor``/``has_more`` on full, partial,
  empty, and past-the-end pages;
- the full chain-view record fields with stored values, byte-identical repeat
  pages, no reread after new inserts, strict read-only behavior, and
  persistence across a restart.
"""
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


T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"

CHANGES_PATH = "/policy-rules/changes"

VISIBLE_KEYS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)
CHAIN_KEYS = VISIBLE_KEYS + ("previous_rule_id", "content_hash", "chain_hash")

ENVELOPE_KEYS = ["limit", "records", "next_cursor", "has_more"]


def changes_url(*, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += f"&cursor={cursor}"
    return f"{CHANGES_PATH}?{query}"


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=None):
    """Insert a policy rule directly with a fixed id and timestamp."""
    if priority is None:
        # Derive a unique priority per id so the (action_type,
        # resource_pattern, priority) uniqueness constraint is never hit.
        priority = int(rule_id.rsplit("-", 1)[1])
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rule_id,
                "action_type": "read",
                "resource_pattern": "res/*",
                "effect": "allow",
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at or created_at,
            },
        )


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
        "",                              # limit missing
        "?limit=",                       # blank
        "?limit=0",                      # below range
        "?limit=101",                    # above range
        "?limit=-1",
        "?limit=1.0",                    # decimal form
        "?limit=1.5",
        "?limit=true",                   # booleans are not integers
        "?limit=false",
        "?limit=abc",
        "?limit= 1",
        "?limit=1 ",
        "?limit=0x1",
        "?limit=1&limit=2",              # repeated parameter
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
        "",                                  # empty
        "not-a-cursor",                      # no separator
        "|",                                 # empty segments
        f"{T0}|",                            # missing uuid
        f"|{rid(1)}",                        # missing timestamp
        f"{T0}|not-a-uuid",                  # bad uuid segment
        f"garbage|{rid(1)}",                 # non-timestamp position
        "2026-03-01T00:00:00|" + rid(1),     # missing Z
        f"2026-03-01T00:00:00+00:00|{rid(1)}",  # offset form
        f"2026-13-01T00:00:00Z|{rid(1)}",    # out-of-range calendar
        f"{T0}||{rid(1)}",
        f"{T0}{rid(1)}",                     # no separator
        f"{T0}/{rid(1)}",
        f"  {T0}|{rid(1)}",                  # whitespace on timestamp
    ],
)
def test_bad_cursor_is_422(client, cursor):
    response = client.get(changes_url(limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    response = client.get(f"{CHANGES_PATH}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_is_invalid_query(client):
    for content in ("{}", b'{"limit": 10}', "x"):
        response = client.request("GET", changes_url(limit=10), content=content)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_unlocatable_cursor_is_422(client):
    # A well-formed cursor against an empty table locates nothing.
    response = client.get(changes_url(limit=10, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}

    insert_rule_row(client, rid(1), T1)

    # A well-formed cursor whose id no rule carries locates nothing.
    response = client.get(changes_url(limit=10, cursor=f"{T1}|{rid(2)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}

    # The id exists but the creation text differs: not the issued position.
    response = client.get(changes_url(limit=10, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}

    # The exact stored position locates the rule and pages after it.
    response = client.get(changes_url(limit=10, cursor=f"{T1}|{rid(1)}"))
    assert response.status_code == 200
    assert response.json()["records"] == []


def test_only_get_is_accepted_on_changes_path(client):
    url = changes_url(limit=10)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Result shape, ordering, paging
# --------------------------------------------------------------------------- #


def test_empty_table_returns_full_empty_page(client):
    response = client.get(changes_url(limit=25))
    assert response.status_code == 200
    assert response.json() == {
        "limit": 25,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_envelope_field_order_and_newline(client):
    insert_rule_row(client, rid(1), T1)
    response = client.get(changes_url(limit=10))
    assert response.status_code == 200
    assert list(response.json().keys()) == ENVELOPE_KEYS
    # Compact JSON terminated by exactly one newline.
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b": " not in response.content


def test_single_page_smaller_than_limit(client):
    insert_rule_row(client, rid(1), T1)
    insert_rule_row(client, rid(2), T3)

    response = client.get(changes_url(limit=10))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == set(ENVELOPE_KEYS)
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client):
    for n, stamp in enumerate((T0, T1), start=1):
        insert_rule_row(client, rid(n), stamp)

    body = client.get(changes_url(limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client):
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_rule_row(client, rid(n), stamp)

    records, pages = fetch_all_pages(client, limit=2)
    assert pages == 3
    assert [r["id"] for r in records] == [rid(2), rid(4), rid(3), rid(5), rid(1)]
    assert [r["created_at"] for r in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client):
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_rule_row(client, rid(n), stamp)

    first = client.get(changes_url(limit=2)).json()
    assert [r["id"] for r in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor points just after the page's last record.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(changes_url(limit=2, cursor=first["next_cursor"])).json()
    assert [r["id"] for r in second["records"]] == [rid(3), rid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page again; nothing reread.
    repeated = client.get(changes_url(limit=2, cursor=first["next_cursor"])).json()
    assert repeated == second


def test_cursor_at_the_tail_returns_empty_page(client):
    insert_rule_row(client, rid(1), T0)

    body = client.get(changes_url(limit=10, cursor=f"{T0}|{rid(1)}")).json()
    assert body["records"] == []
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_exact_second_sorts_before_fractional_same_second(client):
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_rule_row(client, rid(2), fractional)
    insert_rule_row(client, rid(1), T0)

    body = client.get(changes_url(limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_rule_id_and_cursor(client):
    insert_rule_row(client, rid(30), T2, priority=30)
    insert_rule_row(client, rid(20), T2, priority=20)
    insert_rule_row(client, rid(10), T3, priority=10)

    first = client.get(changes_url(limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(20)]
    assert first["next_cursor"] == f"{T2}|{rid(20)}"

    second = client.get(changes_url(limit=1, cursor=first["next_cursor"])).json()
    assert [r["id"] for r in second["records"]] == [rid(30)]

    third = client.get(changes_url(limit=1, cursor=second["next_cursor"])).json()
    assert [r["id"] for r in third["records"]] == [rid(10)]
    assert third["next_cursor"] is None


def test_unparseable_created_at_sorts_last_and_is_kept(client):
    insert_rule_row(client, rid(1), T1)
    insert_rule_row(client, rid(2), "not-a-timestamp")
    insert_rule_row(client, rid(3), T0)

    body = client.get(changes_url(limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(3), rid(1), rid(2)]
    # The damaged record is returned with its stored text untouched.
    damaged = body["records"][2]
    assert damaged["created_at"] == "not-a-timestamp"
    assert damaged["updated_at"] == "not-a-timestamp"


def test_records_carry_the_full_chain_view_fields(client):
    first = create_rule(client, priority=1)
    second = create_rule(client, action_type="write", effect="deny", priority=2)

    chain = client.get("/policy-rules/chain").json()
    body = client.get(changes_url(limit=10)).json()

    assert body["records"] == chain
    for record, created in zip(body["records"], (first, second), strict=True):
        assert list(record.keys()) == list(CHAIN_KEYS)
        assert record == created


def test_stored_values_are_emitted_verbatim(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rid(7),
                "action_type": "  Read ",
                "resource_pattern": " res/* ",
                "effect": "ALLOW",
                "priority": 9,
                "created_at": T0,
                "updated_at": "2026-04-01T00:00:00Z",
            },
        )

    [record] = client.get(changes_url(limit=10)).json()["records"]
    assert record["action_type"] == "  Read "
    assert record["resource_pattern"] == " res/* "
    assert record["effect"] == "ALLOW"
    assert record["priority"] == 9
    assert record["updated_at"] == "2026-04-01T00:00:00Z"
    # Chain columns were never backfilled for this direct insert; the stored
    # nulls are emitted as stored, never recomputed by the read-only query.
    assert record["previous_rule_id"] is None
    assert record["content_hash"] is None
    assert record["chain_hash"] is None


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client):
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_rule_row(client, rid(n), stamp)

    first_response = client.get(changes_url(limit=2))
    cursor = first_response.json()["next_cursor"]

    page_one = client.get(changes_url(limit=2, cursor=cursor))
    page_two = client.get(changes_url(limit=2, cursor=cursor))
    assert page_one.status_code == 200
    assert page_one.content == page_two.content


def test_repeat_calls_are_byte_identical(client):
    for n, stamp in enumerate((T0, T1), start=1):
        insert_rule_row(client, rid(n), stamp)

    one = client.get(changes_url(limit=100))
    two = client.get(changes_url(limit=100))
    assert one.status_code == 200
    assert one.content == two.content


def test_query_is_read_only(client):
    insert_rule_row(client, rid(1), T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    client.get(changes_url(limit=1))
    client.get(changes_url(limit=1, cursor=f"{T1}|{rid(1)}"))
    after = table_state()
    assert before == after


def test_new_inserts_do_not_revisit_returned_pages(client):
    insert_rule_row(client, rid(1), T2)
    insert_rule_row(client, rid(2), T4)

    first = client.get(changes_url(limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert both a record before the cursor position and one after it.
    insert_rule_row(client, rid(3), T1)  # sorts before T2
    insert_rule_row(client, rid(4), T3)  # sorts between T2/T4

    # Resuming from the old cursor returns only records after it, in the
    # established order; rid(1) and the earlier rid(3) never resurface.
    second = client.get(changes_url(limit=10, cursor=old_cursor)).json()
    assert [r["id"] for r in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the new complete order.
    records, _ = fetch_all_pages(client, limit=10)
    assert [r["id"] for r in records] == [rid(3), rid(1), rid(4), rid(2)]


def test_has_more_reflects_records_after_position_only(client):
    insert_rule_row(client, rid(1), T0)
    insert_rule_row(client, rid(2), T1)
    insert_rule_row(client, rid(3), T2)

    # One record left after a full first page.
    body = client.get(changes_url(limit=2)).json()
    assert body["has_more"] is True

    # A large limit straight from the start: nothing after.
    body = client.get(changes_url(limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    # limit=1 landing exactly on the last record: no more.
    body = client.get(changes_url(limit=1, cursor=f"{T1}|{rid(2)}")).json()
    assert [r["id"] for r in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        for priority in (1, 2, 3):
            create_rule(first, priority=priority)
        first_page = first.get(changes_url(limit=2))
        cursor = first_page.json()["next_cursor"]
        expected_second = first.get(changes_url(limit=2, cursor=cursor)).content

    with TestClient(app) as second:
        response = second.get(changes_url(limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert len(response.json()["records"]) == 1


def test_existing_policy_rule_semantics_unchanged(client):
    """The new entry point leaves the existing policy-rule surfaces intact."""
    created = create_rule(client, priority=1)

    listing = client.get("/policy-rules")
    assert listing.status_code == 200
    assert [rule["id"] for rule in listing.json()] == [created["id"]]
    assert set(listing.json()[0].keys()) == set(VISIBLE_KEYS)

    chain = client.get("/policy-rules/chain")
    assert chain.status_code == 200
    assert chain.json()[0]["chain_hash"] == created["chain_hash"]

    integrity = client.get("/policy-rules/integrity")
    assert integrity.status_code == 200
    assert integrity.json() == {
        "valid": True,
        "checked_count": 1,
        "broken_policy_rule_id": None,
    }
