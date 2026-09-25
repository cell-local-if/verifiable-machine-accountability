"""Tests for the stable incremental privacy-access ``changes`` query.

Covers `GET /machines/{machine_id}/privacy-accesses/changes`:

- query validation before any machine/record read: ``bad_limit`` for a
  missing, non-integer, boolean, or out-of-range ``limit``,
  ``invalid_cursor`` for a malformed/non-string/shape-mismatching ``cursor``,
  ``invalid_query`` for unknown parameters, all reported as 422 even against a
  missing machine;
- ``404 not_found`` for valid parameters against a missing machine, with no
  access records in the response; GET-only ``405``;
- keyset pagination over the machine's own records ordered by the actual UTC
  instant of ``accessed_at`` then record id, exact-second before
  fractional-second within a second;
- exclusive cursor semantics, ``next_cursor``/``has_more`` on full, partial,
  empty, and past-the-end pages;
- byte-identical repeat pages, no reread after new inserts, strict machine
  isolation, the seven visible fields only, and persistence across a restart.
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


MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def changes_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/changes"


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += f"&cursor={cursor}"
    return f"{changes_path(machine_id)}?{query}"


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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_access_row(
    client,
    machine_id,
    access_id,
    accessed_at,
    *,
    window_start=T1,
    window_end=T2,
    result="success",
    matches_count=1,
):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO privacy_accesses "
                "(id, machine_id, accessed_at, window_start, window_end, "
                "result, matches_count) VALUES "
                "(:id, :machine_id, :accessed_at, :window_start, :window_end, "
                ":result, :matches_count)"
            ),
            {
                "id": access_id,
                "machine_id": machine_id,
                "accessed_at": accessed_at,
                "window_start": window_start,
                "window_end": window_end,
                "result": result,
                "matches_count": matches_count,
            },
        )


ACCESS_KEYS = {
    "id",
    "machine_id",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
}

ENVELOPE_KEYS = {"machine_id", "limit", "records", "next_cursor", "has_more"}


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
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"{changes_path(machine_id)}?limit=10&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = changes_path(MISSING_ID)

    bad_limit = client.get(f"{base}")
    assert bad_limit.status_code == 422
    assert bad_limit.json() == {"error": {"code": "bad_limit"}}

    bad_limit_value = client.get(f"{base}?limit=0")
    assert bad_limit_value.status_code == 422
    assert bad_limit_value.json() == {"error": {"code": "bad_limit"}}

    bad_cursor = client.get(f"{base}?limit=10&cursor=garbage")
    assert bad_cursor.status_code == 422
    assert bad_cursor.json() == {"error": {"code": "invalid_cursor"}}

    unknown = client.get(f"{base}?limit=10&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_valid_params_for_missing_machine_are_404_without_records(client):
    response = client.get(changes_url(MISSING_ID, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    # A well-formed cursor must not change the 404 and no records leak.
    response = client.get(changes_url(MISSING_ID, limit=10, cursor=f"{T0}|{rid(1)}"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted_on_changes_path(client):
    machine_id = create_machine(client)
    url = changes_url(machine_id, limit=10)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Result shape, ordering, paging
# --------------------------------------------------------------------------- #


def test_empty_machine_returns_full_empty_result(client):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=25))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "limit": 25,
        "records": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_single_page_smaller_than_limit(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T1)
    insert_access_row(client, machine_id, rid(2), T3)

    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]


def test_exact_page_size_has_no_more(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1), start=1):
        insert_access_row(client, machine_id, rid(n), stamp)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client):
    machine_id = create_machine(client)
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_access_row(client, machine_id, rid(n), stamp)

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [r["id"] for r in records] == [rid(2), rid(4), rid(3), rid(5), rid(1)]
    assert [r["accessed_at"] for r in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_access_row(client, machine_id, rid(n), stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["id"] for r in first["records"]] == [rid(1), rid(2)]
    assert first["has_more"] is True
    # The cursor points just after the page's last record.
    assert first["next_cursor"] == f"{T1}|{rid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [rid(3), rid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page again; nothing reread.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_returns_empty_page(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T0)

    body = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|{rid(999)}")
    ).json()
    assert body["records"] == []
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_access_row(client, machine_id, rid(2), fractional)
    insert_access_row(client, machine_id, rid(1), T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [rid(1), rid(2)]
    # The cursor carries the accessed_at text verbatim.
    assert body["next_cursor"] is None


def test_same_instant_tie_breaks_by_record_id_and_cursor(client):
    machine_id = create_machine(client)
    insert_access_row(
        client, machine_id, rid(30), T2, window_start=T0, window_end=T1
    )
    insert_access_row(
        client, machine_id, rid(20), T2, window_start=T3, window_end=T4
    )
    insert_access_row(client, machine_id, rid(10), T3)

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


def test_records_carry_exactly_the_seven_visible_fields(client):
    machine_id = create_machine(client)
    insert_access_row(
        client,
        machine_id,
        rid(7),
        "2026-03-01T00:00:00.250Z",
        window_start="2026-03-02T00:00:00Z",
        window_end="2026-03-03T00:00:00Z",
        result="failed",
        matches_count=0,
    )
    record = client.get(changes_url(machine_id, limit=10)).json()["records"][0]
    assert set(record.keys()) == ACCESS_KEYS
    assert record == {
        "id": rid(7),
        "machine_id": machine_id,
        "accessed_at": "2026-03-01T00:00:00.250Z",
        "window_start": "2026-03-02T00:00:00Z",
        "window_end": "2026-03-03T00:00:00Z",
        "result": "failed",
        "matches_count": 0,
    }
    # Chain columns and any sensitive material never appear.
    assert "content_hash" not in record
    assert "chain_hash" not in record
    assert "previous_access_id" not in record


def test_machine_isolation_on_every_page(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), T1)
    insert_access_row(client, machine_two, rid(2), T0)
    insert_access_row(client, machine_one, rid(3), T3)

    for machine_id, expected in (
        (machine_one, [rid(1), rid(3)]),
        (machine_two, [rid(2)]),
    ):
        records, _ = fetch_all_pages(client, machine_id, limit=1)
        assert [r["id"] for r in records] == expected
        assert all(r["machine_id"] == machine_id for r in records)


def test_empty_page_keeps_machine_isolation(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_two, rid(2), T0)

    # A cursor past machine one's (empty) tail never surfaces machine two's
    # record.
    body = client.get(
        changes_url(machine_one, limit=10, cursor=f"{T5}|{rid(999)}")
    ).json()
    assert body["records"] == []
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_access_row(client, machine_id, rid(n), stamp)

    first_response = client.get(changes_url(machine_id, limit=2))
    cursor = first_response.json()["next_cursor"]

    page_one = client.get(changes_url(machine_id, limit=2, cursor=cursor))
    page_two = client.get(changes_url(machine_id, limit=2, cursor=cursor))
    assert page_one.status_code == 200
    assert page_one.content == page_two.content


def test_query_is_read_only(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM privacy_accesses")))

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T5}|{rid(9)}"))
    after = table_state()
    assert before == after


def test_new_inserts_do_not_revisit_returned_pages(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2)
    insert_access_row(client, machine_id, rid(2), T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [rid(1)]
    old_cursor = first["next_cursor"]

    # Insert both a record before the cursor position and one after it.
    insert_access_row(client, machine_id, rid(3), T1)  # sorts before T2
    insert_access_row(client, machine_id, rid(4), T3)  # sorts between T2/T4

    # Resuming from the old cursor returns only records after it, in the
    # established order; rid(1) and the earlier rid(3) never resurface.
    second = client.get(changes_url(machine_id, limit=10, cursor=old_cursor)).json()
    assert [r["id"] for r in second["records"]] == [rid(4), rid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the new complete order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [r["id"] for r in records] == [rid(3), rid(1), rid(4), rid(2)]


def test_has_more_reflects_records_after_position_only(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T0)
    insert_access_row(client, machine_id, rid(2), T1)
    insert_access_row(client, machine_id, rid(3), T2)

    # One record left after a full first page.
    body = client.get(changes_url(machine_id, limit=2)).json()
    assert body["has_more"] is True

    # A large limit straight from the start: nothing after.
    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    # limit=1 landing exactly on the last record: no more.
    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{rid(2)}")
    ).json()
    assert [r["id"] for r in body["records"]] == [rid(3)]
    assert body["has_more"] is False


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        for n, stamp in enumerate((T0, T1, T2), start=1):
            insert_access_row(first, machine_id, rid(n), stamp)
        first_page = first.get(changes_url(machine_id, limit=2))
        cursor = first_page.json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [r["id"] for r in response.json()["records"]] == [rid(3)]
