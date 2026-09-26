"""Tests for the stable incremental authorization decision event ``changes``
query under the privacy-export sub-entry.

Covers
`GET /machines/{machine_id}/authorization-decision-events/privacy-export/changes`:

- query validation before any machine/event read: ``bad_limit`` for a
  missing, decimal, boolean, or out-of-range ``limit``, ``invalid_cursor``
  for an empty/non-string/shape-mismatching/unparseable ``cursor``, and
  ``invalid_query`` for unknown parameters, all 422 even against a missing
  machine (unknown parameters first, then the 404);
- ``404 not_found`` for valid parameters against a missing machine;
  GET-only ``405``;
- keyset pagination over the machine's own events ordered by the actual UTC
  instant of ``created_at`` then event id, exact-second before
  fractional-second within a second;
- exclusive cursor semantics, ``next_cursor``/``has_more`` on full, partial,
  empty, exact-size, and past-the-end pages;
- byte-identical repeat pages, no reread after earlier-timestamped inserts,
  strict machine isolation, the ten privacy-export fields only with raw
  action/resource never emitted, damaged/misowned rows kept verbatim, the
  compact newline-terminated body, and persistence across a restart.
"""
import hashlib
import json

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

HASH_A = "a" * 64
HASH_B = "b" * 64


def changes_path(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "privacy-export/changes"
    )


def changes_url(machine_id, *, limit=100, cursor=None):
    query = f"limit={limit}"
    if cursor is not None:
        query += f"&cursor={cursor}"
    return f"{changes_path(machine_id)}?{query}"


def create_machine(client, external_id="machine-1", public_key="key-1"):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "Machine One",
            "public_key": public_key,
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()


def eid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_event_row(
    client,
    machine_id,
    event_id,
    created_at,
    *,
    action_type="read",
    resource="res/x",
    allowed=1,
    reason="allowed_by_policy",
    previous_event_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert an event row directly with a fixed id and timestamp.

    Chain columns are supplied explicitly so the startup NULL-only backfill
    leaves them exactly as given; the changes query never recomputes them.
    """
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at, previous_event_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :action_type, :resource, :allowed, "
                ":reason, :created_at, :previous_event_id, :content_hash, "
                ":chain_hash)"
            ),
            {
                "id": event_id,
                "machine_id": machine_id,
                "action_type": action_type,
                "resource": resource,
                "allowed": allowed,
                "reason": reason,
                "created_at": created_at,
                "previous_event_id": previous_event_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        )


EVENT_KEYS = [
    "id",
    "machine_id",
    "action_ref",
    "resource_ref",
    "allowed",
    "reason",
    "created_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
]

ENVELOPE_KEYS = ["machine_id", "limit", "records", "next_cursor", "has_more"]


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


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",  # limit missing
        "?limit=",  # blank
        "?limit=0",  # below range
        "?limit=101",  # above range
        "?limit=-1",
        "?limit=1.0",  # decimal form
        "?limit=1.5",
        "?limit=true",  # booleans are not integers
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
        "",  # empty
        "not-a-cursor",  # no separator
        "|",  # empty segments
        f"{T0}|",  # missing uuid
        f"|{eid(1)}",  # missing timestamp
        f"{T0}|not-a-uuid",  # bad uuid segment
        f"garbage|{eid(1)}",  # non-timestamp position
        "2026-03-01T00:00:00|" + eid(1),  # missing Z
        f"2026-03-01T00:00:00+00:00|{eid(1)}",  # offset form
        f"2026-13-01T00:00:00Z|{eid(1)}",  # out-of-range calendar
        f"2026-03-01T24:00:00Z|{eid(1)}",  # out-of-range time
        f"{T0}||{eid(1)}",
        f"{T0}{eid(1)}",  # no separator
        f"{T0}/{eid(1)}",
        f"  {T0}|{eid(1)}",  # whitespace on timestamp
    ],
)
def test_bad_cursor_is_422(client, cursor):
    machine_id = create_machine(client)
    response = client.get(changes_url(machine_id, limit=10, cursor=cursor))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(f"{changes_path(machine_id)}?limit=10&unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_parameter_is_rejected_before_machine_lookup(client):
    base = changes_path(MISSING_ID)

    # An unknown parameter wins over a bad limit, a bad cursor, and the
    # missing machine; validation never reads event data.
    unknown_with_bad_limit = client.get(f"{base}?limit=0&x=1")
    assert unknown_with_bad_limit.status_code == 422
    assert unknown_with_bad_limit.json() == {"error": {"code": "invalid_query"}}

    unknown_with_bad_cursor = client.get(f"{base}?limit=10&cursor=nope&x=1")
    assert unknown_with_bad_cursor.status_code == 422
    assert unknown_with_bad_cursor.json() == {"error": {"code": "invalid_query"}}

    unknown_alone = client.get(f"{base}?x=1")
    assert unknown_alone.status_code == 422
    assert unknown_alone.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = changes_path(MISSING_ID)

    missing_limit = client.get(base)
    assert missing_limit.status_code == 422
    assert missing_limit.json() == {"error": {"code": "bad_limit"}}

    bad_limit_value = client.get(f"{base}?limit=0")
    assert bad_limit_value.status_code == 422
    assert bad_limit_value.json() == {"error": {"code": "bad_limit"}}

    bad_cursor = client.get(f"{base}?limit=10&cursor=garbage")
    assert bad_cursor.status_code == 422
    assert bad_cursor.json() == {"error": {"code": "invalid_cursor"}}


def test_valid_params_for_missing_machine_are_404_without_records(client):
    response = client.get(changes_url(MISSING_ID, limit=10))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    # A well-formed cursor must not change the 404 and no records leak.
    response = client.get(
        changes_url(MISSING_ID, limit=10, cursor=f"{T0}|{eid(1)}")
    )
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
    insert_event_row(client, machine_id, eid(1), T1)
    insert_event_row(client, machine_id, eid(2), T3)

    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["limit"] == 10
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert [r["id"] for r in body["records"]] == [eid(1), eid(2)]


def test_exact_page_size_has_no_more(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1), start=1):
        insert_event_row(client, machine_id, eid(n), stamp)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert len(body["records"]) == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_pagination_walks_every_record_in_order(client):
    machine_id = create_machine(client)
    stamps = [T4, T0, T2, T1, T3]
    for n, stamp in enumerate(stamps, start=1):
        insert_event_row(client, machine_id, eid(n), stamp)

    records, pages = fetch_all_pages(client, machine_id, limit=2)
    assert pages == 3
    assert [r["id"] for r in records] == [eid(2), eid(4), eid(3), eid(5), eid(1)]
    assert [r["created_at"] for r in records] == [T0, T1, T2, T3, T4]


def test_page_cursors_are_exclusive(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_event_row(client, machine_id, eid(n), stamp)

    first = client.get(changes_url(machine_id, limit=2)).json()
    assert [r["id"] for r in first["records"]] == [eid(1), eid(2)]
    assert first["has_more"] is True
    # The cursor carries the created_at original text and points just past the
    # page's last record.
    assert first["next_cursor"] == f"{T1}|{eid(2)}"

    second = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [eid(3), eid(4)]
    assert second["has_more"] is False
    assert second["next_cursor"] is None

    # The same cursor returns exactly the same page again; nothing reread.
    repeated = client.get(
        changes_url(machine_id, limit=2, cursor=first["next_cursor"])
    ).json()
    assert repeated == second


def test_cursor_past_the_end_returns_empty_page(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)

    body = client.get(
        changes_url(machine_id, limit=10, cursor=f"{T5}|{eid(999)}")
    ).json()
    assert body["records"] == []
    assert body["next_cursor"] is None
    assert body["has_more"] is False


def test_exact_second_sorts_before_fractional_same_second(client):
    # As text "...:00.5Z" sorts before "...:00Z" ('.' < 'Z'); the changes
    # query must order by the true UTC instant: exact-second first.
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(20), "2026-03-01T00:00:00.900000Z")
    insert_event_row(client, machine_id, eid(10), "2026-03-01T00:00:00.500000Z")
    insert_event_row(client, machine_id, eid(1), T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [eid(1), eid(10), eid(20)]


def test_same_instant_tie_breaks_by_event_id_and_cursor(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(30), T2)
    insert_event_row(client, machine_id, eid(20), T2)
    insert_event_row(client, machine_id, eid(10), T3)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [eid(20)]
    assert first["next_cursor"] == f"{T2}|{eid(20)}"

    second = client.get(
        changes_url(machine_id, limit=1, cursor=first["next_cursor"])
    ).json()
    assert [r["id"] for r in second["records"]] == [eid(30)]

    third = client.get(
        changes_url(machine_id, limit=1, cursor=second["next_cursor"])
    ).json()
    assert [r["id"] for r in third["records"]] == [eid(10)]
    assert third["next_cursor"] is None


def test_has_more_reflects_records_after_position_only(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_event_row(client, machine_id, eid(2), T1)
    insert_event_row(client, machine_id, eid(3), T2)

    body = client.get(changes_url(machine_id, limit=2)).json()
    assert body["has_more"] is True

    body = client.get(changes_url(machine_id, limit=100)).json()
    assert len(body["records"]) == 3
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    # limit=1 landing exactly on the last record: no more.
    body = client.get(
        changes_url(machine_id, limit=1, cursor=f"{T1}|{eid(2)}")
    ).json()
    assert [r["id"] for r in body["records"]] == [eid(3)]
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Privacy-export record shape and damaged rows
# --------------------------------------------------------------------------- #


def test_records_carry_exactly_the_privacy_export_fields(client):
    machine_id = create_machine(client)
    recorded = record_event(client, machine_id, "read", "res/secret")
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()[0]

    body = client.get(changes_url(machine_id, limit=10)).json()
    (record,) = body["records"]
    assert list(record.keys()) == EVENT_KEYS
    assert record["id"] == recorded["id"] == listed["id"]
    assert record["machine_id"] == machine_id
    assert record["allowed"] == recorded["allowed"]
    assert record["reason"] == recorded["reason"]
    assert record["created_at"] == recorded["created_at"]
    assert record["previous_event_id"] is None
    assert record["content_hash"] == listed["content_hash"]
    assert record["chain_hash"] == listed["chain_hash"]
    assert record["action_ref"] == expected_ref("action", machine_id, "read")
    assert record["resource_ref"] == expected_ref(
        "resource", machine_id, "res/secret"
    )


def test_raw_action_and_resource_never_appear_in_response(client):
    machine_id = create_machine(client)
    secret_action = "delete-everything"
    secret_resource = "res/top-secret"
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type=f"  {secret_action}  ", resource=secret_resource,
    )
    response = client.get(changes_url(machine_id, limit=10))
    assert response.status_code == 200
    assert secret_action not in response.text
    assert secret_resource not in response.text
    assert '"action_type"' not in response.text
    assert '"resource"' not in response.text


def test_blank_and_non_string_values_keep_record_with_null_refs(client):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type="   ", resource="res/x",
    )
    # BLOBs survive SQLite TEXT affinity, giving genuinely non-string values.
    insert_event_row(
        client, machine_id, eid(2), T2,
        action_type=b"\xff\xfe binary", resource=b"\x00\x01 bin",
    )

    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [eid(1), eid(2)]
    assert rows[0]["action_ref"] is None
    assert rows[0]["resource_ref"] == expected_ref(
        "resource", machine_id, "res/x"
    )
    assert rows[1]["action_ref"] is None
    assert rows[1]["resource_ref"] is None


def test_damaged_and_misowned_rows_are_kept_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    # A cross-machine previous-event link is kept exactly as stored.
    insert_event_row(client, machine_two, eid(9), T1)
    insert_event_row(
        client, machine_one, eid(1), T2,
        allowed=0, reason="custom_reason",
        previous_event_id=eid(9), content_hash=HASH_A, chain_hash=HASH_B,
    )

    (record,) = client.get(changes_url(machine_one, limit=10)).json()["records"]
    assert record["id"] == eid(1)
    assert record["allowed"] is False
    assert record["reason"] == "custom_reason"
    assert record["previous_event_id"] == eid(9)
    assert record["content_hash"] == HASH_A
    assert record["chain_hash"] == HASH_B


def test_machine_isolation_on_every_page(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_one, eid(1), T1)
    insert_event_row(client, machine_two, eid(2), T0)
    insert_event_row(client, machine_one, eid(3), T3)

    for machine_id, expected in (
        (machine_one, [eid(1), eid(3)]),
        (machine_two, [eid(2)]),
    ):
        records, _ = fetch_all_pages(client, machine_id, limit=1)
        assert [r["id"] for r in records] == expected
        assert all(r["machine_id"] == machine_id for r in records)


def test_empty_page_keeps_machine_isolation(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_two, eid(2), T0)

    # A cursor past machine one's (empty) tail never surfaces machine two's.
    body = client.get(
        changes_url(machine_one, limit=10, cursor=f"{T5}|{eid(999)}")
    ).json()
    assert body["records"] == []
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Stability, inserts, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_cursor_is_byte_stable_when_data_unchanged(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1, T2, T3), start=1):
        insert_event_row(client, machine_id, eid(n), stamp)

    first_response = client.get(changes_url(machine_id, limit=2))
    cursor = first_response.json()["next_cursor"]

    page_one = client.get(changes_url(machine_id, limit=2, cursor=cursor))
    page_two = client.get(changes_url(machine_id, limit=2, cursor=cursor))
    assert page_one.status_code == 200
    assert page_one.content == page_two.content


def test_query_is_read_only(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(
                conn.execute(text("SELECT * FROM authorization_decision_events"))
            )

    before = table_state()
    client.get(changes_url(machine_id, limit=1))
    client.get(changes_url(machine_id, limit=1, cursor=f"{T5}|{eid(9)}"))
    after = table_state()
    assert before == after


def test_new_inserts_do_not_revisit_returned_pages(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T2)
    insert_event_row(client, machine_id, eid(2), T4)

    first = client.get(changes_url(machine_id, limit=1)).json()
    assert [r["id"] for r in first["records"]] == [eid(1)]
    old_cursor = first["next_cursor"]

    # Insert one record before the cursor position and one between pages.
    insert_event_row(client, machine_id, eid(3), T1)  # sorts before T2
    insert_event_row(client, machine_id, eid(4), T3)  # between T2/T4

    # Resuming from the old cursor returns only records after it; eid(1) and
    # the earlier eid(3) never resurface, and the current page keeps order.
    second = client.get(
        changes_url(machine_id, limit=10, cursor=old_cursor)
    ).json()
    assert [r["id"] for r in second["records"]] == [eid(4), eid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the new complete order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [r["id"] for r in records] == [eid(3), eid(1), eid(4), eid(2)]


def test_body_is_compact_newline_terminated_json_with_fixed_field_order(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)

    response = client.get(changes_url(machine_id, limit=1))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    assert b'": ' not in raw
    assert b", " not in raw

    expected_item = {
        "id": eid(1),
        "machine_id": machine_id,
        "action_ref": expected_ref("action", machine_id, "read"),
        "resource_ref": expected_ref("resource", machine_id, "res/x"),
        "allowed": True,
        "reason": "allowed_by_policy",
        "created_at": T1,
        "previous_event_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    expected = {
        "machine_id": machine_id,
        "limit": 1,
        "records": [expected_item],
        "next_cursor": None,
        "has_more": False,
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_changes_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        for n, stamp in enumerate((T0, T1, T2), start=1):
            insert_event_row(first, machine_id, eid(n), stamp)
        first_page = first.get(changes_url(machine_id, limit=2))
        cursor = first_page.json()["next_cursor"]
        expected_second = first.get(
            changes_url(machine_id, limit=2, cursor=cursor)
        ).content

    with TestClient(app) as second:
        response = second.get(changes_url(machine_id, limit=2, cursor=cursor))

    assert response.status_code == 200
    assert response.content == expected_second
    assert [r["id"] for r in response.json()["records"]] == [eid(3)]
