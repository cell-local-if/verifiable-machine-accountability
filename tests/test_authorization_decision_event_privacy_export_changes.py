"""Tests for the stable incremental authorization-decision-event ``changes``
query over the desensitized privacy-export view.

Covers `GET
/machines/{machine_id}/authorization-decision-events/privacy-export/changes`:

- query validation before any machine/event read: ``bad_limit`` for a
  missing, non-integer, boolean, fractional, or out-of-range ``limit``,
  ``invalid_cursor`` for an empty/non-string/shape-mismatching/unparseable
  ``cursor``, ``invalid_query`` for unknown parameters (checked before the
  machine), all reported as 422 even against a missing machine;
- ``404 not_found`` for valid parameters against a missing machine, with no
  event records in the response; GET-only ``405``;
- keyset pagination over the machine's own events ordered by the actual UTC
  instant of ``created_at`` then event id, exact-second before
  fractional-second within a second;
- the fixed ``{machine_id, limit, records, next_cursor, has_more}`` envelope
  and the ten desensitized privacy-export record fields only — raw
  action/resource never appear, only their digests;
- exclusive cursor semantics, ``next_cursor``/``has_more`` on full, partial,
  empty, and past-the-end pages;
- byte-identical repeat pages, no reread after new earlier inserts, strict
  machine isolation, damaged or misowned records kept verbatim, strict
  read-only compact newline-terminated JSON, and persistence across a
  restart.
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
        f"/machines/{machine_id}/authorization-decision-events/privacy-export"
        "/changes"
    )


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

    The chain columns are supplied explicitly so the startup backfill (which
    only fills NULLs) leaves them exactly as given; the changes query never
    recomputes or normalizes them.
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
        f"|{eid(1)}",                        # missing timestamp
        f"{T0}|not-a-uuid",                  # bad uuid segment
        f"garbage|{eid(1)}",                 # non-timestamp position
        "2026-03-01T00:00:00|" + eid(1),     # missing Z
        f"2026-03-01T00:00:00+00:00|{eid(1)}",  # offset form
        f"2026-13-01T00:00:00Z|{eid(1)}",    # out-of-range calendar
        f"{T0}||{eid(1)}",
        f"{T0}{eid(1)}",                     # no separator
        f"{T0}/{eid(1)}",
        f"  {T0}|{eid(1)}",                  # whitespace on timestamp
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


def test_unknown_parameter_is_rejected_before_machine_lookup(client):
    # Even a well-formed limit/cursor pair is rejected for an unknown name,
    # and the check must precede the machine lookup (422, never 404).
    base = changes_path(MISSING_ID)

    unknown = client.get(f"{base}?limit=10&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}

    unknown_with_cursor = client.get(
        f"{base}?limit=10&cursor={T0}|{eid(1)}&x=1"
    )
    assert unknown_with_cursor.status_code == 422
    assert unknown_with_cursor.json() == {"error": {"code": "invalid_query"}}


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
    # The cursor points just after the page's last record and carries the
    # stored created_at text verbatim.
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
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_event_row(client, machine_id, eid(2), fractional)
    insert_event_row(client, machine_id, eid(1), T0)

    body = client.get(changes_url(machine_id, limit=10)).json()
    assert [r["id"] for r in body["records"]] == [eid(1), eid(2)]
    assert body["next_cursor"] is None


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


def test_records_carry_exactly_the_desensitized_fields(client):
    machine_id = create_machine(client)
    insert_event_row(
        client,
        machine_id,
        eid(7),
        "2026-03-01T00:00:00.250Z",
        action_type="write",
        resource="res/7",
        allowed=0,
        reason="denied_by_policy",
        previous_event_id=eid(3),
        content_hash=HASH_A,
        chain_hash=HASH_B,
    )
    record = client.get(changes_url(machine_id, limit=10)).json()["records"][0]
    assert list(record.keys()) == EVENT_KEYS
    assert record == {
        "id": eid(7),
        "machine_id": machine_id,
        "action_ref": expected_ref("action", machine_id, "write"),
        "resource_ref": expected_ref("resource", machine_id, "res/7"),
        "allowed": False,
        "reason": "denied_by_policy",
        "created_at": "2026-03-01T00:00:00.250Z",
        "previous_event_id": eid(3),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    # Raw business fields never appear under their stored names.
    assert "action_type" not in record
    assert "resource" not in record


def test_digest_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type="  read\t", resource="\n res/x  ",
    )
    (record,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert record["action_ref"] == expected_ref("action", machine_id, "read")
    assert record["resource_ref"] == expected_ref(
        "resource", machine_id, "res/x"
    )


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_value_digest_is_null_but_record_kept(client, blank):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1, action_type=blank, resource="res/x"
    )
    insert_event_row(
        client, machine_id, eid(2), T2, action_type="read", resource=blank
    )
    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert [r["id"] for r in rows] == [eid(1), eid(2)]
    assert rows[0]["action_ref"] is None
    assert rows[0]["resource_ref"] == expected_ref(
        "resource", machine_id, "res/x"
    )
    assert rows[1]["action_ref"] == expected_ref("action", machine_id, "read")
    assert rows[1]["resource_ref"] is None


def test_non_string_value_digest_is_null_but_record_kept(client):
    # SQLite TEXT affinity would coerce a numeric literal to text, so store
    # BLOBs (which TEXT affinity leaves untouched) to get genuinely non-string
    # values. The changes view must still return the record with null refs
    # rather than coercing or leaking either value.
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type=b"\xff\xfe binary-action", resource=b"\x00\x01 bin-res",
    )
    rows = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert len(rows) == 1
    assert rows[0]["action_ref"] is None
    assert rows[0]["resource_ref"] is None
    assert rows[0]["id"] == eid(1)


def test_raw_action_and_resource_never_appear_in_response(client):
    machine_id = create_machine(client)
    secret_action = "delete-everything"
    secret_resource = "res/top-secret"
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type=f"  {secret_action}  ", resource=secret_resource,
    )
    first = client.get(changes_url(machine_id, limit=1))
    assert first.status_code == 200
    # Follow the cursor chain; no page may carry the raw text or stored name.
    cursor = first.json()["next_cursor"]
    bodies = [first.text]
    while cursor is not None:
        page = client.get(changes_url(machine_id, limit=1, cursor=cursor))
        bodies.append(page.text)
        cursor = page.json()["next_cursor"]
    whole = "".join(bodies)
    assert secret_action not in whole
    assert secret_resource not in whole
    assert '"action_type"' not in whole
    # A record field literally named "resource" (vs resource_ref) never shows.
    assert '"resource"' not in whole


def test_digest_is_scoped_to_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_one, eid(1), T1)
    insert_event_row(client, machine_two, eid(2), T1)
    ref_one = client.get(changes_url(machine_one, limit=10)).json()["records"][0][
        "action_ref"
    ]
    ref_two = client.get(changes_url(machine_two, limit=10)).json()["records"][0][
        "action_ref"
    ]
    assert ref_one == expected_ref("action", machine_one, "read")
    assert ref_two == expected_ref("action", machine_two, "read")
    assert ref_one != ref_two


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

    # A cursor past machine one's (empty) tail never surfaces machine two's
    # event.
    body = client.get(
        changes_url(machine_one, limit=10, cursor=f"{T5}|{eid(999)}")
    ).json()
    assert body["records"] == []
    assert body["has_more"] is False


# --------------------------------------------------------------------------- #
# Damaged or misowned records are kept exactly as stored
# --------------------------------------------------------------------------- #


def test_damaged_chain_fields_are_emitted_unmodified(client):
    machine_id = create_machine(client)
    insert_event_row(
        client,
        machine_id,
        eid(1),
        T1,
        previous_event_id=eid(7),
        content_hash=HASH_A,
        chain_hash=HASH_B,
    )
    (record,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert record["previous_event_id"] == eid(7)
    assert record["content_hash"] == HASH_A
    assert record["chain_hash"] == HASH_B


def test_misowned_previous_event_link_is_emitted_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    # A record of machine two that machine one's record points at: the dangling
    # cross-machine link is kept exactly as stored, never repaired or filtered.
    insert_event_row(client, machine_two, eid(9), T1)
    insert_event_row(
        client, machine_one, eid(1), T2, previous_event_id=eid(9)
    )

    (record,) = client.get(changes_url(machine_one, limit=10)).json()["records"]
    assert record["previous_event_id"] == eid(9)
    # Machine two's own event never enters machine one's page.
    assert all(r["id"] != eid(9) for r in [record])


def test_stored_fields_are_emitted_verbatim(client):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        allowed=0, reason="custom_reason",
        previous_event_id=eid(7), content_hash=HASH_A, chain_hash=HASH_B,
    )
    (record,) = client.get(changes_url(machine_id, limit=10)).json()["records"]
    assert record["allowed"] is False
    assert record["reason"] == "custom_reason"
    assert record["previous_event_id"] == eid(7)
    assert record["content_hash"] == HASH_A
    assert record["chain_hash"] == HASH_B


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
            return list(conn.execute(text("SELECT * FROM authorization_decision_events")))

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

    # Insert both an event before the cursor position and one after it.
    insert_event_row(client, machine_id, eid(3), T1)  # sorts before T2
    insert_event_row(client, machine_id, eid(4), T3)  # sorts between T2/T4

    # Resuming from the old cursor returns only events after it, in the
    # established order; eid(1) and the earlier eid(3) never resurface.
    second = client.get(
        changes_url(machine_id, limit=10, cursor=old_cursor)
    ).json()
    assert [r["id"] for r in second["records"]] == [eid(4), eid(2)]
    assert second["has_more"] is False

    # A fresh walk from the start sees the new complete order.
    records, _ = fetch_all_pages(client, machine_id, limit=10)
    assert [r["id"] for r in records] == [eid(3), eid(1), eid(4), eid(2)]


def test_has_more_reflects_records_after_position_only(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_event_row(client, machine_id, eid(2), T1)
    insert_event_row(client, machine_id, eid(3), T2)

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
        changes_url(machine_id, limit=1, cursor=f"{T1}|{eid(2)}")
    ).json()
    assert [r["id"] for r in body["records"]] == [eid(3)]
    assert body["has_more"] is False


def test_body_is_compact_newline_terminated_json_with_fixed_field_order(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)

    response = client.get(changes_url(machine_id, limit=10))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    # Compact separators: no spaces after ':' or ','.
    assert b'": ' not in raw
    assert b", " not in raw
    # Fixed envelope and item field order, verifiable by exact reconstruction.
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
        "limit": 10,
        "records": [expected_item],
        "next_cursor": None,
        "has_more": False,
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_next_cursor_page_is_byte_stable_across_repeats(client):
    machine_id = create_machine(client)
    for n, stamp in enumerate((T0, T1, T2), start=1):
        insert_event_row(client, machine_id, eid(n), stamp)

    cursor = client.get(changes_url(machine_id, limit=1)).json()["next_cursor"]
    first = client.get(changes_url(machine_id, limit=1, cursor=cursor)).content
    second = client.get(changes_url(machine_id, limit=1, cursor=cursor)).content
    assert first == second
    assert json.loads(first)["records"][0]["id"] == eid(2)


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
