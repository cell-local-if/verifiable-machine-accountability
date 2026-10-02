"""Tests for the read-only execution-receipt compliance window export.

Covers `GET /machines/{machine_id}/execution-receipts/compliance-export`:

- validation before any machine or receipt is read: ``bad_time`` for a
  missing, blank, offset, whitespace-padded, malformed, out-of-range, or
  inverted bound (equal bounds are valid); ``invalid_query`` for unknown
  parameters, repeated bound names, or a carried request body — all 422 and
  taking priority over the machine lookup;
- GET-only ``405`` (including ``HEAD``) without reading receipts,
  ``404 not_found`` for a missing machine carrying no receipts, and
  ``500 internal_error`` with no partial records when the receipts or the
  machine cannot be read;
- the fixed ``{machine_id, from_occurred_at, to_occurred_at,
  execution_receipts}`` envelope, where each record carries exactly the
  complete receipt fields (the ten content fields plus
  ``previous_receipt_id``/``content_hash``/``chain_hash``) exactly as
  stored, in the fixed execution-receipts changes field order;
- closed-window membership on the actual UTC instant of ``occurred_at``
  (both edges included; an exact-second stamp and fractional stamps compare
  as true instants), ordering by that instant then receipt id, and machine
  isolation;
- a receipt with an unparseable ``occurred_at`` never enters a finite
  window while its stored text is left untouched;
- verbatim export of damaged chain content with no repair or
  normalization;
- byte-identical repeat exports, strict read-only behavior, compact
  newline-terminated JSON with no non-finite tokens, and persistence
  across a restart.
"""
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"
T0_FRAC = "2026-03-01T00:00:00.500000Z"

FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"

HASH_A = "a" * 64
HASH_B = "b" * 64

ENVELOPE_KEYS = [
    "machine_id",
    "from_occurred_at",
    "to_occurred_at",
    "execution_receipts",
]
RECORD_KEYS = [
    "id",
    "machine_id",
    "use_id",
    "grant_id",
    "authorization_event_id",
    "action_type",
    "resource",
    "outcome",
    "result_digest",
    "occurred_at",
    "previous_receipt_id",
    "content_hash",
    "chain_hash",
]


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


def export_path(machine_id):
    return f"/machines/{machine_id}/execution-receipts/compliance-export"


def export_url(machine_id, from_occurred_at=FROM_WIDE, to_occurred_at=TO_WIDE):
    return (
        f"{export_path(machine_id)}"
        f"?from_occurred_at={from_occurred_at}&to_occurred_at={to_occurred_at}"
    )


def insert_receipt(
    client,
    machine_id,
    n,
    occurred_at,
    *,
    action_type="read",
    resource=None,
    outcome="succeeded",
    result_digest=HASH_A,
    previous_receipt_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert a receipt row directly with a fixed id and timestamp.

    The chain columns are supplied explicitly and non-null so the row stays
    exactly as given; the export never recomputes, repairs, or adjudicates
    them.
    """
    if resource is None:
        resource = f"res/{n}"
    values = {
        "id": rid(n),
        "machine_id": machine_id,
        "use_id": rid(1000 + n),
        "grant_id": rid(2000 + n),
        "authorization_event_id": rid(3000 + n),
        "action_type": action_type,
        "resource": resource,
        "outcome": outcome,
        "result_digest": result_digest,
        "occurred_at": occurred_at,
        "previous_receipt_id": previous_receipt_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts "
                "(id, machine_id, use_id, grant_id, authorization_event_id, "
                "action_type, resource, outcome, result_digest, occurred_at, "
                "previous_receipt_id, content_hash, chain_hash) "
                "VALUES "
                "(:id, :machine_id, :use_id, :grant_id, "
                ":authorization_event_id, :action_type, :resource, :outcome, "
                ":result_digest, :occurred_at, :previous_receipt_id, "
                ":content_hash, :chain_hash)"
            ),
            values,
        )
    return values


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_occurred_at=2026-03-01T00:00:00Z",
        "?to_occurred_at=2026-03-01T00:00:05Z",
        "?from_occurred_at=&to_occurred_at=2026-03-01T00:00:05Z",
        "?from_occurred_at=%20&to_occurred_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_or_blank_bounds_are_bad_time(client, machine_id, query):
    response = client.get(f"{export_path(machine_id)}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form instead of Z
        "2026-03-01T00:00:00z",           # lowercase suffix
        "2026-03-01 00:00:00Z",           # space separator
        "2026-03-01T00:00:00.Z",          # dot without fraction digits
        "not-a-time",
        "",                               # blank
        "2026-13-01T00:00:00Z",           # invalid month
        "2026-02-30T00:00:00Z",           # invalid calendar day
        "2026-03-01T24:00:00Z",           # invalid hour
        "2026-03-01T00:60:00Z",           # invalid minute
        "2026-03-01T00:00:60Z",           # invalid second
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
    ],
)
def test_invalid_from_bound_is_bad_time(client, machine_id, value):
    response = client.get(export_url(machine_id, value, T4))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "garbage",
        "2026-03-01T00:00:00.123",
        " 2026-03-01T00:00:00Z",
    ],
)
def test_invalid_to_bound_is_bad_time(client, machine_id, value):
    response = client.get(export_url(machine_id, T0, value))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_range_is_bad_time(client, machine_id):
    response = client.get(export_url(machine_id, T4, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_valid(client, machine_id):
    insert_receipt(client, machine_id, 1, T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["execution_receipts"]] == [rid(1)]


def test_fractional_second_bounds_are_valid(client, machine_id):
    response = client.get(export_url(machine_id, T0_FRAC, T0_FRAC))
    assert response.status_code == 200
    assert response.json()["execution_receipts"] == []


def test_unknown_query_parameter_is_invalid_query(client, machine_id):
    response = client.get(
        f"{export_path(machine_id)}"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_bound_names_are_invalid_query(client, machine_id):
    response = client.get(
        export_path(machine_id),
        params=[
            ("from_occurred_at", T0),
            ("from_occurred_at", T1),
            ("to_occurred_at", T4),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        export_path(machine_id),
        params=[
            ("from_occurred_at", T0),
            ("to_occurred_at", T4),
            ("to_occurred_at", T5),
        ],
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_reads(client, machine_id):
    response = client.request(
        "GET",
        export_url(machine_id, T0, T4),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_parameter_errors_take_priority_over_machine_lookup(client):
    # No machine exists; every parameter error is still its own 422.
    assert client.get(export_path(MISSING_MACHINE)).status_code == 422
    assert client.get(export_url(MISSING_MACHINE, T4, T0)).json() == {
        "error": {"code": "bad_time"}
    }
    assert client.get(
        f"{export_path(MISSING_MACHINE)}?from_occurred_at={T0}"
        f"&to_occurred_at={T4}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}
    assert client.get(
        export_path(MISSING_MACHINE),
        params=[("from_occurred_at", T0), ("from_occurred_at", T1),
                ("to_occurred_at", T4)],
    ).json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_records(client, machine_id):
    # With the table dropped, a read would fail; validation-phase errors
    # must still come back as their 422 codes, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))

    assert client.get(export_path(machine_id)).status_code == 422
    assert client.get(export_url(machine_id, T4, T0)).json() == {
        "error": {"code": "bad_time"}
    }
    assert client.get(
        f"{export_path(machine_id)}?from_occurred_at={T0}"
        f"&to_occurred_at={T4}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Routing, machine lookup, read failures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_non_get_methods_return_405_without_reading(client, machine_id, method):
    # Drop the table so any receipt read would 500; method routing wins.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))

    response = getattr(client, method)(export_url(machine_id, T0, T4))
    assert response.status_code == 405


def test_missing_machine_is_404_with_no_receipt_records(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T4))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"execution_receipts" not in response.content


def test_read_failure_is_500_with_no_partial_records(client, machine_id):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))
    response = client.get(export_url(machine_id, T0, T4))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"execution_receipts" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_records(
    client, machine_id
):
    insert_receipt(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(export_url(machine_id, T0, T4))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"execution_receipts" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, windowing, ordering
# --------------------------------------------------------------------------- #


def test_empty_window_returns_empty_array(client, machine_id):
    response = client.get(export_url(machine_id, T2, T3))
    assert response.status_code == 200
    assert list(response.json().keys()) == ENVELOPE_KEYS
    assert response.json() == {
        "machine_id": machine_id,
        "from_occurred_at": T2,
        "to_occurred_at": T3,
        "execution_receipts": [],
    }


def test_empty_machine_serves_empty_window(client, machine_id):
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert response.json()["execution_receipts"] == []


def test_bounds_are_echoed_verbatim(client, machine_id):
    raw_from = "2026-03-01T00:00:00.100000Z"
    raw_to = "2026-03-01T00:00:04Z"
    body = client.get(export_url(machine_id, raw_from, raw_to)).json()
    assert body["from_occurred_at"] == raw_from
    assert body["to_occurred_at"] == raw_to


def test_closed_window_includes_both_edges(client, machine_id):
    insert_receipt(client, machine_id, 1, T0)
    insert_receipt(client, machine_id, 2, T2)
    insert_receipt(client, machine_id, 3, T4)

    body = client.get(export_url(machine_id, T1, T3)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(2)]

    # Receipts exactly on either edge are included (closed interval), and a
    # receipt outside either edge is not.
    body = client.get(export_url(machine_id, T0, T4)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [
        rid(1), rid(2), rid(3)
    ]

    body = client.get(export_url(machine_id, T0, T2)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(1), rid(2)]

    body = client.get(export_url(machine_id, T2, T4)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [
        rid(2), rid(3)
    ]


def test_window_compares_true_instants_with_fractional_bounds(client, machine_id):
    insert_receipt(client, machine_id, 1, T0)
    insert_receipt(client, machine_id, 2, T0_FRAC)
    insert_receipt(client, machine_id, 3, T1)

    # Window ending between the exact-second stamp and the half-second one.
    body = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:00Z",
            "2026-03-01T00:00:00.250000Z",
        )
    ).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(1)]

    # The fractional receipt enters once the upper edge reaches it.
    body = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:00.500000Z",
            "2026-03-01T00:00:00.500000Z",
        )
    ).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(2)]


def test_records_are_ordered_by_instant_then_id(client, machine_id):
    for n, stamp in ((1, T4), (2, T0), (3, T2), (4, T1), (5, T3)):
        insert_receipt(client, machine_id, n, stamp)

    body = client.get(export_url(machine_id, T0, T4)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [
        rid(2), rid(4), rid(3), rid(5), rid(1)
    ]
    assert [r["occurred_at"] for r in body["execution_receipts"]] == [
        T0, T1, T2, T3, T4
    ]


def test_exact_second_sorts_before_fractional_same_second(client, machine_id):
    # Insert so lexicographic text order would put the fractional row first.
    insert_receipt(client, machine_id, 2, T0_FRAC)
    insert_receipt(client, machine_id, 1, T0)

    body = client.get(export_url(machine_id, T0, T1)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(1), rid(2)]


def test_same_instant_tie_breaks_by_receipt_id(client, machine_id):
    insert_receipt(client, machine_id, 30, T2)
    insert_receipt(client, machine_id, 20, T2)
    insert_receipt(client, machine_id, 10, T2)

    body = client.get(export_url(machine_id, T0, T4)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [
        rid(10), rid(20), rid(30)
    ]


def test_records_carry_exactly_the_complete_chain_fields(client, machine_id):
    stored = insert_receipt(
        client, machine_id, 7, "2026-03-01T00:00:00.250Z",
        action_type="write", resource="res/secret", outcome="failed",
        result_digest=HASH_B, previous_receipt_id=rid(3),
        content_hash=HASH_A, chain_hash=HASH_B,
    )
    record = client.get(
        export_url(machine_id, T0, T4)
    ).json()["execution_receipts"][0]
    assert list(record.keys()) == RECORD_KEYS
    assert record == {
        "id": rid(7),
        "machine_id": machine_id,
        "use_id": rid(1007),
        "grant_id": rid(2007),
        "authorization_event_id": rid(3007),
        "action_type": "write",
        "resource": "res/secret",
        "outcome": "failed",
        "result_digest": HASH_B,
        "occurred_at": "2026-03-01T00:00:00.250Z",
        "previous_receipt_id": rid(3),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    assert stored["id"] == record["id"]


def test_only_the_path_machines_records_are_exported(client, machine_id):
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    insert_receipt(client, machine_id, 1, T0)
    insert_receipt(client, other["id"], 2, T1)
    insert_receipt(client, machine_id, 3, T2)

    body = client.get(export_url(machine_id)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(1), rid(3)]
    assert all(
        r["machine_id"] == machine_id for r in body["execution_receipts"]
    )

    other_body = client.get(export_url(other["id"])).json()
    assert [r["id"] for r in other_body["execution_receipts"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Damaged stored values
# --------------------------------------------------------------------------- #


def test_unparseable_occurred_at_never_enters_window_but_is_kept(
    client, machine_id
):
    insert_receipt(client, machine_id, 1, T1)
    insert_receipt(client, machine_id, 2, "not-a-timestamp")
    insert_receipt(client, machine_id, 3, "2026-13-40T99:99:99Z")

    # No finite window contains a stamp with no actual UTC instant.
    body = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    assert [r["id"] for r in body["execution_receipts"]] == [rid(1)]

    # The stored text was not repaired, recomputed, or deleted: the
    # unbounded changes surface still emits it verbatim.
    rows = client.get(
        f"/machines/{machine_id}/execution-receipts/changes?limit=100"
    ).json()["records"]
    by_id = {row["id"]: row for row in rows}
    assert by_id[rid(2)]["occurred_at"] == "not-a-timestamp"
    assert by_id[rid(3)]["occurred_at"] == "2026-13-40T99:99:99Z"

    with client.app.state.engine.connect() as conn:
        stored = list(
            conn.execute(
                text("SELECT id, occurred_at FROM execution_receipts")
            )
        )
    assert {row[0]: row[1] for row in stored}[rid(2)] == "not-a-timestamp"


def test_damaged_chain_fields_are_emitted_unmodified(client, machine_id):
    insert_receipt(
        client, machine_id, 1, T1, previous_receipt_id=rid(7),
        content_hash="z" * 64, chain_hash="q" * 64,
    )
    (record,) = client.get(
        export_url(machine_id, T0, T4)
    ).json()["execution_receipts"]
    assert record["previous_receipt_id"] == rid(7)
    assert record["content_hash"] == "z" * 64
    assert record["chain_hash"] == "q" * 64


def test_misowned_chain_link_is_emitted_unmodified(client, machine_id):
    # A previous-receipt link pointing at another machine's receipt is
    # damaged chain content: the export keeps it exactly as stored rather
    # than repairing, recomputing, or dropping the record.
    other = client.post(
        "/machines",
        json={
            "external_id": "machine-2",
            "display_name": "Machine Two",
            "public_key": "key-9",
        },
    ).json()
    foreign = insert_receipt(client, other["id"], 9, T0)
    insert_receipt(
        client, machine_id, 1, T1, previous_receipt_id=foreign["id"],
    )
    (record,) = client.get(
        export_url(machine_id, T0, T4)
    ).json()["execution_receipts"]
    assert record["previous_receipt_id"] == foreign["id"]
    assert record["machine_id"] == machine_id


def test_duplicate_and_illegal_field_values_are_emitted_verbatim(
    client, machine_id
):
    # Repeated non-key values and illegal outcome/digest values are never
    # normalized, filtered, or re-adjudicated: every in-window row is
    # exported exactly as stored. (The database-level unique constraint
    # prevents a duplicated ``use_id`` from existing in the first place.)
    first = insert_receipt(
        client, machine_id, 1, T1, outcome="bogus",
        result_digest="Z" * 64,
    )
    second = insert_receipt(
        client, machine_id, 2, T2, outcome="bogus",
        result_digest="not-a-digest",
    )
    # Repeat the first receipt's grant/event binding on the second row.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE execution_receipts SET grant_id = :grant_id, "
                "authorization_event_id = :event_id WHERE id = :id"
            ),
            {
                "grant_id": first["grant_id"],
                "event_id": first["authorization_event_id"],
                "id": second["id"],
            },
        )
    rows = client.get(
        export_url(machine_id, T0, T4)
    ).json()["execution_receipts"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
    assert [r["outcome"] for r in rows] == ["bogus", "bogus"]
    assert rows[0]["result_digest"] == "Z" * 64
    assert rows[1]["result_digest"] == "not-a-digest"
    assert rows[1]["grant_id"] == rows[0]["grant_id"]
    assert (
        rows[1]["authorization_event_id"]
        == rows[0]["authorization_event_id"]
    )


# --------------------------------------------------------------------------- #
# Stability, read-only, body format, restart
# --------------------------------------------------------------------------- #


def test_same_query_is_byte_stable_when_data_unchanged(client, machine_id):
    for n, stamp in ((1, T0), (2, T1), (3, T2), (4, T3)):
        insert_receipt(client, machine_id, n, stamp)

    first = client.get(export_url(machine_id, T0, T3)).content
    second = client.get(export_url(machine_id, T0, T3)).content
    assert first == second


def _reject_non_finite(marker):
    # parse_constant only fires for NaN/Infinity tokens; reaching it means a
    # non-finite literal slipped into the body.
    raise AssertionError(f"non-finite token in response: {marker}")


def test_body_is_compact_newline_terminated_json_with_fixed_order(
    client, machine_id
):
    insert_receipt(
        client, machine_id, 1, T1,
        action_type="read", resource="res/1", outcome="succeeded",
        result_digest=HASH_A, previous_receipt_id=None,
        content_hash=HASH_A, chain_hash=HASH_B,
    )
    response = client.get(export_url(machine_id, T0, T4))
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
        "use_id": rid(1001),
        "grant_id": rid(2001),
        "authorization_event_id": rid(3001),
        "action_type": "read",
        "resource": "res/1",
        "outcome": "succeeded",
        "result_digest": HASH_A,
        "occurred_at": T1,
        "previous_receipt_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    expected = {
        "machine_id": machine_id,
        "from_occurred_at": T0,
        "to_occurred_at": T4,
        "execution_receipts": [expected_item],
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    # No NaN/Infinity/-0.0-style non-finite or float content.
    json.loads(raw, parse_constant=_reject_non_finite)


def test_utf8_content_is_round_tripped_byte_stably(client, machine_id):
    # Stored text is emitted verbatim and must survive as compact UTF-8 JSON
    # without ASCII escaping.
    insert_receipt(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE execution_receipts SET resource = :value "
                "WHERE id = :id"
            ),
            {"value": "res/対象-★", "id": rid(1)},
        )

    first = client.get(export_url(machine_id, T0, T4)).content
    second = client.get(export_url(machine_id, T0, T4)).content
    assert first == second
    assert "対象".encode("utf-8") in first
    body = json.loads(first)
    assert body["execution_receipts"][0]["resource"] == "res/対象-★"


def test_query_is_read_only(client, machine_id):
    insert_receipt(client, machine_id, 1, T1)
    insert_receipt(client, machine_id, 2, "not-a-timestamp")

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM execution_receipts")))

    before = table_state()
    client.get(export_url(machine_id, T0, T4))
    client.get(export_url(machine_id, T2, T3))
    client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    after = table_state()
    assert before == after


def test_export_persists_across_restart(tmp_path, monkeypatch):
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
            insert_receipt(first, machine_id, n, stamp)
        expected = first.get(export_url(machine_id, T1, T2)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T1, T2))

    assert response.status_code == 200
    assert response.content == expected
    assert [
        r["id"] for r in response.json()["execution_receipts"]
    ] == [rid(2), rid(3)]
