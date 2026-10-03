"""Tests for the read-only fixed-window execution-receipt compliance export.

Covers `GET /machines/{machine_id}/execution-receipts/compliance-export`:
closed-UTC-window filtering on each receipt's own ``occurred_at``, ordering
by the actual UTC instant then receipt id (exact-second receipts before
fractional-second receipts of the same second; unparseable or non-``Z``
stamps are excluded while left untouched in storage), verbatim export of
all thirteen receipt fields including damaged chain fields and hashes, the
``bad_time`` / ``invalid_query`` / ``not_found`` / ``internal_error``
outcomes, validation-before-machine-lookup precedence, GET-only routing
(including ``HEAD`` -> 405 without reading), strict read-only byte
stability, machine isolation, and persistence across a restart.
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

FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

ENVELOPE_KEYS = [
    "machine_id",
    "from_occurred_at",
    "to_occurred_at",
    "receipts",
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


def export_url(machine_id, from_occurred_at=FROM_WIDE, to_occurred_at=TO_WIDE):
    return (
        f"/machines/{machine_id}/execution-receipts/compliance-export"
        f"?from_occurred_at={from_occurred_at}"
        f"&to_occurred_at={to_occurred_at}"
    )


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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
        "?to_occurred_at=&from_occurred_at=2026-03-01T00:00:00Z",
    ],
)
def test_missing_or_blank_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export{query}"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form instead of Z
        "2026-03-01T00:00:00-00:00",      # negative offset form
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
def test_invalid_from_occurred_at_is_bad_time(client, value):
    machine_id = create_machine(client)

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
def test_invalid_to_occurred_at_is_bad_time(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, value))

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_range_is_bad_time(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T4, T0))

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_second_bounds_are_accepted(client):
    machine_id = create_machine(client)

    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:00.123456Z",
            "2026-03-01T00:00:09.999999Z",
        )
    )

    assert response.status_code == 200
    assert response.json()["receipts"] == []


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T2)

    response = client.get(export_url(machine_id, T2, T2))

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["receipts"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&unexpected=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "params",
    [
        [
            ("from_occurred_at", T0),
            ("from_occurred_at", T1),
            ("to_occurred_at", T4),
        ],
        [
            ("from_occurred_at", T0),
            ("to_occurred_at", T3),
            ("to_occurred_at", T4),
        ],
    ],
)
def test_repeated_parameter_names_are_invalid_query(client, params):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export",
        params=params,
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.request(
        "GET",
        export_url(machine_id, T0, T4),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_takes_priority_over_bad_time(client):
    machine_id = create_machine(client)

    # Unknown name with a malformed bound is invalid_query, not bad_time.
    response = client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export"
        f"?from_occurred_at=garbage&to_occurred_at={T4}&x=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_param_errors_take_precedence_over_missing_machine(client):
    # No machine exists; every parameter error still reports its 422 code and
    # never 404.
    response = client.get(
        f"/machines/{MISSING_MACHINE}/execution-receipts/compliance-export"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(export_url(MISSING_MACHINE, "2026-13-01T00:00:00Z", T4))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(export_url(MISSING_MACHINE, T4, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(
        f"/machines/{MISSING_MACHINE}/execution-receipts/compliance-export"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&x=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_receipts(client):
    # With the receipts table dropped, a read would 500; validation-phase
    # errors must still come back as their 422 codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))

    assert client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export"
    ).status_code == 422
    assert client.get(
        export_url(machine_id, "garbage", T4)
    ).json() == {"error": {"code": "bad_time"}}
    assert client.get(
        f"/machines/{machine_id}/execution-receipts/compliance-export"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Machine existence and read faults
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"receipts" not in response.content


def test_read_failure_is_500_with_no_partial_receipts(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"receipts" not in response.content
    assert rid(1).encode() not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_receipts(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert rid(1).encode() not in response.content


# --------------------------------------------------------------------------- #
# Routing: only GET
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_non_get_methods_return_405_without_reading(client, method):
    machine_id = create_machine(client)
    # Drop the table so any receipt read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))

    response = getattr(client, method)(export_url(machine_id, T0, T4))

    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Response shape and windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T1)

    raw_from = "2026-03-01T00:00:00.100Z"
    raw_to = "2100-01-01T00:00:00.5Z"
    response = client.get(export_url(machine_id, raw_from, raw_to))

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    # The original bound text is echoed verbatim, never renormalized.
    assert body["from_occurred_at"] == raw_from
    assert body["to_occurred_at"] == raw_to
    assert len(body["receipts"]) == 1


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T0)
    insert_receipt(client, machine_id, 2, T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    assert response.json()["receipts"] == []


def test_machine_with_no_receipts_returns_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    assert response.json()["receipts"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 21, T1)
    insert_receipt(client, machine_id, 22, T2)
    insert_receipt(client, machine_id, 23, T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["receipts"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_window_excludes_receipts_outside_bounds(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 20, T0)
    insert_receipt(client, machine_id, 21, T1)
    insert_receipt(client, machine_id, 22, T2)
    insert_receipt(client, machine_id, 23, T3)
    insert_receipt(client, machine_id, 24, T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["receipts"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp is inside [T0, T1] by instant even though the
    # window bounds carry no fraction.
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 21, "2026-03-01T00:00:00.500000Z")
    insert_receipt(client, machine_id, 22, T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["receipts"]] == [rid(21), rid(22)]


def test_receipts_ordered_by_occurred_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_receipt(client, machine_id, 30, T3)
    insert_receipt(client, machine_id, 21, T2)
    insert_receipt(client, machine_id, 20, T2)
    insert_receipt(client, machine_id, 10, T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [r["id"] for r in response.json()["receipts"]] == [
        rid(10),
        rid(20),
        rid(21),
        rid(30),
    ]


def test_same_second_exact_second_sorts_before_fractional_seconds(client):
    # As text, "...:00.5Z" sorts *before* "...:00Z" ('.' < 'Z'), so a naive
    # lexicographic order inverts the true order within one second. The
    # export must order by the actual UTC instant.
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 20, "2026-03-01T00:00:00.900000Z")
    insert_receipt(client, machine_id, 10, "2026-03-01T00:00:00.500000Z")
    insert_receipt(client, machine_id, 1, T0)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["receipts"]] == [
        rid(1),
        rid(10),
        rid(20),
    ]


def test_export_items_have_exactly_the_complete_receipt_fields(client):
    machine_id = create_machine(client)
    values = insert_receipt(client, machine_id, 1, T1)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["receipts"]
    assert len(exported) == 1
    assert list(exported[0].keys()) == RECORD_KEYS
    assert exported[0] == {key: values[key] for key in RECORD_KEYS}


def test_chain_fields_and_hashes_are_emitted_verbatim_not_recomputed(client):
    # Damaged/arbitrary chain data must come out exactly as stored: a
    # mismatched predecessor and non-derived hashes are exported, not fixed.
    machine_id = create_machine(client)
    values = insert_receipt(
        client,
        machine_id,
        1,
        T1,
        previous_receipt_id=rid(999),
        content_hash=HASH_C,
        chain_hash="not-a-hash",
    )

    response = client.get(export_url(machine_id, T0, T4))

    receipt = response.json()["receipts"][0]
    assert receipt["previous_receipt_id"] == rid(999)
    assert receipt["content_hash"] == HASH_C
    assert receipt["chain_hash"] == "not-a-hash"
    assert receipt["content_hash"] == values["content_hash"]


# --------------------------------------------------------------------------- #
# Damaged occurred_at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("damaged", ["not-a-time", "2026-03-01T00:00:01+00:00"])
def test_unparseable_occurred_at_is_excluded_and_request_succeeds(client, damaged):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 10, T1)
    insert_receipt(client, machine_id, 99, damaged)

    # Even the wide finite window excludes the damaged stamp; the parseable
    # receipt is returned and the read does not crash.
    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["receipts"]] == [rid(10)]

    # The damaged row is left exactly as stored (still visible to the
    # tolerant incremental surface, sorted last).
    changes = client.get(
        f"/machines/{machine_id}/execution-receipts/changes?limit=100"
    ).json()
    assert [r["id"] for r in changes["records"]][-1] == rid(99)
    assert changes["records"][-1]["occurred_at"] == damaged


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_receipts(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    insert_receipt(client, machine_one, 1, T1)
    insert_receipt(client, machine_two, 2, T1)

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    receipts = response.json()["receipts"]
    assert [r["id"] for r in receipts] == [rid(1)]
    assert all(r["machine_id"] == machine_one for r in receipts)


# --------------------------------------------------------------------------- #
# Read-only, determinism, serialization, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    insert_receipt(client, machine_id, 1, T1)
    insert_receipt(client, machine_id, 2, T2)
    insert_receipt(client, other_machine, 3, T1)
    insert_receipt(client, machine_id, 99, "not-a-time")

    coverage_url = f"/machines/{machine_id}/execution-receipts/coverage"
    integrity_url = f"/machines/{machine_id}/execution-receipts/integrity"
    before_coverage = client.get(coverage_url).content
    before_integrity = client.get(integrity_url).content

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in ("execution_receipts", "machines")
            }

    before = table_state()
    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    middle = table_state()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert len(first.json()["receipts"]) == 2
    assert before == middle == after
    assert client.get(coverage_url).content == before_coverage
    assert client.get(integrity_url).content == before_integrity


def test_response_is_compact_utf8_json_ending_in_single_newline(client):
    machine_id = create_machine(client)
    insert_receipt(client, machine_id, 1, T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.content
    body.decode("utf-8")  # valid UTF-8
    assert body.endswith(b"\n")
    assert not body.endswith(b"\n\n")
    assert b", " not in body
    assert b'": ' not in body
    parsed = json.loads(body.decode("utf-8"))
    assert list(parsed.keys()) == ENVELOPE_KEYS
    # No float, -0.0, or non-finite token anywhere in the serialization.
    assert b"-0.0" not in body
    assert b"NaN" not in body
    assert b"Infinity" not in body


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_receipt(first, machine_id, 1, T1)
        insert_receipt(first, machine_id, 2, T2)
        expected = first.get(
            export_url(machine_id, FROM_WIDE, TO_WIDE)
        ).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["receipts"]) == 2
