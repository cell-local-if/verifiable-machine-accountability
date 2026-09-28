"""Tests for the read-only machine status-history compliance window export.

Covers `GET /machines/{machine_id}/status-history/compliance-export`:
closed-UTC-window filtering on each record's own ``created_at``, ordering by
the actual UTC instant then record id (exact-second records before
fractional-second records of the same second; unparseable stamps sort last
and never enter a finite window), verbatim export of damaged field values,
the ``bad_time`` / ``invalid_query`` / ``not_found`` / ``internal_error``
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

ENVELOPE_KEYS = ["machine_id", "from_created_at", "to_created_at", "status_history"]
RECORD_KEYS = ["id", "machine_id", "from_status", "to_status", "created_at"]


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


def set_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response


def export_url(machine_id, from_created_at=FROM_WIDE, to_created_at=TO_WIDE):
    return (
        f"/machines/{machine_id}/status-history/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_status_row(
    client,
    machine_id,
    record_id,
    created_at,
    *,
    from_status="active",
    to_status="suspended",
):
    """Insert a machine status event row directly with a fixed id/timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO machine_status_events "
                "(id, machine_id, from_status, to_status, created_at) "
                "VALUES (:id, :machine_id, :from_status, :to_status, :created_at)"
            ),
            {
                "id": record_id,
                "machine_id": machine_id,
                "from_status": from_status,
                "to_status": to_status,
                "created_at": created_at,
            },
        )


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
        "?from_created_at=&to_created_at=2026-03-01T00:00:05Z",
        "?from_created_at=%20&to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_or_blank_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/status-history/compliance-export{query}"
    )

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
def test_invalid_from_created_at_is_bad_time(client, value):
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
def test_invalid_to_created_at_is_bad_time(client, value):
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
    assert response.json()["status_history"] == []


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    listed = client.get(f"/machines/{machine_id}/status-history").json()
    created_at = listed[0]["created_at"]

    response = client.get(export_url(machine_id, created_at, created_at))

    assert response.status_code == 200
    assert [e["id"] for e in response.json()["status_history"]] == [listed[0]["id"]]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/status-history/compliance-export"
        f"?from_created_at={T0}&to_created_at={T4}&unexpected=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "params",
    [
        [("from_created_at", T0), ("from_created_at", T1), ("to_created_at", T4)],
        [("from_created_at", T0), ("to_created_at", T3), ("to_created_at", T4)],
    ],
)
def test_repeated_parameter_names_are_invalid_query(client, params):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/status-history/compliance-export",
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
        f"/machines/{machine_id}/status-history/compliance-export"
        f"?from_created_at=garbage&to_created_at={T4}&x=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_param_errors_take_precedence_over_missing_machine(client):
    # No machine exists; every parameter error still reports its 422 code and
    # never 404.
    response = client.get(
        f"/machines/{MISSING_MACHINE}/status-history/compliance-export"
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
        f"/machines/{MISSING_MACHINE}/status-history/compliance-export"
        f"?from_created_at={T0}&to_created_at={T4}&x=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_records(client):
    # With the history table dropped, a read would 500; validation-phase
    # errors must still come back as their 422 codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    assert client.get(
        f"/machines/{machine_id}/status-history/compliance-export"
    ).status_code == 422
    assert client.get(
        export_url(machine_id, "garbage", T4)
    ).json() == {"error": {"code": "bad_time"}}
    assert client.get(
        f"/machines/{machine_id}/status-history/compliance-export"
        f"?from_created_at={T0}&to_created_at={T4}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Machine existence and read faults
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"status_history" not in response.content


def test_read_failure_is_500_with_no_partial_records(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(1), T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"status_history" not in response.content
    assert rid(1).encode() not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_records(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(1), T1)
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
    # Drop the table so any history read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    response = getattr(client, method)(export_url(machine_id, T0, T4))

    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Response shape and windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")

    raw_from = "2026-03-01T00:00:00.100Z"
    raw_to = "2100-01-01T00:00:00.5Z"
    response = client.get(export_url(machine_id, raw_from, raw_to))

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    # The original bound text is echoed verbatim, never renormalized.
    assert body["from_created_at"] == raw_from
    assert body["to_created_at"] == raw_to
    assert len(body["status_history"]) == 1


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(1), T0)
    insert_status_row(client, machine_id, rid(2), T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    assert response.json()["status_history"] == []


def test_machine_with_no_history_returns_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    assert response.json()["status_history"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(21), T1)
    insert_status_row(client, machine_id, rid(22), T2)
    insert_status_row(client, machine_id, rid(23), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [e["id"] for e in response.json()["status_history"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_window_excludes_records_outside_bounds(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(20), T0)
    insert_status_row(client, machine_id, rid(21), T1)
    insert_status_row(client, machine_id, rid(22), T2)
    insert_status_row(client, machine_id, rid(23), T3)
    insert_status_row(client, machine_id, rid(24), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [e["id"] for e in response.json()["status_history"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp is inside [T0, T1] by instant even though the
    # window bounds carry no fraction.
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(21), "2026-03-01T00:00:00.500000Z")
    insert_status_row(client, machine_id, rid(22), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [e["id"] for e in response.json()["status_history"]] == [rid(21), rid(22)]


def test_records_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_status_row(client, machine_id, rid(30), T3)
    insert_status_row(client, machine_id, rid(21), T2)
    insert_status_row(client, machine_id, rid(20), T2)
    insert_status_row(client, machine_id, rid(10), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [e["id"] for e in response.json()["status_history"]] == [
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
    insert_status_row(client, machine_id, rid(20), "2026-03-01T00:00:00.900000Z")
    insert_status_row(client, machine_id, rid(10), "2026-03-01T00:00:00.500000Z")
    insert_status_row(client, machine_id, rid(1), T0)

    response = client.get(export_url(machine_id, T0, T1))

    assert [e["id"] for e in response.json()["status_history"]] == [
        rid(1),
        rid(10),
        rid(20),
    ]


def test_export_items_have_exactly_the_list_endpoint_fields(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    listed = client.get(f"/machines/{machine_id}/status-history").json()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["status_history"]
    # The export keeps its five-field shape; the status-history list now also
    # carries the chain fields, so compare on the export's field set.
    assert exported == [
        {key: item[key] for key in RECORD_KEYS} for item in listed
    ]
    assert list(exported[0].keys()) == RECORD_KEYS


# --------------------------------------------------------------------------- #
# Damaged data
# --------------------------------------------------------------------------- #


def test_corrupt_status_edge_is_exported_unmodified(client):
    machine_id = create_machine(client)
    insert_status_row(
        client,
        machine_id,
        rid(21),
        T1,
        from_status="suspended",
        to_status="suspended",
    )

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    history = response.json()["status_history"]
    assert [e["id"] for e in history] == [rid(21)]
    assert history[0]["from_status"] == "suspended"
    assert history[0]["to_status"] == "suspended"


def test_unparseable_created_at_never_enters_a_finite_window(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(10), T1)
    insert_status_row(client, machine_id, rid(99), "not-a-time")

    # Even the wide finite window excludes the unparseable stamp; the
    # parseable record is returned and the read does not crash.
    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert [e["id"] for e in response.json()["status_history"]] == [rid(10)]

    # The damaged row is left exactly as stored (still visible to the
    # tolerant incremental surface, sorted last).
    changes = client.get(
        f"/machines/{machine_id}/status-history/changes?limit=100"
    ).json()
    assert [r["id"] for r in changes["records"]][-1] == rid(99)
    assert changes["records"][-1]["created_at"] == "not-a-time"


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_history(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    set_status(client, machine_one, "suspended")
    set_status(client, machine_two, "suspended")

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    history = response.json()["status_history"]
    assert len(history) == 1
    assert all(e["machine_id"] == machine_one for e in history)


def test_other_machine_rows_never_enter_via_direct_insert(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    insert_status_row(client, machine_one, rid(1), T1)
    insert_status_row(client, machine_two, rid(2), T1)

    response = client.get(export_url(machine_one, T0, T4))

    assert [e["id"] for e in response.json()["status_history"]] == [rid(1)]


# --------------------------------------------------------------------------- #
# Read-only, determinism, serialization, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    set_status(client, other_machine, "suspended")

    history_url = f"/machines/{machine_id}/status-history"
    machine_url = f"/machines/{machine_id}"
    before_history = client.get(history_url).content
    before_machine = client.get(machine_url).content

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in ("machine_status_events", "machines")
            }

    before = table_state()
    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    middle = table_state()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert len(first.json()["status_history"]) == 2
    assert before == middle == after
    assert client.get(history_url).content == before_history
    assert client.get(machine_url).content == before_machine


def test_response_is_compact_utf8_json_ending_in_single_newline(client):
    machine_id = create_machine(client)
    insert_status_row(client, machine_id, rid(1), T1)

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
        set_status(first, machine_id, "suspended")
        set_status(first, machine_id, "active")
        expected = first.get(
            export_url(machine_id, FROM_WIDE, TO_WIDE)
        ).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["status_history"]) == 2
