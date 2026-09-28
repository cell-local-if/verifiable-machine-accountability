"""Tests for the read-only machine status-history compliance export.

Covers `GET /machines/{machine_id}/status-history/compliance-export`:

- validation before any machine or status event is read: ``bad_time`` for a
  missing, blank, offset, whitespace-padded, malformed, out-of-range, or
  inverted bound; ``invalid_query`` for unknown parameters, repeated names,
  or a carried request body — all 422 and taking priority over the machine
  lookup;
- GET-only ``405`` (including ``HEAD``) without reading or filtering,
  ``404 not_found`` for a missing machine carrying no records, and
  ``500 internal_error`` with no partial export on a real read failure;
- the fixed ``{machine_id, from_created_at, to_created_at,
  status_history}`` envelope, each record carrying exactly the status-history
  list fields ``{id, machine_id, from_status, to_status, created_at}``
  exactly as stored, and only the path machine's records;
- closed-UTC-window filtering on the record's own ``created_at`` instant,
  ordering by actual UTC instant then record id (exact-second records before
  fractional-second records of the same second), empty windows, and records
  with an unparseable ``created_at`` excluded from every finite window while
  remaining stored verbatim;
- byte-identical repeats, strict read-only behavior, compact
  newline-terminated JSON with no non-finite tokens, machine isolation, and
  persistence across a restart.
"""
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

FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"

RECORD_KEYS = ["id", "machine_id", "from_status", "to_status", "created_at"]
ENVELOPE_KEYS = ["machine_id", "from_created_at", "to_created_at", "status_history"]

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def export_path(machine_id):
    return f"/machines/{machine_id}/status-history/compliance-export"


def export_url(machine_id, from_created_at=FROM_WIDE, to_created_at=TO_WIDE):
    return (
        f"{export_path(machine_id)}?from_created_at={from_created_at}"
        f"&to_created_at={to_created_at}"
    )


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
    assert response.status_code in (200, 409)
    return response


def insert_status_event(
    client,
    machine_id,
    n,
    created_at,
    *,
    from_status="active",
    to_status="suspended",
):
    """Insert a machine status-history row directly with a fixed id/stamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO machine_status_events "
                "(id, machine_id, from_status, to_status, created_at) "
                "VALUES "
                "(:id, :machine_id, :from_status, :to_status, :created_at)"
            ),
            {
                "id": rid(n),
                "machine_id": machine_id,
                "from_status": from_status,
                "to_status": to_status,
                "created_at": created_at,
            },
        )


def list_history(client, machine_id):
    return client.get(f"/machines/{machine_id}/status-history").json()


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
        "?from_created_at=2026-03-01T00:00:00Z&to_created_at=",
    ],
)
def test_missing_or_blank_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)

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
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "not-a-time",
        "2026-13-01T00:00:00Z",           # invalid month
        "2026-02-30T00:00:00Z",           # invalid calendar day
        "2026-03-01T24:00:00Z",           # invalid hour
        "2026-03-01T00:60:00Z",           # invalid minute
        "2026-03-01T00:00:60Z",           # invalid second
    ],
)
def test_malformed_from_created_at_is_bad_time(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, value, T5))

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "garbage",
        "2026-03-01T00:00:00.123",        # fractional but missing Z
        "2026-13-01T00:00:00Z",
    ],
)
def test_malformed_to_created_at_is_bad_time(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, value))

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_range_is_bad_time(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T5, T0))

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
    assert list(response.json().keys()) == ENVELOPE_KEYS
    assert response.json()["status_history"] == []


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    created_at = list_history(client, machine_id)[0]["created_at"]

    response = client.get(export_url(machine_id, created_at, created_at))

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["status_history"]] == [
        list_history(client, machine_id)[0]["id"]
    ]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.get(
        f"{export_path(machine_id)}?from_created_at={T0}"
        f"&to_created_at={T5}&unexpected=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "pairs",
    [
        [
            ("from_created_at", T0),
            ("from_created_at", T1),
            ("to_created_at", T5),
        ],
        [
            ("from_created_at", T0),
            ("to_created_at", T5),
            ("to_created_at", T4),
        ],
    ],
)
def test_repeated_parameter_names_are_invalid_query(client, pairs):
    machine_id = create_machine(client)

    response = client.get(export_path(machine_id), params=pairs)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query_before_reads(client):
    machine_id = create_machine(client)

    response = client.request(
        "GET",
        export_url(machine_id, T0, T5),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    # The zero id names no machine in the fresh database; every parameter
    # error is still its 422 rather than a 404.
    base = export_path(MISSING_MACHINE)

    bad_value = client.get(f"{base}?from_created_at=nope&to_created_at={T5}")
    assert bad_value.status_code == 422
    assert bad_value.json() == {"error": {"code": "bad_time"}}

    inverted = client.get(f"{base}?from_created_at={T5}&to_created_at={T0}")
    assert inverted.status_code == 422
    assert inverted.json() == {"error": {"code": "bad_time"}}

    missing = client.get(base)
    assert missing.status_code == 422
    assert missing.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_records(client):
    # With the history table dropped, a read would fail; validation-phase
    # errors must still come back as their 422 codes, never 500.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    assert client.get(export_path(machine_id)).status_code == 422
    assert client.get(export_url(machine_id, "nope", T5)).json() == {
        "error": {"code": "bad_time"}
    }
    assert client.get(
        f"{export_path(machine_id)}?from_created_at={T0}"
        f"&to_created_at={T5}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}
    assert client.get(
        export_path(machine_id),
        params=[
            ("from_created_at", T0),
            ("from_created_at", T1),
            ("to_created_at", T5),
        ],
    ).json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Machine existence and method routing
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404_with_no_records(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T5))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"status_history" not in response.content


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted_without_reading(client, method):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 1, T1)
    # Drop the table so any history read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    response = getattr(client, method)(export_url(machine_id, T0, T5))

    assert response.status_code == 405


def test_read_failure_is_500_with_no_partial_export(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machine_status_events"))

    response = client.get(export_url(machine_id, T0, T5))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"status_history" not in response.content
    assert b'"id"' not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_export(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(export_url(machine_id, T0, T5))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"status_history" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape, windowing, ordering
# --------------------------------------------------------------------------- #


def test_envelope_shape_echoes_bounds_and_uses_compact_json(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == FROM_WIDE
    assert body["to_created_at"] == TO_WIDE
    assert len(body["status_history"]) == 1

    raw = response.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no ", " or ": " anywhere.
    assert b", " not in raw
    assert b'": ' not in raw
    assert json.loads(raw.decode("utf-8")) == body
    assert b"NaN" not in raw and b"Infinity" not in raw


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 1, T0)
    insert_status_event(client, machine_id, 2, T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    assert response.json()["status_history"] == []
    assert response.content.endswith(b'"status_history":[]}\n')


def test_machine_with_no_history_returns_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["status_history"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 21, T1)
    insert_status_event(client, machine_id, 22, T2)
    insert_status_event(client, machine_id, 23, T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_window_excludes_records_outside_bounds(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 20, T0)
    insert_status_event(client, machine_id, 21, T1)
    insert_status_event(client, machine_id, 22, T2)
    insert_status_event(client, machine_id, 23, T3)
    insert_status_event(client, machine_id, 24, T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(21),
        rid(22),
        rid(23),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # Lexicographically ".500000Z" sorts *before* "Z"; the export must
    # compare parsed instants so the fractional record falls in [T0, T1].
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 21, T0_FRAC)
    insert_status_event(client, machine_id, 22, T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(21),
        rid(22),
    ]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    # Insert so lexicographic text order would put the fractional row first.
    insert_status_event(client, machine_id, 2, T0_FRAC)
    insert_status_event(client, machine_id, 1, T0)

    response = client.get(export_url(machine_id, T0, T5))

    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(1),
        rid(2),
    ]


def test_records_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 30, T3)
    insert_status_event(client, machine_id, 21, T2)
    insert_status_event(client, machine_id, 20, T2)
    insert_status_event(client, machine_id, 10, T1)

    response = client.get(export_url(machine_id, T0, T5))

    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(10),
        rid(20),
        rid(21),
        rid(30),
    ]


def test_export_items_match_status_history_list_fields(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    listed = list_history(client, machine_id)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["status_history"]
    assert exported == listed
    assert [list(item.keys()) for item in exported] == [RECORD_KEYS] * len(
        exported
    )


# --------------------------------------------------------------------------- #
# Damaged stored values are kept, not filtered or repaired within the window
# --------------------------------------------------------------------------- #


def test_corrupt_status_edge_is_exported_unmodified(client):
    machine_id = create_machine(client)
    insert_status_event(
        client,
        machine_id,
        21,
        T1,
        from_status="suspended",
        to_status="active",
    )

    response = client.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    history = response.json()["status_history"]
    assert [r["id"] for r in history] == [rid(21)]
    assert history[0]["from_status"] == "suspended"
    assert history[0]["to_status"] == "active"


@pytest.mark.parametrize("bad_stamp", ["not-a-timestamp", "2026-13-40T99:99:99Z"])
def test_unparseable_created_at_never_enters_a_finite_window(client, bad_stamp):
    machine_id = create_machine(client)
    insert_status_event(client, machine_id, 1, T1)
    insert_status_event(client, machine_id, 2, bad_stamp)
    insert_status_event(client, machine_id, 3, T3)

    # Even a window stretching to the last representable UTC day excludes it:
    # an unparseable instant sorts after every parseable instant and therefore
    # cannot satisfy any closed finite interval.
    response = client.get(
        export_url(machine_id, FROM_WIDE, "9999-12-31T23:59:59Z")
    )

    assert response.status_code == 200
    history = response.json()["status_history"]
    assert [r["id"] for r in history] == [rid(1), rid(3)]

    # A narrow ordinary window behaves the same; the damaged row is still in
    # storage verbatim and the export never repairs or removes it.
    assert [
        r["id"]
        for r in client.get(export_url(machine_id, T0, T5)).json()[
            "status_history"
        ]
    ] == [rid(1), rid(3)]
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT created_at, from_status, to_status "
                "FROM machine_status_events WHERE id = :id"
            ),
            {"id": rid(2)},
        ).one()
    assert row.created_at == bad_stamp
    assert row.from_status == "active"
    assert row.to_status == "suspended"


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_other_machines_records_never_enter_the_export(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_status_event(client, machine_one, 1, T1)
    insert_status_event(client, machine_two, 2, T2)

    one = client.get(export_url(machine_one, T0, T5)).json()
    two = client.get(export_url(machine_two, T0, T5)).json()

    assert [r["id"] for r in one["status_history"]] == [rid(1)]
    assert all(r["machine_id"] == machine_one for r in one["status_history"])
    assert [r["id"] for r in two["status_history"]] == [rid(2)]
    assert all(r["machine_id"] == machine_two for r in two["status_history"])


# --------------------------------------------------------------------------- #
# Read-only, determinism, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, "machine-2")
    insert_status_event(client, machine_id, 1, T1)
    insert_status_event(client, machine_id, 2, T3)
    insert_status_event(client, other_machine, 3, T2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM machine_status_events")))

    before = table_state()
    first = client.get(export_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(export_url(machine_id, T0, T5))
    third = client.get(export_url(machine_id, T1, T2))
    third_again = client.get(export_url(machine_id, T1, T2))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert third.content == third_again.content
    assert before == middle == after
    # The other machine's record never appears, regardless of the window.
    assert rid(3).encode() not in first.content


def test_real_transitions_export_through_the_full_lifecycle(client):
    machine_id = create_machine(client)
    set_status(client, machine_id, "suspended")
    set_status(client, machine_id, "active")
    set_status(client, machine_id, "suspended")

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    history = response.json()["status_history"]
    assert [(r["from_status"], r["to_status"]) for r in history] == [
        ("active", "suspended"),
        ("suspended", "active"),
        ("active", "suspended"),
    ]
    assert all(r["machine_id"] == machine_id for r in history)


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_status_event(first, machine_id, 1, T1)
        insert_status_event(first, machine_id, 2, T3)
        expected = first.get(export_url(machine_id, T0, T5)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.content == expected
    assert [r["id"] for r in response.json()["status_history"]] == [
        rid(1),
        rid(2),
    ]


def test_empty_database_needs_no_migration_and_exports_empty(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "from_created_at": T0,
        "to_created_at": T5,
        "status_history": [],
    }
