"""Tests for the read-only fixed-window grant-lifecycle compliance export.

Covers
`GET /machines/{machine_id}/authorization-grant-lifecycle-events/compliance-export`:
closed-UTC-window filtering on each event's own ``occurred_at``, ordering
by the actual UTC instant then event id (exact-second events before
fractional-second events of the same second; unparseable, offset-form, or
non-``Z`` stamps are excluded while left untouched in storage), verbatim
export of all nine lifecycle fields including damaged references, type,
chain fields, and hashes, the ``bad_time`` / ``invalid_query`` /
``not_found`` / ``internal_error`` outcomes, validation-before-machine-
lookup precedence, GET-only routing (including ``HEAD`` -> 405 without
reading), strict read-only byte stability, machine isolation, and
persistence across a restart.
"""
import json
from datetime import datetime, timedelta

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
    "events",
]
EVENT_KEYS = [
    "id",
    "machine_id",
    "grant_id",
    "authorization_event_id",
    "type",
    "occurred_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
]

TABLE = "authorization_grant_lifecycle_events"


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
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
        f"?from_occurred_at={from_occurred_at}"
        f"&to_occurred_at={to_occurred_at}"
    )


def eid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_event(
    client,
    machine_id,
    n,
    occurred_at,
    *,
    type_="issued",
    previous_event_id=None,
    content_hash=HASH_A,
    chain_hash=HASH_B,
):
    """Insert a lifecycle row directly with a fixed id and timestamp.

    The reference and chain columns are supplied explicitly and non-null so
    the row stays exactly as given; the export never recomputes, repairs,
    or adjudicates them.
    """
    values = {
        "id": eid(n),
        "machine_id": machine_id,
        "grant_id": eid(2000 + n),
        "authorization_event_id": eid(3000 + n),
        "type": type_,
        "occurred_at": occurred_at,
        "previous_event_id": previous_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {TABLE} "
                "(id, machine_id, grant_id, authorization_event_id, type, "
                "occurred_at, previous_event_id, content_hash, chain_hash) "
                "VALUES "
                "(:id, :machine_id, :grant_id, :authorization_event_id, "
                ":type, :occurred_at, :previous_event_id, :content_hash, "
                ":chain_hash)"
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
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
        f"{query}"
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
    assert response.json()["events"] == []


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T2)

    response = client.get(export_url(machine_id, T2, T2))

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["events"]] == [eid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
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
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export",
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
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
        f"?from_occurred_at=garbage&to_occurred_at={T4}&x=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_param_errors_take_precedence_over_missing_machine(client):
    # No machine exists; every parameter error still reports its 422 code and
    # never 404.
    response = client.get(
        f"/machines/{MISSING_MACHINE}"
        "/authorization-grant-lifecycle-events/compliance-export"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(
        export_url(MISSING_MACHINE, "2026-13-01T00:00:00Z", T4)
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(export_url(MISSING_MACHINE, T4, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(
        f"/machines/{MISSING_MACHINE}"
        "/authorization-grant-lifecycle-events/compliance-export"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&x=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_errors_do_not_read_events(client):
    # With the events table dropped, a read would 500; validation-phase
    # errors must still come back as their 422 codes.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {TABLE}"))

    assert client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
    ).status_code == 422
    assert client.get(
        export_url(machine_id, "garbage", T4)
    ).json() == {"error": {"code": "bad_time"}}
    assert client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/compliance-export"
        f"?from_occurred_at={T0}&to_occurred_at={T4}&x=1"
    ).json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Machine existence and read faults
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"events" not in response.content


def test_read_failure_is_500_with_no_partial_events(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {TABLE}"))

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"events" not in response.content
    assert eid(1).encode() not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_events(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert eid(1).encode() not in response.content


# --------------------------------------------------------------------------- #
# Routing: only GET
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_non_get_methods_return_405_without_reading(client, method):
    machine_id = create_machine(client)
    # Drop the table so any event read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {TABLE}"))

    response = getattr(client, method)(export_url(machine_id, T0, T4))

    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Response shape and windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)

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
    assert len(body["events"]) == 1


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T0)
    insert_event(client, machine_id, 2, T4)

    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:05Z",
            "2026-03-01T00:00:09Z",
        )
    )

    assert response.status_code == 200
    assert response.json()["events"] == []


def test_machine_with_no_events_returns_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    assert response.json()["events"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 21, T1)
    insert_event(client, machine_id, 22, T2)
    insert_event(client, machine_id, 23, T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(21),
        eid(22),
        eid(23),
    ]


def test_window_excludes_events_outside_bounds(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 20, T0)
    insert_event(client, machine_id, 21, T1)
    insert_event(client, machine_id, 22, T2)
    insert_event(client, machine_id, 23, T3)
    insert_event(client, machine_id, 24, T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(21),
        eid(22),
        eid(23),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp is inside [T0, T1] by instant even though the
    # window bounds carry no fraction.
    machine_id = create_machine(client)
    insert_event(client, machine_id, 21, "2026-03-01T00:00:00.500000Z")
    insert_event(client, machine_id, 22, T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["events"]] == [eid(21), eid(22)]


def test_events_ordered_by_occurred_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_event(client, machine_id, 30, T3)
    insert_event(client, machine_id, 21, T2)
    insert_event(client, machine_id, 20, T2)
    insert_event(client, machine_id, 10, T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(10),
        eid(20),
        eid(21),
        eid(30),
    ]


def test_same_second_exact_second_sorts_before_fractional_seconds(client):
    # As text, "...:00.5Z" sorts *before* "...:00Z" ('.' < 'Z'), so a naive
    # lexicographic order inverts the true order within one second. The
    # export must order by the actual UTC instant.
    machine_id = create_machine(client)
    insert_event(client, machine_id, 20, "2026-03-01T00:00:00.900000Z")
    insert_event(client, machine_id, 10, "2026-03-01T00:00:00.500000Z")
    insert_event(client, machine_id, 1, T0)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(1),
        eid(10),
        eid(20),
    ]


def test_export_items_have_exactly_the_complete_event_fields(client):
    machine_id = create_machine(client)
    values = insert_event(client, machine_id, 1, T1)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["events"]
    assert len(exported) == 1
    assert list(exported[0].keys()) == EVENT_KEYS
    assert exported[0] == {key: values[key] for key in EVENT_KEYS}


def test_chain_fields_and_hashes_are_emitted_verbatim_not_recomputed(client):
    # Damaged/arbitrary chain data must come out exactly as stored: a
    # mismatched predecessor, a bogus type, and non-derived hashes are
    # exported, not fixed.
    machine_id = create_machine(client)
    values = insert_event(
        client,
        machine_id,
        1,
        T1,
        type_="revoked",
        previous_event_id=eid(999),
        content_hash=HASH_C,
        chain_hash="not-a-hash",
    )

    response = client.get(export_url(machine_id, T0, T4))

    event = response.json()["events"][0]
    assert event["type"] == "revoked"
    assert event["previous_event_id"] == eid(999)
    assert event["content_hash"] == HASH_C
    assert event["chain_hash"] == "not-a-hash"
    assert event["content_hash"] == values["content_hash"]


# --------------------------------------------------------------------------- #
# Damaged occurred_at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "damaged",
    [
        "not-a-time",
        "2026-03-01T00:00:01+00:00",   # offset form
        "2026-03-01T00:00:01.5+00:00",  # fractional offset form
        "2026-13-01T00:00:01Z",         # well-shaped but out of range
        "2026-03-01T00:00:01",          # missing Z
        "",                             # blank
    ],
)
def test_unparseable_occurred_at_is_excluded_and_request_succeeds(
    client, damaged
):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 10, T1)
    insert_event(client, machine_id, 99, damaged)

    # Even the wide finite window excludes the damaged stamp; the parseable
    # event is returned and the read does not crash.
    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["events"]] == [eid(10)]

    # The damaged row is left exactly as stored (still visible to the
    # tolerant incremental surface, sorted last).
    changes = client.get(
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes?limit=100"
    ).json()
    assert [r["id"] for r in changes["records"]][-1] == eid(99)
    assert changes["records"][-1]["occurred_at"] == damaged


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_events(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    insert_event(client, machine_one, 1, T1)
    insert_event(client, machine_two, 2, T1)

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    events = response.json()["events"]
    assert [r["id"] for r in events] == [eid(1)]
    assert all(r["machine_id"] == machine_one for r in events)


# --------------------------------------------------------------------------- #
# Read-only, determinism, serialization, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    insert_event(client, machine_id, 1, T1)
    insert_event(client, machine_id, 2, T2)
    insert_event(client, other_machine, 3, T1)
    insert_event(client, machine_id, 99, "not-a-time")

    integrity_path = (
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/integrity"
    )
    changes_path = (
        f"/machines/{machine_id}"
        "/authorization-grant-lifecycle-events/changes?limit=100"
    )
    before_integrity = client.get(integrity_path).content
    before_changes = client.get(changes_path).content

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (TABLE, "machines")
            }

    before = table_state()
    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    middle = table_state()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert len(first.json()["events"]) == 2
    assert before == middle == after
    assert client.get(integrity_path).content == before_integrity
    assert client.get(changes_path).content == before_changes


def test_response_is_compact_utf8_json_ending_in_single_newline(client):
    machine_id = create_machine(client)
    insert_event(client, machine_id, 1, T1)

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
        insert_event(first, machine_id, 1, T1)
        insert_event(first, machine_id, 2, T2)
        expected = first.get(
            export_url(machine_id, FROM_WIDE, TO_WIDE)
        ).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["events"]) == 2


# --------------------------------------------------------------------------- #
# End-to-end: events written by real grant actions
# --------------------------------------------------------------------------- #


def test_real_grant_lifecycle_events_export_in_window(client):
    machine_id = create_machine(client)
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
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
    decision = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    grant = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": decision["id"], "ttl_seconds": 60},
    ).json()
    use = client.post(
        f"/machines/{machine_id}/authorization-grants/{grant['id']}/consume"
    ).json()

    issued_at = grant["issued_at"]
    consumed_at = use["consumed_at"]

    # A window covering only the issuance returns just the issued event.
    response = client.get(
        export_url(machine_id, issued_at, issued_at)
    )
    assert response.status_code == 200
    body = response.json()
    assert [event["type"] for event in body["events"]] == ["issued"]
    event = body["events"][0]
    assert list(event.keys()) == EVENT_KEYS
    assert event["occurred_at"] == issued_at
    assert event["grant_id"] == grant["id"]
    assert event["authorization_event_id"] == decision["id"]
    assert event["previous_event_id"] is None

    # A window ending strictly before the issuance is empty.
    before_issuance = (
        datetime.fromisoformat(issued_at[:-1]) - timedelta(seconds=1)
    ).isoformat() + "Z"
    assert client.get(
        export_url(machine_id, FROM_WIDE, before_issuance)
    ).json()["events"] == []

    # The full window holds both events in chronological order and is
    # byte-stable across repeats.
    wide = client.get(
        export_url(machine_id, FROM_WIDE, TO_WIDE)
    )
    assert [event["type"] for event in wide.json()["events"]] == [
        "issued",
        "consumed",
    ]
    assert wide.json()["events"][1]["occurred_at"] == consumed_at
    assert client.get(
        export_url(machine_id, FROM_WIDE, TO_WIDE)
    ).content == wide.content
