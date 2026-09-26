"""Tests for the read-only desensitized authorization decision event export.

Covers `GET /machines/{machine_id}/authorization-decision-events/privacy-export`:
the strict query validation (``bad_time`` / ``invalid_query`` before any machine
or event is read), ``404 not_found``, GET-only ``405`` routing, closed-UTC-
window filtering on each event's own ``created_at``, ordering by the actual
UTC instant then event id (exact-second before fractional-second), the
``action_ref`` / ``resource_ref`` SHA-256 desensitizing digests (including
``null`` for non-string or blank values), the absence of raw action/resource
text, verbatim export of missing, misowned, duplicated, or chain-damaged
records, machine isolation, the fixed-field-order compact newline-terminated
body, strict read-only byte stability, and persistence across a restart.
"""
import hashlib
import json
import sqlite3

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


WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def export_url(machine_id, from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/authorization-decision-events/privacy-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


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


HASH_A = "a" * 64
HASH_B = "b" * 64


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
    only fills NULLs) leaves them exactly as given; the privacy export never
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


def expected_ref(kind, machine_id, value):
    return hashlib.sha256(
        f"privacy:v1|{kind}{machine_id}{value}".encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/privacy-export{query}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form
        "2026-03-01T00:00:00z",           # lowercase suffix
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "2026-03-01 00:00:00Z",           # space separator
        "2026-03-01T00:00:00.Z",          # dot without fraction digits
        "garbage",
        "",                               # blank
        "   ",                            # whitespace only
        "2026-13-01T00:00:00Z",           # bad month
        "2026-02-30T00:00:00Z",           # bad day
        "2026-03-01T24:00:00Z",           # bad hour
        "2026-03-01T00:60:00Z",           # bad minute
        "2026-03-01T00:00:60Z",           # bad second
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.get(export_url(machine_id, T0, value))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:00:00.5Z",
        "2026-03-01T00:00:00.123456789Z",
    ],
)
def test_fractional_second_bounds_are_accepted(client, value):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, value, WIDE[1]))
    assert response.status_code == 200


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["events"]] == [eid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/privacy-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_parameter_is_rejected_before_machine_lookup(client):
    # The invalid-query check precedes both time validation and the machine
    # lookup, and must never read event data.
    missing = "00000000-0000-0000-0000-000000000000"
    base = f"/machines/{missing}/authorization-decision-events/privacy-export"

    unknown_with_bad_time = client.get(
        f"{base}?from_created_at=nope&to_created_at={T5}&x=1"
    )
    assert unknown_with_bad_time.status_code == 422
    assert unknown_with_bad_time.json() == {"error": {"code": "invalid_query"}}

    unknown_alone = client.get(f"{base}?x=1")
    assert unknown_alone.status_code == 422
    assert unknown_alone.json() == {"error": {"code": "invalid_query"}}


def test_bad_time_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    bad_time = client.get(
        f"/machines/{missing}/authorization-decision-events/privacy-export"
        f"?from_created_at=nope&to_created_at={T5}"
    )
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}


def test_missing_machine_returns_404_without_event_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and desensitized fields
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_events(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "machine_id",
        "from_created_at",
        "to_created_at",
        "events",
    ]
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == WIDE[0]
    assert body["to_created_at"] == WIDE[1]
    assert body["events"] == []


def test_event_exported_with_desensitized_fields(client):
    machine_id = create_machine(client)
    recorded = record_event(client, machine_id, "read", "res/secret")
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()[0]

    body = client.get(export_url(machine_id)).json()
    (exported,) = body["events"]
    assert list(exported.keys()) == EVENT_KEYS
    assert exported["id"] == recorded["id"] == listed["id"]
    assert exported["machine_id"] == machine_id
    assert exported["allowed"] == recorded["allowed"]
    assert exported["reason"] == recorded["reason"]
    assert exported["created_at"] == recorded["created_at"]
    assert exported["previous_event_id"] is None
    assert exported["content_hash"] == listed["content_hash"]
    assert exported["chain_hash"] == listed["chain_hash"]
    assert exported["action_ref"] == expected_ref("action", machine_id, "read")
    assert exported["resource_ref"] == expected_ref(
        "resource", machine_id, "res/secret"
    )


def test_digest_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type="  read\t", resource="\n res/x  ",
    )
    (exported,) = client.get(export_url(machine_id)).json()["events"]
    assert exported["action_ref"] == expected_ref("action", machine_id, "read")
    assert exported["resource_ref"] == expected_ref(
        "resource", machine_id, "res/x"
    )


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_value_digest_is_null_but_record_kept(client, blank):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type=blank, resource="res/x",
    )
    insert_event_row(
        client, machine_id, eid(2), T2,
        action_type="read", resource=blank,
    )
    rows = client.get(export_url(machine_id)).json()["events"]
    assert [r["id"] for r in rows] == [eid(1), eid(2)]
    assert rows[0]["action_ref"] is None
    assert rows[0]["resource_ref"] == expected_ref(
        "resource", machine_id, "res/x"
    )
    assert rows[1]["action_ref"] == expected_ref("action", machine_id, "read")
    assert rows[1]["resource_ref"] is None


def test_non_string_value_digest_is_null_but_record_kept(client):
    machine_id = create_machine(client)
    # SQLite TEXT affinity would coerce a numeric literal to text, so store
    # BLOBs (which TEXT affinity leaves untouched) to get genuinely non-string
    # values. The privacy view must still return the record with null refs
    # rather than coercing or leaking either value.
    insert_event_row(
        client, machine_id, eid(1), T1,
        action_type=b"\xff\xfe binary-action", resource=b"\x00\x01 bin-res",
    )
    rows = client.get(export_url(machine_id)).json()["events"]
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
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert secret_action not in response.text
    assert secret_resource not in response.text
    assert '"action_type"' not in response.text
    assert '"resource"' not in response.text


def test_digest_is_scoped_to_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(
        client, machine_one, eid(1), T1,
        action_type="read", resource="res/x",
    )
    insert_event_row(
        client, machine_two, eid(2), T1,
        action_type="read", resource="res/x",
    )
    ref_one = client.get(export_url(machine_one)).json()["events"][0][
        "action_ref"
    ]
    ref_two = client.get(export_url(machine_two)).json()["events"][0][
        "action_ref"
    ]
    assert ref_one == expected_ref("action", machine_one, "read")
    assert ref_two == expected_ref("action", machine_two, "read")
    assert ref_one != ref_two


# --------------------------------------------------------------------------- #
# Windowing, ordering, machine isolation
# --------------------------------------------------------------------------- #


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)
    insert_event_row(client, machine_id, eid(2), T2)
    insert_event_row(client, machine_id, eid(3), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(1),
        eid(2),
        eid(3),
    ]


def test_window_excludes_events_outside_bounds(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(10), T0)
    insert_event_row(client, machine_id, eid(11), T1)
    insert_event_row(client, machine_id, eid(12), T2)
    insert_event_row(client, machine_id, eid(13), T3)
    insert_event_row(client, machine_id, eid(14), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(11),
        eid(12),
        eid(13),
    ]


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_event_row(client, machine_id, eid(2), T4)

    response = client.get(export_url(machine_id, T5, "2026-03-01T00:00:09Z"))

    assert response.status_code == 200
    assert response.json()["events"] == []


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(11), "2026-03-01T00:00:00.500000Z")
    insert_event_row(client, machine_id, eid(12), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["events"]] == [eid(11), eid(12)]


def test_events_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_event_row(client, machine_id, eid(30), T3)
    insert_event_row(client, machine_id, eid(21), T2)
    insert_event_row(client, machine_id, eid(20), T2)
    insert_event_row(client, machine_id, eid(10), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(10),
        eid(20),
        eid(21),
        eid(30),
    ]


def test_same_second_exact_second_sorts_before_fractional_seconds(client):
    # As text, "...:00.5Z" sorts *before* "...:00Z" ('.' < 'Z'), so a naive
    # lexicographic order inverts the true order within one second. The export
    # must order by the actual UTC instant: the exact-second record first, then
    # fractional records earliest fraction first.
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(20), "2026-03-01T00:00:00.900000Z")
    insert_event_row(client, machine_id, eid(10), "2026-03-01T00:00:00.500000Z")
    insert_event_row(client, machine_id, eid(1), "2026-03-01T00:00:00Z")

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["events"]] == [
        eid(1),
        eid(10),
        eid(20),
    ]


def test_export_never_contains_other_machine_events(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_one, eid(1), T1)
    insert_event_row(client, machine_two, eid(2), T1)
    insert_event_row(client, machine_two, eid(3), T2)

    response = client.get(export_url(machine_one))

    assert response.status_code == 200
    exported = response.json()["events"]
    assert [r["id"] for r in exported] == [eid(1)]
    assert all(r["machine_id"] == machine_one for r in exported)


# --------------------------------------------------------------------------- #
# Damaged records are exported exactly as stored
# --------------------------------------------------------------------------- #


def test_damaged_chain_fields_are_exported_unmodified(client, tmp_path):
    machine_id = create_machine(client)
    recorded = record_event(client, machine_id)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE authorization_decision_events "
        "SET previous_event_id = NULL, chain_hash = ?, content_hash = ? "
        "WHERE id = ?",
        ("f" * 64, "e" * 64, recorded["id"]),
    )
    connection.commit()
    connection.close()

    response = client.get(export_url(machine_id))

    assert response.status_code == 200
    (exported,) = response.json()["events"]
    assert exported["id"] == recorded["id"]
    assert exported["previous_event_id"] is None
    assert exported["chain_hash"] == "f" * 64
    assert exported["content_hash"] == "e" * 64


def test_misowned_previous_event_link_is_exported_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    # A record of machine two that machine one's record points at: the dangling
    # cross-machine link is kept exactly as stored, never repaired or filtered.
    insert_event_row(client, machine_two, eid(9), T1)
    insert_event_row(
        client, machine_one, eid(1), T2, previous_event_id=eid(9)
    )

    (exported,) = client.get(export_url(machine_one)).json()["events"]

    assert exported["previous_event_id"] == eid(9)


def test_stored_fields_are_emitted_verbatim(client):
    machine_id = create_machine(client)
    insert_event_row(
        client, machine_id, eid(1), T1,
        allowed=0, reason="custom_reason",
        previous_event_id=eid(7), content_hash=HASH_A, chain_hash=HASH_B,
    )
    (exported,) = client.get(export_url(machine_id)).json()["events"]
    assert exported["allowed"] is False
    assert exported["reason"] == "custom_reason"
    assert exported["previous_event_id"] == eid(7)
    assert exported["content_hash"] == HASH_A
    assert exported["chain_hash"] == HASH_B


# --------------------------------------------------------------------------- #
# Body format, read-only byte stability, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_newline_terminated_json_with_fixed_field_order(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)

    response = client.get(export_url(machine_id, T0, T2))

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
        "from_created_at": T0,
        "to_created_at": T2,
        "events": [expected_item],
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    record_event(client, machine_id, "read", "res/1")
    record_event(client, machine_id, "write", "res/2")
    record_event(client, other_machine, "read", "res/3")

    events_url = f"/machines/{machine_id}/authorization-decision-events"
    integrity_url = (
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )
    compliance_url = (
        f"/machines/{machine_id}/authorization-decision-events/compliance-export"
        f"?from_created_at={WIDE[0]}&to_created_at={WIDE[1]}"
    )
    machine_url = f"/machines/{machine_id}"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()
    before_compliance = client.get(compliance_url).json()
    before_machine = client.get(machine_url).json()

    first = client.get(export_url(machine_id)).content
    second = client.get(export_url(machine_id)).content
    assert first == second
    assert len(json.loads(first)["events"]) == 2

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    assert client.get(compliance_url).json() == before_compliance
    assert client.get(machine_url).json() == before_machine
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        record_event(first, machine_id, "read", "res/1")
        record_event(first, machine_id, "write", "res/2")
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["events"]) == 2
