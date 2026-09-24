"""Tests for the read-only, privacy-preserving responsibility export.

Covers `GET /machines/{machine_id}/privacy-responsibility/compliance-export`:
closed-UTC-window filtering on each assignment's own ``created_at``, ordering by
the actual UTC instant then record id (exact-second records before
fractional-second records of the same second), replacement of the raw
``party``/``role`` text with versioned, machine-bound SHA-256 ``party_ref``/
``role_ref`` digests (``null`` for non-string or blank-after-trim values),
verbatim retention of damaged, missing, misowned, or duplicated records, the
``bad_time`` / ``invalid_query`` / ``not_found`` outcomes, GET-only routing,
the guarantee that no party/role text or key/secret/policy/identity material is
returned, strict read-only byte stability, machine isolation, and persistence
across a restart.
"""
import hashlib
import re
import uuid

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
        f"/machines/{machine_id}/privacy-responsibility/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
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


def allow_read(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
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


def record_event(client, machine_id, resource="res/1"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201
    return response.json()


def register_incident(client, machine_id, event_id,
                      incident_type="fault", summary="something failed"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def assign_responsibility(client, machine_id, event_id, incident_id,
                          party="ops-oncall", role="incident_commander"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents/{incident_id}/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


HASH_A = "a" * 64
HASH_B = "b" * 64


def expected_party_ref(machine_id, party):
    return hashlib.sha256(
        f"privacy:v1|party|{machine_id}|{party.strip()}".encode("utf-8")
    ).hexdigest()


def expected_role_ref(machine_id, role):
    return hashlib.sha256(
        f"privacy:v1|role|{machine_id}|{role.strip()}".encode("utf-8")
    ).hexdigest()


def insert_assignment_row(client, machine_id, assignment_id, event_id,
                          incident_id, created_at, *, party="ops", role="lead",
                          previous_assignment_id=None, content_hash=HASH_A,
                          chain_hash=HASH_B):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments "
                "(id, machine_id, event_id, incident_id, party, role, "
                "created_at, previous_assignment_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, :party, "
                ":role, :created_at, :previous_assignment_id, :content_hash, "
                ":chain_hash)"
            ),
            {
                "id": assignment_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                "party": party,
                "role": role,
                "created_at": created_at,
                "previous_assignment_id": previous_assignment_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        )


ASSIGNMENT_KEYS = {
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party_ref",
    "role_ref",
    "created_at",
    "previous_assignment_id",
    "content_hash",
    "chain_hash",
}
ENVELOPE_KEYS = {
    "machine_id",
    "from_created_at",
    "to_created_at",
    "responsibility_assignments",
}


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
        f"/machines/{machine_id}/privacy-responsibility/compliance-export{query}"
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
        "garbage",
        "",                               # blank
        "2026-13-01T00:00:00Z",           # bad month
        "2026-02-30T00:00:00Z",           # bad day
        "2026-03-01T24:00:00Z",           # bad hour
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["responsibility_assignments"]] == [
        rid(1)
    ]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-responsibility/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    base = f"/machines/{missing}/privacy-responsibility/compliance-export"

    bad_time = client.get(f"{base}?from_created_at=nope&to_created_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    # An unknown parameter is invalid_query (not not_found): the machine is
    # never looked up and no machine data is read.
    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_with_no_assignment_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert "responsibility_assignments" not in response.json()


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope and item shape
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_array(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == WIDE[0]
    assert body["to_created_at"] == WIDE[1]
    assert body["responsibility_assignments"] == []


def test_assignment_exported_with_digest_fields(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assignment = assign_responsibility(
        client, machine_id, event["id"], incident["id"]
    )

    body = client.get(export_url(machine_id)).json()
    [exported] = body["responsibility_assignments"]
    assert set(exported.keys()) == ASSIGNMENT_KEYS
    # Raw party/role slots are replaced, not carried alongside the digests.
    assert "party" not in exported
    assert "role" not in exported
    assert exported["id"] == assignment["id"]
    assert exported["machine_id"] == machine_id
    assert exported["event_id"] == event["id"]
    assert exported["incident_id"] == incident["id"]
    assert exported["created_at"] == assignment["created_at"]
    # The three chain fields survive exactly as stored.
    assert exported["previous_assignment_id"] is None
    assert exported["content_hash"] == assignment["content_hash"]
    assert exported["chain_hash"] == assignment["chain_hash"]
    # Digests match the versioned, machine-bound construction.
    assert exported["party_ref"] == expected_party_ref(machine_id, "ops-oncall")
    assert exported["role_ref"] == expected_role_ref(
        machine_id, "incident_commander"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", exported["party_ref"])
    assert re.fullmatch(r"[0-9a-f]{64}", exported["role_ref"])


def test_raw_party_role_and_machine_secrets_never_returned(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assign_responsibility(
        client,
        machine_id,
        event["id"],
        incident["id"],
        party="secret-party-42",
        role="secret-role-42",
    )

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert "secret-party-42" not in response.text
    assert "secret-role-42" not in response.text
    # Machine identity material (public key, display name) is not exposed.
    assert "key-1" not in response.text
    assert "Machine One" not in response.text
    body = response.json()
    assert set(body.keys()) == ENVELOPE_KEYS
    for item in body["responsibility_assignments"]:
        assert set(item.keys()) == ASSIGNMENT_KEYS


# --------------------------------------------------------------------------- #
# Digest construction
# --------------------------------------------------------------------------- #


def test_party_and_role_use_distinct_versioned_prefixes(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party="same-value", role="same-value",
    )
    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    assert exported["party_ref"] == expected_party_ref(machine_id, "same-value")
    assert exported["role_ref"] == expected_role_ref(machine_id, "same-value")
    # Domain separation: identical text yields distinct party/role digests.
    assert exported["party_ref"] != exported["role_ref"]


def test_digest_is_bound_to_machine_id(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(
        client, machine_one, rid(1), rid(100), rid(200), T1, party="ops", role="lead"
    )
    insert_assignment_row(
        client, machine_two, rid(2), rid(101), rid(201), T1, party="ops", role="lead"
    )

    one = client.get(export_url(machine_one)).json()["responsibility_assignments"][0]
    two = client.get(export_url(machine_two)).json()["responsibility_assignments"][0]
    assert one["party_ref"] == expected_party_ref(machine_one, "ops")
    assert two["party_ref"] == expected_party_ref(machine_two, "ops")
    assert one["party_ref"] != two["party_ref"]
    assert one["role_ref"] != two["role_ref"]


def test_digest_trims_surrounding_whitespace_but_keeps_interior(client):
    machine_id = create_machine(client)
    # Surrounding whitespace is removed before hashing; interior whitespace is
    # part of the value and must survive.
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party="  ops on call\t", role="  lead role ",
    )

    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    assert exported["party_ref"] == expected_party_ref(machine_id, "ops on call")
    assert exported["role_ref"] == expected_role_ref(machine_id, "lead role")
    # Whitespace-only/empty preimages are not possible here, but make the
    # interior-space preservation explicit against an over-trimmed digest.
    assert exported["party_ref"] == expected_party_ref(machine_id, "ops on call")
    assert exported["party_ref"] != expected_party_ref(machine_id, "opsoncall")


def test_non_string_or_blank_values_yield_null_refs_but_record_remains(client):
    machine_id = create_machine(client)
    # BLOB (a non-string storage class, which SQLite admits into a
    # TEXT-affinity NOT NULL column) party and role.
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party=b"\x00\x01binary", role=b"\x02\x03blob",
    )
    # BLOB party with a normal string role.
    insert_assignment_row(
        client, machine_id, rid(2), rid(101), rid(201), T2,
        party=b"binary-party", role="lead",
    )
    # Blank-after-trim strings.
    insert_assignment_row(
        client, machine_id, rid(3), rid(102), rid(202), T3,
        party="   ", role="\t\n ",
    )
    # One healthy record.
    insert_assignment_row(
        client, machine_id, rid(4), rid(103), rid(203), T4,
        party="ops", role="lead",
    )

    records = client.get(export_url(machine_id, T0, T5)).json()[
        "responsibility_assignments"
    ]
    by_id = {record["id"]: record for record in records}
    assert [r["id"] for r in records] == [rid(1), rid(2), rid(3), rid(4)]

    assert by_id[rid(1)]["party_ref"] is None
    assert by_id[rid(1)]["role_ref"] is None
    assert by_id[rid(2)]["party_ref"] is None
    assert by_id[rid(2)]["role_ref"] == expected_role_ref(machine_id, "lead")
    assert by_id[rid(3)]["party_ref"] is None
    assert by_id[rid(3)]["role_ref"] is None
    assert by_id[rid(4)]["party_ref"] == expected_party_ref(machine_id, "ops")
    assert by_id[rid(4)]["role_ref"] == expected_role_ref(machine_id, "lead")

    # Raw non-string bytes never leak into the serialized body.
    assert "binary-party" not in client.get(export_url(machine_id, T0, T5)).text


# --------------------------------------------------------------------------- #
# Windowing and ordering
# --------------------------------------------------------------------------- #


def test_window_is_closed_on_assignment_created_at(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200), T0)
    insert_assignment_row(client, machine_id, rid(2), rid(101), rid(201), T2)
    insert_assignment_row(client, machine_id, rid(3), rid(102), rid(202), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(2)]

    # Equal bounds include the boundary record.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["responsibility_assignments"] == []


def test_window_ignores_referenced_event_timestamps(client):
    """Membership is the assignment's own created_at, even when the referenced
    event/incident would never exist in any window."""
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200), T2)

    body = client.get(export_url(machine_id, T2, T2)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(1)]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_assignment_row(
        client, machine_id, rid(2), rid(101), rid(201), fractional,
        party="ops-two", role="lead",
    )
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T0,
        party="ops-one", role="lead",
    )

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(1),
        rid(2),
    ]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(30), rid(130), rid(230), T2)
    insert_assignment_row(client, machine_id, rid(20), rid(120), rid(220), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(20),
        rid(30),
    ]


# --------------------------------------------------------------------------- #
# Machine isolation and verbatim retention
# --------------------------------------------------------------------------- #


def test_other_machine_assignments_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(client, machine_one, rid(1), rid(100), rid(200), T1)
    insert_assignment_row(client, machine_two, rid(2), rid(101), rid(201), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(1)]
    assert all(
        r["machine_id"] == machine_one
        for r in body["responsibility_assignments"]
    )

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(2)]


def test_dangling_and_misowned_references_are_exported_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    foreign_incident = register_incident(
        client, machine_two, foreign_event["id"]
    )
    dangling_event_id = str(uuid.uuid4())
    dangling_incident_id = str(uuid.uuid4())

    # Both related ids missing entirely.
    insert_assignment_row(
        client, machine_one, rid(1), dangling_event_id, dangling_incident_id, T1
    )
    # Related ids owned by another machine.
    insert_assignment_row(
        client, machine_one, rid(2), foreign_event["id"],
        foreign_incident["id"], T2,
    )

    records = client.get(export_url(machine_one, T0, T5)).json()[
        "responsibility_assignments"
    ]
    assert [r["id"] for r in records] == [rid(1), rid(2)]
    assert records[0]["event_id"] == dangling_event_id
    assert records[0]["incident_id"] == dangling_incident_id
    assert records[1]["event_id"] == foreign_event["id"]
    assert records[1]["incident_id"] == foreign_incident["id"]
    assert all(r["machine_id"] == machine_one for r in records)


def test_duplicated_and_corrupt_records_are_retained_unchanged(client):
    machine_id = create_machine(client)
    # Same (party, role) pair attributed twice (to different incidents): the
    # duplicate attribution is not collapsed or filtered, and the digests are
    # identical.
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party="ops", role="lead",
    )
    insert_assignment_row(
        client, machine_id, rid(2), rid(101), rid(201), T2,
        party="ops", role="lead",
    )
    # Corrupt chain fields (wrong link/hashes) ship exactly as stored.
    insert_assignment_row(
        client, machine_id, rid(3), rid(102), rid(202), T3,
        previous_assignment_id=rid(999), content_hash=HASH_A, chain_hash=HASH_B,
    )
    # Missing chain fields (NULL from a writer that bypassed the backfill) are
    # preserved as null rather than repaired.
    insert_assignment_row(
        client, machine_id, rid(4), rid(103), rid(203), T4,
        previous_assignment_id=None, content_hash=None, chain_hash=None,
    )

    records = client.get(export_url(machine_id, T0, T5)).json()[
        "responsibility_assignments"
    ]
    assert [r["id"] for r in records] == [rid(1), rid(2), rid(3), rid(4)]
    assert records[0]["party_ref"] == records[1]["party_ref"]
    assert records[0]["role_ref"] == records[1]["role_ref"]
    assert records[2]["previous_assignment_id"] == rid(999)
    assert records[2]["content_hash"] == HASH_A
    assert records[2]["chain_hash"] == HASH_B
    assert records[3]["previous_assignment_id"] is None
    assert records[3]["content_hash"] is None
    assert records[3]["chain_hash"] is None


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assign_responsibility(client, machine_id, event["id"], incident["id"])
    insert_assignment_row(client, machine_id, rid(99), rid(1), rid(2), T1)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in ("incident_responsibility_assignments",)
            }

    before = table_state()
    first = client.get(export_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(export_url(machine_id, T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_empty_database_needs_no_migration_and_exports_empty(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json()["responsibility_assignments"] == []


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        allow_read(first, machine_id)
        event = record_event(first, machine_id)
        incident = register_incident(first, machine_id, event["id"])
        assignment = assign_responsibility(
            first, machine_id, event["id"], incident["id"]
        )
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    [exported] = response.json()["responsibility_assignments"]
    assert exported["id"] == assignment["id"]
    assert exported["party_ref"] == expected_party_ref(machine_id, "ops-oncall")
    assert exported["role_ref"] == expected_role_ref(
        machine_id, "incident_commander"
    )
