"""Tests for the read-only privacy-preserving responsibility compliance export.

Covers `GET /machines/{machine_id}/privacy-responsibility/compliance-export`:
closed-UTC-window filtering on the assignment's own ``created_at``, ordering by
the actual UTC instant then record id (exact-second assignments before
fractional-second assignments of the same second), machine-bound SHA-256
``party_ref``/``role_ref`` digests that never leak raw party/role text, ``null``
references for non-string or blank-after-trim values while the record stays,
verbatim export when an event/incident is missing, misowned, duplicated, or
otherwise damaged, the ``bad_time`` / ``invalid_query`` / ``not_found``
outcomes, GET-only routing, strict read-only byte stability, machine
isolation, and persistence across a restart.
"""
import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import (
    PARTY_REF_PREFIX,
    ROLE_REF_PREFIX,
    app,
)


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


def register_incident(client, machine_id, event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents",
        json={"incident_type": "fault", "summary": "something failed"},
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


def insert_assignment_row(client, assignment_id, machine_id, event_id,
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


def insert_assignment_row_raw(client, values):
    """Insert one assignment row with arbitrary, possibly damaged values.

    SQLite does not enforce NOT NULL/type constraints, so a direct raw insert
    can place a NULL party/role the API could never write; through the ORM the
    column loads as ``None`` (a true non-string value).
    """
    columns = (
        "id, machine_id, event_id, incident_id, party, role, created_at, "
        "previous_assignment_id, content_hash, chain_hash"
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO incident_responsibility_assignments ({columns}) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, :party, "
                ":role, :created_at, :previous_assignment_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
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


def expected_ref(prefix, machine_id, raw_value):
    return hashlib.sha256(
        (prefix + machine_id + raw_value.strip()).encode("utf-8")
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


def test_fractional_seconds_are_accepted(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, rid(1), machine_id, rid(100), rid(200), T2)
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.250Z",
            "2026-03-01T00:00:03.750000Z",
        )
    )
    assert response.status_code == 200
    assert [a["id"] for a in response.json()["responsibility_assignments"]] == [
        rid(1)
    ]


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, rid(1), machine_id, rid(100), rid(200), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [a["id"] for a in response.json()["responsibility_assignments"]] == [
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

    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_with_no_assignment_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"responsibility_assignments" not in response.content


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405
        # A rejected method performs no digest work: the references are not
        # computed into any response body.
        assert b"party_ref" not in response.content


# --------------------------------------------------------------------------- #
# Envelope shape and digests
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_assignments(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "responsibility_assignments",
    }
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
        client, machine_id, event["id"], incident["id"],
        party="alice", role="owner",
    )

    body = client.get(export_url(machine_id)).json()
    assert len(body["responsibility_assignments"]) == 1
    exported = body["responsibility_assignments"][0]
    assert set(exported.keys()) == ASSIGNMENT_KEYS
    # party_ref/role_ref occupy the old party/role positions.
    assert list(exported.keys())[4:6] == ["party_ref", "role_ref"]
    assert exported == {
        "id": assignment["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "incident_id": incident["id"],
        "party_ref": expected_ref(PARTY_REF_PREFIX, machine_id, "alice"),
        "role_ref": expected_ref(ROLE_REF_PREFIX, machine_id, "owner"),
        "created_at": assignment["created_at"],
        "previous_assignment_id": None,
        "content_hash": assignment["content_hash"],
        "chain_hash": assignment["chain_hash"],
    }


def test_refs_are_lowercase_hex_and_prefix_distinguished(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, rid(1), machine_id, rid(100), rid(200), T1,
        party="same", role="same",
    )
    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    for ref in (exported["party_ref"], exported["role_ref"]):
        assert isinstance(ref, str)
        assert len(ref) == 64
        assert ref == ref.lower()
        assert all(c in "0123456789abcdef" for c in ref)
    # The party and role prefixes make identical source text diverge.
    assert exported["party_ref"] != exported["role_ref"]
    assert exported["party_ref"] == expected_ref(
        PARTY_REF_PREFIX, machine_id, "same"
    )
    assert exported["role_ref"] == expected_ref(
        ROLE_REF_PREFIX, machine_id, "same"
    )


def test_refs_are_bound_to_the_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(
        client, rid(1), machine_one, rid(100), rid(200), T1,
        party="alice", role="owner",
    )
    insert_assignment_row(
        client, rid(2), machine_two, rid(101), rid(201), T1,
        party="alice", role="owner",
    )

    one = client.get(export_url(machine_one)).json()[
        "responsibility_assignments"
    ][0]
    two = client.get(export_url(machine_two)).json()[
        "responsibility_assignments"
    ][0]
    assert one["party_ref"] == expected_ref(
        PARTY_REF_PREFIX, machine_one, "alice"
    )
    assert two["party_ref"] == expected_ref(
        PARTY_REF_PREFIX, machine_two, "alice"
    )
    assert one["party_ref"] != two["party_ref"]
    assert one["role_ref"] != two["role_ref"]


def test_surrounding_whitespace_is_trimmed_inside_the_digest(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, rid(1), machine_id, rid(100), rid(200), T1,
        party="  alice\t", role="\n owner  ",
    )
    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    assert exported["party_ref"] == expected_ref(
        PARTY_REF_PREFIX, machine_id, "alice"
    )
    assert exported["role_ref"] == expected_ref(
        ROLE_REF_PREFIX, machine_id, "owner"
    )


def test_non_string_or_blank_values_yield_null_but_record_stays(client):
    machine_id = create_machine(client)
    base = {
        "machine_id": machine_id,
        "created_at": None,
        "previous_assignment_id": None,
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    # A BLOB stored in the TEXT column loads through the ORM as bytes, a true
    # non-string value (integers are coerced to strings by the column's result
    # processing); role stays sound.
    insert_assignment_row_raw(
        client,
        {**base, "id": rid(1), "event_id": rid(100),
         "incident_id": rid(200), "created_at": T1,
         "party": b"\xde\xad\xbe\xef", "role": "owner"},
    )
    # Blank-after-trim stored role; party stays sound.
    insert_assignment_row_raw(
        client,
        {**base, "id": rid(2), "event_id": rid(101),
         "incident_id": rid(201), "created_at": T2,
         "party": "alice", "role": "   "},
    )
    # Both unusable: record is still exported.
    insert_assignment_row_raw(
        client,
        {**base, "id": rid(3), "event_id": rid(102),
         "incident_id": rid(202), "created_at": T3,
         "party": b"\x00\x01", "role": "\t\n "},
    )

    body = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ]
    assert [a["id"] for a in body] == [rid(1), rid(2), rid(3)]
    assert body[0]["party_ref"] is None
    assert body[0]["role_ref"] == expected_ref(
        ROLE_REF_PREFIX, machine_id, "owner"
    )
    assert body[1]["party_ref"] == expected_ref(
        PARTY_REF_PREFIX, machine_id, "alice"
    )
    assert body[1]["role_ref"] is None
    assert body[2]["party_ref"] is None
    assert body[2]["role_ref"] is None


def test_raw_party_and_role_never_appear_in_response(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, rid(1), machine_id, rid(100), rid(200), T1,
        party="super-secret-party", role="super-secret-role",
    )
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert b"super-secret-party" not in response.content
    assert b"super-secret-role" not in response.content
    assert b'"party"' not in response.content
    assert b'"role"' not in response.content


# --------------------------------------------------------------------------- #
# Windowing, ordering, and machine isolation
# --------------------------------------------------------------------------- #


def test_window_is_closed_on_assignment_created_at(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, rid(1), machine_id, rid(100), rid(200), T0)
    insert_assignment_row(client, rid(2), machine_id, rid(101), rid(201), T2)
    insert_assignment_row(client, rid(3), machine_id, rid(102), rid(202), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [rid(2)]

    # Equal bounds include the boundary assignment.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["responsibility_assignments"] == []


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_assignment_row(
        client, rid(2), machine_id, rid(101), rid(201), fractional,
        party="b", role="b",
    )
    insert_assignment_row(
        client, rid(1), machine_id, rid(100), rid(200), T0,
        party="a", role="a",
    )

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [
        rid(1),
        rid(2),
    ]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, rid(30), machine_id, rid(130), rid(230), T2)
    insert_assignment_row(client, rid(20), machine_id, rid(120), rid(220), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [
        rid(20),
        rid(30),
    ]


def test_other_machine_assignments_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(client, rid(1), machine_one, rid(100), rid(200), T1)
    insert_assignment_row(client, rid(2), machine_two, rid(101), rid(201), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [rid(1)]

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [a["id"] for a in body["responsibility_assignments"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Verbatim export despite missing / misowned / duplicated / damaged records
# --------------------------------------------------------------------------- #


def test_damaged_assignments_are_exported_verbatim(client):
    import uuid

    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    dangling_event_id = str(uuid.uuid4())
    dangling_incident_id = str(uuid.uuid4())

    # Missing event and incident entirely.
    insert_assignment_row(
        client, rid(1), machine_one, dangling_event_id, dangling_incident_id,
        T1, previous_assignment_id=None, content_hash=HASH_A, chain_hash=HASH_B,
    )
    # Event owned by another machine; incident does not exist.
    insert_assignment_row(
        client, rid(2), machine_one, foreign_event["id"], str(uuid.uuid4()),
        T2, previous_assignment_id=rid(99), content_hash="not-a-hash",
        chain_hash=None,
    )
    # Duplicated attribution: identical (party, role) pair on two dangling
    # incidents. Both records stay.
    insert_assignment_row(
        client, rid(3), machine_one, dangling_event_id, str(uuid.uuid4()),
        T3, party="dup", role="same",
    )
    insert_assignment_row(
        client, rid(4), machine_one, dangling_event_id, str(uuid.uuid4()),
        T4, party="dup", role="same",
    )

    body = client.get(export_url(machine_one, T0, T5)).json()
    assignments = body["responsibility_assignments"]
    assert [a["id"] for a in assignments] == [rid(1), rid(2), rid(3), rid(4)]
    assert all(a["machine_id"] == machine_one for a in assignments)
    assert assignments[0]["event_id"] == dangling_event_id
    assert assignments[0]["incident_id"] == dangling_incident_id
    assert assignments[1]["event_id"] == foreign_event["id"]
    # Chain fields come through exactly as stored, damaged values included.
    assert assignments[1]["previous_assignment_id"] == rid(99)
    assert assignments[1]["content_hash"] == "not-a-hash"
    assert assignments[1]["chain_hash"] is None
    assert assignments[2]["party_ref"] == assignments[3]["party_ref"]
    assert assignments[2]["role_ref"] == assignments[3]["role_ref"]


def test_chain_fields_exported_exactly_as_stored(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, rid(1), machine_id, rid(100), rid(200), T1,
        previous_assignment_id=rid(7),
        content_hash="c" * 64,
        chain_hash="d" * 64,
    )
    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    assert exported["previous_assignment_id"] == rid(7)
    assert exported["content_hash"] == "c" * 64
    assert exported["chain_hash"] == "d" * 64


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, rid(1), machine_id, rid(100), rid(200), T1)
    insert_assignment_row(
        client, rid(2), machine_id, rid(101), rid(201), T2,
        party="  bo  ", role=9,
    )

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (
                    "incident_responsibility_assignments",
                    "machines",
                )
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
        insert_assignment_row(
            first, rid(1), machine_id, rid(100), rid(200), T1,
            party="alice", role="owner",
        )
        insert_assignment_row(
            first, rid(2), machine_id, rid(101), rid(201), T3,
            party="bob", role="reviewer",
        )
        expected = first.get(export_url(machine_id, T0, T5)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.content == expected
    assert [a["id"] for a in response.json()["responsibility_assignments"]] == [
        rid(1),
        rid(2),
    ]


def test_existing_responsibility_endpoints_keep_raw_party_and_role(client):
    """The new view must not change the existing assignment semantics."""
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assignment = assign_responsibility(
        client, machine_id, event["id"], incident["id"],
        party="alice", role="owner",
    )

    list_response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event['id']}"
        f"/incidents/{incident['id']}/responsibility-assignments"
    )
    assert list_response.status_code == 200
    assert list_response.json()[0]["party"] == "alice"
    assert list_response.json()[0]["role"] == "owner"

    export_response = client.get(
        f"/machines/{machine_id}/responsibility-assignments/compliance-export"
        f"?from_created_at={WIDE[0]}&to_created_at={WIDE[1]}"
    )
    assert export_response.status_code == 200
    assert export_response.json()["assignments"][0]["party"] == "alice"
    assert export_response.json()["assignments"][0]["id"] == assignment["id"]
