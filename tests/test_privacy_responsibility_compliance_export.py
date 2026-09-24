"""Tests for the read-only desensitized privacy responsibility export.

Covers `GET /machines/{machine_id}/authorization-decision-events/
privacy-responsibility/compliance-export`: the strict query validation
(``bad_time`` / ``invalid_query`` before any machine or assignment read),
``404 not_found``, GET-only ``405`` routing, closed-UTC-window filtering on
each assignment's own ``created_at``, ordering by the actual UTC instant then
record id (exact-second before fractional-second), the ``party_ref`` /
``role_ref`` SHA-256 desensitizing digests (including ``null`` for non-string
or blank values), the absence of raw ``party``/``role`` text, verbatim export
when the related event/incident is missing, misowned, duplicated, or
chain-damaged, machine isolation, strict read-only byte stability, and
persistence across a restart.
"""
import hashlib
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
        f"/machines/{machine_id}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export"
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
    "created_at",
    "previous_assignment_id",
    "content_hash",
    "chain_hash",
    "party_ref",
    "role_ref",
}


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
        f"/machines/{machine_id}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export{query}"
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
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200), T2)
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.250Z",
            "2026-03-01T00:00:03.750000Z",
        )
    )
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["responsibility_assignments"]] == [
        rid(1)
    ]


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
        f"/machines/{machine_id}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    base = (
        f"/machines/{missing}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export"
    )

    bad_time = client.get(f"{base}?from_created_at=nope&to_created_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_assignment_data(client):
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


def test_assignment_exported_with_desensitized_fields(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assignment = assign_responsibility(
        client, machine_id, event["id"], incident["id"],
        party="alice", role="lead",
    )

    body = client.get(export_url(machine_id)).json()
    exported = body["responsibility_assignments"][0]
    assert set(exported.keys()) == ASSIGNMENT_KEYS
    assert exported["id"] == assignment["id"]
    assert exported["machine_id"] == machine_id
    assert exported["event_id"] == event["id"]
    assert exported["incident_id"] == incident["id"]
    assert exported["created_at"] == assignment["created_at"]
    assert exported["previous_assignment_id"] is None
    assert exported["content_hash"] == assignment["content_hash"]
    assert exported["chain_hash"] == assignment["chain_hash"]
    assert exported["party_ref"] == expected_ref("party", machine_id, "alice")
    assert exported["role_ref"] == expected_ref("role", machine_id, "lead")


def test_digest_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party="  alice\t", role="\n lead  ",
    )
    exported = client.get(export_url(machine_id)).json()[
        "responsibility_assignments"
    ][0]
    assert exported["party_ref"] == expected_ref("party", machine_id, "alice")
    assert exported["role_ref"] == expected_ref("role", machine_id, "lead")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_party_or_role_digest_is_null_but_record_kept(client, blank):
    machine_id = create_machine(client)
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party=blank, role="lead",
    )
    insert_assignment_row(
        client, machine_id, rid(2), rid(100), rid(200), T2,
        party="alice", role=blank,
    )
    rows = client.get(export_url(machine_id)).json()["responsibility_assignments"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
    assert rows[0]["party_ref"] is None
    assert rows[0]["role_ref"] == expected_ref("role", machine_id, "lead")
    assert rows[1]["party_ref"] == expected_ref("party", machine_id, "alice")
    assert rows[1]["role_ref"] is None


def test_non_string_party_or_role_digest_is_null_but_record_kept(client):
    machine_id = create_machine(client)
    # SQLite TEXT affinity would coerce a numeric literal to text, so store
    # BLOBs (which TEXT affinity leaves untouched) to get genuinely non-string
    # values. The privacy view must still return the record with null refs
    # rather than coercing or leaking either value.
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party=b"\xff\xfe binary-party", role=b"\x00\x01 binary-role",
    )
    rows = client.get(export_url(machine_id)).json()["responsibility_assignments"]
    assert len(rows) == 1
    assert rows[0]["party_ref"] is None
    assert rows[0]["role_ref"] is None
    assert rows[0]["id"] == rid(1)


def test_raw_party_and_role_never_appear_in_response(client):
    machine_id = create_machine(client)
    secret_party = "the-secret-party-name"
    secret_role = "the-secret-role-name"
    insert_assignment_row(
        client, machine_id, rid(1), rid(100), rid(200), T1,
        party=f"  {secret_party}  ", role=secret_role,
    )
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert secret_party not in response.text
    assert secret_role not in response.text


def test_digest_is_scoped_to_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(
        client, machine_one, rid(1), rid(100), rid(200), T1,
        party="alice", role="lead",
    )
    insert_assignment_row(
        client, machine_two, rid(2), rid(101), rid(201), T1,
        party="alice", role="lead",
    )
    ref_one = client.get(export_url(machine_one)).json()[
        "responsibility_assignments"
    ][0]["party_ref"]
    ref_two = client.get(export_url(machine_two)).json()[
        "responsibility_assignments"
    ][0]["party_ref"]
    assert ref_one == expected_ref("party", machine_one, "alice")
    assert ref_two == expected_ref("party", machine_two, "alice")
    assert ref_one != ref_two


# --------------------------------------------------------------------------- #
# Windowing, ordering, machine isolation
# --------------------------------------------------------------------------- #


def test_window_is_closed_on_assignment_created_at(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200), T0,
                          party="alice", role="lead")
    insert_assignment_row(client, machine_id, rid(2), rid(100), rid(200), T2,
                          party="bob", role="lead")
    insert_assignment_row(client, machine_id, rid(3), rid(100), rid(200), T4,
                          party="carol", role="lead")

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(2)]

    # Equal bounds include the boundary assignment.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["responsibility_assignments"] == []


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_assignment_row(client, machine_id, rid(2), rid(100), rid(200),
                          fractional, party="bob", role="lead")
    insert_assignment_row(client, machine_id, rid(1), rid(100), rid(200),
                          T0, party="alice", role="lead")

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(1),
        rid(2),
    ]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(30), rid(100), rid(200), T2,
                          party="carol", role="lead")
    insert_assignment_row(client, machine_id, rid(20), rid(100), rid(200), T2,
                          party="bob", role="lead")

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(20),
        rid(30),
    ]


def test_damaged_references_and_chain_fields_export_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    foreign_incident = register_incident(
        client, machine_two, foreign_event["id"]
    )
    dangling_event_id = str(uuid.uuid4())
    dangling_incident_id = str(uuid.uuid4())

    # Missing event/incident; foreign event/incident; two rows duplicate the
    # same dangling incident; chain fields deliberately corrupt/arbitrary.
    insert_assignment_row(
        client, machine_one, rid(1), dangling_event_id, dangling_incident_id,
        T1, previous_assignment_id=None, content_hash=HASH_A, chain_hash=HASH_B,
    )
    insert_assignment_row(
        client, machine_one, rid(2), foreign_event["id"],
        foreign_incident["id"], T2, previous_assignment_id=rid(99),
        content_hash="z" * 64, chain_hash="0" * 64,
    )
    insert_assignment_row(
        client, machine_one, rid(3), dangling_event_id, dangling_incident_id,
        T3, party="other-ops", role="scribe",
    )

    rows = client.get(export_url(machine_one, T0, T5)).json()[
        "responsibility_assignments"
    ]
    assert [r["id"] for r in rows] == [rid(1), rid(2), rid(3)]
    assert all(r["machine_id"] == machine_one for r in rows)
    assert rows[0]["event_id"] == dangling_event_id
    assert rows[0]["incident_id"] == dangling_incident_id
    assert rows[1]["event_id"] == foreign_event["id"]
    assert rows[1]["incident_id"] == foreign_incident["id"]
    assert rows[1]["previous_assignment_id"] == rid(99)
    assert rows[1]["content_hash"] == "z" * 64
    assert rows[1]["chain_hash"] == "0" * 64


def test_other_machine_assignments_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_assignment_row(client, machine_one, rid(1), rid(100), rid(200), T1)
    insert_assignment_row(client, machine_two, rid(2), rid(101), rid(201), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(1)]

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    assign_responsibility(client, machine_id, event["id"], incident["id"])
    insert_assignment_row(client, machine_id, rid(99), event["id"], incident["id"],
                          T3, party="  carol  ", role="lead")

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
        assign_responsibility(
            first, machine_id, event["id"], incident["id"],
            party="alice", role="lead",
        )
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    rows = response.json()["responsibility_assignments"]
    assert len(rows) == 1
    assert rows[0]["party_ref"] == expected_ref("party", machine_id, "alice")
    assert rows[0]["role_ref"] == expected_ref("role", machine_id, "lead")
