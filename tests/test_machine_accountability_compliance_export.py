"""Tests for the read-only machine-level accountability compliance export.

Covers `GET /machines/{machine_id}/accountability/compliance-export`: the
five accountability record groups (events, evidence, incidents, status
history, responsibility assignments), closed-UTC-window filtering per group,
ordering by the actual UTC instant then record id (exact-second records
before fractional-second records of the same second), verbatim export when a
related object is missing or misowned, the `bad_time` / `invalid_query` /
`not_found` outcomes, strict read-only byte stability, machine isolation, and
persistence across a restart.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability.chain import backfill_chains
from accountability.evidence_chain import backfill_chains as backfill_evidence_chains


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
        f"/machines/{machine_id}/accountability/compliance-export"
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


def attach_evidence(client, machine_id, event_id, evidence_type="log",
                    content_hash=None):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/evidence",
        json={
            "evidence_type": evidence_type,
            "content_hash": content_hash or uuid.uuid4().hex * 2,
        },
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


def transition_incident(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
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


def insert_event_row(client, machine_id, event_id, created_at, *,
                     action_type="read", resource="res/x", allowed=1,
                     reason="allowed_by_policy"):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at) "
                "VALUES (:id, :machine_id, :action_type, :resource, :allowed, "
                ":reason, :created_at)"
            ),
            {
                "id": event_id,
                "machine_id": machine_id,
                "action_type": action_type,
                "resource": resource,
                "allowed": allowed,
                "reason": reason,
                "created_at": created_at,
            },
        )
    backfill_chains(client.app.state.engine)


def insert_evidence_row(client, machine_id, evidence_id, event_id, created_at, *,
                        evidence_type="log", content_hash=HASH_A):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_evidence "
                "(id, machine_id, event_id, evidence_type, content_hash, "
                "created_at) "
                "VALUES (:id, :machine_id, :event_id, :evidence_type, "
                ":content_hash, :created_at)"
            ),
            {
                "id": evidence_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "evidence_type": evidence_type,
                "content_hash": content_hash,
                "created_at": created_at,
            },
        )
    backfill_evidence_chains(client.app.state.engine)


def insert_incident_row(client, machine_id, incident_id, event_id, created_at, *,
                        incident_type="fault", summary="summary",
                        status="open"):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_incidents "
                "(id, machine_id, event_id, incident_type, summary, status, "
                "created_at) "
                "VALUES (:id, :machine_id, :event_id, :incident_type, "
                ":summary, :status, :created_at)"
            ),
            {
                "id": incident_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_type": incident_type,
                "summary": summary,
                "status": status,
                "created_at": created_at,
            },
        )


def insert_history_row(client, machine_id, history_id, event_id, incident_id,
                       created_at, *, from_status="open", to_status="acknowledged"):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events "
                "(id, machine_id, event_id, incident_id, from_status, "
                "to_status, created_at) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, "
                ":from_status, :to_status, :created_at)"
            ),
            {
                "id": history_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                "from_status": from_status,
                "to_status": to_status,
                "created_at": created_at,
            },
        )


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


EVENT_KEYS = {
    "id",
    "machine_id",
    "action_type",
    "resource",
    "allowed",
    "reason",
    "created_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
}
EVIDENCE_KEYS = {
    "id",
    "machine_id",
    "event_id",
    "evidence_type",
    "content_hash",
    "created_at",
    "previous_evidence_id",
    "chain_hash",
}
INCIDENT_KEYS = {
    "id",
    "machine_id",
    "event_id",
    "incident_type",
    "summary",
    "status",
    "created_at",
}
HISTORY_KEYS = {
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "from_status",
    "to_status",
    "created_at",
}
ASSIGNMENT_KEYS = {
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party",
    "role",
    "created_at",
    "previous_assignment_id",
    "content_hash",
    "chain_hash",
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
        f"/machines/{machine_id}/accountability/compliance-export{query}"
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
    insert_event_row(client, machine_id, rid(1), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [e["id"] for e in response.json()["events"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/accountability/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    bad_time = client.get(
        f"/machines/{missing}/accountability/compliance-export"
        f"?from_created_at=nope&to_created_at={T5}"
    )
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(
        f"/machines/{missing}/accountability/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Envelope shape and group contents
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_five_empty_groups(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "events",
        "evidence",
        "incidents",
        "status_history",
        "responsibility_assignments",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == WIDE[0]
    assert body["to_created_at"] == WIDE[1]
    for group in (
        "events",
        "evidence",
        "incidents",
        "status_history",
        "responsibility_assignments",
    ):
        assert body[group] == []


def test_all_five_groups_exported_with_list_endpoint_fields(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    evidence = attach_evidence(client, machine_id, event["id"])
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(
        client, machine_id, event["id"], incident["id"], "acknowledged"
    )
    assignment = assign_responsibility(
        client, machine_id, event["id"], incident["id"]
    )

    body = client.get(export_url(machine_id)).json()

    exported_event = body["events"][0]
    assert set(exported_event.keys()) == EVENT_KEYS
    assert exported_event["id"] == event["id"]
    # The authorization result and the integrity-chain fields both survive.
    assert exported_event["allowed"] is True
    assert exported_event["reason"] == "allowed_by_policy"
    assert exported_event["previous_event_id"] is None
    assert exported_event["content_hash"] == event["content_hash"]
    assert exported_event["chain_hash"] == event["chain_hash"]

    exported_evidence = body["evidence"][0]
    assert set(exported_evidence.keys()) == EVIDENCE_KEYS
    assert exported_evidence["id"] == evidence["id"]
    # The fingerprint is exported raw, exactly as stored.
    assert exported_evidence["content_hash"] == evidence["content_hash"]

    exported_incident = body["incidents"][0]
    assert set(exported_incident.keys()) == INCIDENT_KEYS
    assert exported_incident["id"] == incident["id"]
    assert exported_incident["incident_type"] == "fault"
    assert exported_incident["summary"] == "something failed"
    assert exported_incident["status"] == "acknowledged"

    exported_history = body["status_history"][0]
    assert set(exported_history.keys()) == HISTORY_KEYS
    assert exported_history["incident_id"] == incident["id"]
    assert exported_history["from_status"] == "open"
    assert exported_history["to_status"] == "acknowledged"

    exported_assignment = body["responsibility_assignments"][0]
    assert set(exported_assignment.keys()) == ASSIGNMENT_KEYS
    assert exported_assignment["id"] == assignment["id"]
    assert exported_assignment["party"] == "ops-oncall"
    assert exported_assignment["role"] == "incident_commander"
    assert exported_assignment["previous_assignment_id"] is None
    assert exported_assignment["content_hash"] == assignment["content_hash"]
    assert exported_assignment["chain_hash"] == assignment["chain_hash"]


def test_incident_exports_current_lifecycle_status(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(
        client, machine_id, event["id"], incident["id"], "acknowledged"
    )
    resolved = transition_incident(
        client, machine_id, event["id"], incident["id"], "resolved"
    )
    assert resolved["status"] == "resolved"

    body = client.get(export_url(machine_id)).json()
    assert body["incidents"][0]["status"] == "resolved"
    assert [h["to_status"] for h in body["status_history"]] == [
        "acknowledged",
        "resolved",
    ]


# --------------------------------------------------------------------------- #
# Windowing and ordering per group
# --------------------------------------------------------------------------- #


def test_window_is_closed_per_group(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, rid(1), T0)
    insert_event_row(client, machine_id, rid(2), T2)
    insert_event_row(client, machine_id, rid(3), T4)
    insert_evidence_row(client, machine_id, rid(11), rid(1), T1)
    insert_evidence_row(client, machine_id, rid(12), rid(2), T3)
    insert_incident_row(client, machine_id, rid(21), rid(1), T2)
    insert_history_row(client, machine_id, rid(31), rid(1), rid(21), T3)
    insert_assignment_row(client, machine_id, rid(41), rid(1), rid(21), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["events"]] == [rid(2)]
    assert [r["id"] for r in body["evidence"]] == [rid(12)]
    assert [r["id"] for r in body["incidents"]] == [rid(21)]
    assert [r["id"] for r in body["status_history"]] == [rid(31)]
    assert body["responsibility_assignments"] == []

    # Equal bounds include the boundary record in every group.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["events"]] == [rid(3)]
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(41)]


def test_exact_second_sorts_before_fractional_same_second_in_every_group(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_event_row(client, machine_id, rid(2), fractional)
    insert_event_row(client, machine_id, rid(1), T0)
    insert_evidence_row(client, machine_id, rid(12), rid(1), fractional,
                        content_hash=HASH_B)
    insert_evidence_row(client, machine_id, rid(11), rid(1), T0)
    insert_incident_row(client, machine_id, rid(22), rid(1), fractional,
                        summary="later failure")
    insert_incident_row(client, machine_id, rid(21), rid(1), T0,
                        summary="first failure")
    insert_history_row(client, machine_id, rid(32), rid(1), rid(21), fractional)
    insert_history_row(client, machine_id, rid(31), rid(1), rid(21), T0)
    insert_assignment_row(client, machine_id, rid(42), rid(1), rid(21), fractional,
                          party="ops-two", role="lead", chain_hash=HASH_A)
    insert_assignment_row(client, machine_id, rid(41), rid(1), rid(21), T0,
                          party="ops-one", role="lead", chain_hash=HASH_B)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["events"]] == [rid(1), rid(2)]
    assert [r["id"] for r in body["evidence"]] == [rid(11), rid(12)]
    assert [r["id"] for r in body["incidents"]] == [rid(21), rid(22)]
    assert [r["id"] for r in body["status_history"]] == [rid(31), rid(32)]
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(41),
        rid(42),
    ]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, rid(30), T2)
    insert_event_row(client, machine_id, rid(20), T2)
    insert_evidence_row(client, machine_id, rid(60), rid(20), T2,
                        content_hash=HASH_B)
    insert_evidence_row(client, machine_id, rid(50), rid(20), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["events"]] == [rid(20), rid(30)]
    assert [r["id"] for r in body["evidence"]] == [rid(50), rid(60)]


def test_dependent_groups_follow_machine_ownership_not_event_window(client):
    """Evidence/history/assignments follow the machine/entity ownership that
    already exists; they are not filtered by whether the referenced event is
    in this export's event set.
    """
    machine_id = create_machine(client)
    # Event sits outside the window [T3, T5]; evidence inside it still ships.
    insert_event_row(client, machine_id, rid(1), T0)
    insert_evidence_row(client, machine_id, rid(11), rid(1), T4)
    insert_incident_row(client, machine_id, rid(21), rid(1), T0)
    insert_history_row(client, machine_id, rid(31), rid(1), rid(21), T4)
    insert_assignment_row(client, machine_id, rid(41), rid(1), rid(21), T4)

    body = client.get(export_url(machine_id, T3, T5)).json()
    assert body["events"] == []
    assert [r["id"] for r in body["evidence"]] == [rid(11)]
    assert body["incidents"] == []
    assert [r["id"] for r in body["status_history"]] == [rid(31)]
    assert [r["id"] for r in body["responsibility_assignments"]] == [rid(41)]


# --------------------------------------------------------------------------- #
# Verbatim export despite missing or misowned related objects
# --------------------------------------------------------------------------- #


def test_dangling_and_foreign_references_are_exported_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)

    dangling_event_id = str(uuid.uuid4())
    insert_evidence_row(client, machine_one, rid(11), dangling_event_id, T1)
    insert_evidence_row(client, machine_one, rid(12), foreign_event["id"], T2)
    insert_incident_row(client, machine_one, rid(21), dangling_event_id, T1)
    insert_incident_row(client, machine_one, rid(22), foreign_event["id"], T2)
    insert_history_row(
        client, machine_one, rid(31), dangling_event_id, str(uuid.uuid4()), T1
    )
    insert_history_row(
        client, machine_one, rid(32), foreign_event["id"], str(uuid.uuid4()), T2
    )
    insert_assignment_row(
        client, machine_one, rid(41), dangling_event_id, str(uuid.uuid4()), T1
    )
    insert_assignment_row(
        client, machine_one, rid(42), foreign_event["id"], str(uuid.uuid4()), T2
    )

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["evidence"]] == [rid(11), rid(12)]
    assert body["evidence"][0]["event_id"] == dangling_event_id
    assert body["evidence"][1]["event_id"] == foreign_event["id"]
    assert body["evidence"][1]["content_hash"] == HASH_A
    assert [r["id"] for r in body["incidents"]] == [rid(21), rid(22)]
    assert [r["id"] for r in body["status_history"]] == [rid(31), rid(32)]
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        rid(41),
        rid(42),
    ]
    for group in (
        body["events"],
        body["evidence"],
        body["incidents"],
        body["status_history"],
        body["responsibility_assignments"],
    ):
        assert all(r["machine_id"] == machine_one for r in group)


def test_other_machine_records_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_one, rid(1), T1)
    insert_event_row(client, machine_two, rid(2), T1)
    insert_evidence_row(client, machine_one, rid(11), rid(1), T1)
    insert_evidence_row(client, machine_two, rid(12), rid(2), T1)
    insert_incident_row(client, machine_one, rid(21), rid(1), T1)
    insert_incident_row(client, machine_two, rid(22), rid(2), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["events"]] == [rid(1)]
    assert [r["id"] for r in body["evidence"]] == [rid(11)]
    assert [r["id"] for r in body["incidents"]] == [rid(21)]
    assert body["status_history"] == []
    assert body["responsibility_assignments"] == []


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    attach_evidence(client, machine_id, event["id"])
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(
        client, machine_id, event["id"], incident["id"], "acknowledged"
    )
    assign_responsibility(client, machine_id, event["id"], incident["id"])

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (
                    "authorization_decision_events",
                    "authorization_decision_evidence",
                    "authorization_decision_incidents",
                    "incident_status_events",
                    "incident_responsibility_assignments",
                )
            }

    before = table_state()
    first = client.get(export_url(machine_id))
    middle = table_state()
    second = client.get(export_url(machine_id))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        allow_read(first, machine_id)
        event = record_event(first, machine_id)
        attach_evidence(first, machine_id, event["id"])
        incident = register_incident(first, machine_id, event["id"])
        transition_incident(
            first, machine_id, event["id"], incident["id"], "acknowledged"
        )
        assign_responsibility(first, machine_id, event["id"], incident["id"])
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    body = response.json()
    assert len(body["events"]) == 1
    assert len(body["evidence"]) == 1
    assert len(body["incidents"]) == 1
    assert len(body["status_history"]) == 1
    assert len(body["responsibility_assignments"]) == 1
