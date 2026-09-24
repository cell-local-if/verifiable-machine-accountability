"""Tests for the read-only machine-level integrity summary.

Covers `GET /machines/{machine_id}/integrity-summary`: the four audit blocks
(authorization events, key rotations, evidence, incidents), the overall valid
flag, the empty-machine shape, the `invalid_query` / `not_found` outcomes and
their precedence, method routing, machine isolation, strict read-only byte
stability, and persistence across a restart.
"""
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


SUMMARY_PATH = "/integrity-summary"

T0 = "2026-03-01T00:00:00Z"

HASH_A = "a" * 64
HASH_B = "b" * 64


def summary_url(machine_id):
    return f"/machines/{machine_id}{SUMMARY_PATH}"


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


def rotate_key(client, machine_id, new_public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={
            "public_key": new_public_key,
            "expected_version": expected_version,
        },
    )
    assert response.status_code == 200
    return response.json()


def attach_evidence(client, machine_id, event_id, content_hash=HASH_A):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/evidence",
        json={"evidence_type": "log", "content_hash": content_hash},
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


def transition_incident(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def assign_responsibility(client, machine_id, event_id, incident_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents/{incident_id}/responsibility-assignments",
        json={"party": "ops-oncall", "role": "incident_commander"},
    )
    assert response.status_code == 201
    return response.json()


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


# --------------------------------------------------------------------------- #
# Response shape
# --------------------------------------------------------------------------- #


def test_empty_machine_returns_four_valid_zero_blocks(client):
    machine_id = create_machine(client)
    response = client.get(summary_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "valid": True,
        "events": {"valid": True, "checked_count": 0, "broken_event_id": None},
        "rotations": {
            "valid": True,
            "checked_count": 0,
            "broken_rotation_id": None,
        },
        "evidence": {
            "valid": True,
            "checked_count": 0,
            "broken_evidence_id": None,
        },
        "incidents": {
            "valid": True,
            "checked_count": 0,
            "broken_incident_id": None,
        },
    }


def test_response_exposes_only_integrity_conclusions_and_ids(client):
    machine_id = create_machine(client)
    body = client.get(summary_url(machine_id)).json()
    assert set(body.keys()) == {
        "machine_id",
        "valid",
        "events",
        "rotations",
        "evidence",
        "incidents",
    }
    assert set(body["events"].keys()) == {
        "valid",
        "checked_count",
        "broken_event_id",
    }
    assert set(body["rotations"].keys()) == {
        "valid",
        "checked_count",
        "broken_rotation_id",
    }
    assert set(body["evidence"].keys()) == {
        "valid",
        "checked_count",
        "broken_evidence_id",
    }
    assert set(body["incidents"].keys()) == {
        "valid",
        "checked_count",
        "broken_incident_id",
    }
    # No key content, policy text, or identity material leaks into the summary.
    assert "key-1" not in client.get(summary_url(machine_id)).text


def test_healthy_machine_all_blocks_valid_with_counts(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event_one = record_event(client, machine_id, resource="res/a")
    event_two = record_event(client, machine_id, resource="res/b")
    rotate_key(client, machine_id, "key-2", expected_version=1)
    rotate_key(client, machine_id, "key-3", expected_version=2)
    attach_evidence(client, machine_id, event_one["id"])
    attach_evidence(client, machine_id, event_two["id"], content_hash=HASH_B)
    incident = register_incident(client, machine_id, event_one["id"])
    transition_incident(
        client, machine_id, event_one["id"], incident["id"], "acknowledged"
    )
    transition_incident(
        client, machine_id, event_one["id"], incident["id"], "resolved"
    )
    assign_responsibility(client, machine_id, event_one["id"], incident["id"])

    body = client.get(summary_url(machine_id)).json()
    assert body["machine_id"] == machine_id
    assert body["valid"] is True
    assert body["events"] == {
        "valid": True,
        "checked_count": 2,
        "broken_event_id": None,
    }
    assert body["rotations"] == {
        "valid": True,
        "checked_count": 2,
        "broken_rotation_id": None,
    }
    assert body["evidence"] == {
        "valid": True,
        "checked_count": 2,
        "broken_evidence_id": None,
    }
    assert body["incidents"] == {
        "valid": True,
        "checked_count": 1,
        "broken_incident_id": None,
    }


# --------------------------------------------------------------------------- #
# Error semantics and precedence
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404_with_no_summary(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(summary_url(missing))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("query", ["?x=1", "?unexpected=value", "?machine_id=m"])
def test_extra_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(f"{summary_url(machine_id)}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"{summary_url(missing)}?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = summary_url(machine_id)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Each block surfaces its existing audit's first anomaly
# --------------------------------------------------------------------------- #


def test_broken_event_chain_fails_events_block_and_overall(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    events = [
        record_event(client, machine_id, resource=f"res/{i}") for i in range(3)
    ]
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_events SET reason = 'forged' "
                "WHERE id = :id"
            ),
            {"id": events[1]["id"]},
        )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["events"] == {
        "valid": False,
        "checked_count": 3,
        "broken_event_id": events[1]["id"],
    }
    # The other three blocks are unaffected by the event-chain damage.
    assert body["rotations"]["valid"] is True
    assert body["evidence"]["valid"] is True
    assert body["incidents"]["valid"] is True


def test_broken_rotation_chain_fails_rotations_block(client):
    machine_id = create_machine(client)
    rotate_key(client, machine_id, "key-2", expected_version=1)
    with client.app.state.engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM key_rotation_events")
        ).first()
        rotation_id = row[0]
        conn.execute(
            text(
                "UPDATE key_rotation_events SET new_public_key = 'forged' "
                "WHERE id = :id"
            ),
            {"id": rotation_id},
        )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["rotations"] == {
        "valid": False,
        "checked_count": 1,
        "broken_rotation_id": rotation_id,
    }
    assert body["events"]["valid"] is True
    assert body["evidence"]["valid"] is True
    assert body["incidents"]["valid"] is True


def test_bad_evidence_fingerprint_fails_evidence_block(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    # An uppercase fingerprint can only arrive via direct storage; the audit
    # must fail it exactly as stored, never case-fold it into a legal value.
    insert_evidence_row(
        client,
        machine_id,
        str(uuid.uuid4()),
        event["id"],
        T0,
        content_hash="A" * 64,
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["evidence"]["valid"] is False
    assert body["evidence"]["checked_count"] == 1
    assert body["evidence"]["broken_evidence_id"] is not None


def test_foreign_event_evidence_fails_evidence_block(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    evidence_id = str(uuid.uuid4())
    insert_evidence_row(
        client, machine_one, evidence_id, foreign_event["id"], T0
    )

    body = client.get(summary_url(machine_one)).json()
    assert body["valid"] is False
    assert body["evidence"] == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence_id,
    }


def test_unclosed_resolved_incident_fails_incidents_block(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(
        client, machine_id, event["id"], incident["id"], "acknowledged"
    )
    transition_incident(
        client, machine_id, event["id"], incident["id"], "resolved"
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["incidents"] == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }


def test_first_broken_incident_id_is_reported(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event_one = record_event(client, machine_id, resource="res/a")
    event_two = record_event(client, machine_id, resource="res/b")
    first = register_incident(client, machine_id, event_one["id"])
    second = register_incident(client, machine_id, event_two["id"])
    for incident in (first, second):
        transition_incident(
            client, machine_id, incident["event_id"], incident["id"],
            "acknowledged",
        )
        transition_incident(
            client, machine_id, incident["event_id"], incident["id"],
            "resolved",
        )

    body = client.get(summary_url(machine_id)).json()
    assert body["incidents"]["valid"] is False
    assert body["incidents"]["checked_count"] == 2
    # The first incident in (created_at, id) order keeps the broken id.
    assert body["incidents"]["broken_incident_id"] == first["id"]


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_other_machine_damage_never_affects_summary(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_one)
    allow_read(client, machine_two)
    record_event(client, machine_one)
    damaged = record_event(client, machine_two)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_events SET reason = 'forged' "
                "WHERE id = :id"
            ),
            {"id": damaged["id"]},
        )

    body = client.get(summary_url(machine_one)).json()
    assert body["valid"] is True
    assert body["events"]["checked_count"] == 1
    assert body["events"]["broken_event_id"] is None
    for block in ("rotations", "evidence", "incidents"):
        assert body[block]["checked_count"] == 0
        assert body[block]["valid"] is True

    damaged_body = client.get(summary_url(machine_two)).json()
    assert damaged_body["valid"] is False
    assert damaged_body["events"]["broken_event_id"] == damaged["id"]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def _table_state(client):
    with client.app.state.engine.connect() as conn:
        names = [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
        ]
        return {
            name: list(conn.execute(text(f"SELECT * FROM {name}")))
            for name in names
        }


def test_summary_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    rotate_key(client, machine_id, "key-2", expected_version=1)
    attach_evidence(client, machine_id, event["id"])
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(
        client, machine_id, event["id"], incident["id"], "acknowledged"
    )

    before = _table_state(client)
    first = client.get(summary_url(machine_id))
    middle = _table_state(client)
    second = client.get(summary_url(machine_id))
    third = client.get(summary_url(machine_id))
    after = _table_state(client)

    assert first.status_code == 200
    assert first.content == second.content == third.content
    assert before == middle == after


def test_summary_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        allow_read(first, machine_id)
        event = record_event(first, machine_id)
        rotate_key(first, machine_id, "key-2", expected_version=1)
        attach_evidence(first, machine_id, event["id"])
        expected = first.get(summary_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(summary_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    body = response.json()
    assert body["valid"] is True
    assert body["events"]["checked_count"] == 1
    assert body["rotations"]["checked_count"] == 1
    assert body["evidence"]["checked_count"] == 1
    assert body["incidents"]["checked_count"] == 0
