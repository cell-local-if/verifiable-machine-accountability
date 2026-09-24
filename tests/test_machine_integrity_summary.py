"""Tests for the read-only machine-level integrity summary.

Covers `GET /machines/{machine_id}/integrity-summary`: the complete
all-zero response for an empty machine, aggregation of the four existing
single audits (authorization event chain, key rotation chain, evidence,
incident lifecycle/responsibility closure) in their existing shapes, the
overall ``valid`` conjunction, first-anomaly retention per block, machine
isolation, strict read-only stability, persistence across a restart, the
``invalid_query`` / ``not_found`` outcomes, validation-before-lookup
ordering, and GET-only routing.
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


MISSING = "00000000-0000-0000-0000-000000000000"
VALID_HASH = "a" * 64


def summary_url(machine_id):
    return f"/machines/{machine_id}/integrity-summary"


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


def allow_read(client, machine_id, resource_pattern="res/*"):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": resource_pattern,
            "enabled": True,
        },
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": resource_pattern,
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


def rotate_key(client, machine_id, public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": public_key, "expected_version": expected_version},
    )
    assert response.status_code == 200
    return response.json()


def add_evidence(client, machine_id, event_id, content_hash=VALID_HASH):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/evidence",
        json={"evidence_type": "log", "content_hash": content_hash},
    )
    assert response.status_code == 201
    return response.json()


def create_resolved_incident_with_owner(client, machine_id, event_id):
    """Drive one incident through a fully closed, audit-sound lifecycle."""
    incident = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents",
        json={"incident_type": "breach", "summary": "something happened"},
    )
    assert incident.status_code == 201
    incident_id = incident.json()["id"]
    base = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents/{incident_id}"
    )
    assert client.post(f"{base}/status", json={"status": "acknowledged"}).status_code == 200
    assignment = client.post(
        f"{base}/responsibility-assignments",
        json={"party": "ops", "role": "owner"},
    )
    assert assignment.status_code == 201
    assert client.post(f"{base}/status", json={"status": "resolved"}).status_code == 200
    return incident.json()


# --------------------------------------------------------------------------- #
# Response shape
# --------------------------------------------------------------------------- #


def test_empty_machine_returns_complete_all_zero_summary(client):
    machine_id = create_machine(client)

    response = client.get(summary_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "valid": True,
        "events": {"valid": True, "checked_count": 0, "broken_event_id": None},
        "key_rotations": {
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


def test_response_has_exactly_the_documented_keys(client):
    machine_id = create_machine(client)
    body = client.get(summary_url(machine_id)).json()
    assert set(body) == {"machine_id", "valid", "events", "key_rotations",
                        "evidence", "incidents"}
    assert set(body["events"]) == {"valid", "checked_count", "broken_event_id"}
    assert set(body["key_rotations"]) == {
        "valid", "checked_count", "broken_rotation_id"
    }
    assert set(body["evidence"]) == {
        "valid", "checked_count", "broken_evidence_id"
    }
    assert set(body["incidents"]) == {
        "valid", "checked_count", "broken_incident_id"
    }


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404_with_no_summary_data(client):
    response = client.get(summary_url(MISSING))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_any_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    for suffix in ("?unexpected=1", "?x=", "?valid=true&y=2"):
        response = client.get(summary_url(machine_id) + suffix)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    response = client.get(summary_url(MISSING) + "?unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = summary_url(machine_id)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Aggregation of sound data
# --------------------------------------------------------------------------- #


def test_sound_machine_reports_all_blocks_valid_with_counts(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    events = [record_event(client, machine_id, resource=f"res/{i}")
              for i in range(3)]
    rotate_key(client, machine_id, "key-2", 1)
    rotate_key(client, machine_id, "key-3", 2)
    add_evidence(client, machine_id, events[0]["id"], content_hash="a" * 64)
    add_evidence(client, machine_id, events[1]["id"], content_hash="b" * 64)
    create_resolved_incident_with_owner(client, machine_id, events[2]["id"])

    body = client.get(summary_url(machine_id)).json()
    assert body["machine_id"] == machine_id
    assert body["valid"] is True
    assert body["events"] == {
        "valid": True, "checked_count": 3, "broken_event_id": None
    }
    assert body["key_rotations"]["valid"] is True
    assert body["key_rotations"]["checked_count"] == 2
    assert body["key_rotations"]["broken_rotation_id"] is None
    assert body["evidence"] == {
        "valid": True, "checked_count": 2, "broken_evidence_id": None
    }
    assert body["incidents"] == {
        "valid": True, "checked_count": 1, "broken_incident_id": None
    }


def test_blocks_match_the_individual_integrity_endpoints(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    rotate_key(client, machine_id, "key-2", 1)
    add_evidence(client, machine_id, event["id"])
    create_resolved_incident_with_owner(client, machine_id, event["id"])

    body = client.get(summary_url(machine_id)).json()
    events_check = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    rotations_check = client.get(
        f"/machines/{machine_id}/key-rotation-events/integrity"
    ).json()
    evidence_check = client.get(
        f"/machines/{machine_id}/authorization-decision-events/evidence/integrity"
    ).json()
    incidents_check = client.get(
        f"/machines/{machine_id}/authorization-decision-events/incidents/integrity"
    ).json()

    assert body["events"] == events_check
    assert body["key_rotations"] == rotations_check
    assert body["evidence"] == evidence_check
    assert body["incidents"] == incidents_check


# --------------------------------------------------------------------------- #
# Each block detects its own anomaly and drives the overall flag
# --------------------------------------------------------------------------- #


def _tamper(db_path, statement, params):
    connection = sqlite3.connect(db_path)
    connection.execute(statement, params)
    connection.commit()
    connection.close()


def test_broken_event_chain_fails_events_block_and_overall(client, tmp_path):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    events = [record_event(client, machine_id, resource=f"res/{i}")
              for i in range(2)]

    _tamper(
        tmp_path / "test.db",
        "UPDATE authorization_decision_events SET reason = 'forged' WHERE id = ?",
        (events[1]["id"],),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["events"] == {
        "valid": False,
        "checked_count": 2,
        "broken_event_id": events[1]["id"],
    }
    assert body["key_rotations"]["valid"] is True
    assert body["evidence"]["valid"] is True
    assert body["incidents"]["valid"] is True


def test_broken_rotation_chain_fails_key_rotations_block_and_overall(
    client, tmp_path
):
    machine_id = create_machine(client)
    rotate_key(client, machine_id, "key-2", 1)
    rotation = client.get(
        f"/machines/{machine_id}/key-rotation-events"
    ).json()[0]

    _tamper(
        tmp_path / "test.db",
        "UPDATE key_rotation_events SET new_public_key = 'forged' WHERE id = ?",
        (rotation["id"],),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["key_rotations"] == {
        "valid": False,
        "checked_count": 1,
        "broken_rotation_id": rotation["id"],
    }
    assert body["events"]["valid"] is True
    assert body["evidence"]["valid"] is True
    assert body["incidents"]["valid"] is True


def test_bad_evidence_fingerprint_fails_evidence_block_and_overall(
    client, tmp_path
):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    evidence = add_evidence(client, machine_id, event["id"])

    # Uppercase hex is not the exact lowercase fingerprint format; the audit
    # must report it rather than normalizing the stored value.
    _tamper(
        tmp_path / "test.db",
        "UPDATE authorization_decision_evidence SET content_hash = ? WHERE id = ?",
        ("A" * 64, evidence["id"]),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["evidence"] == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": evidence["id"],
    }
    assert body["events"]["valid"] is True
    assert body["key_rotations"]["valid"] is True
    assert body["incidents"]["valid"] is True


def test_resolved_incident_without_owner_fails_incidents_block_and_overall(
    client,
):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)

    incident = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents",
        json={"incident_type": "breach", "summary": "unowned resolution"},
    ).json()
    base = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents/{incident['id']}"
    )
    client.post(f"{base}/status", json={"status": "acknowledged"})
    client.post(f"{base}/status", json={"status": "resolved"})

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["incidents"] == {
        "valid": False,
        "checked_count": 1,
        "broken_incident_id": incident["id"],
    }
    assert body["events"]["valid"] is True
    assert body["key_rotations"]["valid"] is True
    assert body["evidence"]["valid"] is True


def test_multiple_broken_blocks_all_report_and_overall_is_false(
    client, tmp_path
):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    events = [record_event(client, machine_id, resource=f"res/{i}")
              for i in range(2)]
    rotate_key(client, machine_id, "key-2", 1)
    evidence = add_evidence(client, machine_id, events[0]["id"])

    incident = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{events[1]['id']}/incidents",
        json={"incident_type": "breach", "summary": "open"},
    ).json()
    # Detach the incident's event so the incident block fails ownership.
    _tamper(
        tmp_path / "test.db",
        "UPDATE authorization_decision_incidents SET event_id = ? WHERE id = ?",
        (MISSING, incident["id"]),
    )
    # Break the event chain and the evidence fingerprint too.
    _tamper(
        tmp_path / "test.db",
        "UPDATE authorization_decision_events SET resource = 'res/x' WHERE id = ?",
        (events[0]["id"],),
    )
    _tamper(
        tmp_path / "test.db",
        "UPDATE authorization_decision_evidence SET content_hash = ? WHERE id = ?",
        ("z" * 64, evidence["id"]),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["valid"] is False
    assert body["events"]["valid"] is False
    assert body["events"]["broken_event_id"] == events[0]["id"]
    assert body["evidence"]["valid"] is False
    assert body["evidence"]["broken_evidence_id"] == evidence["id"]
    assert body["incidents"]["valid"] is False
    assert body["incidents"]["broken_incident_id"] == incident["id"]
    # The one rotation is sound.
    assert body["key_rotations"] == {
        "valid": True, "checked_count": 1, "broken_rotation_id": None
    }


# --------------------------------------------------------------------------- #
# Machine isolation, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_only_path_machine_records_are_counted_and_checked(client):
    good_machine = create_machine(client, external_id="good")
    bad_machine = create_machine(client, external_id="bad")
    for machine_id in (good_machine, bad_machine):
        allow_read(client, machine_id)
    good_events = [
        record_event(client, good_machine, resource="res/g"),
    ]
    bad_event = record_event(client, bad_machine, resource="res/b")

    add_evidence(client, good_machine, good_events[0]["id"], content_hash="a" * 64)
    add_evidence(client, bad_machine, bad_event["id"], content_hash="b" * 64)
    # A healthy closed incident on the good machine.
    create_resolved_incident_with_owner(client, good_machine, good_events[0]["id"])
    # A broken (resolved, unowned) incident on the other machine.
    other_incident = client.post(
        f"/machines/{bad_machine}/authorization-decision-events/"
        f"{bad_event['id']}/incidents",
        json={"incident_type": "breach", "summary": "x"},
    ).json()
    base = (
        f"/machines/{bad_machine}/authorization-decision-events/"
        f"{bad_event['id']}/incidents/{other_incident['id']}"
    )
    client.post(f"{base}/status", json={"status": "acknowledged"})
    client.post(f"{base}/status", json={"status": "resolved"})

    good_summary = client.get(summary_url(good_machine)).json()
    assert good_summary["valid"] is True
    assert good_summary["events"]["checked_count"] == 1
    assert good_summary["evidence"]["checked_count"] == 1
    assert good_summary["incidents"]["checked_count"] == 1

    bad_summary = client.get(summary_url(bad_machine)).json()
    assert bad_summary["valid"] is False
    assert bad_summary["incidents"]["checked_count"] == 1
    assert bad_summary["incidents"]["broken_incident_id"] == other_incident["id"]


def test_summary_is_strictly_read_only_and_stable_on_repeat_calls(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event = record_event(client, machine_id)
    rotate_key(client, machine_id, "key-2", 1)
    add_evidence(client, machine_id, event["id"])
    create_resolved_incident_with_owner(client, machine_id, event["id"])

    snapshots = {
        "events": client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json(),
        "rotations": client.get(
            f"/machines/{machine_id}/key-rotation-events"
        ).json(),
        "evidence": client.get(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event['id']}/evidence"
        ).json(),
        "incidents": client.get(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{event['id']}/incidents"
        ).json(),
    }

    first = client.get(summary_url(machine_id))
    second = client.get(summary_url(machine_id))
    assert first.status_code == 200
    assert second.json() == first.json()

    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json() == snapshots["events"]
    assert client.get(
        f"/machines/{machine_id}/key-rotation-events"
    ).json() == snapshots["rotations"]
    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/evidence"
    ).json() == snapshots["evidence"]
    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents"
    ).json() == snapshots["incidents"]


def test_summary_persists_across_application_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        allow_read(first, machine_id)
        event = record_event(first, machine_id)
        rotate_key(first, machine_id, "key-2", 1)
        add_evidence(first, machine_id, event["id"])
        create_resolved_incident_with_owner(first, machine_id, event["id"])
        before = first.get(summary_url(machine_id)).json()

    with TestClient(app) as second:
        after = second.get(summary_url(machine_id)).json()
        assert after == before
        assert after["valid"] is True

    with TestClient(app) as third:
        assert third.get(summary_url(machine_id)).json() == before


def test_existing_endpoints_remain_available_alongside_summary(client):
    machine_id = create_machine(client)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get(f"/machines/{machine_id}").status_code == 200
    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).status_code == 200
    assert client.get(
        f"/machines/{machine_id}/key-rotation-events/integrity"
    ).status_code == 200
    assert client.get(summary_url(machine_id)).status_code == 200
