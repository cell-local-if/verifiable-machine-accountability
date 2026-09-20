import re

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def create_incident(
    client, machine_id, event_id, incident_type="breach", summary="something happened"
):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def assignments_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    )


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = create_incident(client, machine_id, event_id)
    return machine_id, event_id, record["id"]


def test_create_assignment_returns_full_record(client, incident):
    machine_id, event_id, incident_id = incident
    response = client.post(
        assignments_url(machine_id, event_id, incident_id),
        json={"party": "team-alpha", "role": "responder"},
    )
    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.fullmatch(body["id"])
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event_id
    assert body["incident_id"] == incident_id
    assert body["party"] == "team-alpha"
    assert body["role"] == "responder"
    assert RFC3339_Z_RE.fullmatch(body["created_at"])


def test_create_assignment_strips_surrounding_whitespace(client, incident):
    machine_id, event_id, incident_id = incident
    response = client.post(
        assignments_url(machine_id, event_id, incident_id),
        json={"party": "  team-alpha  ", "role": "\tresponder\n"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["party"] == "team-alpha"
    assert body["role"] == "responder"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"party": "team-alpha"},
        {"role": "responder"},
        {"party": "", "role": "responder"},
        {"party": "   ", "role": "responder"},
        {"party": "team-alpha", "role": ""},
        {"party": "team-alpha", "role": "  \t "},
        {"party": 1, "role": "responder"},
        {"party": "team-alpha", "role": None},
    ],
)
def test_invalid_body_is_422_before_path_validation(client, payload):
    # Even with every path segment missing, an invalid body is a 422.
    response = client.post(
        assignments_url(MISSING_ID, MISSING_ID, MISSING_ID), json=payload
    )
    assert response.status_code == 422


def test_missing_machine_event_or_incident_is_404(client, incident):
    machine_id, event_id, incident_id = incident
    payload = {"party": "team-alpha", "role": "responder"}

    assert (
        client.post(
            assignments_url(MISSING_ID, event_id, incident_id), json=payload
        ).status_code
        == 404
    )
    assert (
        client.post(
            assignments_url(machine_id, MISSING_ID, incident_id), json=payload
        ).status_code
        == 404
    )
    assert (
        client.post(
            assignments_url(machine_id, event_id, MISSING_ID), json=payload
        ).status_code
        == 404
    )

    response = client.post(
        assignments_url(machine_id, event_id, MISSING_ID), json=payload
    )
    assert response.json() == {"error": {"code": "not_found"}}


def test_ownership_mismatch_is_404(client, incident):
    machine_id, event_id, incident_id = incident
    other_machine = create_machine(client, external_id="machine-2")
    other_event = record_event(client, other_machine)
    other_incident = create_incident(client, other_machine, other_event)["id"]
    payload = {"party": "team-alpha", "role": "responder"}

    # Incident belongs to a different event/machine than the path claims.
    assert (
        client.post(
            assignments_url(machine_id, event_id, other_incident), json=payload
        ).status_code
        == 404
    )
    assert (
        client.post(
            assignments_url(other_machine, event_id, incident_id), json=payload
        ).status_code
        == 404
    )
    assert (
        client.post(
            assignments_url(machine_id, other_event, incident_id), json=payload
        ).status_code
        == 404
    )


def test_duplicate_party_role_is_409_and_writes_nothing(client, incident):
    machine_id, event_id, incident_id = incident
    url = assignments_url(machine_id, event_id, incident_id)
    payload = {"party": "team-alpha", "role": "responder"}

    assert client.post(url, json=payload).status_code == 201
    response = client.post(url, json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_assignment"}}

    listed = client.get(url)
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_same_party_different_role_is_allowed(client, incident):
    machine_id, event_id, incident_id = incident
    url = assignments_url(machine_id, event_id, incident_id)

    assert (
        client.post(
            url, json={"party": "team-alpha", "role": "responder"}
        ).status_code
        == 201
    )
    assert (
        client.post(url, json={"party": "team-alpha", "role": "owner"}).status_code
        == 201
    )
    assert (
        client.post(url, json={"party": "team-beta", "role": "responder"}).status_code
        == 201
    )
    assert len(client.get(url).json()) == 3


def test_list_returns_records_in_created_at_id_order(client, incident):
    machine_id, event_id, incident_id = incident
    url = assignments_url(machine_id, event_id, incident_id)

    created = []
    for party, role in (
        ("team-alpha", "responder"),
        ("team-beta", "owner"),
        ("team-gamma", "observer"),
    ):
        response = client.post(url, json={"party": party, "role": role})
        assert response.status_code == 201
        created.append(response.json())

    listed = client.get(url).json()
    assert [r["id"] for r in listed] == [r["id"] for r in created]
    assert listed == sorted(created, key=lambda r: (r["created_at"], r["id"]))
    assert set(listed[0]) == {
        "id",
        "machine_id",
        "event_id",
        "incident_id",
        "party",
        "role",
        "created_at",
    }


def test_list_empty_is_empty_list(client, incident):
    machine_id, event_id, incident_id = incident
    response = client.get(assignments_url(machine_id, event_id, incident_id))
    assert response.status_code == 200
    assert response.json() == []


def test_list_404_semantics_match_create(client, incident):
    machine_id, event_id, incident_id = incident
    for path in (
        assignments_url(MISSING_ID, event_id, incident_id),
        assignments_url(machine_id, MISSING_ID, incident_id),
        assignments_url(machine_id, event_id, MISSING_ID),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


def test_assignments_are_isolated_per_incident(client):
    machine_a = create_machine(client, external_id="machine-a")
    event_a = record_event(client, machine_a)
    incident_a = create_incident(client, machine_a, event_a)["id"]

    machine_b = create_machine(client, external_id="machine-b")
    event_b = record_event(client, machine_b)
    incident_b = create_incident(client, machine_b, event_b)["id"]

    payload = {"party": "team-alpha", "role": "responder"}
    url_a = assignments_url(machine_a, event_a, incident_a)
    url_b = assignments_url(machine_b, event_b, incident_b)

    assert client.post(url_a, json=payload).status_code == 201
    # The same (party, role) pair on a different incident is not a duplicate.
    assert client.post(url_b, json=payload).status_code == 201

    assert len(client.get(url_a).json()) == 1
    assert len(client.get(url_b).json()) == 1


def test_assignments_survive_restart(client, incident, tmp_path, monkeypatch):
    machine_id, event_id, incident_id = incident
    url = assignments_url(machine_id, event_id, incident_id)
    created = client.post(url, json={"party": "team-alpha", "role": "responder"})
    assert created.status_code == 201

    # Rebuild the app state against the same database file (a "restart").
    with TestClient(app) as restarted:
        listed = restarted.get(url)
        assert listed.status_code == 200
        assert listed.json() == [created.json()]


def test_existing_incident_data_is_not_rewritten(client, incident):
    machine_id, event_id, incident_id = incident
    before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"
    ).json()

    response = client.post(
        assignments_url(machine_id, event_id, incident_id),
        json={"party": "team-alpha", "role": "responder"},
    )
    assert response.status_code == 201

    after = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"
    ).json()
    assert after == before
