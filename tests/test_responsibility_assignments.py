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
    return machine_id, event_id, record


def test_create_assignment_returns_201_with_full_record(client, incident):
    machine_id, event_id, record = incident

    response = client.post(
        assignments_url(machine_id, event_id, record["id"]),
        json={"party": "alice", "role": "owner"},
    )

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {
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
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event_id
    assert body["incident_id"] == record["id"]
    assert body["party"] == "alice"
    assert body["role"] == "owner"
    assert RFC3339_Z_RE.match(body["created_at"])


def test_create_assignment_strips_field_whitespace(client, incident):
    machine_id, event_id, record = incident

    response = client.post(
        assignments_url(machine_id, event_id, record["id"]),
        json={"party": "  alice\t", "role": " owner \n"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["party"] == "alice"
    assert body["role"] == "owner"


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "owner"},
        {"party": "alice"},
        {},
        {"party": "   ", "role": "owner"},
        {"party": "", "role": "owner"},
        {"party": None, "role": "owner"},
        {"party": 1, "role": "owner"},
        {"party": ["alice"], "role": "owner"},
        {"party": "alice", "role": "   "},
        {"party": "alice", "role": ""},
        {"party": "alice", "role": None},
        {"party": "alice", "role": 1},
        {"party": "alice", "role": {"value": "owner"}},
    ],
)
def test_create_assignment_rejects_invalid_payload_with_422(client, incident, payload):
    machine_id, event_id, record = incident

    response = client.post(
        assignments_url(machine_id, event_id, record["id"]), json=payload
    )

    assert response.status_code == 422


def test_invalid_body_is_422_before_path_lookup(client):
    response = client.post(
        assignments_url(MISSING_ID, MISSING_ID, MISSING_ID),
        json={"party": "  ", "role": ""},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        ("missing_machine", "existing_event", "existing_incident"),
        ("existing_machine", "missing_event", "existing_incident"),
        ("existing_machine", "existing_event", "missing_incident"),
    ],
)
def test_post_assignment_missing_owner_returns_404(client, incident, path):
    machine_id, event_id, record = incident
    which_machine, which_event, which_incident = path
    mid = MISSING_ID if which_machine == "missing_machine" else machine_id
    eid = MISSING_ID if which_event == "missing_event" else event_id
    iid = MISSING_ID if which_incident == "missing_incident" else record["id"]

    response = client.post(
        assignments_url(mid, eid, iid), json={"party": "alice", "role": "owner"}
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_post_assignment_event_of_other_machine_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_machine = create_machine(client, external_id="machine-2")

    response = client.post(
        assignments_url(other_machine, event_id, record["id"]),
        json={"party": "alice", "role": "owner"},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert (
        client.get(
            assignments_url(machine_id, event_id, record["id"])
        ).json()
        == []
    )


def test_post_assignment_incident_of_other_event_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_event = record_event(client, machine_id, resource="res/other")

    response = client.post(
        assignments_url(machine_id, other_event, record["id"]),
        json={"party": "alice", "role": "owner"},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_duplicate_party_and_role_returns_409_and_writes_nothing(client, incident):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])
    first = client.post(url, json={"party": "alice", "role": "owner"}).json()

    response = client.post(url, json={"party": "alice", "role": "owner"})

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_assignment"}}
    assert client.get(url).json() == [first]


def test_duplicate_match_uses_trimmed_values(client, incident):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])
    client.post(url, json={"party": "alice", "role": "owner"})

    response = client.post(url, json={"party": "  alice\t", "role": "owner\n"})

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_assignment"}}


def test_different_party_or_role_allowed_on_same_incident(client, incident):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])

    assert client.post(
        url, json={"party": "alice", "role": "owner"}
    ).status_code == 201
    assert client.post(
        url, json={"party": "bob", "role": "owner"}
    ).status_code == 201
    assert client.post(
        url, json={"party": "alice", "role": "reviewer"}
    ).status_code == 201


def test_same_party_and_role_allowed_on_different_incidents(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_one = create_incident(client, machine_id, event_id, summary="one")
    incident_two = create_incident(client, machine_id, event_id, summary="two")

    assert (
        client.post(
            assignments_url(machine_id, event_id, incident_one["id"]),
            json={"party": "alice", "role": "owner"},
        ).status_code
        == 201
    )
    assert (
        client.post(
            assignments_url(machine_id, event_id, incident_two["id"]),
            json={"party": "alice", "role": "owner"},
        ).status_code
        == 201
    )


def test_list_assignments_empty_returns_empty_list(client, incident):
    machine_id, event_id, record = incident

    response = client.get(assignments_url(machine_id, event_id, record["id"]))

    assert response.status_code == 200
    assert response.json() == []


def test_list_assignments_returns_records_ordered_by_created_at_then_id(
    client, incident
):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])
    created = [
        client.post(
            url, json={"party": f"party-{n}", "role": f"role-{n}"}
        ).json()
        for n in range(3)
    ]

    response = client.get(url)

    assert response.status_code == 200
    records = response.json()
    expected = sorted(created, key=lambda r: (r["created_at"], r["id"]))
    assert [r["id"] for r in records] == [r["id"] for r in expected]
    ordering_key = [(r["created_at"], r["id"]) for r in records]
    assert ordering_key == sorted(ordering_key)
    for entry in records:
        assert entry["machine_id"] == machine_id
        assert entry["event_id"] == event_id
        assert entry["incident_id"] == record["id"]
        assert set(entry.keys()) == {
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


def test_list_assignments_is_scoped_to_the_incident(client, incident):
    machine_id, event_id, record = incident
    other = create_incident(client, machine_id, event_id, summary="second")
    client.post(
        assignments_url(machine_id, event_id, record["id"]),
        json={"party": "alice", "role": "owner"},
    )
    client.post(
        assignments_url(machine_id, event_id, other["id"]),
        json={"party": "bob", "role": "reviewer"},
    )

    records = client.get(
        assignments_url(machine_id, event_id, record["id"])
    ).json()

    assert [r["party"] for r in records] == ["alice"]


def test_list_assignments_missing_owner_returns_404(client, incident):
    machine_id, event_id, record = incident

    for path in (
        assignments_url(MISSING_ID, event_id, record["id"]),
        assignments_url(machine_id, MISSING_ID, record["id"]),
        assignments_url(machine_id, event_id, MISSING_ID),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


def test_assignments_do_not_modify_incident_event_evidence_chain_or_links(
    client, incident
):
    machine_id, event_id, record = incident
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    incidents_url = f"{events_url}/{event_id}/incidents"
    incidents_before = client.get(incidents_url).json()
    events_before = client.get(events_url).json()
    integrity_before = client.get(f"{events_url}/integrity").json()
    evidence_before = client.get(f"{events_url}/{event_id}/evidence").json()
    links_before = client.get(f"{events_url}/{event_id}/causal-links").json()

    url = assignments_url(machine_id, event_id, record["id"])
    client.post(url, json={"party": "alice", "role": "owner"})
    client.post(url, json={"party": "bob", "role": "reviewer"})
    client.get(url)

    assert client.get(incidents_url).json() == incidents_before
    assert client.get(events_url).json() == events_before
    assert client.get(f"{events_url}/integrity").json() == integrity_before
    assert client.get(f"{events_url}/{event_id}/evidence").json() == evidence_before
    assert client.get(f"{events_url}/{event_id}/causal-links").json() == links_before
    assert integrity_before["valid"] is True


def test_assignments_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        record = first.post(
            assignments_url(machine_id, event_id, incident["id"]),
            json={"party": "alice", "role": "owner"},
        ).json()

    with TestClient(app) as second:
        response = second.get(
            assignments_url(machine_id, event_id, incident["id"])
        )

    assert response.status_code == 200
    assert response.json() == [record]
