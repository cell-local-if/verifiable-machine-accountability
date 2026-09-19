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


def incidents_url(machine_id, event_id):
    return f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"


def create_incident(
    client, machine_id, event_id, incident_type="breach", summary="something happened"
):
    return client.post(
        incidents_url(machine_id, event_id),
        json={"incident_type": incident_type, "summary": summary},
    )


def test_create_incident_returns_201_with_full_record(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_incident(client, machine_id, event_id)

    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event_id
    assert body["incident_type"] == "breach"
    assert body["summary"] == "something happened"
    assert body["status"] == "open"
    assert RFC3339_Z_RE.match(body["created_at"])
    assert set(body.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_type",
        "summary",
        "status",
        "created_at",
    }


def test_create_incident_strips_field_whitespace(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_incident(
        client, machine_id, event_id, incident_type="  breach\t", summary=" note \n"
    )

    assert response.status_code == 201
    body = response.json()
    assert body["incident_type"] == "breach"
    assert body["summary"] == "note"


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": "note"},
        {"incident_type": "breach"},
        {},
        {"incident_type": "   ", "summary": "note"},
        {"incident_type": "", "summary": "note"},
        {"incident_type": None, "summary": "note"},
        {"incident_type": 1, "summary": "note"},
        {"incident_type": "breach", "summary": "   "},
        {"incident_type": "breach", "summary": ""},
        {"incident_type": "breach", "summary": None},
        {"incident_type": "breach", "summary": 1},
    ],
)
def test_create_incident_rejects_invalid_payload_with_422(client, payload):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.post(incidents_url(machine_id, event_id), json=payload)

    assert response.status_code == 422


def test_create_incident_invalid_payload_is_422_before_path_lookup(client):
    response = client.post(
        incidents_url(MISSING_ID, MISSING_ID),
        json={"incident_type": "  ", "summary": ""},
    )

    assert response.status_code == 422


def test_create_incident_missing_machine_returns_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_incident(client, MISSING_ID, event_id)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_incident_missing_event_returns_404(client):
    machine_id = create_machine(client)

    response = create_incident(client, machine_id, MISSING_ID)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_incident_event_of_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_id = record_event(client, machine_one)

    response = create_incident(client, machine_two, event_id)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # And nothing was written under the event's own machine either.
    assert client.get(incidents_url(machine_one, event_id)).json() == []


def test_create_incident_duplicate_type_and_summary_returns_409_and_writes_nothing(
    client,
):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = create_incident(client, machine_id, event_id).json()

    response = create_incident(client, machine_id, event_id)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_incident"}}
    records = client.get(incidents_url(machine_id, event_id)).json()
    assert records == [first]


def test_duplicate_match_uses_trimmed_values(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id, incident_type="breach", summary="note")

    response = create_incident(
        client, machine_id, event_id, incident_type="  breach", summary="note\t"
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_incident"}}


def test_different_type_or_summary_allowed_on_same_event(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    assert create_incident(
        client, machine_id, event_id, incident_type="breach", summary="one"
    ).status_code == 201
    assert create_incident(
        client, machine_id, event_id, incident_type="breach", summary="two"
    ).status_code == 201
    assert create_incident(
        client, machine_id, event_id, incident_type="other", summary="one"
    ).status_code == 201


def test_same_type_and_summary_allowed_on_different_events(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")

    assert create_incident(client, machine_id, event_one).status_code == 201
    assert create_incident(client, machine_id, event_two).status_code == 201


def test_list_incidents_empty_returns_empty_list(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.get(incidents_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == []


def test_list_incidents_returns_records_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    created = [
        create_incident(client, machine_id, event_id, summary=f"note {n}").json()
        for n in range(3)
    ]

    response = client.get(incidents_url(machine_id, event_id))

    assert response.status_code == 200
    records = response.json()
    expected = sorted(created, key=lambda r: (r["created_at"], r["id"]))
    assert [r["id"] for r in records] == [r["id"] for r in expected]
    ordering_key = [(r["created_at"], r["id"]) for r in records]
    assert ordering_key == sorted(ordering_key)
    for record in records:
        assert record["machine_id"] == machine_id
        assert record["event_id"] == event_id
        assert record["status"] == "open"


def test_list_incidents_is_scoped_to_the_event(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    create_incident(client, machine_id, event_one, summary="one")
    create_incident(client, machine_id, event_two, summary="two")

    records = client.get(incidents_url(machine_id, event_one)).json()

    assert [r["summary"] for r in records] == ["one"]


def test_list_incidents_is_scoped_to_the_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    create_incident(client, machine_one, event_one)

    assert client.get(incidents_url(machine_two, event_one)).status_code == 404


def test_list_incidents_missing_machine_or_event_returns_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    for path in (
        incidents_url(MISSING_ID, event_id),
        incidents_url(machine_id, MISSING_ID),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


def test_incidents_do_not_modify_event_evidence_chain_or_links(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).json()
    integrity_before = client.get(f"{events_url}/integrity").json()
    evidence_before = client.get(incidents_url(machine_id, event_id).replace(
        "/incidents", "/evidence"
    )).json()
    links_before = client.get(
        f"{events_url}/{event_id}/causal-links"
    ).json()

    create_incident(client, machine_id, event_id)
    create_incident(client, machine_id, event_id, summary="second")
    client.get(incidents_url(machine_id, event_id))

    assert client.get(events_url).json() == events_before
    assert client.get(f"{events_url}/integrity").json() == integrity_before
    assert (
        client.get(
            incidents_url(machine_id, event_id).replace("/incidents", "/evidence")
        ).json()
        == evidence_before
    )
    assert client.get(f"{events_url}/{event_id}/causal-links").json() == links_before
    assert integrity_before["valid"] is True


def test_incidents_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        record = create_incident(first, machine_id, event_id).json()

    with TestClient(app) as second:
        response = second.get(incidents_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == [record]
