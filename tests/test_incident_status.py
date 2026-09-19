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


def create_incident(client, machine_id, event_id, summary="something happened"):
    response = client.post(
        incidents_url(machine_id, event_id),
        json={"incident_type": "breach", "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def status_url(machine_id, event_id, incident_id):
    return f"{incidents_url(machine_id, event_id)}/{incident_id}/status"


def history_url(machine_id, event_id, incident_id):
    return f"{incidents_url(machine_id, event_id)}/{incident_id}/status-history"


def transition(client, machine_id, event_id, incident_id, status):
    return client.post(
        status_url(machine_id, event_id, incident_id), json={"status": status}
    )


def test_transition_open_to_acknowledged_returns_200_with_updated_incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    response = transition(client, machine_id, event_id, incident["id"], "acknowledged")

    assert response.status_code == 200
    body = response.json()
    assert body == {**incident, "status": "acknowledged"}


def test_transition_acknowledged_to_resolved_returns_200(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    transition(client, machine_id, event_id, incident["id"], "acknowledged")

    response = transition(client, machine_id, event_id, incident["id"], "resolved")

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    # The incident list endpoint reflects the new status too.
    records = client.get(incidents_url(machine_id, event_id)).json()
    assert [r["status"] for r in records] == ["resolved"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": None},
        {"status": 1},
        {"status": True},
        {"status": ["acknowledged"]},
        {"status": {"value": "acknowledged"}},
        {"status": ""},
        {"status": "   "},
        {"status": "open"},
        {"status": "closed"},
        {"status": "Acknowledged"},
        {"status": " acknowledged"},
        {"status": "acknowledged "},
    ],
)
def test_transition_invalid_payload_returns_422(client, payload):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    response = client.post(
        status_url(machine_id, event_id, incident["id"]), json=payload
    )

    assert response.status_code == 422
    # Nothing was written: status and history are untouched.
    records = client.get(incidents_url(machine_id, event_id)).json()
    assert [r["status"] for r in records] == ["open"]
    assert (
        client.get(history_url(machine_id, event_id, incident["id"])).json() == []
    )


def test_transition_invalid_payload_is_422_before_path_lookup(client):
    response = client.post(
        status_url(MISSING_ID, MISSING_ID, MISSING_ID), json={"status": "open"}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("method", ["post", "get"])
def test_missing_machine_event_or_incident_returns_404(client, method):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    paths = [
        (MISSING_ID, event_id, incident["id"]),
        (machine_id, MISSING_ID, incident["id"]),
        (machine_id, event_id, MISSING_ID),
    ]
    for machine, event, inc in paths:
        if method == "post":
            response = client.post(
                status_url(machine, event, inc), json={"status": "acknowledged"}
            )
        else:
            response = client.get(history_url(machine, event, inc))
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["post", "get"])
def test_ownership_mismatch_returns_404(client, method):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one, resource="res/1")
    event_two = record_event(client, machine_one, resource="res/2")
    incident = create_incident(client, machine_one, event_one)

    # Incident belongs to event_one, not event_two; event_one is not on
    # machine_two.
    for machine, event in ((machine_one, event_two), (machine_two, event_one)):
        if method == "post":
            response = client.post(
                status_url(machine, event, incident["id"]),
                json={"status": "acknowledged"},
            )
        else:
            response = client.get(history_url(machine, event, incident["id"]))
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}

    # Nothing was written under the incident's real path either.
    assert (
        client.get(history_url(machine_one, event_one, incident["id"])).json() == []
    )
    records = client.get(incidents_url(machine_one, event_one)).json()
    assert [r["status"] for r in records] == ["open"]


@pytest.mark.parametrize(
    "transitions,attempt",
    [
        ([], "resolved"),  # open -> resolved
        (["acknowledged"], "acknowledged"),  # acknowledged -> acknowledged
        (["acknowledged", "resolved"], "acknowledged"),  # resolved -> acknowledged
        (["acknowledged", "resolved"], "resolved"),  # resolved -> resolved
    ],
)
def test_disallowed_transition_returns_409_and_writes_nothing(
    client, transitions, attempt
):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    for status in transitions:
        assert (
            transition(client, machine_id, event_id, incident["id"], status).status_code
            == 200
        )
    history_before = client.get(
        history_url(machine_id, event_id, incident["id"])
    ).json()
    incidents_before = client.get(incidents_url(machine_id, event_id)).json()

    response = transition(client, machine_id, event_id, incident["id"], attempt)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    assert (
        client.get(history_url(machine_id, event_id, incident["id"])).json()
        == history_before
    )
    assert client.get(incidents_url(machine_id, event_id)).json() == incidents_before


def test_status_history_empty_returns_empty_list(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    response = client.get(history_url(machine_id, event_id, incident["id"]))

    assert response.status_code == 200
    assert response.json() == []


def test_status_history_records_full_trail_in_order(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")

    response = client.get(history_url(machine_id, event_id, incident["id"]))

    assert response.status_code == 200
    records = response.json()
    assert len(records) == 2
    assert [(r["from_status"], r["to_status"]) for r in records] == [
        ("open", "acknowledged"),
        ("acknowledged", "resolved"),
    ]
    ordering_key = [(r["created_at"], r["id"]) for r in records]
    assert ordering_key == sorted(ordering_key)
    for record in records:
        assert UUID_RE.match(record["id"])
        assert record["machine_id"] == machine_id
        assert record["event_id"] == event_id
        assert record["incident_id"] == incident["id"]
        assert RFC3339_Z_RE.match(record["created_at"])
        assert set(record.keys()) == {
            "id",
            "machine_id",
            "event_id",
            "incident_id",
            "from_status",
            "to_status",
            "created_at",
        }


def test_status_history_is_append_only_and_scoped_to_the_incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_one = create_incident(client, machine_id, event_id, summary="one")
    incident_two = create_incident(client, machine_id, event_id, summary="two")

    transition(client, machine_id, event_id, incident_one["id"], "acknowledged")
    history_after_first = client.get(
        history_url(machine_id, event_id, incident_one["id"])
    ).json()
    transition(client, machine_id, event_id, incident_one["id"], "resolved")
    transition(client, machine_id, event_id, incident_two["id"], "acknowledged")

    history_one = client.get(
        history_url(machine_id, event_id, incident_one["id"])
    ).json()
    # Existing records were not rewritten by the later transition.
    assert history_one[:1] == history_after_first
    assert len(history_one) == 2
    # The other incident has only its own record.
    history_two = client.get(
        history_url(machine_id, event_id, incident_two["id"])
    ).json()
    assert [(r["from_status"], r["to_status"]) for r in history_two] == [
        ("open", "acknowledged")
    ]
    assert history_two[0]["incident_id"] == incident_two["id"]


def test_transitions_do_not_modify_event_evidence_chain_or_links(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).json()
    integrity_before = client.get(f"{events_url}/integrity").json()
    evidence_before = client.get(f"{events_url}/{event_id}/evidence").json()
    links_before = client.get(f"{events_url}/{event_id}/causal-links").json()

    transition(client, machine_id, event_id, incident["id"], "acknowledged")
    transition(client, machine_id, event_id, incident["id"], "resolved")
    client.get(history_url(machine_id, event_id, incident["id"]))

    assert client.get(events_url).json() == events_before
    assert client.get(f"{events_url}/integrity").json() == integrity_before
    assert client.get(f"{events_url}/{event_id}/evidence").json() == evidence_before
    assert client.get(f"{events_url}/{event_id}/causal-links").json() == links_before
    assert integrity_before["valid"] is True


def test_status_and_history_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        transition(first, machine_id, event_id, incident["id"], "acknowledged")
        transition(first, machine_id, event_id, incident["id"], "resolved")
        history = first.get(history_url(machine_id, event_id, incident["id"])).json()

    with TestClient(app) as second:
        incidents = second.get(incidents_url(machine_id, event_id)).json()
        response = second.get(history_url(machine_id, event_id, incident["id"]))

    assert [r["status"] for r in incidents] == ["resolved"]
    assert response.status_code == 200
    assert response.json() == history
