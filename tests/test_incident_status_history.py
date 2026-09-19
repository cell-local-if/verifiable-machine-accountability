import re
from concurrent.futures import ThreadPoolExecutor

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


def status_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status"
    )


def history_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status-history"
    )


def incidents_url(machine_id, event_id):
    return f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents"


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = create_incident(client, machine_id, event_id)
    return machine_id, event_id, record


def test_acknowledge_returns_200_with_updated_incident(client, incident):
    machine_id, event_id, record = incident

    response = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_type",
        "summary",
        "status",
        "created_at",
    }
    assert body["id"] == record["id"]
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event_id
    assert body["incident_type"] == record["incident_type"]
    assert body["summary"] == record["summary"]
    assert body["status"] == "acknowledged"
    assert body["created_at"] == record["created_at"]


def test_resolve_after_acknowledge_returns_200(client, incident):
    machine_id, event_id, record = incident

    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )
    response = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "resolved"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": None},
        {"status": "open"},
        {"status": "resolved_after"},
        {"status": "acknowledged "},
        {"status": " ACKNOWLEDGED"},
        {"status": ""},
        {"status": 1},
        {"status": 1.0},
        {"status": True},
        {"status": ["acknowledged"]},
        {"status": {"value": "acknowledged"}},
        "acknowledged",
        ["acknowledged"],
    ],
)
def test_invalid_status_body_returns_422(client, incident, payload):
    machine_id, event_id, record = incident

    response = client.post(
        status_url(machine_id, event_id, record["id"]), json=payload
    )

    assert response.status_code == 422


def test_invalid_body_is_422_before_path_lookup(client):
    response = client.post(
        status_url(MISSING_ID, MISSING_ID, MISSING_ID),
        json={"status": "open"},
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
def test_post_status_missing_owner_returns_404(client, incident, path):
    machine_id, event_id, record = incident
    which_machine, which_event, which_incident = path
    mid = MISSING_ID if which_machine == "missing_machine" else machine_id
    eid = MISSING_ID if which_event == "missing_event" else event_id
    iid = MISSING_ID if which_incident == "missing_incident" else record["id"]

    response = client.post(status_url(mid, eid, iid), json={"status": "acknowledged"})

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_post_status_event_of_other_machine_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_machine = create_machine(client, external_id="machine-2")

    response = client.post(
        status_url(other_machine, event_id, record["id"]),
        json={"status": "acknowledged"},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_post_status_incident_of_other_event_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_event = record_event(client, machine_id, resource="res/other")

    response = client.post(
        status_url(machine_id, other_event, record["id"]),
        json={"status": "acknowledged"},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize(
    "current_status,target",
    [
        ("open", "resolved"),
        ("acknowledged", "acknowledged"),
        ("resolved", "resolved"),
        ("resolved", "acknowledged"),
    ],
)
def test_illegal_transition_returns_409_and_writes_nothing(
    client, incident, current_status, target
):
    machine_id, event_id, record = incident
    if current_status in ("acknowledged", "resolved"):
        client.post(
            status_url(machine_id, event_id, record["id"]),
            json={"status": "acknowledged"},
        )
    if current_status == "resolved":
        client.post(
            status_url(machine_id, event_id, record["id"]),
            json={"status": "resolved"},
        )
    history_before = client.get(
        history_url(machine_id, event_id, record["id"])
    ).json()

    response = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": target},
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "invalid_status_transition"}}
    # The incident status is untouched and no history record was appended.
    stored = client.get(incidents_url(machine_id, event_id)).json()
    assert [i for i in stored if i["id"] == record["id"]][0]["status"] == (
        current_status
    )
    history_after = client.get(
        history_url(machine_id, event_id, record["id"])
    ).json()
    assert history_after == history_before


def test_successful_transition_atomically_updates_status_and_history(
    client, incident
):
    machine_id, event_id, record = incident

    response = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )

    assert response.status_code == 200
    stored = client.get(incidents_url(machine_id, event_id)).json()
    assert stored[0]["status"] == "acknowledged"
    history = client.get(history_url(machine_id, event_id, record["id"])).json()
    assert len(history) == 1
    entry = history[0]
    assert set(entry.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_id",
        "from_status",
        "to_status",
        "created_at",
    }
    assert UUID_RE.match(entry["id"])
    assert entry["machine_id"] == machine_id
    assert entry["event_id"] == event_id
    assert entry["incident_id"] == record["id"]
    assert entry["from_status"] == "open"
    assert entry["to_status"] == "acknowledged"
    assert RFC3339_Z_RE.match(entry["created_at"])


def test_full_lifecycle_records_two_ordered_history_entries(client, incident):
    machine_id, event_id, record = incident

    ack = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )
    resolve = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "resolved"},
    )
    assert ack.status_code == 200
    assert resolve.status_code == 200

    history = client.get(history_url(machine_id, event_id, record["id"])).json()
    assert [h["from_status"] for h in history] == ["open", "acknowledged"]
    assert [h["to_status"] for h in history] == ["acknowledged", "resolved"]
    ordering_key = [(h["created_at"], h["id"]) for h in history]
    assert ordering_key == sorted(ordering_key)


def test_history_entries_are_immutable_after_later_transition(client, incident):
    machine_id, event_id, record = incident
    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )
    first = client.get(
        history_url(machine_id, event_id, record["id"])
    ).json()[0]

    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "resolved"},
    )
    history = client.get(history_url(machine_id, event_id, record["id"])).json()

    assert history[0] == first
    assert len(history) == 2


def test_status_history_empty_for_fresh_incident(client, incident):
    machine_id, event_id, record = incident

    response = client.get(history_url(machine_id, event_id, record["id"]))

    assert response.status_code == 200
    assert response.json() == []


def test_get_history_missing_owner_returns_404(client, incident):
    machine_id, event_id, record = incident

    for path in (
        history_url(MISSING_ID, event_id, record["id"]),
        history_url(machine_id, MISSING_ID, record["id"]),
        history_url(machine_id, event_id, MISSING_ID),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


def test_get_history_event_of_other_machine_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_machine = create_machine(client, external_id="machine-2")

    response = client.get(
        history_url(other_machine, event_id, record["id"])
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_history_is_isolated_between_incidents(client, incident):
    machine_id, event_id, record = incident
    other = create_incident(
        client, machine_id, event_id, incident_type="breach", summary="second"
    )
    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )

    own_history = client.get(
        history_url(machine_id, event_id, record["id"])
    ).json()
    other_history = client.get(
        history_url(machine_id, event_id, other["id"])
    ).json()

    assert len(own_history) == 1
    assert other_history == []


def test_failed_transition_leaves_other_incidents_untouched(client, incident):
    machine_id, event_id, record = incident
    other = create_incident(
        client, machine_id, event_id, incident_type="breach", summary="second"
    )
    client.post(
        status_url(machine_id, event_id, other["id"]),
        json={"status": "acknowledged"},
    )

    # open -> resolved is illegal on the first incident.
    response = client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "resolved"},
    )

    assert response.status_code == 409
    incidents = {
        i["id"]: i
        for i in client.get(incidents_url(machine_id, event_id)).json()
    }
    assert incidents[record["id"]]["status"] == "open"
    assert incidents[other["id"]]["status"] == "acknowledged"
    assert (
        client.get(history_url(machine_id, event_id, record["id"])).json() == []
    )
    assert len(
        client.get(history_url(machine_id, event_id, other["id"])).json()
    ) == 1


def test_concurrent_acknowledgements_transition_exactly_once(client, incident):
    machine_id, event_id, record = incident

    def post_acknowledge(_):
        return client.post(
            status_url(machine_id, event_id, record["id"]),
            json={"status": "acknowledged"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(post_acknowledge, range(8)))

    status_codes = sorted(r.status_code for r in responses)
    assert status_codes.count(200) == 1
    assert status_codes.count(409) == 7
    stored = client.get(incidents_url(machine_id, event_id)).json()
    assert stored[0]["status"] == "acknowledged"
    history = client.get(history_url(machine_id, event_id, record["id"])).json()
    assert len(history) == 1
    assert history[0]["from_status"] == "open"
    assert history[0]["to_status"] == "acknowledged"


def test_status_transitions_do_not_modify_event_evidence_chain_or_links(
    client, incident
):
    machine_id, event_id, record = incident
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).json()
    integrity_before = client.get(f"{events_url}/integrity").json()
    evidence_before = client.get(
        f"{events_url}/{event_id}/evidence"
    ).json()
    links_before = client.get(f"{events_url}/{event_id}/causal-links").json()

    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "acknowledged"},
    )
    client.post(
        status_url(machine_id, event_id, record["id"]),
        json={"status": "resolved"},
    )
    client.get(history_url(machine_id, event_id, record["id"]))

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
        record = create_incident(first, machine_id, event_id)
        first.post(
            status_url(machine_id, event_id, record["id"]),
            json={"status": "acknowledged"},
        )

    with TestClient(app) as second:
        incident_response = second.get(incidents_url(machine_id, event_id))
        history_response = second.get(
            history_url(machine_id, event_id, record["id"])
        )

    assert incident_response.status_code == 200
    assert incident_response.json()[0]["status"] == "acknowledged"
    history = history_response.json()
    assert len(history) == 1
    entry = history[0]
    assert entry["machine_id"] == machine_id
    assert entry["event_id"] == event_id
    assert entry["incident_id"] == record["id"]
    assert entry["from_status"] == "open"
    assert entry["to_status"] == "acknowledged"
    assert RFC3339_Z_RE.match(entry["created_at"])
