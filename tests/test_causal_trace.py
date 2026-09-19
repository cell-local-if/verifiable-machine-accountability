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
    return response.json()


def create_link(client, machine_id, cause_event_id, effect_event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{cause_event_id}/causal-links",
        json={"effect_event_id": effect_event_id},
    )
    assert response.status_code == 201
    return response.json()


def trace_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/causal-trace"
    )


def get_trace(client, machine_id, event_id, **params):
    return client.get(trace_url(machine_id, event_id), params=params)


def build_chain(client, machine_id, length):
    events = [
        record_event(client, machine_id, resource=f"res/{index}")
        for index in range(length)
    ]
    for first, second in zip(events, events[1:]):
        create_link(client, machine_id, first["id"], second["id"])
    return events


def test_downstream_trace_walks_cause_to_effect(client):
    machine_id = create_machine(client)
    events = build_chain(client, machine_id, 4)

    response = get_trace(
        client, machine_id, events[0]["id"],
        direction="downstream", max_depth=20,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == events[0]["id"]
    assert body["direction"] == "downstream"
    assert body["max_depth"] == 20
    assert body["events"] == [
        {"event_id": events[1]["id"], "depth": 1},
        {"event_id": events[2]["id"], "depth": 2},
        {"event_id": events[3]["id"], "depth": 3},
    ]


def test_upstream_trace_walks_effect_to_cause(client):
    machine_id = create_machine(client)
    events = build_chain(client, machine_id, 4)

    response = get_trace(
        client, machine_id, events[3]["id"],
        direction="upstream", max_depth=20,
    )

    assert response.status_code == 200
    assert response.json()["events"] == [
        {"event_id": events[2]["id"], "depth": 1},
        {"event_id": events[1]["id"], "depth": 2},
        {"event_id": events[0]["id"], "depth": 3},
    ]


def test_max_depth_bounds_reachable_events(client):
    machine_id = create_machine(client)
    events = build_chain(client, machine_id, 4)

    response = get_trace(
        client, machine_id, events[0]["id"],
        direction="downstream", max_depth=2,
    )

    assert response.status_code == 200
    assert response.json()["max_depth"] == 2
    assert response.json()["events"] == [
        {"event_id": events[1]["id"], "depth": 1},
        {"event_id": events[2]["id"], "depth": 2},
    ]


def test_branching_trace_reports_minimum_depth(client):
    machine_id = create_machine(client)
    root = record_event(client, machine_id, resource="res/root")
    middle = record_event(client, machine_id, resource="res/middle")
    leaf = record_event(client, machine_id, resource="res/leaf")
    create_link(client, machine_id, root["id"], middle["id"])
    create_link(client, machine_id, middle["id"], leaf["id"])
    create_link(client, machine_id, root["id"], leaf["id"])

    response = get_trace(
        client, machine_id, root["id"], direction="downstream", max_depth=20
    )

    assert response.status_code == 200
    assert response.json()["events"] == [
        {"event_id": middle["id"], "depth": 1},
        {"event_id": leaf["id"], "depth": 1},
    ]


def test_trace_excludes_start_event(client):
    machine_id = create_machine(client)
    events = build_chain(client, machine_id, 3)

    response = get_trace(
        client, machine_id, events[1]["id"],
        direction="downstream", max_depth=20,
    )

    assert response.status_code == 200
    assert response.json()["events"] == [{"event_id": events[2]["id"], "depth": 1}]


def test_trace_without_reachable_events_returns_empty_list(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = get_trace(
        client, machine_id, event["id"], direction="downstream", max_depth=5
    )

    assert response.status_code == 200
    assert response.json() == {
        "event_id": event["id"],
        "direction": "downstream",
        "max_depth": 5,
        "events": [],
    }


def test_trace_does_not_cross_machine_boundary(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    events_one = build_chain(client, machine_one, 2)
    events_two = build_chain(client, machine_two, 2)

    response = get_trace(
        client, machine_one, events_one[0]["id"],
        direction="downstream", max_depth=20,
    )

    assert response.status_code == 200
    reached = {item["event_id"] for item in response.json()["events"]}
    assert reached == {events_one[1]["id"]}
    assert events_two[0]["id"] not in reached
    assert events_two[1]["id"] not in reached


def test_trace_missing_event_returns_404(client):
    machine_id = create_machine(client)

    response = get_trace(
        client,
        machine_id,
        "00000000-0000-0000-0000-000000000000",
        direction="downstream",
        max_depth=3,
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_trace_event_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign = record_event(client, machine_two)

    response = get_trace(
        client, machine_one, foreign["id"], direction="upstream", max_depth=3
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"direction": "downstream"},
        {"max_depth": 3},
        {"direction": "sideways", "max_depth": 3},
        {"direction": "DOWNSTREAM", "max_depth": 3},
        {"direction": "downstream", "max_depth": 0},
        {"direction": "downstream", "max_depth": 21},
        {"direction": "downstream", "max_depth": "three"},
        {"direction": "downstream", "max_depth": "1.5"},
        {"direction": "downstream", "max_depth": "true"},
        {"direction": "downstream", "max_depth": ""},
    ],
)
def test_trace_invalid_params_return_422(client, params):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = get_trace(client, machine_id, event["id"], **params)

    assert response.status_code == 422


def test_trace_invalid_params_checked_before_missing_event(client):
    response = get_trace(
        client,
        "00000000-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000000",
        direction="sideways",
        max_depth=99,
    )

    assert response.status_code == 422


def test_trace_is_read_only(client):
    machine_id = create_machine(client)
    events = build_chain(client, machine_id, 3)
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    links_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{events[0]['id']}/causal-links"
    ).json()
    integrity_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    response = get_trace(
        client, machine_id, events[0]["id"],
        direction="downstream", max_depth=20,
    )
    assert response.status_code == 200

    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        == events_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/"
            f"{events[0]['id']}/causal-links"
        ).json()
        == links_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).json()
        == integrity_before
    )


def test_trace_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        events = build_chain(first, machine_id, 3)

    with TestClient(app) as second:
        response = get_trace(
            second, machine_id, events[0]["id"],
            direction="downstream", max_depth=20,
        )

    assert response.status_code == 200
    assert response.json()["events"] == [
        {"event_id": events[1]["id"], "depth": 1},
        {"event_id": events[2]["id"], "depth": 2},
    ]
