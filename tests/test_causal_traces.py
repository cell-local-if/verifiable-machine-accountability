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


def trace_url(machine_id, event_id, direction, max_depth):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/causal-trace?direction={direction}&max_depth={max_depth}"
    )


def insert_link_row(client, machine_id, cause_event_id, effect_event_id):
    """Insert a causal link row directly, bypassing creation rules."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :machine_id, :cause, :effect, :created_at)"
            ),
            {
                "id": str(uuid.uuid4()),
                "machine_id": machine_id,
                "cause": cause_event_id,
                "effect": effect_event_id,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "?direction=downstream", "?max_depth=3"])
def test_missing_params_return_422(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/causal-trace{query}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize("direction", ["sideways", "DOWNSTREAM", "Upstream", "both"])
def test_invalid_direction_returns_422(client, direction):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/causal-trace?direction={direction}&max_depth=3"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "max_depth",
    ["0", "21", "-1", "100", "1.5", "3.0", "true", "false", "True", "abc", "", "   "],
)
def test_invalid_max_depth_returns_422(client, max_depth):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/causal-trace?direction=downstream&max_depth={max_depth}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize("max_depth", ["1", "20"])
def test_boundary_depths_are_accepted(client, max_depth):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        trace_url(machine_id, event["id"], "downstream", max_depth)
    )

    assert response.status_code == 200
    assert response.json()["max_depth"] == int(max_depth)


def test_invalid_params_take_precedence_over_missing_machine(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/"
        "authorization-decision-events/"
        "00000000-0000-0000-0000-000000000000/causal-trace"
        "?direction=sideways&max_depth=99"
    )

    assert response.status_code == 422


def test_missing_params_take_precedence_over_missing_machine(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/"
        "authorization-decision-events/"
        "00000000-0000-0000-0000-000000000000/causal-trace"
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Existence / ownership
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/"
        "authorization-decision-events/"
        "00000000-0000-0000-0000-000000000000/causal-trace"
        "?direction=downstream&max_depth=3"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_returns_404(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        "00000000-0000-0000-0000-000000000000/causal-trace"
        "?direction=downstream&max_depth=3"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_event_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign = record_event(client, machine_two)

    response = client.get(
        trace_url(machine_one, foreign["id"], "downstream", 3)
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Downstream traversal
# --------------------------------------------------------------------------- #


def test_downstream_trace_follows_chain_and_reports_depths(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)
    create_link(client, machine_id, c, d)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == a
    assert body["direction"] == "downstream"
    assert body["max_depth"] == 20
    assert body["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": c, "depth": 2},
        {"event_id": d, "depth": 3},
    ]


def test_downstream_trace_respects_max_depth(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)
    create_link(client, machine_id, c, d)

    response = client.get(trace_url(machine_id, a, "downstream", 1))

    assert response.json()["events"] == [{"event_id": b, "depth": 1}]

    response = client.get(trace_url(machine_id, a, "downstream", 2))

    assert response.json()["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": c, "depth": 2},
    ]


def test_downstream_trace_starts_from_middle_event(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)
    create_link(client, machine_id, c, d)

    response = client.get(trace_url(machine_id, b, "downstream", 20))

    assert response.json()["events"] == [
        {"event_id": c, "depth": 1},
        {"event_id": d, "depth": 2},
    ]


def test_downstream_trace_never_includes_start(client):
    machine_id = create_machine(client)
    a = record_event(client, machine_id)["id"]

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    assert response.status_code == 200
    assert response.json()["events"] == []
    assert all(item["event_id"] != a for item in response.json()["events"])


def test_downstream_diamond_dedups_and_keeps_minimum_depth(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, d)  # d reachable at depth 2 ...
    create_link(client, machine_id, a, d)  # ... and directly at depth 1
    create_link(client, machine_id, a, c)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    depths = {item["event_id"]: item["depth"] for item in response.json()["events"]}
    assert depths == {b: 1, c: 1, d: 1}
    assert len(response.json()["events"]) == 3


def test_downstream_trace_orders_by_depth_then_created_at_then_id(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    by_id = {e["id"]: e for e in events}
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, a, c)
    create_link(client, machine_id, b, d)
    create_link(client, machine_id, c, d)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    items = response.json()["events"]
    # b and c share depth 1 and must sort by (created_at, id); d is depth 2.
    depth_one = sorted([b, c], key=lambda eid: (by_id[eid]["created_at"], eid))
    assert [item["event_id"] for item in items] == depth_one + [d]
    depths = {item["event_id"]: item["depth"] for item in items}
    assert depths == {b: 1, c: 1, d: 2}
    sorted_keys = [
        (item["depth"], by_id[item["event_id"]]["created_at"], item["event_id"])
        for item in items
    ]
    assert sorted_keys == sorted(sorted_keys)


# --------------------------------------------------------------------------- #
# Upstream traversal
# --------------------------------------------------------------------------- #


def test_upstream_trace_walks_edges_in_reverse(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)

    response = client.get(trace_url(machine_id, c, "upstream", 20))

    assert response.json()["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": a, "depth": 2},
    ]


def test_upstream_trace_respects_max_depth(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)

    response = client.get(trace_url(machine_id, c, "upstream", 1))

    assert response.json()["events"] == [{"event_id": b, "depth": 1}]


def test_upstream_trace_without_causes_returns_empty(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)

    response = client.get(trace_url(machine_id, a, "upstream", 20))

    assert response.status_code == 200
    assert response.json()["events"] == []


def test_upstream_diamond_dedups_and_orders(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    by_id = {e["id"]: e for e in events}
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, a, c)
    create_link(client, machine_id, b, d)
    create_link(client, machine_id, c, d)

    response = client.get(trace_url(machine_id, d, "upstream", 20))

    items = response.json()["events"]
    depth_one = sorted([b, c], key=lambda eid: (by_id[eid]["created_at"], eid))
    assert [item["event_id"] for item in items] == depth_one + [a]
    assert {item["event_id"]: item["depth"] for item in items} == {
        b: 1,
        c: 1,
        a: 2,
    }


# --------------------------------------------------------------------------- #
# Cycles, machine boundaries, dangling targets
# --------------------------------------------------------------------------- #


def test_trace_terminates_on_a_cycle_through_start(client):
    # Link creation forbids cycles, so inject one directly: a -> b -> c -> a.
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{ch}") for ch in "abc"]
    a, b, c = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, a)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    assert response.status_code == 200
    assert response.json()["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": c, "depth": 2},
    ]


def test_trace_terminates_on_a_cycle_not_through_start(client):
    # a -> b -> c -> b forms a ring downstream of a.
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{ch}") for ch in "abc"]
    a, b, c = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, b)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    assert response.json()["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": c, "depth": 2},
    ]


def test_upstream_trace_terminates_on_a_cycle(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{ch}") for ch in "abc"]
    a, b, c = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, a)

    response = client.get(trace_url(machine_id, a, "upstream", 20))

    assert response.json()["events"] == [
        {"event_id": c, "depth": 1},
        {"event_id": b, "depth": 2},
    ]


def test_trace_does_not_cross_machine_boundary(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    local = record_event(client, machine_one, resource="res/local")["id"]
    foreign = record_event(client, machine_two, resource="res/foreign")["id"]
    foreign_child = record_event(client, machine_two, resource="res/foreign-child")["id"]
    # A link row owned by machine one pointing at another machine's event.
    insert_link_row(client, machine_one, local, foreign)
    create_link(client, machine_two, foreign, foreign_child)

    response = client.get(trace_url(machine_one, local, "downstream", 20))

    assert response.status_code == 200
    assert response.json()["events"] == []


def test_trace_ignores_links_owned_by_other_machines(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    a1 = record_event(client, machine_one, resource="res/a1")["id"]
    b1 = record_event(client, machine_one, resource="res/b1")["id"]
    record_event(client, machine_two, resource="res/a2")
    # A link row owned by machine two that happens to reference machine one's
    # events (allowed by the bare event foreign keys). It must never enter
    # machine one's graph.
    insert_link_row(client, machine_two, a1, b1)

    response = client.get(trace_url(machine_one, a1, "downstream", 20))
    assert response.json()["events"] == []

    # A legitimate, locally owned link is traced normally.
    create_link(client, machine_one, a1, b1)
    response = client.get(trace_url(machine_one, a1, "downstream", 20))
    assert response.json()["events"] == [{"event_id": b1, "depth": 1}]


def test_trace_excludes_dangling_link_targets(client):
    machine_id = create_machine(client)
    a = record_event(client, machine_id, resource="res/a")["id"]
    b = record_event(client, machine_id, resource="res/b")["id"]
    missing = "00000000-0000-0000-0000-000000000000"
    insert_link_row(client, machine_id, a, missing)
    create_link(client, machine_id, a, b)

    response = client.get(trace_url(machine_id, a, "downstream", 20))

    assert response.json()["events"] == [{"event_id": b, "depth": 1}]


# --------------------------------------------------------------------------- #
# Shape, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_trace_response_shape(client):
    machine_id = create_machine(client)
    a = record_event(client, machine_id, resource="res/a")["id"]
    b = record_event(client, machine_id, resource="res/b")["id"]
    create_link(client, machine_id, a, b)

    response = client.get(trace_url(machine_id, a, "upstream", 5))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"event_id", "direction", "max_depth", "events"}
    assert body == {
        "event_id": a,
        "direction": "upstream",
        "max_depth": 5,
        "events": [],
    }

    response = client.get(trace_url(machine_id, a, "downstream", 5))
    item = response.json()["events"][0]
    assert set(item.keys()) == {"event_id", "depth"}


def test_trace_is_read_only(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)

    events_url = f"/machines/{machine_id}/authorization-decision-events"
    integrity_url = f"{events_url}/integrity"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()
    before_links = {
        e["id"]: client.get(
            f"{events_url}/{e['id']}/causal-links"
        ).json()
        for e in events
    }

    for direction in ("downstream", "upstream"):
        for depth in (1, 2, 20):
            first = client.get(trace_url(machine_id, b, direction, depth)).json()
            second = client.get(trace_url(machine_id, b, direction, depth)).json()
            assert first == second

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    for event in events:
        after_links = client.get(f"{events_url}/{event['id']}/causal-links").json()
        assert after_links == before_links[event["id"]]
    assert before_integrity["valid"] is True


def test_trace_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        events = [
            record_event(first, machine_id, resource=f"res/{c}") for c in "abc"
        ]
        a, b, c = [e["id"] for e in events]
        create_link(first, machine_id, a, b)
        create_link(first, machine_id, b, c)

    with TestClient(app) as second:
        response = second.get(trace_url(machine_id, a, "downstream", 20))

    assert response.status_code == 200
    assert response.json()["events"] == [
        {"event_id": b, "depth": 1},
        {"event_id": c, "depth": 2},
    ]
