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


MISSING_ID = "00000000-0000-0000-0000-000000000000"


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
    assert response.status_code == 201, response.text
    return response.json()["id"]


def path_url(machine_id, event_id, target_event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/causal-path/{target_event_id}"
    )


def insert_link_row(client, machine_id, cause_event_id, effect_event_id, link_id=None):
    """Insert a causal link row directly, bypassing creation rules.

    Returns the inserted link id so cycle-exit tests can identify an edge.
    """
    link_id = link_id or str(uuid.uuid4())
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :machine_id, :cause, :effect, :created_at)"
            ),
            {
                "id": link_id,
                "machine_id": machine_id,
                "cause": cause_event_id,
                "effect": effect_event_id,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )
    return link_id


def fixed_id(first, rest="0"):
    """A deterministic 36-char link id whose ordering is set by ``first``."""
    return first + rest * 35


# --------------------------------------------------------------------------- #
# Method routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    response = getattr(client, method)(path_url(machine_id, events[0]["id"], events[1]["id"]))
    assert response.status_code == 405


def test_method_routing_does_not_read_causal_records(client):
    # With the causal tables dropped, only routing is in play: non-GET must
    # still be 405 without touching a causal record.
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_causal_links"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert (
            getattr(client, method)(
                path_url(machine_id, events[0]["id"], events[1]["id"])
            ).status_code
            == 405
        )


# --------------------------------------------------------------------------- #
# Query-string / body validation (before any machine or event lookup)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    ["?x=1", "?limit=10", "?x=1&x=2", "?=", "?foo", "?event_id=x"],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    response = client.get(
        path_url(machine_id, events[0]["id"], events[1]["id"]) + query
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_repeated_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    response = client.get(
        path_url(machine_id, events[0]["id"], events[1]["id"]) + "?x=1&x=2"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    response = client.request(
        "GET",
        path_url(machine_id, events[0]["id"], events[1]["id"]),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_and_event_lookup(client):
    response = client.get(path_url(MISSING_ID, MISSING_ID, MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_machine_and_event_lookup(client):
    response = client.request(
        "GET",
        path_url(MISSING_ID, MISSING_ID, MISSING_ID),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_error_does_not_read_causal_records(client):
    # With the tables dropped, a read would 500; validation wins.
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_causal_links"))
    assert (
        client.get(path_url(machine_id, events[0]["id"], events[1]["id"]) + "?x=1").status_code
        == 422
    )


# --------------------------------------------------------------------------- #
# Existence / ownership
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    response = client.get(path_url(MISSING_ID, MISSING_ID, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_source_event_returns_404(client):
    machine_id = create_machine(client)
    target = record_event(client, machine_id)
    response = client.get(path_url(machine_id, MISSING_ID, target["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_target_event_returns_404(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id)
    response = client.get(path_url(machine_id, source["id"], MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_source_event_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_source = record_event(client, machine_two, resource="res/foreign-a")
    local_target = record_event(client, machine_one, resource="res/local-b")

    response = client.get(path_url(machine_one, foreign_source["id"], local_target["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_target_event_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    local_source = record_event(client, machine_one, resource="res/local-a")
    foreign_target = record_event(client, machine_two, resource="res/foreign-b")

    response = client.get(path_url(machine_one, local_source["id"], foreign_target["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_not_found_returns_no_partial_path(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id)
    response = client.get(path_url(machine_id, source["id"], MISSING_ID))
    assert response.json() == {"error": {"code": "not_found"}}
    assert "path" not in response.json()


# --------------------------------------------------------------------------- #
# Success shape: self path, direct link, chain, no path
# --------------------------------------------------------------------------- #


def test_source_equals_target_returns_zero_edge_path(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(path_url(machine_id, event["id"], event["id"]))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": event["id"],
        "target_event_id": event["id"],
        "found": True,
        "depth": 0,
        "path": [{"event_id": event["id"], "causal_link_id": None}],
    }


def test_direct_link_path(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, source["id"], target["id"])

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": source["id"],
        "target_event_id": target["id"],
        "found": True,
        "depth": 1,
        "path": [
            {"event_id": source["id"], "causal_link_id": None},
            {"event_id": target["id"], "causal_link_id": link},
        ],
    }


def test_path_follows_a_chain_and_reports_depth(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    link_ab = create_link(client, machine_id, a, b)
    link_bc = create_link(client, machine_id, b, c)
    link_cd = create_link(client, machine_id, c, d)

    response = client.get(path_url(machine_id, a, d))

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["depth"] == 3
    assert [node["event_id"] for node in body["path"]] == [a, b, c, d]
    assert [node["causal_link_id"] for node in body["path"]] == [
        None,
        link_ab,
        link_bc,
        link_cd,
    ]


def test_no_directed_path_is_found_false_not_an_error(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    # Edges point c -> b -> a: from a nothing downstream is reachable.
    create_link(client, machine_id, c, b)
    create_link(client, machine_id, b, a)

    response = client.get(path_url(machine_id, a, c))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": a,
        "target_event_id": c,
        "found": False,
        "depth": None,
        "path": [],
    }


def test_disconnected_events_have_no_path(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/z")

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["depth"] is None
    assert body["path"] == []


def test_success_response_has_exactly_five_fields(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/b")
    create_link(client, machine_id, source["id"], target["id"])

    body = client.get(path_url(machine_id, source["id"], target["id"])).json()
    assert set(body.keys()) == {
        "source_event_id",
        "target_event_id",
        "found",
        "depth",
        "path",
    }
    for node in body["path"]:
        assert set(node.keys()) == {"event_id", "causal_link_id"}


# --------------------------------------------------------------------------- #
# Shortest-path selection and equal-depth tie-breaking
# --------------------------------------------------------------------------- #


def test_shortest_path_is_chosen_over_longer(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abd"]
    a, b, d = [e["id"] for e in events]
    direct = create_link(client, machine_id, a, d)
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, d)

    response = client.get(path_url(machine_id, a, d))

    body = response.json()
    assert body["depth"] == 1
    assert body["path"] == [
        {"event_id": a, "causal_link_id": None},
        {"event_id": d, "causal_link_id": direct},
    ]


def test_equal_depth_paths_tie_break_on_first_link_id(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "axyt"]
    a, x, y, t = [e["id"] for e in events]
    # Via x: first edge sorts AFTER via y's first edge, even though its second
    # edge sorts earlier. The first edge must decide.
    ax = insert_link_row(client, machine_id, a, x, fixed_id("b"))
    insert_link_row(client, machine_id, x, t, fixed_id("a", "1"))
    ay = insert_link_row(client, machine_id, a, y, fixed_id("a"))
    yt = insert_link_row(client, machine_id, y, t, fixed_id("z"))

    body = client.get(path_url(machine_id, a, t)).json()

    assert body["depth"] == 2
    assert [node["event_id"] for node in body["path"]] == [a, y, t]
    assert [node["causal_link_id"] for node in body["path"]] == [None, ay, yt]


def test_equal_depth_paths_with_shared_prefix_tie_break_later(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "axpqt"]
    a, x, p, q, t = [e["id"] for e in events]
    # Both paths share a -> x; they diverge at x and reconverge at t, so the
    # first differing link is the second edge. Via p uses the larger second
    # edge; via q the smaller one — but q's tail edge is itself larger. The
    # second edge alone must decide, so via q wins.
    ax = insert_link_row(client, machine_id, a, x, fixed_id("m", "0"))
    xp = insert_link_row(client, machine_id, x, p, fixed_id("n"))
    insert_link_row(client, machine_id, p, t, fixed_id("a"))
    xq = insert_link_row(client, machine_id, x, q, fixed_id("m", "1"))
    qt = insert_link_row(client, machine_id, q, t, fixed_id("z"))

    body = client.get(path_url(machine_id, a, t)).json()

    assert body["depth"] == 3
    assert [node["event_id"] for node in body["path"]] == [a, x, q, t]
    assert [node["causal_link_id"] for node in body["path"]] == [None, ax, xq, qt]


def test_tie_break_result_is_deterministic_across_calls(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "axyt"]
    a, x, y, t = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, x, fixed_id("b"))
    insert_link_row(client, machine_id, x, t, fixed_id("a", "1"))
    insert_link_row(client, machine_id, a, y, fixed_id("a"))
    insert_link_row(client, machine_id, y, t, fixed_id("z"))

    url = path_url(machine_id, a, t)
    first = client.get(url).json()
    second = client.get(url).json()
    assert first == second


# --------------------------------------------------------------------------- #
# Cycles, dangling endpoints, machine boundaries
# --------------------------------------------------------------------------- #


def test_path_terminates_through_a_cycle_and_uses_exit(client):
    # Link creation forbids cycles, so inject one: a -> b -> c -> a, with an
    # exit c -> d.
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, a)
    exit_link = insert_link_row(client, machine_id, c, d)

    response = client.get(path_url(machine_id, a, d))

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["depth"] == 3
    assert [node["event_id"] for node in body["path"]] == [a, b, c, d]
    link_ids = [node["causal_link_id"] for node in body["path"]]
    assert link_ids[0] is None
    assert all(link_ids[1:-1])  # earlier edges are real stored link ids
    assert link_ids[-1] == exit_link


def test_path_search_terminates_on_cycle_when_target_unreachable(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcz"]
    a, b, c, z = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, a)

    response = client.get(path_url(machine_id, a, z))

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert response.json()["depth"] is None
    assert response.json()["path"] == []


def test_self_path_in_a_cyclic_graph_is_depth_zero(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abc"]
    a, b, c = [e["id"] for e in events]
    insert_link_row(client, machine_id, a, b)
    insert_link_row(client, machine_id, b, c)
    insert_link_row(client, machine_id, c, a)

    body = client.get(path_url(machine_id, a, a)).json()
    assert body["found"] is True
    assert body["depth"] == 0
    assert body["path"] == [{"event_id": a, "causal_link_id": None}]


def test_dangling_edge_is_never_a_path_edge(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    a, b = [e["id"] for e in events]
    missing = MISSING_ID
    insert_link_row(client, machine_id, a, missing)
    valid_link = create_link(client, machine_id, a, b)

    body = client.get(path_url(machine_id, a, b)).json()
    assert body["found"] is True
    assert body["path"][-1]["causal_link_id"] == valid_link

    # Nothing can reach the dangling target; the endpoint event itself does
    # not exist, so the request is a 404 rather than a found-false answer.
    response = client.get(path_url(machine_id, a, missing))
    assert response.status_code == 404


def test_dangling_middle_node_breaks_that_route_only(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "ab"]
    a, b = [e["id"] for e in events]
    missing = MISSING_ID
    # a -> missing and missing -> b cannot chain (missing is no event), while
    # the real a -> b edge still answers.
    insert_link_row(client, machine_id, a, missing)
    insert_link_row(client, machine_id, missing, b)
    valid_link = create_link(client, machine_id, a, b)

    body = client.get(path_url(machine_id, a, b)).json()
    assert body["depth"] == 1
    assert body["path"][-1]["causal_link_id"] == valid_link


def test_cross_machine_link_is_never_a_path_edge(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    local_a = record_event(client, machine_one, resource="res/a1")["id"]
    local_b = record_event(client, machine_one, resource="res/b1")["id"]
    foreign = record_event(client, machine_two, resource="res/foreign")["id"]
    # Link owned by machine one pointing at another machine's event.
    insert_link_row(client, machine_one, local_a, foreign)
    valid_link = create_link(client, machine_one, local_a, local_b)

    body = client.get(path_url(machine_one, local_a, local_b)).json()
    assert body["found"] is True
    assert body["path"][-1]["causal_link_id"] == valid_link


def test_links_owned_by_other_machines_are_ignored(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    a1 = record_event(client, machine_one, resource="res/a1")["id"]
    b1 = record_event(client, machine_one, resource="res/b1")["id"]
    # A link row owned by machine two referencing machine one's events must
    # never enter machine one's graph.
    insert_link_row(client, machine_two, a1, b1)

    body = client.get(path_url(machine_one, a1, b1)).json()
    assert body["found"] is False
    assert body["path"] == []

    # A locally owned edge is then used normally.
    create_link(client, machine_one, a1, b1)
    body = client.get(path_url(machine_one, a1, b1)).json()
    assert body["found"] is True
    assert body["depth"] == 1


# --------------------------------------------------------------------------- #
# Read-only behavior, determinism, persistence
# --------------------------------------------------------------------------- #


def test_causal_path_is_read_only(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{c}") for c in "abcd"]
    a, b, c, d = [e["id"] for e in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)
    create_link(client, machine_id, c, d)

    events_url = f"/machines/{machine_id}/authorization-decision-events"
    integrity_url = f"{events_url}/causal-links/integrity"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()
    before_links = {
        e["id"]: client.get(f"{events_url}/{e['id']}/causal-links").json()
        for e in events
    }

    for source, target in ((a, d), (a, a), (d, a), (b, d), (a, b)):
        first = client.get(path_url(machine_id, source, target)).json()
        second = client.get(path_url(machine_id, source, target)).json()
        assert first == second

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    for event in events:
        after_links = client.get(f"{events_url}/{event['id']}/causal-links").json()
        assert after_links == before_links[event["id"]]
    assert before_integrity["valid"] is True


def test_causal_path_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        events = [
            record_event(first, machine_id, resource=f"res/{ch}") for ch in "abc"
        ]
        a, b, c = [e["id"] for e in events]
        create_link(first, machine_id, a, b)
        create_link(first, machine_id, b, c)

    with TestClient(app) as second:
        response = second.get(path_url(machine_id, a, c))

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["depth"] == 2
    assert [node["event_id"] for node in body["path"]] == [a, b, c]
