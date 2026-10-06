import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


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
    return response.json()


def create_link(client, machine_id, cause_event_id, effect_event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{cause_event_id}/causal-links",
        json={"effect_event_id": effect_event_id},
    )
    assert response.status_code == 201
    return response.json()


def insert_link_row(
    client, machine_id, cause_event_id, effect_event_id, link_id=None
):
    """Insert a causal link row directly, bypassing creation rules."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :machine_id, :cause, :effect, :created_at)"
            ),
            {
                "id": link_id or str(uuid.uuid4()),
                "machine_id": machine_id,
                "cause": cause_event_id,
                "effect": effect_event_id,
                "created_at": "2026-01-01T00:00:00Z",
            },
        )


def path_url(machine_id, event_id, target_event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/causal-path/{target_event_id}"
    )


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["?foo=bar", "?limit=1", "?direction=downstream"])
def test_any_query_param_returns_422_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(path_url(machine_id, event["id"], event["id"]) + query)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_query_param_returns_422_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        path_url(machine_id, event["id"], event["id"]) + "?foo=1&foo=2"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_returns_422_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.request(
        "GET", path_url(machine_id, event["id"], event["id"]), content=b"{}"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_takes_precedence_over_missing_machine(client):
    response = client.get(path_url(MISSING_ID, MISSING_ID, MISSING_ID) + "?foo=bar")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_takes_precedence_over_missing_machine(client):
    response = client.request(
        "GET", path_url(MISSING_ID, MISSING_ID, MISSING_ID), content=b"{}"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete", "head"])
def test_non_get_methods_return_405(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = getattr(client, method)(path_url(machine_id, event["id"], event["id"]))

    assert response.status_code == 405


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
    foreign = record_event(client, machine_two)
    target = record_event(client, machine_one)

    response = client.get(path_url(machine_one, foreign["id"], target["id"]))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_target_event_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    source = record_event(client, machine_one)
    foreign = record_event(client, machine_two)

    response = client.get(path_url(machine_one, source["id"], foreign["id"]))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Path finding
# --------------------------------------------------------------------------- #


def test_source_equal_target_returns_zero_depth_path(client):
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


def test_direct_link_returns_depth_one_path(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, cause["id"], effect["id"])

    response = client.get(path_url(machine_id, cause["id"], effect["id"]))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": cause["id"],
        "target_event_id": effect["id"],
        "found": True,
        "depth": 1,
        "path": [
            {"event_id": cause["id"], "causal_link_id": None},
            {"event_id": effect["id"], "causal_link_id": link["id"]},
        ],
    }


def test_multi_hop_path_records_each_entering_link(client):
    machine_id = create_machine(client)
    first = record_event(client, machine_id, resource="res/a")
    middle = record_event(client, machine_id, resource="res/b")
    last = record_event(client, machine_id, resource="res/c")
    link_one = create_link(client, machine_id, first["id"], middle["id"])
    link_two = create_link(client, machine_id, middle["id"], last["id"])

    response = client.get(path_url(machine_id, first["id"], last["id"]))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": first["id"],
        "target_event_id": last["id"],
        "found": True,
        "depth": 2,
        "path": [
            {"event_id": first["id"], "causal_link_id": None},
            {"event_id": middle["id"], "causal_link_id": link_one["id"]},
            {"event_id": last["id"], "causal_link_id": link_two["id"]},
        ],
    }


def test_response_contains_exactly_the_five_fixed_fields(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    create_link(client, machine_id, cause["id"], effect["id"])

    response = client.get(path_url(machine_id, cause["id"], effect["id"]))

    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "source_event_id",
        "target_event_id",
        "found",
        "depth",
        "path",
    ]
    for step in response.json()["path"]:
        assert set(step.keys()) == {"event_id", "causal_link_id"}


def test_no_directed_path_is_a_result_not_an_error(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/b")

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json() == {
        "source_event_id": source["id"],
        "target_event_id": target["id"],
        "found": False,
        "depth": None,
        "path": [],
    }


def test_links_are_followed_only_in_cause_to_effect_direction(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    create_link(client, machine_id, cause["id"], effect["id"])

    response = client.get(path_url(machine_id, effect["id"], cause["id"]))

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert response.json()["depth"] is None
    assert response.json()["path"] == []


def test_shortest_path_is_chosen(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    middle = record_event(client, machine_id, resource="res/b")
    target = record_event(client, machine_id, resource="res/c")
    create_link(client, machine_id, source["id"], middle["id"])
    create_link(client, machine_id, middle["id"], target["id"])
    direct = create_link(client, machine_id, source["id"], target["id"])

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["depth"] == 1
    assert response.json()["path"] == [
        {"event_id": source["id"], "causal_link_id": None},
        {"event_id": target["id"], "causal_link_id": direct["id"]},
    ]


def test_equal_depth_paths_choose_smallest_link_id_sequence(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    via_one = record_event(client, machine_id, resource="res/b")
    via_two = record_event(client, machine_id, resource="res/c")
    target = record_event(client, machine_id, resource="res/d")
    # Two depth-2 paths. The link-id sequences are compared from the first
    # hop on, so the path through via_one (first hop "111...") wins even
    # though the other path's second hop ("aaa...") is the smallest link id
    # in the graph.
    insert_link_row(
        client, machine_id, source["id"], via_one["id"],
        link_id="11111111-1111-1111-1111-111111111111",
    )
    insert_link_row(
        client, machine_id, via_one["id"], target["id"],
        link_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    )
    insert_link_row(
        client, machine_id, source["id"], via_two["id"],
        link_id="zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
    )
    insert_link_row(
        client, machine_id, via_two["id"], target["id"],
        link_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["depth"] == 2
    assert response.json()["path"] == [
        {"event_id": source["id"], "causal_link_id": None},
        {
            "event_id": via_one["id"],
            "causal_link_id": "11111111-1111-1111-1111-111111111111",
        },
        {
            "event_id": target["id"],
            "causal_link_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        },
    ]


def test_first_hop_link_id_decides_before_later_hops(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    via_one = record_event(client, machine_id, resource="res/b")
    via_two = record_event(client, machine_id, resource="res/c")
    target = record_event(client, machine_id, resource="res/d")
    # The path entering via "000..." wins from the first hop on, regardless
    # of the later hop's link id.
    insert_link_row(
        client, machine_id, source["id"], via_one["id"],
        link_id="00000000-0000-0000-0000-000000000000",
    )
    insert_link_row(
        client, machine_id, via_one["id"], target["id"],
        link_id="ffffffff-ffff-ffff-ffff-ffffffffffff",
    )
    insert_link_row(
        client, machine_id, source["id"], via_two["id"],
        link_id="00000000-0000-0000-0000-000000000001",
    )
    insert_link_row(
        client, machine_id, via_two["id"], target["id"],
        link_id="00000000-0000-0000-0000-000000000002",
    )

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["path"] == [
        {"event_id": source["id"], "causal_link_id": None},
        {
            "event_id": via_one["id"],
            "causal_link_id": "00000000-0000-0000-0000-000000000000",
        },
        {
            "event_id": target["id"],
            "causal_link_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
        },
    ]


def test_stored_cycle_still_terminates(client):
    machine_id = create_machine(client)
    first = record_event(client, machine_id, resource="res/a")
    second = record_event(client, machine_id, resource="res/b")
    # A ring created by direct row inserts (the create endpoint forbids it).
    insert_link_row(client, machine_id, first["id"], second["id"])
    insert_link_row(client, machine_id, second["id"], first["id"])

    response = client.get(path_url(machine_id, first["id"], second["id"]))

    assert response.status_code == 200
    assert response.json()["found"] is True
    assert response.json()["depth"] == 1


def test_disconnected_cycle_reports_not_found(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/b")
    ring_one = record_event(client, machine_id, resource="res/c")
    ring_two = record_event(client, machine_id, resource="res/d")
    insert_link_row(client, machine_id, ring_one["id"], ring_two["id"])
    insert_link_row(client, machine_id, ring_two["id"], ring_one["id"])

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert response.json()["path"] == []


def test_dangling_links_are_not_path_edges(client):
    machine_id = create_machine(client)
    source = record_event(client, machine_id, resource="res/a")
    target = record_event(client, machine_id, resource="res/b")
    ghost = MISSING_ID
    # A two-hop "path" through an event id that does not exist must not
    # connect source to target.
    insert_link_row(client, machine_id, source["id"], ghost)
    insert_link_row(client, machine_id, ghost, target["id"])

    response = client.get(path_url(machine_id, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert response.json()["path"] == []


def test_cross_machine_links_are_not_path_edges(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    source = record_event(client, machine_one, resource="res/a")
    target = record_event(client, machine_one, resource="res/b")
    # A link between the two events stored under the other machine's id.
    insert_link_row(client, machine_two, source["id"], target["id"])

    response = client.get(path_url(machine_one, source["id"], target["id"]))

    assert response.status_code == 200
    assert response.json()["found"] is False
    assert response.json()["path"] == []


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #


def test_repeated_reads_are_identical(client):
    machine_id = create_machine(client)
    first = record_event(client, machine_id, resource="res/a")
    middle = record_event(client, machine_id, resource="res/b")
    last = record_event(client, machine_id, resource="res/c")
    create_link(client, machine_id, first["id"], middle["id"])
    create_link(client, machine_id, middle["id"], last["id"])

    url = path_url(machine_id, first["id"], last["id"])
    first_response = client.get(url)
    second_response = client.get(url)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()


def test_path_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first_client:
        machine_id = create_machine(first_client)
        cause = record_event(first_client, machine_id, resource="res/a")
        effect = record_event(first_client, machine_id, resource="res/b")
        create_link(first_client, machine_id, cause["id"], effect["id"])
        expected = first_client.get(
            path_url(machine_id, cause["id"], effect["id"])
        ).json()

    with TestClient(app) as second_client:
        response = second_client.get(path_url(machine_id, cause["id"], effect["id"]))

    assert response.status_code == 200
    assert response.json() == expected
    assert response.json()["found"] is True
    assert response.json()["depth"] == 1
