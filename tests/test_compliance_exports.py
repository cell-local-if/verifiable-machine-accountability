import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability.chain import backfill_chains


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


def export_url(machine_id, from_created_at, to_created_at):
    return (
        f"/machines/{machine_id}/authorization-decision-events/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


def insert_event_row(
    client,
    machine_id,
    event_id,
    created_at,
    *,
    action_type="read",
    resource="res/x",
    allowed=1,
    reason="allowed_by_policy",
):
    """Insert an event row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, created_at) "
                "VALUES (:id, :machine_id, :action_type, :resource, :allowed, "
                ":reason, :created_at)"
            ),
            {
                "id": event_id,
                "machine_id": machine_id,
                "action_type": action_type,
                "resource": resource,
                "allowed": allowed,
                "reason": reason,
                "created_at": created_at,
            },
        )
    # Normalize chain columns exactly as startup backfill would for rows
    # written by an external writer.
    backfill_chains(client.app.state.engine)


def insert_link_row(client, machine_id, link_id, cause_event_id, effect_event_id,
                    created_at):
    """Insert a causal link row directly with a fixed id and timestamp."""
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
                "created_at": created_at,
            },
        )


def eid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"

# Wide window for events minted by the API at wall-clock "now".
FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_params_return_422(client, query):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"compliance-export{query}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form instead of Z
        "2026-03-01T00:00:00z",           # lowercase suffix
        "2026-03-01 00:00:00Z",           # space separator
        "2026-03-01T00:00:00.Z",          # dot without fraction digits
        "not-a-time",
        "2026-13-01T00:00:00Z",           # invalid month
        "2026-02-30T00:00:00Z",           # invalid calendar day
        "2026-03-01T24:00:00Z",           # invalid hour
        "2026-03-01T00:60:00Z",           # invalid minute
        "2026-03-01T00:00:60Z",           # invalid second
        " 2026-03-01T00:00:00Z",          # surrounding whitespace
    ],
)
def test_invalid_from_created_at_returns_422(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, value, T4))

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "garbage",
        "2026-03-01T00:00:00.123",
    ],
)
def test_invalid_to_created_at_returns_422(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, value))

    assert response.status_code == 422


def test_inverted_range_returns_422(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T4, T0))

    assert response.status_code == 422


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(
        export_url(machine_id, event["created_at"], event["created_at"])
    )

    assert response.status_code == 200
    assert [e["id"] for e in response.json()["events"]] == [event["id"]]


def test_invalid_params_take_precedence_over_missing_machine(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(export_url(missing_machine, "2026-13-01T00:00:00Z", T4))
    assert response.status_code == 422

    response = client.get(
        f"/machines/{missing_machine}/authorization-decision-events/compliance-export"
    )
    assert response.status_code == 422

    response = client.get(export_url(missing_machine, T4, T0))
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Machine existence
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(export_url(missing_machine, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Response shape and event windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "events",
        "causal_links",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == FROM_WIDE
    assert body["to_created_at"] == TO_WIDE
    assert len(body["events"]) == 1


def test_empty_window_returns_empty_arrays(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_event_row(client, machine_id, eid(2), T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["causal_links"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)
    insert_event_row(client, machine_id, eid(2), T2)
    insert_event_row(client, machine_id, eid(3), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [event["id"] for event in response.json()["events"]] == [
        eid(1),
        eid(2),
        eid(3),
    ]


def test_window_excludes_events_outside_bounds(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(0), T0)
    insert_event_row(client, machine_id, eid(1), T1)
    insert_event_row(client, machine_id, eid(2), T2)
    insert_event_row(client, machine_id, eid(3), T3)
    insert_event_row(client, machine_id, eid(4), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [event["id"] for event in response.json()["events"]] == [
        eid(1),
        eid(2),
        eid(3),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp sorts *after* "...:00Z" lexicographically only
    # by accident; the implementation must compare parsed instants so the
    # event falls inside [T0, T1].
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), "2026-03-01T00:00:00.500000Z")
    insert_event_row(client, machine_id, eid(2), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [event["id"] for event in response.json()["events"]] == [eid(1), eid(2)]


def test_events_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_event_row(client, machine_id, eid(30), T3)
    insert_event_row(client, machine_id, eid(21), T2)
    insert_event_row(client, machine_id, eid(20), T2)
    insert_event_row(client, machine_id, eid(10), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [event["id"] for event in response.json()["events"]] == [
        eid(10),
        eid(20),
        eid(21),
        eid(30),
    ]


def test_export_event_items_match_list_endpoint_items(client):
    machine_id = create_machine(client)
    record_event(client, machine_id, resource="res/a")
    record_event(client, machine_id, resource="res/b")
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["events"] == listed
    assert set(response.json()["events"][0].keys()) == set(listed[0].keys())


# --------------------------------------------------------------------------- #
# Causal links
# --------------------------------------------------------------------------- #


def test_causal_links_require_both_endpoints_in_window(client):
    machine_id = create_machine(client)
    for i, ts in enumerate((T0, T1, T2, T3, T4)):
        insert_event_row(client, machine_id, eid(i), ts)
    create_link(client, machine_id, eid(1), eid(2))  # both inside
    create_link(client, machine_id, eid(2), eid(3))  # both inside (boundary)
    create_link(client, machine_id, eid(0), eid(1))  # cause outside
    create_link(client, machine_id, eid(2), eid(4))  # effect outside
    create_link(client, machine_id, eid(0), eid(4))  # both outside

    response = client.get(export_url(machine_id, T1, T3))

    assert response.status_code == 200
    pairs = {
        (link["cause_event_id"], link["effect_event_id"])
        for link in response.json()["causal_links"]
    }
    assert pairs == {(eid(1), eid(2)), (eid(2), eid(3))}


def test_causal_links_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    for i, ts in enumerate((T1, T2, T3)):
        insert_event_row(client, machine_id, eid(i + 10), ts)
    link_t2_b = "11111111-0000-0000-0000-000000000002"
    link_t2_a = "11111111-0000-0000-0000-000000000001"
    link_t1 = "11111111-0000-0000-0000-000000000000"
    insert_link_row(client, machine_id, link_t2_b, eid(12), eid(11), T2)
    insert_link_row(client, machine_id, link_t2_a, eid(11), eid(12), T2)
    insert_link_row(client, machine_id, link_t1, eid(10), eid(11), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [link["id"] for link in response.json()["causal_links"]] == [
        link_t1,
        link_t2_a,
        link_t2_b,
    ]


def test_causal_link_items_match_list_endpoint_items(client):
    machine_id = create_machine(client)
    a = record_event(client, machine_id, resource="res/a")
    b = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, a["id"], b["id"])

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    links = response.json()["causal_links"]
    assert links == [link]
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{a['id']}"
        "/causal-links"
    ).json()
    assert links == listed


def test_links_absent_when_window_has_events_but_none_connected(client):
    machine_id = create_machine(client)
    for i, ts in enumerate((T0, T1, T2)):
        insert_event_row(client, machine_id, eid(i), ts)
    # Link exists, but neither endpoint is in the export window.
    create_link(client, machine_id, eid(0), eid(1))

    response = client.get(export_url(machine_id, T2, T4))

    assert [event["id"] for event in response.json()["events"]] == [eid(2)]
    assert response.json()["causal_links"] == []


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_data(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    for i, ts in enumerate((T1, T2, T3)):
        insert_event_row(client, machine_one, eid(100 + i), ts, resource="res/one")
        insert_event_row(client, machine_two, eid(200 + i), ts, resource="res/two")
    # Legitimate link inside machine two ...
    create_link(client, machine_two, eid(200), eid(201))
    # ... and a link row owned by machine one that references machine two's
    # events. It must never appear in machine one's export.
    insert_link_row(
        client, machine_one, str(uuid.uuid4()), eid(200), eid(201), T2
    )
    # A link owned by machine two referencing machine one's events likewise
    # never enters machine one's export.
    insert_link_row(
        client, machine_two, str(uuid.uuid4()), eid(100), eid(101), T2
    )

    response = client.get(export_url(machine_one, T0, T4))
    assert response.status_code == 200
    body = response.json()
    assert [event["id"] for event in body["events"]] == [
        eid(100),
        eid(101),
        eid(102),
    ]
    assert all(event["machine_id"] == machine_one for event in body["events"])
    assert body["causal_links"] == []

    response_two = client.get(export_url(machine_two, T0, T4))
    assert [event["id"] for event in response_two.json()["events"]] == [
        eid(200),
        eid(201),
        eid(202),
    ]
    assert len(response_two.json()["causal_links"]) == 1
    assert response_two.json()["causal_links"][0]["machine_id"] == machine_two


# --------------------------------------------------------------------------- #
# Read-only, determinism, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_deterministic(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    events = [
        record_event(client, machine_id, resource=f"res/{c}") for c in "abc"
    ]
    a, b, c = [event["id"] for event in events]
    create_link(client, machine_id, a, b)
    create_link(client, machine_id, b, c)
    record_event(client, other_machine, resource="res/other")

    events_url = f"/machines/{machine_id}/authorization-decision-events"
    integrity_url = f"{events_url}/integrity"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()
    before_links = {
        event["id"]: client.get(f"{events_url}/{event['id']}/causal-links").json()
        for event in events
    }
    before_other = client.get(
        f"/machines/{other_machine}/authorization-decision-events"
    ).json()

    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    assert first == second
    assert [event["id"] for event in first["events"]] == [a, b, c]

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    for event in events:
        after = client.get(f"{events_url}/{event['id']}/causal-links").json()
        assert after == before_links[event["id"]]
    assert (
        client.get(
            f"/machines/{other_machine}/authorization-decision-events"
        ).json()
        == before_other
    )
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        response = first.post(
            "/machines",
            json={
                "external_id": "machine-2",
                "display_name": "Machine Two",
                "public_key": "key-2",
            },
        )
        other_machine = response.json()["id"]
        events = [
            record_event(first, machine_id, resource=f"res/{ch}") for ch in "abc"
        ]
        a, b, c = [event["id"] for event in events]
        create_link(first, machine_id, a, b)
        create_link(first, machine_id, b, c)
        record_event(first, other_machine, resource="res/other")
        expected = first.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json() == expected
    body = response.json()
    assert [event["id"] for event in body["events"]] == [a, b, c]
    assert {
        (link["cause_event_id"], link["effect_event_id"])
        for link in body["causal_links"]
    } == {(a, b), (b, c)}
