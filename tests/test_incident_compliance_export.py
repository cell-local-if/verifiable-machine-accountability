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


def register_incident(client, machine_id, event_id, incident_type="fraud",
                      summary="suspicious decision"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def export_url(machine_id, from_created_at, to_created_at):
    return (
        f"/machines/{machine_id}/authorization-decision-events/incidents/"
        f"compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


def insert_event_row(client, machine_id, event_id, created_at):
    """Insert an event row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at) "
                "VALUES (:id, :machine_id, 'read', 'res/x', 1, "
                "'allowed_by_policy', :created_at)"
            ),
            {"id": event_id, "machine_id": machine_id, "created_at": created_at},
        )
    backfill_chains(client.app.state.engine)


def insert_incident_row(
    client,
    machine_id,
    incident_id,
    event_id,
    created_at,
    *,
    incident_type="fraud",
    summary=None,
    status="open",
):
    """Insert an incident row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_incidents "
                "(id, machine_id, event_id, incident_type, summary, status, "
                "created_at) "
                "VALUES (:id, :machine_id, :event_id, :incident_type, "
                ":summary, :status, :created_at)"
            ),
            {
                "id": incident_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_type": incident_type,
                # Default to a per-row summary so several direct inserts on
                # one event never trip the (event, type, summary) uniqueness.
                "summary": summary if summary is not None else f"summary-{incident_id}",
                "status": status,
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

# Wide window for incidents minted by the API at wall-clock "now".
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
        f"/machines/{machine_id}/authorization-decision-events/incidents/"
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
    incident = register_incident(client, machine_id, event["id"])

    response = client.get(
        export_url(machine_id, incident["created_at"], incident["created_at"])
    )

    assert response.status_code == 200
    assert [i["id"] for i in response.json()["incidents"]] == [incident["id"]]


def test_invalid_params_take_precedence_over_missing_machine(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(export_url(missing_machine, "2026-13-01T00:00:00Z", T4))
    assert response.status_code == 422

    response = client.get(
        f"/machines/{missing_machine}/authorization-decision-events/incidents/"
        "compliance-export"
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
# Response shape and windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    register_incident(client, machine_id, event["id"])

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "incidents",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == FROM_WIDE
    assert body["to_created_at"] == TO_WIDE
    assert len(body["incidents"]) == 1


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_incident_row(client, machine_id, eid(11), eid(1), T0)
    insert_incident_row(client, machine_id, eid(12), eid(1), T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    assert response.json()["incidents"] == []


def test_machine_with_no_incidents_returns_empty_array(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["incidents"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_incident_row(client, machine_id, eid(11), eid(1), T1)
    insert_incident_row(client, machine_id, eid(12), eid(1), T2)
    insert_incident_row(client, machine_id, eid(13), eid(1), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [i["id"] for i in response.json()["incidents"]] == [
        eid(11),
        eid(12),
        eid(13),
    ]


def test_window_excludes_incidents_outside_bounds(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_incident_row(client, machine_id, eid(10), eid(1), T0)
    insert_incident_row(client, machine_id, eid(11), eid(1), T1)
    insert_incident_row(client, machine_id, eid(12), eid(1), T2)
    insert_incident_row(client, machine_id, eid(13), eid(1), T3)
    insert_incident_row(client, machine_id, eid(14), eid(1), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [i["id"] for i in response.json()["incidents"]] == [
        eid(11),
        eid(12),
        eid(13),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp sorts *after* "...:00Z" lexicographically only
    # by accident; the implementation must compare parsed instants so the
    # record falls inside [T0, T1].
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_incident_row(
        client, machine_id, eid(11), eid(1), "2026-03-01T00:00:00.500000Z"
    )
    insert_incident_row(client, machine_id, eid(12), eid(1), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [i["id"] for i in response.json()["incidents"]] == [eid(11), eid(12)]


def test_incidents_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_incident_row(client, machine_id, eid(30), eid(1), T3)
    insert_incident_row(client, machine_id, eid(21), eid(1), T2)
    insert_incident_row(client, machine_id, eid(20), eid(1), T2)
    insert_incident_row(client, machine_id, eid(10), eid(1), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [i["id"] for i in response.json()["incidents"]] == [
        eid(10),
        eid(20),
        eid(21),
        eid(30),
    ]


def test_export_items_match_list_endpoint_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    register_incident(client, machine_id, event["id"], incident_type="fraud")
    register_incident(client, machine_id, event["id"], incident_type="misuse")
    listed = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents"
    ).json()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["incidents"] == listed
    assert set(response.json()["incidents"][0].keys()) == set(listed[0].keys())


# --------------------------------------------------------------------------- #
# Broken or foreign references are exported exactly as stored
# --------------------------------------------------------------------------- #


def test_dangling_event_id_is_exported_unmodified(client):
    machine_id = create_machine(client)
    dangling_event_id = str(uuid.uuid4())
    insert_incident_row(
        client, machine_id, eid(11), dangling_event_id, T1,
        incident_type="fraud",
    )

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    incidents = response.json()["incidents"]
    assert [i["id"] for i in incidents] == [eid(11)]
    assert incidents[0]["event_id"] == dangling_event_id
    assert incidents[0]["machine_id"] == machine_id


def test_foreign_owned_event_reference_is_exported_unmodified(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_event = record_event(client, machine_two)
    insert_incident_row(client, machine_one, eid(11), foreign_event["id"], T1)

    response = client.get(export_url(machine_one, T0, T4))

    assert response.status_code == 200
    incidents = response.json()["incidents"]
    assert [i["id"] for i in incidents] == [eid(11)]
    assert incidents[0]["event_id"] == foreign_event["id"]


def test_broken_status_history_is_exported_unmodified(client):
    # An incident whose status/history pair is corrupt (status moved without
    # any history record) is still exported exactly as stored.
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_incident_row(
        client, machine_id, eid(11), eid(1), T1, status="resolved"
    )

    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/"
        "incidents/integrity"
    ).json()
    assert integrity["valid"] is False

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    incidents = response.json()["incidents"]
    assert [i["id"] for i in incidents] == [eid(11)]
    assert incidents[0]["status"] == "resolved"


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_incidents(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    own = register_incident(client, machine_one, event_one["id"])
    register_incident(client, machine_two, event_two["id"])

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    incidents = response.json()["incidents"]
    assert [i["id"] for i in incidents] == [own["id"]]
    assert all(i["machine_id"] == machine_one for i in incidents)


# --------------------------------------------------------------------------- #
# Read-only, determinism, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_deterministic(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id)
    register_incident(client, machine_id, event["id"], incident_type="fraud")
    register_incident(client, machine_id, event["id"], incident_type="misuse")
    other_event = record_event(client, other_machine)
    register_incident(client, other_machine, other_event["id"])

    incidents_url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event['id']}/incidents"
    )
    integrity_url = (
        f"/machines/{machine_id}/authorization-decision-events/incidents/integrity"
    )
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    before_incidents = client.get(incidents_url).json()
    before_integrity = client.get(integrity_url).json()
    before_events = client.get(events_url).json()

    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    assert first == second
    assert len(first["incidents"]) == 2

    assert client.get(incidents_url).json() == before_incidents
    assert client.get(integrity_url).json() == before_integrity
    assert client.get(events_url).json() == before_events
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = record_event(first, machine_id)
        register_incident(first, machine_id, event["id"], incident_type="fraud")
        register_incident(first, machine_id, event["id"], incident_type="misuse")
        expected = first.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json() == expected
    assert len(response.json()["incidents"]) == 2
