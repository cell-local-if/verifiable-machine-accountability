"""Tests for the machine-level closed-loop accountability compliance export.

Covers `GET /machines/{machine_id}/accountability/compliance-export`: the
five accountability record groups (events, evidence, incidents, status
history, responsibility assignments) plus the causal links whose endpoints
are both exported events, closed-UTC-window filtering, (instant, id)
ordering with exact-second records before fractional records of the same
second, the `bad_time` / `invalid_query` / `not_found` outcomes, verbatim
export of records with missing or mis-owned references, strict read-only
byte stability, per-machine isolation, and persistence across restarts.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability import assignment_chain
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


def attach_evidence(client, machine_id, event_id, evidence_type="log"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/evidence",
        json={"evidence_type": evidence_type, "content_hash": uuid.uuid4().hex * 2},
    )
    assert response.status_code == 201
    return response.json()


def register_incident(client, machine_id, event_id, incident_type="alert",
                      summary="needs review"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def transition_incident(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/"
        f"incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def assign_responsibility(client, machine_id, event_id, incident_id,
                          party="team-a", role="operator"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/"
        f"incidents/{incident_id}/responsibility-assignments",
        json={"party": party, "role": role},
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
        f"/machines/{machine_id}/accountability/compliance-export"
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


def insert_evidence_row(client, machine_id, evidence_id, event_id, created_at,
                        *, evidence_type="log", content_hash=None):
    """Insert an evidence row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_evidence "
                "(id, machine_id, event_id, evidence_type, content_hash, "
                "created_at) "
                "VALUES (:id, :machine_id, :event_id, :evidence_type, "
                ":content_hash, :created_at)"
            ),
            {
                "id": evidence_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "evidence_type": evidence_type,
                "content_hash": content_hash or uuid.uuid4().hex * 2,
                "created_at": created_at,
            },
        )


def insert_incident_row(client, machine_id, incident_id, event_id, created_at,
                        *, status="open", incident_type=None, summary=None):
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
                # Unique per row: (event_id, incident_type, summary) is a
                # unique constraint.
                "incident_type": incident_type or f"alert-{incident_id}",
                "summary": summary or "needs review",
                "status": status,
                "created_at": created_at,
            },
        )


def insert_history_row(client, machine_id, history_id, event_id, incident_id,
                       created_at, *, from_status="open",
                       to_status="acknowledged"):
    """Insert a status-history row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events "
                "(id, machine_id, event_id, incident_id, from_status, "
                "to_status, created_at) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, "
                ":from_status, :to_status, :created_at)"
            ),
            {
                "id": history_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                "from_status": from_status,
                "to_status": to_status,
                "created_at": created_at,
            },
        )


def insert_assignment_row(client, machine_id, assignment_id, event_id,
                          incident_id, created_at, *, party=None,
                          role="operator"):
    """Insert a responsibility-assignment row directly (no chain columns)."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments "
                "(id, machine_id, event_id, incident_id, party, role, "
                "created_at) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, :party, "
                ":role, :created_at)"
            ),
            {
                "id": assignment_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                # Unique per row: (incident_id, party, role) is a unique
                # constraint.
                "party": party or f"team-{assignment_id}",
                "role": role,
                "created_at": created_at,
            },
        )
    # Fill chain columns exactly as startup backfill would for rows written
    # by an external writer.
    assignment_chain.backfill_chains(client.app.state.engine)


def insert_link_row(client, machine_id, link_id, cause_event_id,
                    effect_event_id, created_at):
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

# Wide window for records minted by the API at wall-clock "now".
FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


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
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/accountability/compliance-export{query}"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


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
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "",                               # blank
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, value, T4))

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time_and_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)

    inverted = client.get(export_url(machine_id, T4, T0))
    assert inverted.status_code == 422
    assert inverted.json() == {"error": {"code": "bad_time"}}

    equal = client.get(export_url(machine_id, T2, T2))
    assert equal.status_code == 200


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/accountability/compliance-export"
        f"?from_created_at={T0}&to_created_at={T4}&unexpected=1"
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_params_take_precedence_over_missing_machine(client):
    bad_time = client.get(export_url(MISSING_MACHINE, "2026-13-01T00:00:00Z", T4))
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(
        f"/machines/{MISSING_MACHINE}/accountability/compliance-export"
        f"?from_created_at={T0}&to_created_at={T4}&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Machine existence
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404_not_found(client):
    response = client.get(export_url(MISSING_MACHINE, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Response shape and closed-loop content
# --------------------------------------------------------------------------- #


def build_closed_loop(client, machine_id):
    """Create one event with evidence, a resolved incident with history and a
    responsibility assignment, plus a second event and a causal link."""
    event = record_event(client, machine_id, resource="res/a")
    evidence = attach_evidence(client, machine_id, event["id"])
    incident = register_incident(client, machine_id, event["id"])
    transition_incident(client, machine_id, event["id"], incident["id"],
                        "acknowledged")
    transition_incident(client, machine_id, event["id"], incident["id"],
                        "resolved")
    assignment = assign_responsibility(
        client, machine_id, event["id"], incident["id"]
    )
    other_event = record_event(client, machine_id, resource="res/b")
    link = create_link(client, machine_id, event["id"], other_event["id"])
    history = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event['id']}/"
        f"incidents/{incident['id']}/status-history"
    ).json()
    return {
        "event": event,
        "evidence": evidence,
        "incident": incident,
        "assignment": assignment,
        "other_event": other_event,
        "link": link,
        "history": history,
    }


def test_export_response_shape_echoes_params_and_empty_groups(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "events",
        "causal_links",
        "evidence",
        "incidents",
        "status_history",
        "assignments",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == FROM_WIDE
    assert body["to_created_at"] == TO_WIDE
    assert body["events"] == []
    assert body["causal_links"] == []
    assert body["evidence"] == []
    assert body["incidents"] == []
    assert body["status_history"] == []
    assert body["assignments"] == []


def test_export_contains_the_whole_closed_loop_as_stored(client):
    machine_id = create_machine(client)
    loop = build_closed_loop(client, machine_id)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()

    # Events keep the authorization result and the integrity-chain fields.
    assert [e["id"] for e in body["events"]] == [
        loop["event"]["id"],
        loop["other_event"]["id"],
    ]
    first = body["events"][0]
    assert first["allowed"] is False
    assert first["reason"] == "no_enabled_declaration"
    assert set(first.keys()) == {
        "id", "machine_id", "action_type", "resource", "allowed", "reason",
        "created_at", "previous_event_id", "content_hash", "chain_hash",
    }
    assert first["previous_event_id"] is None
    assert body["events"][1]["previous_event_id"] == first["id"]

    # Evidence keeps the original fingerprint.
    assert body["evidence"] == [loop["evidence"]]
    assert len(body["evidence"][0]["content_hash"]) == 64

    # The incident keeps its registration content and current lifecycle
    # status (resolved after the two transitions).
    assert len(body["incidents"]) == 1
    incident = body["incidents"][0]
    assert incident["id"] == loop["incident"]["id"]
    assert incident["incident_type"] == "alert"
    assert incident["summary"] == "needs review"
    assert incident["status"] == "resolved"

    # Status history keeps the from/to status of every transition.
    assert body["status_history"] == loop["history"]
    assert [(h["from_status"], h["to_status"]) for h in body["status_history"]] == [
        ("open", "acknowledged"),
        ("acknowledged", "resolved"),
    ]

    # The assignment keeps party, role, and the chain fields.
    assert body["assignments"] == [loop["assignment"]]
    assignment = body["assignments"][0]
    assert assignment["party"] == "team-a"
    assert assignment["role"] == "operator"
    assert assignment["previous_assignment_id"] is None
    assert len(assignment["content_hash"]) == 64
    assert len(assignment["chain_hash"]) == 64

    # The causal link between the two exported events is included.
    assert body["causal_links"] == [loop["link"]]


# --------------------------------------------------------------------------- #
# Windowing and ordering
# --------------------------------------------------------------------------- #


def test_window_is_closed_and_filters_every_group(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T1)
    insert_event_row(client, machine_id, eid(2), T2)
    insert_event_row(client, machine_id, eid(3), T4)
    insert_evidence_row(client, machine_id, eid(11), eid(1), T1)
    insert_evidence_row(client, machine_id, eid(12), eid(1), T3)
    insert_incident_row(client, machine_id, eid(21), eid(1), T2)
    insert_incident_row(client, machine_id, eid(22), eid(1), T4)
    insert_history_row(client, machine_id, eid(31), eid(1), eid(21), T2)
    insert_history_row(client, machine_id, eid(32), eid(1), eid(21), T4)
    insert_assignment_row(client, machine_id, eid(41), eid(1), eid(21), T1)
    insert_assignment_row(client, machine_id, eid(42), eid(1), eid(21), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["events"]] == [eid(1), eid(2)]
    assert [e["id"] for e in body["evidence"]] == [eid(11), eid(12)]
    assert [i["id"] for i in body["incidents"]] == [eid(21)]
    assert [h["id"] for h in body["status_history"]] == [eid(31)]
    assert [a["id"] for a in body["assignments"]] == [eid(41), eid(42)]


def test_empty_window_returns_empty_groups(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, eid(1), T0)
    insert_evidence_row(client, machine_id, eid(11), eid(1), T0)
    insert_incident_row(client, machine_id, eid(21), eid(1), T0)
    insert_history_row(client, machine_id, eid(31), eid(1), eid(21), T0)
    insert_assignment_row(client, machine_id, eid(41), eid(1), eid(21), T0)

    response = client.get(export_url(machine_id, T1, T4))

    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["causal_links"] == []
    assert body["evidence"] == []
    assert body["incidents"] == []
    assert body["status_history"] == []
    assert body["assignments"] == []


def test_exact_second_records_sort_before_fractional_records(client):
    machine_id = create_machine(client)
    # The exact-second stamp T1 sorts before the fractional stamp of the same
    # second even though "...:01.5Z" < "...:01Z" would be false as text and
    # true only when parsed as instants.
    insert_event_row(client, machine_id, eid(2), "2026-03-01T00:00:01.500000Z")
    insert_event_row(client, machine_id, eid(1), T1)
    insert_evidence_row(
        client, machine_id, eid(12), eid(1), "2026-03-01T00:00:01.500000Z"
    )
    insert_evidence_row(client, machine_id, eid(11), eid(1), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["events"]] == [eid(1), eid(2)]
    assert [e["id"] for e in body["evidence"]] == [eid(11), eid(12)]


def test_groups_ordered_by_instant_then_id(client):
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


# --------------------------------------------------------------------------- #
# Causal links require both endpoints in the exported event set
# --------------------------------------------------------------------------- #


def test_causal_links_require_both_endpoints_in_event_set(client):
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


# --------------------------------------------------------------------------- #
# Missing or mis-owned references are exported exactly as stored
# --------------------------------------------------------------------------- #


def test_records_with_missing_references_are_exported_unmodified(client):
    machine_id = create_machine(client)
    dangling_event_id = str(uuid.uuid4())
    dangling_incident_id = str(uuid.uuid4())
    insert_evidence_row(client, machine_id, eid(11), dangling_event_id, T1)
    insert_incident_row(client, machine_id, eid(21), dangling_event_id, T1)
    insert_history_row(
        client, machine_id, eid(31), dangling_event_id, dangling_incident_id, T1
    )
    insert_assignment_row(
        client, machine_id, eid(41), dangling_event_id, dangling_incident_id, T1
    )

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["evidence"]] == [eid(11)]
    assert body["evidence"][0]["event_id"] == dangling_event_id
    assert [i["id"] for i in body["incidents"]] == [eid(21)]
    assert body["incidents"][0]["event_id"] == dangling_event_id
    assert [h["id"] for h in body["status_history"]] == [eid(31)]
    assert body["status_history"][0]["incident_id"] == dangling_incident_id
    assert [a["id"] for a in body["assignments"]] == [eid(41)]
    assert body["assignments"][0]["incident_id"] == dangling_incident_id


def test_records_with_foreign_owned_references_are_exported_unmodified(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_event = record_event(client, machine_two)
    # Rows owned by machine one that reference machine two's event: exported
    # verbatim under machine one, never rewritten or filtered out.
    insert_evidence_row(client, machine_one, eid(11), foreign_event["id"], T1)
    insert_incident_row(client, machine_one, eid(21), foreign_event["id"], T1)

    response = client.get(export_url(machine_one, T0, T4))

    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert [e["id"] for e in body["evidence"]] == [eid(11)]
    assert body["evidence"][0]["event_id"] == foreign_event["id"]
    assert [i["id"] for i in body["incidents"]] == [eid(21)]
    assert body["incidents"][0]["event_id"] == foreign_event["id"]


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_data(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    loop_two = build_closed_loop(client, machine_two)
    event_one = record_event(client, machine_one, resource="res/one")

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["events"]] == [event_one["id"]]
    assert body["causal_links"] == []
    assert body["evidence"] == []
    assert body["incidents"] == []
    assert body["status_history"] == []
    assert body["assignments"] == []

    response_two = client.get(export_url(machine_two, FROM_WIDE, TO_WIDE))
    body_two = response_two.json()
    assert len(body_two["events"]) == 2
    assert len(body_two["evidence"]) == 1
    assert len(body_two["incidents"]) == 1
    assert len(body_two["status_history"]) == 2
    assert len(body_two["assignments"]) == 1
    assert body_two["causal_links"] == [loop_two["link"]]


# --------------------------------------------------------------------------- #
# Read-only, byte stability, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    build_closed_loop(client, machine_id)

    events_url = f"/machines/{machine_id}/authorization-decision-events"
    integrity_url = f"{events_url}/integrity"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()

    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))
    assert first.status_code == second.status_code == 200
    # Identical data and window give byte-identical output.
    assert first.content == second.content

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        build_closed_loop(first, machine_id)
        expected = first.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json() == expected
    body = response.json()
    assert len(body["events"]) == 2
    assert len(body["evidence"]) == 1
    assert len(body["incidents"]) == 1
    assert len(body["status_history"]) == 2
    assert len(body["assignments"]) == 1
    assert len(body["causal_links"]) == 1
