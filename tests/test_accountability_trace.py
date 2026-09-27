"""Tests for the read-only single-event closed-loop accountability trace.

Covers `GET /machines/{machine_id}/authorization-decision-events/{event_id}/
accountability-trace`: GET-only 405 (including HEAD) without reading records,
``invalid_query`` 422 for any query parameter, a repeated parameter, or a
carried body before the machine/event lookup, ``not_found`` 404 for a missing
machine/event and for an event owned by another machine with no closed-loop
data, ``internal_error`` 500 on a real read fault with no event summary or
association arrays, the fixed six-group response (the event summary object
plus the evidence, incidents, status-history, responsibility-assignment, and
causal-link arrays), per-machine and per-event selection, causal membership on
either endpoint with the other endpoint preserved verbatim even when dangling,
child records kept when a parent is damaged, complete stored fields emitted
without repair, tolerant (instant, id) ordering with damaged stamps last,
empty-group preservation, machine isolation, compact single-newline JSON,
read-only byte-identical repeats, and persistence across restarts.
"""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T0_HALF = "2026-03-01T00:00:00.5Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
GHOST_EVENT_ID = "99999999-9999-9999-9999-999999999999"
GHOST_INCIDENT_ID = "deadbeef-dead-dead-dead-deaddeaddead"

GROUP_ORDER = (
    "event_summary",
    "evidence",
    "incidents",
    "status_history",
    "responsibility_assignments",
    "causal_links",
)


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


def declare(client, machine_id, action_type="read", resource_pattern="res/*"):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": True,
        },
    )
    assert response.status_code == 201


def create_rule(client, action_type="read", resource_pattern="res/*", priority=0):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": "allow",
            "priority": priority,
        },
    )
    assert response.status_code == 201


def record_event(client, machine_id, resource="res/a", allow=False):
    if allow:
        declare(client, machine_id, resource_pattern="res/*")
        create_rule(client, resource_pattern="res/*")
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201, response.text
    return response.json()


def trace_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/accountability-trace"
    )


def trace(client, machine_id, event_id):
    response = client.get(trace_url(machine_id, event_id))
    assert response.status_code == 200, response.text
    return response


def add_evidence(client, machine_id, event_id, content_hash="a" * 64,
                 evidence_type="log"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/evidence",
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )
    assert response.status_code == 201, response.text
    return response.json()


def add_incident(client, machine_id, event_id, incident_type="fault",
                 summary="summary"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201, response.text
    return response.json()


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/"
        f"incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200, response.text
    # The transition responds with the updated incident; the new history row
    # is read back from the status-history endpoint.
    history = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/"
        f"incidents/{incident_id}/status-history"
    )
    assert history.status_code == 200
    return history.json()[-1]


def assign(client, machine_id, event_id, incident_id, party="ops", role="owner"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/"
        f"incidents/{incident_id}/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201, response.text
    return response.json()


def add_link(client, machine_id, cause_event_id, effect_event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{cause_event_id}/causal-links",
        json={"effect_event_id": effect_event_id},
    )
    assert response.status_code == 201, response.text
    return response.json()


def execute_sql(client, statement, parameters=None):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement), parameters or {})


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = getattr(client, method)(trace_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"event_summary" not in response.content


def test_method_routing_does_not_read_records(client):
    # With every table the trace could read dropped, only routing is in play.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        for table in (
            "authorization_decision_causal_links",
            "incident_responsibility_assignments",
            "incident_status_events",
            "authorization_decision_incidents",
            "authorization_decision_evidence",
            "authorization_decision_events",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(trace_url(machine_id, event["id"])).status_code == 405


@pytest.mark.parametrize(
    "query",
    ["?x=1", "?limit=10", "?x=1&x=2", "?=", "?foo"],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_repeated_parameter_is_invalid_query_even_with_a_value(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + "?x=1&x=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.request(
        "GET",
        trace_url(machine_id, event["id"]),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    response = client.get(trace_url(MISSING_ID, MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_machine_lookup(client):
    response = client.request(
        "GET",
        trace_url(MISSING_ID, MISSING_ID),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_and_method_errors_do_not_read_records(client):
    # With the tables dropped, a read would 500; validation and routing win.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_evidence"))
        conn.execute(text("DROP TABLE authorization_decision_events"))
    assert client.get(trace_url(machine_id, event["id"]) + "?x=1").status_code == 422
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(trace_url(machine_id, event["id"])).status_code == 405


def test_missing_machine_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(MISSING_ID, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content
    assert b"causal_links" not in response.content


def test_missing_event_is_404_without_trace_data(client):
    machine_id = create_machine(client)
    response = client.get(trace_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content
    assert b"evidence" not in response.content


def test_event_owned_by_another_machine_is_404(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    event = record_event(client, machine_id)
    response = client.get(trace_url(other, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content


def test_event_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_events"))
    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"evidence" not in response.content
    assert b"causal_links" not in response.content


def test_associated_read_failure_is_500_with_no_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_evidence"))
    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"incidents" not in response.content


# --------------------------------------------------------------------------- #
# Success shape and the six fixed groups
# --------------------------------------------------------------------------- #


def test_event_with_no_associations_has_summary_and_five_empty_arrays(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = trace(client, machine_id, event["id"])
    body = response.json()
    assert list(body.keys()) == list(GROUP_ORDER)
    assert isinstance(body["event_summary"], dict)
    for group in GROUP_ORDER[1:]:
        assert body[group] == []


def test_event_summary_carries_result_reason_moment_and_chain_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, allow=True)

    body = trace(client, machine_id, event["id"]).json()
    summary = body["event_summary"]
    assert list(summary.keys()) == [
        "allowed",
        "reason",
        "created_at",
        "previous_event_id",
        "content_hash",
        "chain_hash",
    ]
    assert summary["allowed"] is event["allowed"]
    assert summary["reason"] == event["reason"]
    assert summary["created_at"] == event["created_at"]
    assert summary["previous_event_id"] == event["previous_event_id"]
    assert summary["content_hash"] == event["content_hash"]
    assert summary["chain_hash"] == event["chain_hash"]
    # Identifying/context fields are intentionally not part of the summary.
    assert "id" not in summary
    assert "machine_id" not in summary
    assert "action_type" not in summary
    assert "resource" not in summary


def test_denied_event_summary_preserves_deny_result(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    assert event["allowed"] is False

    summary = trace(client, machine_id, event["id"]).json()["event_summary"]
    assert summary["allowed"] is False
    assert summary["reason"] == "no_enabled_declaration"


# --------------------------------------------------------------------------- #
# Group membership, machine isolation, parent anomalies
# --------------------------------------------------------------------------- #


def test_all_five_association_groups_are_collected_for_the_event(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    downstream = record_event(client, machine_id, resource="res/b")
    upstream = record_event(client, machine_id, resource="res/c")

    evidence = add_evidence(client, machine_id, event["id"])
    incident = add_incident(client, machine_id, event["id"])
    history = transition(client, machine_id, event["id"], incident["id"], "acknowledged")
    assignment = assign(client, machine_id, event["id"], incident["id"])
    outgoing = add_link(client, machine_id, event["id"], downstream["id"])
    incoming = add_link(client, machine_id, upstream["id"], event["id"])

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["evidence"]] == [evidence["id"]]
    assert [item["id"] for item in body["incidents"]] == [incident["id"]]
    assert [item["id"] for item in body["status_history"]] == [history["id"]]
    assert [item["id"] for item in body["responsibility_assignments"]] == [
        assignment["id"]
    ]
    link_ids = {item["id"] for item in body["causal_links"]}
    assert link_ids == {outgoing["id"], incoming["id"]}


def test_evidence_records_emit_complete_fields_and_chain_columns(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    evidence = add_evidence(client, machine_id, event["id"])

    # The per-record content digest is stored on the row but is not part of
    # the registration response, so read it straight from storage to compare.
    with client.app.state.engine.connect() as conn:
        stored_content_digest = conn.execute(
            text(
                "SELECT content_digest FROM authorization_decision_evidence "
                "WHERE id = :id"
            ),
            {"id": evidence["id"]},
        ).scalar_one()

    item = trace(client, machine_id, event["id"]).json()["evidence"][0]
    assert list(item.keys()) == [
        "id",
        "machine_id",
        "event_id",
        "evidence_type",
        "content_hash",
        "created_at",
        "previous_evidence_id",
        "content_digest",
        "chain_hash",
    ]
    assert item == {
        "id": evidence["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "evidence_type": evidence["evidence_type"],
        "content_hash": evidence["content_hash"],
        "created_at": evidence["created_at"],
        "previous_evidence_id": evidence["previous_evidence_id"],
        "content_digest": stored_content_digest,
        "chain_hash": evidence["chain_hash"],
    }


def test_status_history_and_assignments_use_their_own_event_and_machine_columns(
    client,
):
    # Directly inserted rows are selected by the status/assignment rows' own
    # (machine_id, event_id) even when their stored incident parent does not
    # exist: a damaged parent never filters the child out.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    execute_sql(
        client,
        "INSERT INTO incident_status_events (id, machine_id, event_id, "
        "incident_id, from_status, to_status, created_at, "
        "previous_status_event_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :e, :inc, 'open', 'acknowledged', :at, NULL, NULL, NULL)",
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "m": machine_id,
            "e": event["id"],
            "inc": GHOST_INCIDENT_ID,
            "at": T1,
        },
    )
    execute_sql(
        client,
        "INSERT INTO incident_responsibility_assignments (id, machine_id, "
        "event_id, incident_id, party, role, created_at, "
        "previous_assignment_id, content_hash, chain_hash) VALUES "
        "(:id, :m, :e, :inc, 'ops', 'owner', :at, NULL, NULL, NULL)",
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "m": machine_id,
            "e": event["id"],
            "inc": GHOST_INCIDENT_ID,
            "at": T2,
        },
    )

    body = trace(client, machine_id, event["id"]).json()
    assert [r["id"] for r in body["status_history"]] == [
        "11111111-1111-1111-1111-111111111111"
    ]
    assert body["status_history"][0]["incident_id"] == GHOST_INCIDENT_ID
    assert [r["id"] for r in body["responsibility_assignments"]] == [
        "22222222-2222-2222-2222-222222222222"
    ]
    assert body["responsibility_assignments"][0]["incident_id"] == GHOST_INCIDENT_ID


def test_causal_links_match_either_endpoint_and_keep_the_other_endpoint_verbatim(
    client,
):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    other_event = record_event(client, machine_id, resource="res/b")

    # Selected event is the effect; the cause endpoint is a real event.
    incoming = add_link(client, machine_id, other_event["id"], event["id"])
    # Selected event is the cause; the effect endpoint is a dangling ghost.
    execute_sql(
        client,
        "INSERT INTO authorization_decision_causal_links (id, machine_id, "
        "cause_event_id, effect_event_id, created_at) VALUES "
        "(:id, :m, :cause, :effect, :at)",
        {
            "id": "33333333-3333-3333-3333-333333333333",
            "m": machine_id,
            "cause": event["id"],
            "effect": GHOST_EVENT_ID,
            "at": T1,
        },
    )
    # Two incoming links from distinct dangling causes: repeated references
    # to the selected event are both kept, ghost endpoints verbatim.
    execute_sql(
        client,
        "INSERT INTO authorization_decision_causal_links (id, machine_id, "
        "cause_event_id, effect_event_id, created_at) VALUES "
        "(:id, :m, :cause, :effect, :at)",
        {
            "id": "44444444-4444-4444-4444-444444444444",
            "m": machine_id,
            "cause": "88888888-8888-8888-8888-888888888888",
            "effect": event["id"],
            "at": T2,
        },
    )

    links = trace(client, machine_id, event["id"]).json()["causal_links"]
    by_id = {link["id"]: link for link in links}
    assert set(by_id) == {
        incoming["id"],
        "33333333-3333-3333-3333-333333333333",
        "44444444-4444-4444-4444-444444444444",
    }
    assert by_id["33333333-3333-3333-3333-333333333333"]["effect_event_id"] == GHOST_EVENT_ID
    assert (
        by_id["44444444-4444-4444-4444-444444444444"]["cause_event_id"]
        == "88888888-8888-8888-8888-888888888888"
    )
    for link in links:
        assert list(link.keys()) == [
            "id",
            "machine_id",
            "cause_event_id",
            "effect_event_id",
            "created_at",
        ]


def test_records_of_another_event_and_another_machine_never_enter(client):
    machine_id = create_machine(client, "machine-1")
    other = create_machine(client, "machine-2")
    event = record_event(client, machine_id)
    sibling = record_event(client, machine_id, resource="res/b")
    foreign = record_event(client, other)
    foreign_sibling = record_event(client, other, resource="res/d")

    add_evidence(client, machine_id, sibling["id"], content_hash="b" * 64)
    add_incident(client, machine_id, sibling["id"], summary="other")
    add_evidence(client, other, foreign["id"], content_hash="c" * 64)
    add_incident(client, other, foreign["id"], summary="foreign")
    add_link(client, other, foreign["id"], foreign_sibling["id"])
    # A foreign-machine link that happens to point at the selected event is
    # still other-machine data and must never enter.
    execute_sql(
        client,
        "INSERT INTO authorization_decision_causal_links (id, machine_id, "
        "cause_event_id, effect_event_id, created_at) VALUES "
        "(:id, :m, :cause, :effect, :at)",
        {
            "id": "55555555-5555-5555-5555-555555555555",
            "m": other,
            "cause": foreign["id"],
            "effect": event["id"],
            "at": T1,
        },
    )

    body = trace(client, machine_id, event["id"]).json()
    assert body["evidence"] == []
    assert body["incidents"] == []
    assert body["status_history"] == []
    assert body["responsibility_assignments"] == []
    assert body["causal_links"] == []


def test_tracing_one_event_does_not_disturb_an_unrelated_event_trace(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    sibling = record_event(client, machine_id, resource="res/b")
    add_evidence(client, machine_id, event["id"], content_hash="a" * 64)
    add_evidence(client, machine_id, sibling["id"], content_hash="b" * 64)

    first = trace(client, machine_id, event["id"]).json()
    second = trace(client, machine_id, sibling["id"]).json()
    assert len(first["evidence"]) == 1
    assert len(second["evidence"]) == 1
    assert first["evidence"][0]["content_hash"] == "a" * 64
    assert second["evidence"][0]["content_hash"] == "b" * 64


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _insert_evidence(client, record_id, machine_id, event_id, created_at,
                     content_hash):
    execute_sql(
        client,
        "INSERT INTO authorization_decision_evidence (id, machine_id, event_id, "
        "evidence_type, content_hash, created_at, previous_evidence_id, "
        "content_digest, chain_hash) VALUES "
        "(:id, :m, :e, 'log', :hash, :at, NULL, NULL, NULL)",
        {
            "id": record_id,
            "m": machine_id,
            "e": event_id,
            "hash": content_hash,
            "at": created_at,
        },
    )


def test_exact_second_sorts_before_fractional_and_damaged_stamp_sorts_last(
    client,
):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    # Insert in a deliberately non-chronological order.
    _insert_evidence(client, "33333333-3333-3333-3333-333333333333",
                     machine_id, event["id"], "not-a-time", "d" * 64)
    _insert_evidence(client, "22222222-2222-2222-2222-222222222222",
                     machine_id, event["id"], T0_HALF, "c" * 64)
    _insert_evidence(client, "11111111-1111-1111-1111-111111111111",
                     machine_id, event["id"], T0, "b" * 64)

    items = trace(client, machine_id, event["id"]).json()["evidence"]
    assert [item["id"] for item in items] == [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]
    # The damaged stamp is emitted verbatim, not repaired.
    assert items[2]["created_at"] == "not-a-time"


def test_same_instant_ties_break_by_id_in_every_group(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    other_event = record_event(client, machine_id, resource="res/b")

    _insert_evidence(client, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                     machine_id, event["id"], T0, "b" * 64)
    _insert_evidence(client, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                     machine_id, event["id"], T0, "a" * 64)

    execute_sql(
        client,
        "INSERT INTO authorization_decision_causal_links (id, machine_id, "
        "cause_event_id, effect_event_id, created_at) VALUES "
        "(:id, :m, :cause, :effect, :at)",
        {
            "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "m": machine_id,
            "cause": event["id"],
            "effect": other_event["id"],
            "at": T0,
        },
    )
    execute_sql(
        client,
        "INSERT INTO authorization_decision_causal_links (id, machine_id, "
        "cause_event_id, effect_event_id, created_at) VALUES "
        "(:id, :m, :cause, :effect, :at)",
        {
            "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "m": machine_id,
            "cause": event["id"],
            "effect": GHOST_EVENT_ID,
            "at": T0,
        },
    )

    body = trace(client, machine_id, event["id"]).json()
    assert [item["id"] for item in body["evidence"]] == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]
    assert [item["id"] for item in body["causal_links"]] == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]


# --------------------------------------------------------------------------- #
# Serialization, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_a_single_newline(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = trace(client, machine_id, event["id"])
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    json.loads(response.content)


def test_group_field_order_is_fixed(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    add_evidence(client, machine_id, event["id"])

    raw = trace(client, machine_id, event["id"]).content.decode("utf-8")
    positions = [raw.index(f'"{name}"') for name in GROUP_ORDER]
    assert positions == sorted(positions)


def test_repeated_calls_are_byte_identical_and_read_only(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    downstream = record_event(client, machine_id, resource="res/b")
    incident = add_incident(client, machine_id, event["id"])
    transition(client, machine_id, event["id"], incident["id"], "acknowledged")
    assign(client, machine_id, event["id"], incident["id"])
    add_evidence(client, machine_id, event["id"])
    add_link(client, machine_id, event["id"], downstream["id"])

    def snapshot():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(text(f"SELECT * FROM {table}")).fetchall()
                for table in (
                    "authorization_decision_events",
                    "authorization_decision_evidence",
                    "authorization_decision_incidents",
                    "incident_status_events",
                    "incident_responsibility_assignments",
                    "authorization_decision_causal_links",
                )
            }

    before = snapshot()
    first = trace(client, machine_id, event["id"]).content
    second = trace(client, machine_id, event["id"]).content
    assert first == second
    assert snapshot() == before


def test_trace_survives_restart_byte_identical(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = record_event(first, machine_id, allow=True)
        downstream = record_event(first, machine_id, resource="res/b")
        incident = add_incident(first, machine_id, event["id"])
        transition(first, machine_id, event["id"], incident["id"], "acknowledged")
        assign(first, machine_id, event["id"], incident["id"])
        add_evidence(first, machine_id, event["id"])
        add_link(first, machine_id, event["id"], downstream["id"])
        body_before = trace(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        body_after = trace(second, machine_id, event["id"]).content
        assert body_after == body_before
        assert list(json.loads(body_after).keys()) == list(GROUP_ORDER)


def test_trace_body_has_no_floating_point_tokens(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    add_evidence(client, machine_id, event["id"])

    raw = trace(client, machine_id, event["id"]).content
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    assert b"-0.0" not in raw
