"""Tests for the read-only single-event accountability trace.

Covers `GET /machines/{machine_id}/authorization-decision-events/{event_id}/
accountability-trace`: GET-only ``405`` (including ``HEAD``) without reading
records, ``invalid_query`` 422 for any query parameter, repeated parameter, or
carried body before the machine lookup, ``not_found`` 404 for a missing
machine/event or a foreign-owned event with no trace data, ``internal_error``
500 on a read fault with neither the event summary nor any association array,
the fixed six-group shape and order, membership by the record's own machine
ownership and event association (parents' damage never filters children),
causal-link selection by either endpoint with the other end kept verbatim,
verbatim output of complete stored fields (damage, duplicates, dangling
references never crash or are repaired), ordering by the actual created_at
instant then id with damaged stamps last, machine isolation, read-only
byte-identical stability, and persistence across restarts.
"""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


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


def add_evidence(client, machine_id, event_id, content_hash=None,
                 evidence_type="log"):
    content_hash = content_hash or ("a" * 64)
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/evidence",
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )
    assert response.status_code == 201
    return response.json()


def add_incident(client, machine_id, event_id, incident_type="failure",
                 summary="something happened"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
        f"{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def assign(client, machine_id, event_id, incident_id, party="ops", role="owner"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
        f"{incident_id}/responsibility-assignments",
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


def trace_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/accountability-trace"
    )


def trace(client, machine_id, event_id):
    response = client.get(trace_url(machine_id, event_id))
    assert response.status_code == 200
    return response.json()


def tamper(db_path, statement, parameters=()):
    connection = sqlite3.connect(db_path)
    connection.execute(statement, parameters)
    connection.commit()
    connection.close()


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = getattr(client, method)(trace_url(machine_id, event["id"]))
    assert response.status_code == 405


@pytest.mark.parametrize("query", ["?x=1", "?x=", "?machine_id=z", "?limit=1"])
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    response = client.get(trace_url(machine_id, event["id"]) + "?x=1&x=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_and_event_lookup(client):
    response = client.get(trace_url(MISSING_ID, MISSING_ID) + "?x=1")
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


def test_get_with_a_body_is_invalid_query_before_machine_lookup(client):
    response = client.request(
        "GET",
        trace_url(MISSING_ID, MISSING_ID),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_and_method_errors_do_not_read_records(client):
    # Drop every related table: validation-phase and routing errors must
    # still come back as their codes without a read.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    assert client.get(trace_url(machine_id, event["id"]) + "?x=1").json() == {
        "error": {"code": "invalid_query"}
    }
    for method in ("head", "post", "put", "patch", "delete"):
        assert getattr(client, method)(trace_url(machine_id, event["id"])).status_code == 405


# --------------------------------------------------------------------------- #
# Existence and ownership
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404_with_no_trace_data(client):
    response = client.get(trace_url(MISSING_ID, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content
    assert b"evidence" not in response.content
    assert b"causal_links" not in response.content


def test_missing_event_returns_404(client):
    machine_id = create_machine(client)
    response = client.get(trace_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_event_owned_by_another_machine_returns_404(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    foreign = record_event(client, machine_two)

    response = client.get(trace_url(machine_one, foreign["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content


def test_machine_read_failure_is_500_without_trace_data(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))

    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content
    assert b"evidence" not in response.content


def test_group_read_failure_is_500_without_any_trace_data(client):
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
# Shape and membership
# --------------------------------------------------------------------------- #


def test_empty_trace_shape_and_order(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    result = trace(client, machine_id, event["id"])
    assert list(result.keys()) == [
        "event_summary",
        "evidence",
        "incidents",
        "status_history",
        "responsibility_assignments",
        "causal_links",
    ]
    assert isinstance(result["event_summary"], dict)
    assert result["evidence"] == []
    assert result["incidents"] == []
    assert result["status_history"] == []
    assert result["responsibility_assignments"] == []
    assert result["causal_links"] == []


def test_event_summary_carries_result_reason_moment_and_chain_fields(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    result = trace(client, machine_id, event["id"])
    summary = result["event_summary"]
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


def test_all_five_groups_collect_only_this_event_records(client):
    machine_id = create_machine(client)
    target = record_event(client, machine_id, resource="res/target")
    other = record_event(client, machine_id, resource="res/other")

    target_evidence = add_evidence(client, machine_id, target["id"])
    add_evidence(client, machine_id, other["id"], content_hash="b" * 64)
    target_incident = add_incident(client, machine_id, target["id"])
    add_incident(client, machine_id, other["id"], incident_type="other")
    transition(client, machine_id, target["id"], target_incident["id"],
               "acknowledged")
    target_assignment = assign(client, machine_id, target["id"],
                               target_incident["id"])
    create_link(client, machine_id, target["id"], other["id"])
    # The reverse edge closes a two-edge cycle, which the write API rejects;
    # inject the stored row directly so the trace's either-endpoint selection
    # is exercised on real stored data.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :m, :c, :e, :t)"
            ),
            {
                "id": "88888888-8888-8888-8888-888888888888",
                "m": machine_id,
                "c": other["id"],
                "e": target["id"],
                "t": T0,
            },
        )

    result = trace(client, machine_id, target["id"])

    assert [row["id"] for row in result["evidence"]] == [target_evidence["id"]]
    assert [row["id"] for row in result["incidents"]] == [target_incident["id"]]
    assert len(result["status_history"]) == 1
    assert result["status_history"][0]["incident_id"] == target_incident["id"]
    assert [row["id"] for row in result["responsibility_assignments"]] == [
        target_assignment["id"]
    ]
    # Either endpoint naming the selected event: both links are included.
    assert len(result["causal_links"]) == 2
    endpoints = {
        (row["cause_event_id"], row["effect_event_id"])
        for row in result["causal_links"]
    }
    assert endpoints == {
        (target["id"], other["id"]),
        (other["id"], target["id"]),
    }


def test_records_of_other_machines_never_enter(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    target = record_event(client, machine_one)
    foreign_event = record_event(client, machine_two)
    add_evidence(client, machine_two, foreign_event["id"])
    add_incident(client, machine_two, foreign_event["id"])
    # Foreign-owned link rows that happen to reference machine one's event
    # are injected directly (the write API forbids cross-machine endpoints).
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :m, :c, :e, :t)"
            ),
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "m": machine_two,
                "c": foreign_event["id"],
                "e": target["id"],
                "t": T0,
            },
        )
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :m, :c, :e, :t)"
            ),
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "m": machine_one,
                "c": target["id"],
                "e": foreign_event["id"],
                "t": T1,
            },
        )

    result = trace(client, machine_one, target["id"])
    assert result["evidence"] == []
    assert result["incidents"] == []
    assert result["status_history"] == []
    assert result["responsibility_assignments"] == []
    assert len(result["causal_links"]) == 1
    assert result["causal_links"][0]["id"] == "22222222-2222-2222-2222-222222222222"


def test_causal_links_keep_dangling_or_foreign_other_endpoint_verbatim(client):
    machine_id = create_machine(client)
    target = record_event(client, machine_id)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :m, :c, :e, :t)"
            ),
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "m": machine_id,
                "c": target["id"],
                "e": MISSING_ID,
                "t": T0,
            },
        )
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :m, :c, :e, :t)"
            ),
            {
                "id": "44444444-4444-4444-4444-444444444444",
                "m": machine_id,
                "c": MISSING_ID,
                "e": target["id"],
                "t": T1,
            },
        )

    result = trace(client, machine_id, target["id"])
    by_id = {row["id"]: row for row in result["causal_links"]}
    assert by_id["33333333-3333-3333-3333-333333333333"]["effect_event_id"] == MISSING_ID
    assert by_id["44444444-4444-4444-4444-444444444444"]["cause_event_id"] == MISSING_ID


def test_child_records_kept_when_parent_incident_is_damaged(client):
    # Status history and assignments are selected by their own machine/event
    # association; a missing or misowned incident id does not filter them.
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_status_events (id, machine_id, event_id, "
                "incident_id, from_status, to_status, created_at, "
                "previous_status_event_id, content_hash, chain_hash) VALUES "
                "(:id, :m, :e, :i, 'open', 'acknowledged', :t, NULL, NULL, NULL)"
            ),
            {"id": "55555555-5555-5555-5555-555555555555", "m": machine_id,
             "e": event["id"], "i": MISSING_ID, "t": T0},
        )
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments (id, machine_id, "
                "event_id, incident_id, party, role, created_at, "
                "previous_assignment_id, content_hash, chain_hash) VALUES "
                "(:id, :m, :e, :i, 'ops', 'owner', :t, NULL, NULL, NULL)"
            ),
            {"id": "66666666-6666-6666-6666-666666666666", "m": machine_id,
             "e": event["id"], "i": MISSING_ID, "t": T1},
        )

    result = trace(client, machine_id, event["id"])
    assert [row["id"] for row in result["status_history"]] == [
        "55555555-5555-5555-5555-555555555555"
    ]
    assert [row["id"] for row in result["responsibility_assignments"]] == [
        "66666666-6666-6666-6666-666666666666"
    ]


def test_records_complete_stored_fields_are_emitted(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    evidence = add_evidence(client, machine_id, event["id"])
    incident = add_incident(client, machine_id, event["id"])
    transition(client, machine_id, event["id"], incident["id"], "acknowledged")
    assignment = assign(client, machine_id, event["id"], incident["id"])
    link = create_link(client, machine_id, event["id"],
                       record_event(client, machine_id)["id"])

    result = trace(client, machine_id, event["id"])

    with client.app.state.engine.connect() as conn:
        stored_digest = conn.execute(
            text(
                "SELECT content_digest FROM authorization_decision_evidence "
                "WHERE id = :id"
            ),
            {"id": evidence["id"]},
        ).scalar()

    assert result["evidence"][0] == {
        "id": evidence["id"],
        "machine_id": machine_id,
        "event_id": event["id"],
        "evidence_type": "log",
        "content_hash": "a" * 64,
        "created_at": evidence["created_at"],
        "previous_evidence_id": evidence["previous_evidence_id"],
        "content_digest": stored_digest,
        "chain_hash": evidence["chain_hash"],
    }
    assert set(result["incidents"][0].keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_type",
        "summary",
        "status",
        "created_at",
    }
    assert set(result["status_history"][0].keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_id",
        "from_status",
        "to_status",
        "created_at",
        "previous_status_event_id",
        "content_hash",
        "chain_hash",
    }
    assert set(result["responsibility_assignments"][0].keys()) == {
        "id",
        "machine_id",
        "event_id",
        "incident_id",
        "party",
        "role",
        "created_at",
        "previous_assignment_id",
        "content_hash",
        "chain_hash",
    }
    assert set(result["causal_links"][0].keys()) == {
        "id",
        "machine_id",
        "cause_event_id",
        "effect_event_id",
        "created_at",
    }
    assert result["causal_links"][0]["id"] == link["id"]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_groups_order_by_created_at_instant_then_id(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    # Insert evidence rows with controlled created_at, including fractional.
    with client.app.state.engine.begin() as conn:
        for idx, stamp in enumerate((T2, T0, "2026-03-01T00:00:00.5Z", T1)):
            conn.execute(
                text(
                    "INSERT INTO authorization_decision_evidence (id, machine_id, "
                    "event_id, evidence_type, content_hash, created_at, "
                    "previous_evidence_id, content_digest, chain_hash) VALUES "
                    "(:id, :m, :e, 'log', :h, :t, NULL, NULL, NULL)"
                ),
                {
                    "id": f"0000000{idx}-0000-0000-0000-000000000000",
                    "m": machine_id,
                    "e": event["id"],
                    "h": f"{idx:064x}",
                    "t": stamp,
                },
            )

    result = trace(client, machine_id, event["id"])
    stamps = [row["created_at"] for row in result["evidence"]]
    # Exact-second T0 sorts before the fractional .5 of the same second.
    assert stamps == [T0, "2026-03-01T00:00:00.5Z", T1, T2]


def test_same_instant_tie_breaks_by_id(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    ids = []
    with client.app.state.engine.begin() as conn:
        for suffix in ("b", "a", "c"):
            row_id = f"00000000-0000-0000-0000-00000000000{suffix}"
            conn.execute(
                text(
                    "INSERT INTO authorization_decision_evidence (id, machine_id, "
                    "event_id, evidence_type, content_hash, created_at, "
                    "previous_evidence_id, content_digest, chain_hash) VALUES "
                    "(:id, :m, :e, 'log', :h, :t, NULL, NULL, NULL)"
                ),
                {"id": row_id, "m": machine_id, "e": event["id"],
                 "h": suffix * 64, "t": T0},
            )
            ids.append(row_id)

    result = trace(client, machine_id, event["id"])
    assert [row["id"] for row in result["evidence"]] == sorted(ids)


def test_damaged_created_at_sorts_last_and_does_not_crash(client, db_path):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    good = add_evidence(client, machine_id, event["id"])
    bad = add_evidence(client, machine_id, event["id"], content_hash="c" * 64)
    tamper(
        db_path,
        "UPDATE authorization_decision_evidence SET created_at = 'not-a-time' "
        "WHERE id = ?",
        (bad["id"],),
    )

    result = trace(client, machine_id, event["id"])
    assert [row["id"] for row in result["evidence"]] == [good["id"], bad["id"]]
    assert result["evidence"][1]["created_at"] == "not-a-time"


def test_duplicated_and_damaged_rows_are_emitted_verbatim(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    # The schema's primary key forbids two stored rows sharing an id, so
    # recreate the table without constraints to model the duplicated/damaged
    # storage the trace must tolerate without crashing or deduplicating.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE authorization_decision_evidence RENAME TO "
                "authorization_decision_evidence_backup"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE authorization_decision_evidence ("
                "id VARCHAR, machine_id VARCHAR, event_id VARCHAR, "
                "evidence_type VARCHAR, content_hash VARCHAR, created_at, "
                "previous_evidence_id VARCHAR, content_digest VARCHAR, "
                "chain_hash VARCHAR)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO authorization_decision_evidence SELECT * FROM "
                "authorization_decision_evidence_backup"
            )
        )
        # Two rows sharing every key (id included), then a row with damaged
        # non-text stored values.
        for stamp in (T0, T1):
            conn.execute(
                text(
                    "INSERT INTO authorization_decision_evidence (id, machine_id, "
                    "event_id, evidence_type, content_hash, created_at, "
                    "previous_evidence_id, content_digest, chain_hash) VALUES "
                    "(:id, :m, :e, 'log', :h, :t, NULL, NULL, NULL)"
                ),
                {"id": "77777777-7777-7777-7777-777777777777", "m": machine_id,
                 "e": event["id"], "h": "d" * 64, "t": stamp},
            )
        conn.execute(
            text(
                "INSERT INTO authorization_decision_evidence (id, machine_id, "
                "event_id, evidence_type, content_hash, created_at, "
                "previous_evidence_id, content_digest, chain_hash) VALUES "
                "(:id, :m, :e, 'log', NULL, 12345, NULL, NULL, NULL)"
            ),
            {"id": "99999999-9999-9999-9999-999999999999", "m": machine_id,
             "e": event["id"]},
        )

    result = trace(client, machine_id, event["id"])
    rows = result["evidence"]
    assert len(rows) == 3
    duplicated = [row for row in rows if row["id"] == "77777777-7777-7777-7777-777777777777"]
    assert len(duplicated) == 2
    assert [row["created_at"] for row in duplicated] == [T0, T1]
    # The non-text damaged stamp sorts after every parseable record and is
    # emitted verbatim (as a JSON number here).
    damaged = next(
        row for row in rows if row["id"] == "99999999-9999-9999-9999-999999999999"
    )
    assert rows[-1] is damaged
    assert damaged["created_at"] == 12345
    assert damaged["content_hash"] is None


# --------------------------------------------------------------------------- #
# Serialization, read-only, persistence
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_ending_in_single_newline(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = client.get(trace_url(machine_id, event["id"]))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b", " not in response.content
    assert b": " not in response.content
    json.loads(response.content.decode("utf-8"))


def test_trace_is_read_only_and_byte_identical(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)
    add_evidence(client, machine_id, event["id"])
    incident = add_incident(client, machine_id, event["id"])
    transition(client, machine_id, event["id"], incident["id"], "acknowledged")
    assign(client, machine_id, event["id"], incident["id"])

    first = client.get(trace_url(machine_id, event["id"]))
    second = client.get(trace_url(machine_id, event["id"]))
    assert first.content == second.content

    # Existing surfaces are unchanged by the trace reads.
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json() == events


def test_trace_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event = record_event(first, machine_id)
        add_evidence(first, machine_id, event["id"])
        incident = add_incident(first, machine_id, event["id"])
        transition(first, machine_id, event["id"], incident["id"],
                   "acknowledged")
        body_before = first.get(trace_url(machine_id, event["id"])).content

    with TestClient(app) as second:
        body_after = second.get(trace_url(machine_id, event["id"])).content

    assert body_after == body_before
