"""Tests for the read-only decision-basis snapshot consistency audit.

The independent cross-check entry is

    GET /machines/{machine_id}/authorization-decision-events/{event_id}/decision-basis/integrity

It answers exactly four conclusions — ``valid``, ``checked_count``,
``broken_basis_id``, ``reason`` — and never trusts the snapshot blindly: the
stored basis is checked against the committed event and the point-in-time
status/declaration/rule records. These tests cover the five decision
shapes, verbatim event-summary matching, status/read-flag consistency,
declaration participation and ordering, policy candidate relations and the
judgement they support, the fixed tamper categories, the
``snapshot_not_found`` case, point-in-time independence from later records,
validation/routing outcomes (422/404/405/500), byte-identical repeats,
machine isolation, restart persistence, and strict read-only behavior.
"""
import json

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


def create_rule(client, action_type="read", resource_pattern="res/*",
                effect="allow", priority=0):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )
    assert response.status_code == 201
    return response.json()


def declare(client, machine_id, action_type="read", resource_pattern="res/*",
            enabled=True):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201
    return response.json()


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def set_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response


def integrity_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/decision-basis/integrity"
    )


def get_integrity(client, machine_id, event_id):
    return client.get(integrity_url(machine_id, event_id))


def tamper_snapshot(client, machine_id, event_id, mutate):
    """Replace the event's stored basis document after applying ``mutate``."""
    with client.app.state.engine.connect() as conn:
        document_text = conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :id AND machine_id = :machine"
            ).bindparams(id=event_id, machine=machine_id)
        ).scalar_one()
    document = json.loads(document_text)
    mutate(document)
    new_text = json.dumps(
        document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id AND machine_id = :machine"
            ).bindparams(doc=new_text, id=event_id, machine=machine_id)
        )


def write_snapshot_raw(client, machine_id, event_id, raw_text):
    """Overwrite the stored basis document with arbitrary text."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id AND machine_id = :machine"
            ).bindparams(doc=raw_text, id=event_id, machine=machine_id)
        )


# --------------------------------------------------------------------------- #
# Sound snapshots: every decision shape audits valid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "setup,expected_reason,expected_allowed",
    [
        (lambda c, m: (declare(c, m), create_rule(c)), "allowed_by_policy", True),
        (
            lambda c, m: (
                declare(c, m, resource_pattern="res/*"),
                create_rule(c, resource_pattern="res/x", effect="deny", priority=1),
                create_rule(c, resource_pattern="res/*", effect="allow", priority=5),
            ),
            "denied_by_policy",
            False,
        ),
        (
            lambda c, m: (
                declare(c, m, resource_pattern="res/*"),
                create_rule(c, resource_pattern="res/*", effect="allow", priority=2),
                create_rule(c, resource_pattern="res/x", effect="deny", priority=2),
            ),
            "denied_by_policy",
            False,
        ),
        (lambda c, m: None, "no_enabled_declaration", False),
        (
            lambda c, m: declare(c, m, resource_pattern="res/*"),
            "no_matching_policy",
            False,
        ),
    ],
)
def test_sound_snapshots_audit_valid(client, setup, expected_reason, expected_allowed):
    machine_id = create_machine(client)
    setup(client, machine_id)
    event = record_event(client, machine_id).json()
    assert (event["allowed"], event["reason"]) == (
        expected_allowed,
        expected_reason,
    )

    response = get_integrity(client, machine_id, event["id"])
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 1,
        "broken_basis_id": None,
        "reason": None,
    }


def test_suspended_event_audits_valid_and_stays_valid_after_reactivation(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    set_status(client, machine_id, "suspended")
    event = record_event(client, machine_id).json()
    assert event["reason"] == "machine_suspended"

    # The status history, not the machine's current row, fixes the status at
    # capture time: reactivating afterwards leaves the historical audit valid.
    set_status(client, machine_id, "active")
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body == {
        "valid": True,
        "checked_count": 1,
        "broken_basis_id": None,
        "reason": None,
    }


def test_active_event_audits_valid_after_a_later_suspension(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    set_status(client, machine_id, "suspended")

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["valid"] is True


# --------------------------------------------------------------------------- #
# Encoding: four fields, fixed order, compact, byte-identical repeats
# --------------------------------------------------------------------------- #


def test_response_has_exactly_four_conclusions_in_fixed_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    raw = get_integrity(client, machine_id, event["id"]).content
    assert raw == b'{"valid":true,"checked_count":1,"broken_basis_id":null,"reason":null}\n'
    assert list(json.loads(raw)) == [
        "valid",
        "checked_count",
        "broken_basis_id",
        "reason",
    ]


def test_repeated_queries_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    first = get_integrity(client, machine_id, event["id"]).content
    for _ in range(3):
        assert get_integrity(client, machine_id, event["id"]).content == first


# --------------------------------------------------------------------------- #
# Missing snapshot
# --------------------------------------------------------------------------- #


def test_missing_snapshot_is_invalid_count_zero_snapshot_not_found(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event["id"])
        )

    response = get_integrity(client, machine_id, event["id"])
    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 0,
        "broken_basis_id": event["id"],
        "reason": "snapshot_not_found",
    }


# --------------------------------------------------------------------------- #
# Tampered snapshots: stable first-anomaly categories
# --------------------------------------------------------------------------- #


def test_malformed_json_document_is_document_structure(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    write_snapshot_raw(client, machine_id, event["id"], "{not json")

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_basis_id": event["id"],
        "reason": "document_structure",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.pop("decision"),
        lambda doc: doc.update({"extra": 1}),
        lambda doc: doc.__setitem__("event_summary", []),
        lambda doc: doc["status_basis"].pop("status"),
        lambda doc: doc["declaration_basis"].pop("read"),
        lambda doc: doc["policy_candidates"].__setitem__("winners", {}),
    ],
)
def test_damaged_document_shape_is_document_structure(client, mutate):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(client, machine_id, event["id"], mutate)

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["valid"] is False
    assert body["checked_count"] == 1
    assert body["broken_basis_id"] == event["id"]
    assert body["reason"] == "document_structure"


def test_event_summary_field_change_is_event_summary_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["event_summary"].__setitem__("reason", "other"),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "event_summary_mismatch"
    assert body["valid"] is False and body["checked_count"] == 1


def test_event_summary_extra_key_is_event_summary_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["event_summary"].__setitem__("public_key", "k"),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "event_summary_mismatch"


def test_event_row_change_is_event_summary_mismatch_not_recomputed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # Damage the committed event row itself; the audit must report the
    # verbatim summary mismatch, never silently re-hash or reconcile it.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_events SET reason = 'other' "
                "WHERE id = :id"
            ).bindparams(id=event["id"])
        )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "event_summary_mismatch"
    assert body["broken_basis_id"] == event["id"]


def test_status_basis_wrong_status_is_status_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: (
            doc["status_basis"].__setitem__("status", "suspended"),
            doc["status_basis"].__setitem__("declarations_read", False),
            doc["status_basis"].__setitem__("policies_read", False),
            doc["declaration_basis"].__setitem__("read", False),
            doc["declaration_basis"].__setitem__("declarations", []),
            doc["policy_candidates"].__setitem__("read", False),
            doc["policy_candidates"].__setitem__("candidates", []),
            doc["policy_candidates"].__setitem__("winners", []),
            doc["policy_candidates"].__setitem__("conflicts", []),
        ),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "status_basis_mismatch"


def test_suspended_snapshot_carrying_reads_is_status_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    set_status(client, machine_id, "suspended")
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["status_basis"].__setitem__("declarations_read", True),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "status_basis_mismatch"


def test_contradictory_status_history_is_status_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # A history transition that cannot continue the established status makes
    # the point-in-time status undecidable rather than assumed.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO machine_status_events "
                "(id, machine_id, from_status, to_status, created_at) "
                "VALUES (:id, :machine, 'suspended', 'active', :at)"
            ).bindparams(
                id="11111111-1111-1111-1111-111111111111",
                machine=machine_id,
                at=event["created_at"],
            )
        )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "status_basis_mismatch"
    assert body["valid"] is False


def test_extra_declaration_record_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declaration = declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        extra = dict(doc["declaration_basis"]["declarations"][0])
        extra["id"] = "22222222-2222-2222-2222-222222222222"
        doc["declaration_basis"]["declarations"].append(extra)

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "declaration_basis_mismatch"
    # The real declaration id is surfaced nowhere as broken; the event is.
    assert body["broken_basis_id"] == event["id"]
    assert declaration["id"] != event["id"]


def test_missing_declaration_record_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    declare(client, machine_id, resource_pattern="docs/*")
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["declaration_basis"]["declarations"].pop(),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "declaration_basis_mismatch"


def test_wrong_matched_flag_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="docs/*")
    event = record_event(client, machine_id).json()
    assert event["reason"] == "no_enabled_declaration"
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["declaration_basis"]["declarations"][0].__setitem__(
            "matched", True
        ),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "declaration_basis_mismatch"


def test_declaration_order_anomaly_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/a")
    declare(client, machine_id, resource_pattern="res/b")
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        items = doc["declaration_basis"]["declarations"]
        items[:] = list(reversed(items))

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "declaration_basis_mismatch"


def test_disabled_declaration_injected_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    disabled = declare(client, machine_id, resource_pattern="other/*", enabled=False)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        doc["declaration_basis"]["declarations"].append(
            {
                "id": disabled["id"],
                "action_type": "read",
                "resource_pattern": "other/*",
                "enabled": False,
                "created_at": disabled["created_at"],
                "updated_at": disabled["updated_at"],
                "matched": False,
            }
        )

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "declaration_basis_mismatch"


def test_candidate_relation_change_is_policy_candidates_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["policy_candidates"]["candidates"][0].__setitem__(
            "relation", "unmatched"
        ),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "policy_candidates_mismatch"


def test_winner_reference_change_is_policy_candidates_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: doc["policy_candidates"]["winners"][0].__setitem__(
            "effect", "deny"
        ),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "policy_candidates_mismatch"


def test_conflict_group_change_is_policy_candidates_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    allow_rule = create_rule(
        client, resource_pattern="res/*", effect="allow", priority=2
    )
    deny_rule = create_rule(
        client, resource_pattern="res/x", effect="deny", priority=2
    )
    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "denied_by_policy"

    def mutate(doc):
        # Fabricate a winner alongside the real conflict group: internally
        # contradictory and not what the priority semantics produce.
        doc["policy_candidates"]["winners"].append(
            {
                "id": allow_rule["id"],
                "effect": "allow",
                "priority": 2,
                "created_at": allow_rule["created_at"],
            }
        )
        # Leave a stale reference to the deny rule's pair relationship.
        assert doc["policy_candidates"]["conflicts"] == [
            sorted([allow_rule["id"], deny_rule["id"]])
        ]

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "policy_candidates_mismatch"


def test_non_read_policy_section_when_policy_was_read_is_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        machine_id,
        event["id"],
        lambda doc: (
            doc["policy_candidates"].__setitem__("read", False),
            doc["policy_candidates"].__setitem__("candidates", []),
            doc["policy_candidates"].__setitem__("winners", []),
            doc["policy_candidates"].__setitem__("conflicts", []),
        ),
    )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "policy_candidates_mismatch"


def test_decision_section_change_is_decision_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        doc["decision"]["reason"] = "denied_by_policy"
        doc["decision"]["allowed"] = False

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_basis_id": event["id"],
        "reason": "decision_mismatch",
    }


def test_first_anomaly_is_reported_in_check_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # Damage every group at once; the earliest check in the fixed order
    # (event summary) must be the reported category.
    def mutate(doc):
        doc["event_summary"]["reason"] = "x"
        doc["status_basis"]["status"] = "suspended"
        doc["declaration_basis"]["declarations"] = []
        doc["policy_candidates"]["conflicts"] = [["a", "b"]]
        doc["decision"]["reason"] = "x"

    tamper_snapshot(client, machine_id, event["id"], mutate)
    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["reason"] == "event_summary_mismatch"


def test_damaged_participant_rule_field_is_mismatch_not_internal_error(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    rule = create_rule(client, resource_pattern="res/*", effect="allow")
    event = record_event(client, machine_id, resource="res/x").json()

    # An externally corrupted participant row contradicts the sound
    # snapshot; the audit reports a stable category, never a 500 or partial
    # result, and never repairs the row.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET priority = 'abc' WHERE id = :id"),
            {"id": rule["id"]},
        )

    response = get_integrity(client, machine_id, event["id"])
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_basis_id": event["id"],
        "reason": "policy_candidates_mismatch",
    }


def test_damaged_non_participant_rule_keeps_historical_audit_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    # The good rule matches and decides; the other-action rule never enters
    # the basis for this action even when its stored effect is later damaged.
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    other = create_rule(
        client, action_type="write", resource_pattern="res/*",
        effect="allow", priority=0,
    )
    event = record_event(client, machine_id, resource="res/x").json()

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET effect = 'weird' WHERE id = :id"),
            {"id": other["id"]},
        )

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["valid"] is True
    assert body["checked_count"] == 1


# --------------------------------------------------------------------------- #
# Point-in-time independence: later records never invalidate an old snapshot
# --------------------------------------------------------------------------- #


def test_later_rules_do_not_invalidate_historical_snapshot(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    event = record_event(client, machine_id, resource="res/x").json()

    # Rules added after the event change later decisions but never enter the
    # historical candidate set.
    create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    later = record_event(client, machine_id, resource="res/x").json()
    assert later["reason"] == "denied_by_policy"

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["valid"] is True


def test_later_declarations_do_not_invalidate_historical_snapshot(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    assert event["reason"] == "no_enabled_declaration"

    # A declaration created after the event did not participate in it.
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client)

    body = get_integrity(client, machine_id, event["id"]).json()
    assert body["valid"] is True
    assert body["checked_count"] == 1


# --------------------------------------------------------------------------- #
# Validation, routing, and lookup outcomes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["?x=1", "?limit=1", "?=", "?foo", "?x=1&x=2"])
def test_any_query_parameter_is_invalid_query_before_lookup(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    missing = "00000000-0000-0000-0000-000000000000"

    response = client.get(integrity_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine/event/snapshot lookup.
    assert (
        client.get(integrity_url(missing, event["id"]) + query).status_code == 422
    )


def test_request_body_is_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    missing = "00000000-0000-0000-0000-000000000000"

    response = client.request(
        "GET",
        integrity_url(machine_id, event["id"]),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        integrity_url(missing, "anything"),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_zero_content_length_get_is_accepted(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = client.request(
        "GET", integrity_url(machine_id, event["id"]), content=b""
    )
    assert response.status_code == 200


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_routed(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = getattr(client, method)(integrity_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"valid" not in response.content


def test_non_get_methods_do_not_read_anything(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    # With every table the audit could read dropped, only routing is in play.
    with client.app.state.engine.begin() as conn:
        for table in (
            "authorization_decision_basis",
            "authorization_decision_events",
            "machine_status_events",
            "behavior_declarations",
            "policy_rules",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine_id, event["id"]))
        assert response.status_code == 405


def test_missing_machine_event_or_ownership_returns_not_found(client):
    machine_id = create_machine(client)
    other = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id).json()
    missing = "00000000-0000-0000-0000-000000000000"

    assert client.get(integrity_url(missing, event["id"])).status_code == 404
    assert client.get(integrity_url(machine_id, "no-such-event")).status_code == 404
    response = client.get(integrity_url(other, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # A not-found response carries no audit conclusion fields.
    assert "valid" not in response.json().get("error", {})


def test_read_failure_returns_internal_error_without_conclusion(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))

    response = get_integrity(client, machine_id, event["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"checked_count" not in response.content


# --------------------------------------------------------------------------- #
# Isolation, persistence, and strict read-only behavior
# --------------------------------------------------------------------------- #


def test_audit_is_strictly_machine_isolated(client):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    for machine_id in (one, two):
        declare(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=0)

    event_one = record_event(client, one, resource="a").json()
    event_two = record_event(client, two, resource="b").json()

    assert get_integrity(client, one, event_one["id"]).json()["valid"] is True
    assert get_integrity(client, two, event_two["id"]).json()["valid"] is True
    # Guessing another machine's event id on this path is a plain 404, never
    # a cross-machine conclusion.
    assert get_integrity(client, two, event_one["id"]).status_code == 404
    assert get_integrity(client, one, event_two["id"]).status_code == 404


def test_audit_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id, resource="res/persist").json()
        first_bytes = get_integrity(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        response = get_integrity(second, machine_id, event["id"])

    assert response.status_code == 200
    assert response.content == first_bytes


def test_get_query_never_writes(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def counts():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar_one()
                for table in (
                    "authorization_decision_basis",
                    "authorization_decision_events",
                    "behavior_declarations",
                    "policy_rules",
                    "machine_status_events",
                )
            }

    before = counts()
    for _ in range(3):
        get_integrity(client, machine_id, event["id"])
    # The not-found and missing-snapshot paths must not heal anything either.
    get_integrity(client, machine_id, "missing-event")
    get_integrity(
        client, create_machine(client, "machine-3"), event["id"]
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event["id"])
        )
    get_integrity(client, machine_id, event["id"])
    after = counts()
    assert after == {**before, "authorization_decision_basis": before[
        "authorization_decision_basis"
    ] - 1}
