"""Tests for the read-only decision-basis snapshot consistency audit.

The independent audit entry is

    GET /machines/{machine_id}/authorization-decision-events/{event_id}/
        decision-basis/integrity

and answers exactly four conclusions in fixed order:
``{valid, checked_count, broken_basis_id, reason}``. These tests cover the
five decision shapes reporting valid, the fixed encoding and byte-identical
repeats, the 422/404/405/500 outcomes (validation before any read), the
missing-snapshot contract, the stable first-anomaly categories for damaged
or contradictory snapshots, machine isolation, strict read-only behavior,
later declaration/rule changes never retroactively breaking a faithful
historical snapshot, and persistence across restarts.
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


def integrity_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/decision-basis/integrity"
    )


def audit(client, machine_id, event_id):
    return client.get(integrity_url(machine_id, event_id))


def load_snapshot(client, event_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT document FROM authorization_decision_basis "
                "WHERE event_id = :id"
            ).bindparams(id=event_id)
        ).scalar_one()


def write_snapshot(client, event_id, document):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_basis SET document = :doc "
                "WHERE event_id = :id"
            ).bindparams(doc=document, id=event_id)
        )


def tamper_snapshot(client, event_id, mutate, *, raw=None):
    """Rewrite one snapshot; ``mutate`` edits the parsed document in place."""
    if raw is not None:
        write_snapshot(client, event_id, raw)
        return
    document = json.loads(load_snapshot(client, event_id))
    mutate(document)
    write_snapshot(
        client,
        event_id,
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
    )


# --------------------------------------------------------------------------- #
# Valid conclusions across every decision shape
# --------------------------------------------------------------------------- #


def _assert_valid(body):
    assert list(body) == [
        "valid",
        "checked_count",
        "broken_basis_id",
        "reason",
    ]
    assert body == {
        "valid": True,
        "checked_count": 1,
        "broken_basis_id": None,
        "reason": None,
    }


def test_allowed_by_policy_snapshot_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    response = audit(client, machine_id, event["id"])
    assert response.status_code == 200
    _assert_valid(response.json())


def test_denied_by_winner_snapshot_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/x", effect="deny", priority=1)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=5)
    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "denied_by_policy"

    assert audit(client, machine_id, event["id"]).json()["valid"] is True


def test_conflict_tier_snapshot_is_valid_and_supports_denial(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=2)
    create_rule(client, resource_pattern="res/x", effect="deny", priority=2)
    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "denied_by_policy"

    body = audit(client, machine_id, event["id"]).json()
    assert body["valid"] is True and body["checked_count"] == 1


def test_no_enabled_declaration_snapshot_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id, enabled=False)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["reason"] == "no_enabled_declaration"

    body = audit(client, machine_id, event["id"]).json()
    assert body["valid"] is True and body["checked_count"] == 1


def test_no_matching_policy_snapshot_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "no_matching_policy"

    body = audit(client, machine_id, event["id"]).json()
    assert body["valid"] is True and body["checked_count"] == 1


def test_suspended_snapshot_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    client.post(
        f"/machines/{machine_id}/status", json={"status": "suspended"}
    )
    event = record_event(client, machine_id).json()
    assert event["reason"] == "machine_suspended"

    body = audit(client, machine_id, event["id"]).json()
    assert body["valid"] is True and body["checked_count"] == 1


def test_successful_body_is_compact_fixed_order_single_newline(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    raw = audit(client, machine_id, event["id"]).content
    assert raw == b'{"valid":true,"checked_count":1,' \
        b'"broken_basis_id":null,"reason":null}\n'
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1


def test_repeated_audits_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="deny")
    event = record_event(client, machine_id).json()
    first = audit(client, machine_id, event["id"]).content
    for _ in range(3):
        assert audit(client, machine_id, event["id"]).content == first


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

    # Validation precedes every business read.
    assert (
        client.get(integrity_url(missing, event["id"]) + query).status_code == 422
    )


def test_request_body_is_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = client.request(
        "GET",
        integrity_url(machine_id, event["id"]),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    missing = "00000000-0000-0000-0000-000000000000"
    assert (
        client.request(
            "GET",
            integrity_url(missing, "anything"),
            content=b"{}",
            headers={"content-type": "application/json"},
        ).status_code
        == 422
    )


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
    assert b"checked_count" not in response.content


def test_non_get_methods_neither_read_nor_write(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    # With every table the audit could read dropped, only routing is in play.
    with client.app.state.engine.begin() as conn:
        for table in (
            "authorization_decision_basis",
            "authorization_decision_events",
            "behavior_declarations",
            "policy_rules",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert (
            getattr(client, method)(integrity_url(machine_id, event["id"])).status_code
            == 405
        )


def test_missing_machine_event_or_ownership_returns_not_found_without_conclusion(
    client,
):
    machine_id = create_machine(client)
    other = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id).json()
    missing = "00000000-0000-0000-0000-000000000000"

    expected_error = {"error": {"code": "not_found"}}
    response = client.get(integrity_url(missing, event["id"]))
    assert response.status_code == 404 and response.json() == expected_error
    response = client.get(integrity_url(machine_id, "no-such-event"))
    assert response.status_code == 404 and response.json() == expected_error
    # An existing event owned by another machine is indistinguishable.
    response = client.get(integrity_url(other, event["id"]))
    assert response.status_code == 404 and response.json() == expected_error
    assert b"checked_count" not in response.content


def test_missing_snapshot_reports_snapshot_not_found_without_fabrication(client):
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

    response = audit(client, machine_id, event["id"])
    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 0,
        "broken_basis_id": event["id"],
        "reason": "snapshot_not_found",
    }


@pytest.mark.parametrize(
    "table",
    [
        "authorization_decision_basis",
        "behavior_declarations",
        "policy_rules",
    ],
)
def test_real_read_failure_is_internal_error_without_partial_result(
    client, table
):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE {table}"))

    response = audit(client, machine_id, event["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"checked_count" not in response.content


# --------------------------------------------------------------------------- #
# Damaged or contradictory snapshots: stable first-anomaly categories
# --------------------------------------------------------------------------- #


def _assert_broken(response, event_id, reason):
    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_basis_id": event_id,
        "reason": reason,
    }


@pytest.mark.parametrize(
    "raw",
    [
        "this is not json",
        '{"a":1,}',
        '{"x":NaN}',
        "[]",
        '"a string"',
        "123",
    ],
)
def test_unparseable_or_non_object_document_is_corrupt(client, raw):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(client, event["id"], None, raw=raw)
    _assert_broken(audit(client, machine_id, event["id"]), event["id"],
                   "corrupt_document")


def test_missing_top_level_group_is_malformed_structure(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        del doc["decision"]

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_structure",
    )


def test_extra_or_reordered_top_level_group_is_malformed_structure(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def add_extra(doc):
        doc["extra"] = 1

    tamper_snapshot(client, event["id"], add_extra)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_structure",
    )

    second = record_event(client, machine_id).json()
    tamper_snapshot(
        client,
        second["id"],
        lambda doc: _reorder(
            doc,
            [
                "decision",
                "event_summary",
                "status_basis",
                "declaration_basis",
                "policy_candidates",
            ],
        ),
    )
    _assert_broken(
        audit(client, machine_id, second["id"]), second["id"],
        "malformed_structure",
    )


def _reorder(doc, keys):
    reordered = {key: doc[key] for key in keys}
    doc.clear()
    doc.update(reordered)


def test_wrong_summary_field_type_is_malformed_event_summary(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["event_summary"].__setitem__(
            "allowed", "yes"
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_event_summary",
    )


def test_wrong_status_field_type_is_malformed_status_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["status_basis"].__setitem__(
            "declarations_read", "x"
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_status_basis",
    )


def test_wrong_declaration_shape_is_malformed_declaration_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["declaration_basis"].__setitem__(
            "declarations", {}
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_declaration_basis",
    )

    # A second, intact snapshot for the wrong member type.
    second = record_event(client, machine_id).json()
    tamper_snapshot(
        client, second["id"], lambda doc: doc["declaration_basis"][
            "declarations"
        ][0].__setitem__("matched", "yes")
    )
    _assert_broken(
        audit(client, machine_id, second["id"]), second["id"],
        "malformed_declaration_basis",
    )


def test_wrong_candidate_shape_is_malformed_policy_candidates(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["policy_candidates"].__setitem__(
            "candidates", []
        )
        or doc["policy_candidates"].__setitem__("winners", [
            {"id": "x", "effect": "allow", "priority": "0",
             "created_at": "t"}
        ])
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_policy_candidates",
    )


def test_wrong_decision_shape_is_malformed_decision(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["decision"].pop("reason")
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "malformed_decision",
    )


def test_summary_that_disagrees_with_event_is_event_summary_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["event_summary"].__setitem__(
            "reason", "denied_by_policy"
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "event_summary_mismatch",
    )


def test_summary_chain_field_and_timestamp_must_be_verbatim(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["event_summary"].__setitem__(
            "chain_hash", "0" * 64
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "event_summary_mismatch",
    )

    # A fresh, intact snapshot with only the captured moment altered.
    second = record_event(client, machine_id).json()
    tamper_snapshot(
        client, second["id"], lambda doc: doc["event_summary"].__setitem__(
            "created_at", "1970-01-01T00:00:00Z"
        )
    )
    _assert_broken(
        audit(client, machine_id, second["id"]), second["id"],
        "event_summary_mismatch",
    )


def test_status_read_flags_must_match_status(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # Active machine recorded as not having read declarations.
    tamper_snapshot(
        client, event["id"], lambda doc: doc["status_basis"].__setitem__(
            "declarations_read", False
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "status_basis_mismatch",
    )


def test_suspended_basis_with_records_or_read_flag_is_status_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    client.post(
        f"/machines/{machine_id}/status", json={"status": "suspended"}
    )
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"], lambda doc: doc["status_basis"].__setitem__(
            "policies_read", True
        )
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "status_basis_mismatch",
    )


def test_extra_declaration_record_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        extra = dict(doc["declaration_basis"]["declarations"][0])
        extra["id"] = "00000000-0000-0000-0000-000000000001"
        doc["declaration_basis"]["declarations"].append(extra)

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "declaration_basis_mismatch",
    )


def test_missing_declaration_record_is_declaration_basis_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    declare(client, machine_id, resource_pattern="docs/*")
    create_rule(client)
    event = record_event(client, machine_id).json()

    tamper_snapshot(
        client, event["id"],
        lambda doc: doc["declaration_basis"]["declarations"].pop(),
    )
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "declaration_basis_mismatch",
    )


def test_wrong_match_flag_is_declaration_match_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        item = doc["declaration_basis"]["declarations"][0]
        assert item["matched"] is True
        item["matched"] = False

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "declaration_match_mismatch",
    )


def test_anomalous_declaration_order_is_declaration_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/a*")
    declare(client, machine_id, resource_pattern="res/b*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        items = doc["declaration_basis"]["declarations"]
        # The capture orders by (created_at instant, id); reversing two
        # distinct records breaks that ordering even on a shared timestamp.
        assert len(items) == 2 and items[0]["id"] != items[1]["id"]
        items.reverse()

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "declaration_order",
    )


def test_wrong_candidate_relation_is_policy_candidate_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/x", effect="deny", priority=1)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=5)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        for candidate in doc["policy_candidates"]["candidates"]:
            if candidate["priority"] == 1:
                assert candidate["relation"] == "winner"
                candidate["relation"] = "unmatched"

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "policy_candidate_mismatch",
    )


def test_other_action_candidate_is_policy_candidate_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        candidate = dict(doc["policy_candidates"]["candidates"][0])
        candidate["id"] = "00000000-0000-0000-0000-000000000009"
        candidate["action_type"] = "write"
        doc["policy_candidates"]["candidates"].append(candidate)

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "policy_candidate_mismatch",
    )


def test_anomalous_candidate_order_is_candidate_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    create_rule(client, resource_pattern="other/*", effect="deny", priority=0)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        doc["policy_candidates"]["candidates"].reverse()

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "candidate_order",
    )


def test_wrong_winner_group_is_policy_winner_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/x", effect="deny", priority=1)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=5)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        doc["policy_candidates"]["winners"][0]["priority"] = 9

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "policy_winner_mismatch",
    )


def test_wrong_conflict_group_is_policy_conflict_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=2)
    create_rule(client, resource_pattern="res/x", effect="deny", priority=2)
    event = record_event(client, machine_id, resource="res/x").json()

    def mutate(doc):
        doc["policy_candidates"]["conflicts"] = []

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "policy_conflict_mismatch",
    )


def test_decision_that_disagrees_with_event_is_decision_mismatch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        doc["decision"]["allowed"] = False
        doc["decision"]["reason"] = "denied_by_policy"

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "decision_mismatch",
    )


def test_first_anomaly_is_reported_when_several_parts_are_broken(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def mutate(doc):
        # Break both the early event summary and the late decision.
        doc["event_summary"]["reason"] = "machine_suspended"
        doc["decision"]["allowed"] = False

    tamper_snapshot(client, event["id"], mutate)
    _assert_broken(
        audit(client, machine_id, event["id"]), event["id"],
        "event_summary_mismatch",
    )


def test_broken_conclusion_is_stable_across_repeats(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    tamper_snapshot(
        client, event["id"], lambda doc: doc["decision"].__setitem__(
            "reason", "machine_suspended"
        )
    )
    first = audit(client, machine_id, event["id"]).content
    for _ in range(3):
        assert audit(client, machine_id, event["id"]).content == first


# --------------------------------------------------------------------------- #
# Isolation, read-only behavior, as-of-then semantics, and persistence
# --------------------------------------------------------------------------- #


def test_audit_is_strictly_isolated_per_machine(client):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    for machine_id in (one, two):
        declare(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=0)
    event_one = record_event(client, one, resource="a").json()
    event_two = record_event(client, two, resource="b").json()

    assert audit(client, one, event_one["id"]).json()["valid"] is True
    assert audit(client, two, event_two["id"]).json()["valid"] is True

    # Tamper one machine's snapshot; the other machine's conclusion and
    # checked count are unaffected, and cross-machine paths stay 404.
    tamper_snapshot(
        client, event_one["id"],
        lambda doc: doc["decision"].__setitem__("allowed", False),
    )
    body_one = audit(client, one, event_one["id"]).json()
    assert body_one["valid"] is False
    assert body_one["broken_basis_id"] == event_one["id"]
    assert body_one["checked_count"] == 1
    body_two = audit(client, two, event_two["id"]).json()
    assert body_two["valid"] is True and body_two["checked_count"] == 1
    assert audit(client, two, event_one["id"]).status_code == 404
    assert audit(client, one, event_two["id"]).status_code == 404


def test_other_machine_snapshot_never_counts(client):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    declare(client, one, resource_pattern="*")
    declare(client, two, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=0)
    event_one = record_event(client, one, resource="a").json()
    event_two = record_event(client, two, resource="b").json()

    # A missing snapshot for one machine reports 0 regardless of the other
    # machine's snapshot, and path isolation keeps cross reads at 404.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event_one["id"])
        )
    body = audit(client, one, event_one["id"]).json()
    assert body["checked_count"] == 0 and body["reason"] == "snapshot_not_found"
    assert audit(client, two, event_two["id"]).json()["checked_count"] == 1


def test_audit_never_writes(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def counts():
        with client.app.state.engine.connect() as conn:
            return {
                name: conn.execute(
                    text(f"SELECT COUNT(*) FROM {name}")
                ).scalar_one()
                for name in (
                    "authorization_decision_basis",
                    "authorization_decision_events",
                    "behavior_declarations",
                    "policy_rules",
                )
            }

    before = counts()
    audit(client, machine_id, event["id"])
    audit(client, machine_id, event["id"])
    audit(client, machine_id, "missing-event")
    audit(client, create_machine(client, "machine-3"), event["id"])
    assert counts() == before


def test_later_rule_change_never_breaks_a_faithful_historical_snapshot(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    first = record_event(client, machine_id, resource="res/x").json()
    assert first["reason"] == "allowed_by_policy"

    # A later deny at a lower priority changes subsequent decisions but the
    # stored basis of the earlier event was captured against the earlier rule
    # set and must still verify.
    create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    second = record_event(client, machine_id, resource="res/x").json()
    assert second["reason"] == "denied_by_policy"

    assert audit(client, machine_id, first["id"]).json()["valid"] is True
    assert audit(client, machine_id, second["id"]).json()["valid"] is True


def test_later_new_declaration_never_breaks_a_historical_snapshot(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client)
    event = record_event(client, machine_id, resource="res/x").json()

    # A declaration added after the event did not participate and must not be
    # required of (nor break) the historical snapshot.
    declare(client, machine_id, resource_pattern="res/later*")
    assert audit(client, machine_id, event["id"]).json()["valid"] is True


def test_snapshot_audit_persists_byte_identically_across_restart(
    tmp_path, monkeypatch
):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)
    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id, resource="res/persist").json()
        first_bytes = audit(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        response = audit(second, machine_id, event["id"])
    assert response.status_code == 200
    assert response.content == first_bytes
