"""Tests for immutable execution receipts bound to consumed grants.

Two entries sit under the machine/grant path plus one machine-level audit:

    POST /machines/{machine_id}/authorization-grants/{grant_id}/execution-receipts
    GET  /machines/{machine_id}/authorization-grants/{grant_id}/execution-receipts
    GET  /machines/{machine_id}/execution-receipts/integrity

A receipt binds what actually happened to a grant's unique consumption record:
each consumed grant carries at most one receipt, the real action/resource are
stored verbatim alongside ``matches_authorization`` (the comparison against
the original authorization event), and receipts form the machine's
tamper-evident per-machine hash chain in (created_at instant, id) order using
the empty-prefix rule. These tests cover the success shape and hashing, the
match/mismatch comparison, both outcomes, every validation outcome and its
ordering before any read, the 404/409 lookup outcomes, exactly-once writes
under concurrency, the single-receipt GET, the integrity audit (empty, sound,
tampered, isolated), 405 routing, restart persistence, and safe migration that
never rewrites old data.
"""
import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

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


def issue(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume(client, machine_id, grant_id):
    return client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def add_evidence(client, machine_id, event_id, fingerprint,
                 evidence_type="log"):
    return client.post(
        f"/machines/{machine_id}"
        f"/authorization-decision-events/{event_id}/evidence",
        json={"evidence_type": evidence_type, "content_hash": fingerprint},
    )


def receipt_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}"
        "/execution-receipts"
    )


def integrity_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/integrity"


FINGERPRINT_ONE = "a" * 64
FINGERPRINT_TWO = "b" * 64

RECEIPT_FIELDS = [
    "id",
    "grant_id",
    "use_id",
    "evidence_id",
    "action_type",
    "resource",
    "outcome",
    "matches_authorization",
    "created_at",
    "previous_receipt_id",
    "content_hash",
    "chain_hash",
]


@pytest.fixture
def consumed_grant(client):
    """A machine with an allowed event, an issued+consumed grant, evidence."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    use = consume(client, machine_id, grant["id"]).json()
    evidence = add_evidence(
        client, machine_id, event["id"], FINGERPRINT_ONE
    ).json()
    return machine_id, event, grant, use, evidence


def post_receipt(client, machine_id, grant_id, **fields):
    body = {
        "evidence_id": fields.get("evidence_id"),
        "action_type": fields.get("action_type", "read"),
        "resource": fields.get("resource", "res/x"),
        "outcome": fields.get("outcome", "succeeded"),
    }
    body = {key: value for key, value in body.items() if value is not None}
    return client.post(receipt_url(machine_id, grant_id), json=body)


def receipt_content_hash(record, machine_id, event_id):
    document = json.dumps(
        {
            "id": record["id"],
            "machine_id": machine_id,
            "grant_id": record["grant_id"],
            "use_id": record["use_id"],
            "event_id": event_id,
            "evidence_id": record["evidence_id"],
            "action_type": record["action_type"],
            "resource": record["resource"],
            "outcome": record["outcome"],
            "matches_authorization": record["matches_authorization"],
            "created_at": record["created_at"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash_of(previous_chain_hash, content_hash):
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


def parse_z(value):
    assert value.endswith("Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# Success shape, comparison, and chaining
# --------------------------------------------------------------------------- #


def test_create_matching_receipt_success(consumed_grant, client):
    machine_id, event, grant, use, evidence = consumed_grant
    response = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id=evidence["id"],
    )
    assert response.status_code == 201
    record = response.json()
    assert list(record.keys()) == RECEIPT_FIELDS
    assert record["grant_id"] == grant["id"]
    assert record["use_id"] == use["use_id"]
    assert record["evidence_id"] == evidence["id"]
    assert record["action_type"] == "read"
    assert record["resource"] == "res/x"
    assert record["outcome"] == "succeeded"
    assert record["matches_authorization"] is True
    parse_z(record["created_at"])  # RFC 3339 UTC Z
    assert record["previous_receipt_id"] is None
    assert record["content_hash"] == receipt_content_hash(
        record, machine_id, event["id"]
    )
    assert record["chain_hash"] == chain_hash_of("", record["content_hash"])


def test_mismatched_action_and_resource_are_kept_verbatim(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    response = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id=evidence["id"],
        action_type="write",
        resource="res/other",
        outcome="failed",
    )
    assert response.status_code == 201
    record = response.json()
    # The real values are preserved; only the flag reports the divergence.
    assert record["action_type"] == "write"
    assert record["resource"] == "res/other"
    assert record["outcome"] == "failed"
    assert record["matches_authorization"] is False
    # The digest covers the true values, so it verifies over the mismatch.
    assert record["content_hash"] == receipt_content_hash(
        record, machine_id, grant_event_id(client, grant)
    )


def grant_event_id(client, grant):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text("SELECT event_id FROM authorization_grants WHERE id = :id"),
            {"id": grant["id"]},
        ).scalar_one()


def test_either_field_diverging_is_a_mismatch(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    only_action = post_receipt(
        client, machine_id, "nonexistent", evidence_id=evidence["id"]
    )
    assert only_action.status_code == 404

    # A second machine+grant with its own event/evidence for the resource case.
    second_machine = create_machine(client, external_id="machine-2")
    declare(client, second_machine)
    create_rule(client, priority=1)
    second_event = record_event(client, second_machine).json()
    second_grant = issue(client, second_machine, second_event["id"]).json()
    consume(client, second_machine, second_grant["id"])
    second_evidence = add_evidence(
        client, second_machine, second_event["id"], FINGERPRINT_TWO
    ).json()

    action_mismatch = post_receipt(
        client,
        second_machine,
        second_grant["id"],
        evidence_id=second_evidence["id"],
        action_type="write",
    ).json()
    assert action_mismatch["matches_authorization"] is False
    assert action_mismatch["resource"] == "res/x"

    third_event = record_event(
        client, second_machine, resource="res/z"
    ).json()
    third_grant = issue(client, second_machine, third_event["id"]).json()
    consume(client, second_machine, third_grant["id"])
    third_evidence = add_evidence(
        client, second_machine, third_event["id"], "c" * 64
    ).json()
    resource_mismatch = post_receipt(
        client,
        second_machine,
        third_grant["id"],
        evidence_id=third_evidence["id"],
        resource="res/different",
    ).json()
    assert resource_mismatch["matches_authorization"] is False
    assert resource_mismatch["action_type"] == "read"


def test_receipts_chain_per_machine_across_grants(consumed_grant, client):
    machine_id, event_one, grant_one, _, evidence_one = consumed_grant
    first = post_receipt(
        client, machine_id, grant_one["id"], evidence_id=evidence_one["id"]
    ).json()

    event_two = record_event(client, machine_id, resource="res/y").json()
    grant_two = issue(client, machine_id, event_two["id"]).json()
    use_two = consume(client, machine_id, grant_two["id"]).json()
    evidence_two = add_evidence(
        client, machine_id, event_two["id"], FINGERPRINT_TWO
    ).json()
    second = post_receipt(
        client,
        machine_id,
        grant_two["id"],
        evidence_id=evidence_two["id"],
        resource="res/y",
    ).json()

    assert second["use_id"] == use_two["use_id"]
    assert second["previous_receipt_id"] == first["id"]
    assert second["content_hash"] == receipt_content_hash(
        second, machine_id, event_two["id"]
    )
    assert second["chain_hash"] == chain_hash_of(
        first["chain_hash"], second["content_hash"]
    )

    integrity = client.get(integrity_url(machine_id)).json()
    assert integrity == {
        "valid": True,
        "checked_count": 2,
        "broken_receipt_id": None,
    }


# --------------------------------------------------------------------------- #
# Single-receipt GET
# --------------------------------------------------------------------------- #


def test_get_returns_the_unique_receipt(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    created = post_receipt(
        client, machine_id, grant["id"], evidence_id=evidence["id"]
    ).json()

    response = client.get(receipt_url(machine_id, grant["id"]))
    assert response.status_code == 200
    assert response.json() == created
    assert list(response.json().keys()) == RECEIPT_FIELDS


def test_get_missing_receipt_is_404(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    # Consumed grant without a receipt.
    response = client.get(receipt_url(machine_id, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "receipt_not_found"}}
    # A missing machine or grant is the same receipt-not-founded outcome.
    assert client.get(
        receipt_url("missing-machine", grant["id"])
    ).status_code == 404
    assert client.get(
        receipt_url(machine_id, "missing-grant")
    ).json() == {"error": {"code": "receipt_not_found"}}


def test_get_rejects_query_and_body(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    post_receipt(client, machine_id, grant["id"], evidence_id=evidence["id"])

    response = client.get(
        receipt_url(machine_id, grant["id"]), params={"x": "1"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.request(
        "GET",
        receipt_url(machine_id, grant["id"]),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    # Validation wins over the lookup, even against a non-existent machine.
    assert client.get(
        receipt_url("missing-machine", grant["id"]), params={"x": "1"}
    ).status_code == 422


# --------------------------------------------------------------------------- #
# Lookup and conflict outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_grant_and_cross_machine_are_404(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()
    consume(client, machine_id, grant["id"])
    evidence = add_evidence(
        client, machine_id, event["id"], FINGERPRINT_ONE
    ).json()

    assert post_receipt(
        client, "missing-machine", grant["id"], evidence_id=evidence["id"]
    ).status_code == 404
    assert post_receipt(
        client, machine_id, "missing-grant", evidence_id=evidence["id"]
    ).status_code == 404

    other_id = create_machine(client, external_id="machine-2")
    response = post_receipt(
        client, other_id, grant["id"], evidence_id=evidence["id"]
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_evidence_must_belong_to_the_grants_event(consumed_grant, client):
    machine_id, event, grant, _, evidence = consumed_grant

    # Evidence on a different event of the same machine.
    other_event = record_event(client, machine_id, resource="res/y").json()
    other_evidence = add_evidence(
        client, machine_id, other_event["id"], FINGERPRINT_TWO
    ).json()
    response = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id=other_evidence["id"],
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}

    # Evidence of another machine.
    other_machine = create_machine(client, external_id="machine-2")
    declare(client, other_machine)
    create_rule(client, priority=1)
    foreign_event = record_event(client, other_machine).json()
    foreign_evidence = add_evidence(
        client, other_machine, foreign_event["id"], "c" * 64
    ).json()
    response = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id=foreign_evidence["id"],
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}

    # An unknown evidence id.
    response = post_receipt(
        client, machine_id, grant["id"], evidence_id="missing-evidence"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}


def test_unconsumed_active_revoked_and_expired_grants_conflict(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    # Active, never consumed.
    active_event = record_event(client, machine_id).json()
    active_grant = issue(client, machine_id, active_event["id"]).json()
    active_evidence = add_evidence(
        client, machine_id, active_event["id"], FINGERPRINT_ONE
    ).json()
    response = post_receipt(
        client,
        machine_id,
        active_grant["id"],
        evidence_id=active_evidence["id"],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_not_consumed"}}

    # Revoked before consumption.
    revoked_event = record_event(client, machine_id, resource="res/r").json()
    revoked_grant = issue(client, machine_id, revoked_event["id"]).json()
    assert client.post(
        f"/machines/{machine_id}/authorization-grants/"
        f"{revoked_grant['id']}/revoke"
    ).status_code == 200
    revoked_evidence = add_evidence(
        client, machine_id, revoked_event["id"], FINGERPRINT_TWO
    ).json()
    response = post_receipt(
        client,
        machine_id,
        revoked_grant["id"],
        evidence_id=revoked_evidence["id"],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_not_consumed"}}

    # Expired before consumption.
    expired_event = record_event(client, machine_id, resource="res/e").json()
    expired_grant = issue(
        client, machine_id, expired_event["id"], ttl_seconds=1
    ).json()
    expired_evidence = add_evidence(
        client, machine_id, expired_event["id"], "c" * 64
    ).json()
    time.sleep(1.1)
    response = post_receipt(
        client,
        machine_id,
        expired_grant["id"],
        evidence_id=expired_evidence["id"],
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_not_consumed"}}


def test_duplicate_receipt_conflicts_and_writes_nothing(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    first = post_receipt(
        client, machine_id, grant["id"], evidence_id=evidence["id"]
    )
    assert first.status_code == 201

    duplicate = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id=evidence["id"],
        action_type="write",
        outcome="failed",
    )
    assert duplicate.status_code == 409
    assert duplicate.json() == {"error": {"code": "duplicate_receipt"}}

    # The stored receipt is unchanged and there is still exactly one row.
    stored = client.get(receipt_url(machine_id, grant["id"])).json()
    assert stored == first.json()
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_execution_receipts "
                "WHERE grant_id = :id"
            ),
            {"id": grant["id"]},
        ).scalar_one()
    assert count == 1


def test_failed_attempt_does_not_change_the_grant(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    assert post_receipt(
        client, machine_id, grant["id"], evidence_id="missing"
    ).status_code == 404
    # The grant stays consumed; a second consumption is still the terminal
    # 409 and no receipt exists.
    assert consume(client, machine_id, grant["id"]).status_code == 409
    assert client.get(
        receipt_url(machine_id, grant["id"])
    ).status_code == 404


def test_concurrent_submissions_have_one_winner(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    barrier = threading.Barrier(8)

    def submit():
        barrier.wait()
        return post_receipt(
            client,
            machine_id,
            grant["id"],
            evidence_id=evidence["id"],
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: submit(), range(8)))
    assert outcomes.count(201) == 1
    assert outcomes.count(409) == 7


# --------------------------------------------------------------------------- #
# Body and query validation, all before any read
# --------------------------------------------------------------------------- #


def test_query_parameter_is_invalid_query_even_for_missing_machine(client):
    response = client.post(
        receipt_url("missing-machine", "missing-grant") + "?x=1",
        json={
            "evidence_id": "e",
            "action_type": "read",
            "resource": "r",
            "outcome": "succeeded",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_or_extract_fields_are_invalid_receipt(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    url = receipt_url(machine_id, grant["id"])
    base = {
        "evidence_id": evidence["id"],
        "action_type": "read",
        "resource": "res/x",
        "outcome": "succeeded",
    }
    for dropped in base:
        body = {key: value for key, value in base.items() if key != dropped}
        response = client.post(url, json=body)
        assert response.status_code == 422, dropped
        assert response.json() == {"error": {"code": "invalid_receipt"}}

    response = client.post(url, json={**base, "extra": 1})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_receipt"}}

    # Non-object and unparseable bodies share invalid_receipt.
    for malformed in ([], ["a"], "text", 1, None):
        response = client.post(url, json=malformed)
        assert response.status_code == 422, malformed
        assert response.json() == {"error": {"code": "invalid_receipt"}}


def test_wrong_types_empty_values_and_outcome_are_invalid_value(
    consumed_grant, client
):
    machine_id, _, grant, _, evidence = consumed_grant
    url = receipt_url(machine_id, grant["id"])
    base = {
        "evidence_id": evidence["id"],
        "action_type": "read",
        "resource": "res/x",
        "outcome": "succeeded",
    }

    def expect_invalid_value(body):
        response = client.post(url, json=body)
        assert response.status_code == 422, body
        assert response.json() == {"error": {"code": "invalid_value"}}

    for field in ("evidence_id", "action_type", "resource", "outcome"):
        for bad in (1, 1.5, True, False, None, [], {}):
            expect_invalid_value({**base, field: bad})
        expect_invalid_value({**base, field: "   "})
        expect_invalid_value({**base, field: "\t\n"})

    expect_invalid_value({**base, "outcome": "succeeded "})
    expect_invalid_value({**base, "outcome": "SUCCEEDED"})
    expect_invalid_value({**base, "outcome": "denied"})


def test_validation_ordering_shape_before_values_before_reads(client):
    machine_id = create_machine(client)
    url = receipt_url(machine_id, "missing-grant")
    # Well-shaped fields but a bad value, against a grant that does not exist:
    # invalid_value still wins over the 404 lookup.
    response = client.post(
        url,
        json={
            "evidence_id": "e",
            "action_type": "read",
            "resource": "r",
            "outcome": "bogus",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}
    # A missing field and a bad value together is the shape failure.
    response = client.post(
        url,
        json={
            "evidence_id": 1,
            "action_type": "read",
            "resource": "r",
            # outcome missing
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_receipt"}}


# --------------------------------------------------------------------------- #
# Integrity audit
# --------------------------------------------------------------------------- #


def test_integrity_empty_chain(client):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "valid",
        "checked_count",
        "broken_receipt_id",
    ]
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
    }


def test_integrity_reports_tampered_record(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    record = post_receipt(
        client, machine_id, grant["id"], evidence_id=evidence["id"]
    ).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET action_type = 'write' WHERE id = :id"
            ).bindparams(id=record["id"])
        )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 1
    assert body["broken_receipt_id"] == record["id"]


def test_integrity_reports_broken_link(consumed_grant, client):
    machine_id, event_one, grant_one, _, evidence_one = consumed_grant
    first = post_receipt(
        client, machine_id, grant_one["id"], evidence_id=evidence_one["id"]
    ).json()

    event_two = record_event(client, machine_id, resource="res/y").json()
    grant_two = issue(client, machine_id, event_two["id"]).json()
    consume(client, machine_id, grant_two["id"])
    evidence_two = add_evidence(
        client, machine_id, event_two["id"], FINGERPRINT_TWO
    ).json()
    second = post_receipt(
        client,
        machine_id,
        grant_two["id"],
        evidence_id=evidence_two["id"],
        resource="res/y",
    ).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET previous_receipt_id = 'bogus' WHERE id = :id"
            ).bindparams(id=second["id"])
        )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["broken_receipt_id"] == second["id"]
    assert body["broken_receipt_id"] != first["id"]


def test_integrity_isolated_per_machine(consumed_grant, client):
    machine_id, _, grant, _, evidence = consumed_grant
    record = post_receipt(
        client, machine_id, grant["id"], evidence_id=evidence["id"]
    ).json()
    other_id = create_machine(client, external_id="machine-2")

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET content_hash = '0' * 64 WHERE id = :id"
            ).bindparams(id=record["id"])
        )

    assert client.get(integrity_url(other_id)).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
    }
    assert client.get(integrity_url(machine_id)).json()["valid"] is False


def test_integrity_rejects_query_and_body(consumed_grant, client):
    machine_id = consumed_grant[0]
    response = client.get(integrity_url(machine_id), params={"x": "1"})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.request(
        "GET",
        integrity_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    assert client.get(
        integrity_url("missing-machine"), params={"x": "1"}
    ).status_code == 422


def test_integrity_missing_machine_is_404(client):
    response = client.get(integrity_url("missing-machine"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Method routing
# --------------------------------------------------------------------------- #


def test_receipt_path_non_routed_methods_are_405(consumed_grant, client):
    machine_id, _, grant, _, _ = consumed_grant
    for method in ("put", "patch", "delete"):
        response = getattr(client, method)(receipt_url(machine_id, grant["id"]))
        assert response.status_code == 405


def test_integrity_non_get_methods_are_405(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Persistence and migration
# --------------------------------------------------------------------------- #


def test_receipts_survive_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"]).json()
        consume(first, machine_id, grant["id"])
        evidence = add_evidence(
            first, machine_id, event["id"], FINGERPRINT_ONE
        ).json()
        receipt = post_receipt(
            first, machine_id, grant["id"], evidence_id=evidence["id"]
        ).json()

    with TestClient(app) as second:
        assert second.get(
            receipt_url(machine_id, grant["id"])
        ).json() == receipt
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
        }


def test_old_database_gets_table_without_touching_old_data(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        grant = issue(first, machine_id, event["id"]).json()
        consume(first, machine_id, grant["id"])

    # Reproduce a database that predates the receipt feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_grant_execution_receipts")
    grant_rows = conn.execute(
        "SELECT id, machine_id, event_id, issued_at, expires_at, status, "
        "consumed_at, revoked_at FROM authorization_grants"
    ).fetchall()
    use_rows = conn.execute(
        "SELECT id, grant_id, machine_id, event_id, consumed_at "
        "FROM authorization_grant_uses"
    ).fetchall()
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # The table is recreated empty; no receipts are fabricated and the
        # old rows are untouched.
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 0,
            "broken_receipt_id": None,
        }
        with second.app.state.engine.connect() as conn:
            assert [tuple(r) for r in conn.execute(text(
                "SELECT id, machine_id, event_id, issued_at, expires_at, "
                "status, consumed_at, revoked_at FROM authorization_grants"
            ))] == grant_rows
            assert [tuple(r) for r in conn.execute(text(
                "SELECT id, grant_id, machine_id, event_id, consumed_at "
                "FROM authorization_grant_uses"
            ))] == use_rows

        # New receipts on the migrated database chain normally.
        evidence = add_evidence(
            second, machine_id, event["id"], FINGERPRINT_ONE
        ).json()
        response = post_receipt(
            second, machine_id, grant["id"], evidence_id=evidence["id"]
        )
        assert response.status_code == 201
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
        }
