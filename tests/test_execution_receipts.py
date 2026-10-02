"""Tests for immutable execution receipts bound to consumed grants.

Two entries sit under the machine grant path and one read-only integrity
entry under the machine path:

    POST /machines/{machine_id}/authorization-grants/{grant_id}/execution-receipts
    GET  /machines/{machine_id}/authorization-grants/{grant_id}/execution-receipts
    GET  /machines/{machine_id}/execution-receipts/integrity

A receipt binds a grant's actual execution to its single consumption record:
at most one receipt per consumed grant, it records the cited evidence, the
action actually performed and its outcome, and the matches_authorization
verdict against the original event. These tests cover the success shape,
match and mismatch, failed outcomes, the validation precedence
(invalid_query -> invalid_receipt -> invalid_value -> reads), every lookup
and conflict outcome, the unique GET, the per-machine integrity chain
(empty/sound/broken), exactly-once concurrency, machine isolation, restart
persistence, safe table creation on an old database, non-interference with
the existing grant views, and 405 for other methods.
"""
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

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


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def receipts_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/"
        f"{grant_id}/execution-receipts"
    )


def integrity_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/integrity"


def add_evidence(client, machine_id, event_id, fingerprint=None,
                 evidence_type="log"):
    if fingerprint is None:
        fingerprint = "a" * 64
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/evidence",
        json={"evidence_type": evidence_type, "content_hash": fingerprint},
    )
    assert response.status_code == 201
    return response.json()


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _fingerprint(seed):
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


RECEIPT_FIELD_ORDER = [
    "evidence_id",
    "action_type",
    "resource",
    "outcome",
    "id",
    "grant_id",
    "use_id",
    "matches_authorization",
    "created_at",
    "previous_receipt_id",
    "content_hash",
    "chain_hash",
]


@pytest.fixture
def consumed_grant(client):
    """Machine + allowed event + evidence + issued and consumed grant."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    evidence = add_evidence(client, machine_id, event["id"])
    grant = issue(client, machine_id, event["id"]).json()
    use = consume(client, machine_id, grant["id"]).json()
    return machine_id, event, evidence, grant, use


def post_receipt(client, machine_id, grant_id, **overrides):
    payload = {
        "evidence_id": overrides.get("evidence_id", "ev"),
        "action_type": overrides.get("action_type", "read"),
        "resource": overrides.get("resource", "res/x"),
        "outcome": overrides.get("outcome", "succeeded"),
    }
    return client.post(receipts_url(machine_id, grant_id), json=payload)


def matching_payload(evidence, **overrides):
    payload = {
        "evidence_id": evidence["id"],
        "action_type": "read",
        "resource": "res/x",
        "outcome": "succeeded",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Success shape and matching
# --------------------------------------------------------------------------- #


def test_receipt_success_shape_and_match(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    before = datetime.now(timezone.utc)
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    after = datetime.now(timezone.utc)

    assert response.status_code == 201
    body = response.json()
    assert list(body.keys()) == RECEIPT_FIELD_ORDER
    assert body["evidence_id"] == evidence["id"]
    assert body["action_type"] == "read"
    assert body["resource"] == "res/x"
    assert body["outcome"] == "succeeded"
    assert body["grant_id"] == grant["id"]
    assert body["use_id"] == use["use_id"]
    assert body["matches_authorization"] is True
    assert body["previous_receipt_id"] is None
    assert isinstance(body["id"], str) and body["id"]
    created = _parse_z(body["created_at"])
    assert body["created_at"].endswith("Z")
    assert created.tzinfo == timezone.utc
    assert before <= created <= after
    for digest_name in ("content_hash", "chain_hash"):
        assert isinstance(body[digest_name], str)
        assert len(body[digest_name]) == 64
        int(body[digest_name], 16)


def test_hashes_follow_the_published_rules(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    body = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    ).json()

    document = json.dumps(
        {
            "id": body["id"],
            "machine_id": machine_id,
            "grant_id": body["grant_id"],
            "use_id": body["use_id"],
            "authorization_event_id": event["id"],
            "evidence_id": body["evidence_id"],
            "action_type": body["action_type"],
            "resource": body["resource"],
            "outcome": body["outcome"],
            "matches_authorization": body["matches_authorization"],
            "created_at": body["created_at"],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    expected_content = hashlib.sha256(document.encode("utf-8")).hexdigest()
    assert body["content_hash"] == expected_content
    # The first receipt chains against the empty prefix.
    expected_chain = hashlib.sha256(
        f":{expected_content}".encode("utf-8")
    ).hexdigest()
    assert body["chain_hash"] == expected_chain


def test_failed_outcome_still_matches(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence, outcome="failed"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["outcome"] == "failed"
    assert body["matches_authorization"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"action_type": "write"},
        {"resource": "res/y"},
        {"action_type": "write", "resource": "res/y"},
    ],
)
def test_action_mismatch_stores_real_values_and_false(
    consumed_grant, client, change
):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence, **change),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["matches_authorization"] is False
    # The real, non-authorized values are persisted, not coerced.
    assert body["action_type"] == change.get("action_type", "read")
    assert body["resource"] == change.get("resource", "res/x")


def test_text_fields_are_trimmed_for_comparison(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(
            evidence, action_type="  read  ", resource=" res/x ",
            evidence_id=f"  {evidence['id']}  ",
        ),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["action_type"] == "read"
    assert body["resource"] == "res/x"
    assert body["evidence_id"] == evidence["id"]
    assert body["matches_authorization"] is True


def test_receipt_persists(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    body = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    ).json()
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, machine_id, grant_id, use_id, "
                "authorization_event_id, evidence_id, action_type, resource, "
                "outcome, matches_authorization, previous_receipt_id, "
                "content_hash, chain_hash FROM "
                "authorization_grant_execution_receipts WHERE id = :id"
            ).bindparams(id=body["id"])
        ).one()
    assert row.machine_id == machine_id
    assert row.grant_id == grant["id"]
    assert row.use_id == use["use_id"]
    assert row.authorization_event_id == event["id"]
    assert row.evidence_id == evidence["id"]
    assert row.outcome == "succeeded"
    assert row.matches_authorization == 1
    assert row.previous_receipt_id is None
    assert row.content_hash == body["content_hash"]
    assert row.chain_hash == body["chain_hash"]


def test_grant_status_unchanged_by_receipt(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    assert client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    ).status_code == 201
    with client.app.state.engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert status == "consumed"


# --------------------------------------------------------------------------- #
# Query validation takes precedence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["?x=1", "?outcome=succeeded", "?=", "?foo"])
def test_query_parameter_is_invalid_query_before_everything(
    consumed_grant, client, query
):
    machine_id, event, evidence, grant, use = consumed_grant
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(
        receipts_url(machine_id, grant["id"]) + query,
        json=matching_payload(evidence),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Even an invalid body with a query is still invalid_query.
    response = client.post(
        receipts_url(machine_id, grant["id"]) + query, json={"bad": 1}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Query validation precedes every lookup.
    response = client.post(
        receipts_url(missing_machine, "anything") + query,
        json=matching_payload(evidence),
    )
    assert response.status_code == 422


def test_repeated_query_parameter_is_invalid_query(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]) + "?x=1&x=2",
        json=matching_payload(evidence),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Body shape: invalid_receipt
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_non_object_body_is_invalid_receipt(
    consumed_grant, client, raw
):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_receipt"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"evidence_id": "e", "action_type": "read", "resource": "r"},
        {"action_type": "read", "resource": "r", "outcome": "succeeded"},
        {"evidence_id": "e", "resource": "r", "outcome": "succeeded"},
        {"evidence_id": "e", "action_type": "read", "outcome": "succeeded"},
        {
            "evidence_id": "e",
            "action_type": "read",
            "resource": "r",
            "outcome": "succeeded",
            "extra": 1,
        },
    ],
)
def test_missing_or_extra_field_is_invalid_receipt(
    consumed_grant, client, payload
):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(receipts_url(machine_id, grant["id"]), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_receipt"}}


def test_shape_check_runs_before_value_check(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    # All four fields present, so it passes the shape check, but the values
    # are illegal; a missing field plus illegal value is invalid_receipt.
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json={
            "evidence_id": None,
            "action_type": "read",
            "resource": "r",
            "outcome": "nope",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}

    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json={"evidence_id": None, "action_type": "read", "resource": "r"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_receipt"}}


def test_body_validation_runs_before_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        receipts_url(missing_machine, "anything"),
        json={
            "evidence_id": {},
            "action_type": "read",
            "resource": "r",
            "outcome": "succeeded",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_value"
    response = client.post(
        receipts_url(missing_machine, "anything"),
        json={"nope": 1},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_receipt"


# --------------------------------------------------------------------------- #
# Field values: invalid_value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,bad",
    [
        ("evidence_id", None),
        ("evidence_id", 123),
        ("evidence_id", True),
        ("evidence_id", []),
        ("evidence_id", {}),
        ("evidence_id", ""),
        ("evidence_id", "   "),
        ("action_type", None),
        ("action_type", 1),
        ("action_type", False),
        ("action_type", ["read"]),
        ("action_type", ""),
        ("action_type", "\t\n"),
        ("resource", None),
        ("resource", 4.5),
        ("resource", True),
        ("resource", ""),
        ("resource", " "),
        ("outcome", None),
        ("outcome", 1),
        ("outcome", True),
        ("outcome", []),
        ("outcome", ""),
        ("outcome", "succeeded "),
        ("outcome", "SUCCEEDED"),
        ("outcome", "done"),
    ],
)
def test_bad_field_value_is_invalid_value(consumed_grant, client, field, bad):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence, **{field: bad}),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_invalid_value_does_not_read_or_write(client):
    # Even against a non-existent machine/grant, value validation stays 422.
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        receipts_url(missing_machine, "anything"),
        json={
            "evidence_id": "e",
            "action_type": "read",
            "resource": "r",
            "outcome": "bogus",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


# --------------------------------------------------------------------------- #
# Lookup and conflict outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client, consumed_grant):
    _, _, evidence, _, _ = consumed_grant
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        receipts_url(missing_machine, "anything"),
        json=matching_payload(evidence),
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_grant_is_not_found(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.post(
        receipts_url(machine_id, "no-such-grant"),
        json=matching_payload(evidence),
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_grant_owned_by_another_machine_is_not_found(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    other = create_machine(client, external_id="machine-2")
    response = client.post(
        receipts_url(other, grant["id"]),
        json=matching_payload(evidence),
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_unconsumed_grant_is_grant_not_consumed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    evidence = add_evidence(client, machine_id, event["id"])
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_not_consumed"}}
    # Nothing was written.
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_execution_receipts"
            )
        ).scalar_one()
    assert count == 0


def test_revoked_grant_is_grant_not_consumed(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    evidence = add_evidence(client, machine_id, event["id"])
    grant = issue(client, machine_id, event["id"]).json()
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 200

    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_not_consumed"}}


def test_missing_evidence_is_evidence_not_found(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = post_receipt(
        client,
        machine_id,
        grant["id"],
        evidence_id="00000000-0000-0000-0000-000000000000",
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}


def test_evidence_of_another_event_is_evidence_not_found(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    first = record_event(client, machine_id, resource="res/1").json()
    second = record_event(client, machine_id, resource="res/2").json()
    # Evidence attached to the FIRST event only.
    evidence = add_evidence(
        client, machine_id, first["id"], fingerprint=_fingerprint("first")
    )
    grant = issue(client, machine_id, second["id"]).json()
    consume(client, machine_id, grant["id"])

    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json={
            "evidence_id": evidence["id"],
            "action_type": "read",
            "resource": "res/2",
            "outcome": "succeeded",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}


def test_evidence_of_another_machine_is_evidence_not_found(client):
    first_machine = create_machine(client, external_id="machine-1")
    declare(client, first_machine)
    create_rule(client)
    foreign_event = record_event(client, first_machine).json()
    foreign_evidence = add_evidence(
        client, first_machine, foreign_event["id"],
        fingerprint=_fingerprint("foreign"),
    )

    second_machine = create_machine(client, external_id="machine-2")
    declare(client, second_machine)
    own_event = record_event(client, second_machine).json()
    add_evidence(
        client, second_machine, own_event["id"],
        fingerprint=_fingerprint("own"),
    )
    grant = issue(client, second_machine, own_event["id"]).json()
    consume(client, second_machine, grant["id"])

    response = client.post(
        receipts_url(second_machine, grant["id"]),
        json={
            "evidence_id": foreign_evidence["id"],
            "action_type": "read",
            "resource": "res/x",
            "outcome": "succeeded",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}


def test_consumed_grant_without_use_record_is_not_found(
    consumed_grant, client
):
    machine_id, event, evidence, grant, use = consumed_grant
    # Simulate the damaged state: consumed status, but the consumption
    # record is gone.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("DELETE FROM authorization_grant_uses WHERE grant_id = :id")
            .bindparams(id=grant["id"])
        )
    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_execution_receipts"
            )
        ).scalar_one()
    assert count == 0


def test_unconsumed_grant_with_bad_evidence_is_evidence_not_found(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    grant = issue(client, machine_id, event["id"]).json()

    response = client.post(
        receipts_url(machine_id, grant["id"]),
        json={
            "evidence_id": "00000000-0000-0000-0000-000000000000",
            "action_type": "read",
            "resource": "res/x",
            "outcome": "succeeded",
        },
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "evidence_not_found"}}


def test_duplicate_receipt_is_conflict_and_unchanged(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    first = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    assert first.status_code == 201

    second = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence, action_type="write"),
    )
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "duplicate_receipt"}}

    # The first receipt is unchanged and still the only one.
    fetched = client.get(receipts_url(machine_id, grant["id"])).json()
    assert fetched == first.json()
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_execution_receipts "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).scalar_one()
    assert count == 1


def test_failed_attempts_write_nothing(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    evidence = add_evidence(client, machine_id, event["id"])
    grant = issue(client, machine_id, event["id"]).json()

    # Unconsumed -> conflict, no row.
    assert (
        client.post(
            receipts_url(machine_id, grant["id"]),
            json=matching_payload(evidence),
        ).status_code
        == 409
    )
    # Consume it, then cite missing evidence -> 404, no row.
    consume(client, machine_id, grant["id"])
    assert (
        post_receipt(
            client, machine_id, grant["id"], evidence_id="missing"
        ).status_code
        == 404
    )
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_grant_execution_receipts"
            )
        ).scalar_one()
    assert count == 0


# --------------------------------------------------------------------------- #
# Concurrency: exactly one winner
# --------------------------------------------------------------------------- #


def test_concurrent_receipts_have_exactly_one_winner(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    count = 20
    gate = threading.Event()

    def hit():
        gate.wait()
        return client.post(
            receipts_url(machine_id, grant["id"]),
            json=matching_payload(evidence),
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit) for _ in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == count - 1
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "duplicate_receipt"}}
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id FROM authorization_grant_execution_receipts "
                "WHERE grant_id = :id"
            ).bindparams(id=grant["id"])
        ).all()
    assert len(rows) == 1


# --------------------------------------------------------------------------- #
# Unique GET
# --------------------------------------------------------------------------- #


def test_get_returns_the_receipt(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    created = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    response = client.get(receipts_url(machine_id, grant["id"]))
    assert response.status_code == 200
    assert response.json() == created.json()
    assert list(response.json().keys()) == RECEIPT_FIELD_ORDER


def test_get_is_byte_identical_on_repeat(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    first = client.get(receipts_url(machine_id, grant["id"])).content
    second = client.get(receipts_url(machine_id, grant["id"])).content
    assert first == second
    assert first.endswith(b"\n")


@pytest.mark.parametrize(
    "setup",
    [
        "missing_machine",
        "missing_grant",
        "unconsumed_grant",
        "consumed_without_receipt",
    ],
)
def test_get_without_receipt_is_receipt_not_found(client, setup):
    if setup == "missing_machine":
        response = client.get(
            receipts_url(
                "00000000-0000-0000-0000-000000000000", "anything"
            )
        )
    else:
        machine_id = create_machine(client)
        declare(client, machine_id)
        create_rule(client)
        event = record_event(client, machine_id).json()
        if setup == "missing_grant":
            response = client.get(receipts_url(machine_id, "no-such-grant"))
        else:
            grant = issue(client, machine_id, event["id"]).json()
            if setup == "unconsumed_grant":
                response = client.get(receipts_url(machine_id, grant["id"]))
            else:
                consume(client, machine_id, grant["id"])
                response = client.get(receipts_url(machine_id, grant["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "receipt_not_found"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?foo"])
def test_get_with_query_is_invalid_query(consumed_grant, client, query):
    machine_id, event, evidence, grant, use = consumed_grant
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.get(receipts_url(machine_id, grant["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.get(receipts_url(missing_machine, "x") + query)
    assert response.status_code == 422


def test_get_with_body_is_invalid_query(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    response = client.request(
        "GET",
        receipts_url(machine_id, grant["id"]),
        content=b'{"x":1}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Integrity chain
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


def test_integrity_missing_machine_is_not_found(client):
    response = client.get(
        integrity_url("00000000-0000-0000-0000-000000000000")
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?valid=true"])
def test_integrity_with_query_is_invalid_query(client, query):
    machine_id = create_machine(client)
    assert client.get(
        integrity_url(machine_id) + query
    ).status_code == 422
    assert (
        client.get(
            integrity_url("00000000-0000-0000-0000-000000000000") + query
        ).status_code
        == 422
    )


def _consume_grant_for_event(client, machine_id, event_id):
    grant = issue(client, machine_id, event_id).json()
    use = consume(client, machine_id, grant["id"]).json()
    return grant, use


def test_integrity_sound_multi_receipt_chain(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    events = [record_event(client, machine_id, resource=f"res/{i}").json()
              for i in range(3)]
    receipt_bodies = []
    previous_id = None
    for index, event in enumerate(events):
        evidence = add_evidence(
            client, machine_id, event["id"], fingerprint=_fingerprint(str(index))
        )
        grant, use = _consume_grant_for_event(client, machine_id, event["id"])
        body = client.post(
            receipts_url(machine_id, grant["id"]),
            json={
                "evidence_id": evidence["id"],
                "action_type": "read",
                "resource": f"res/{index}",
                "outcome": "succeeded" if index % 2 == 0 else "failed",
            },
        ).json()
        assert body["previous_receipt_id"] == previous_id
        previous_id = body["id"]
        receipt_bodies.append(body)

    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 3,
        "broken_receipt_id": None,
    }

    # Chain hashes link back through the empty prefix.
    for index, body in enumerate(receipt_bodies):
        if index == 0:
            expected_previous_chain = ""
        else:
            expected_previous_chain = receipt_bodies[index - 1]["chain_hash"]
        expected = hashlib.sha256(
            f"{expected_previous_chain}:{body['content_hash']}".encode("utf-8")
        ).hexdigest()
        assert body["chain_hash"] == expected


def test_integrity_reports_first_broken_receipt_after_tamper(
    consumed_grant, client
):
    machine_id, event, evidence, grant, use = consumed_grant
    second_event = record_event(client, machine_id, resource="res/z").json()
    second_evidence = add_evidence(
        client, machine_id, second_event["id"], fingerprint=_fingerprint("z")
    )
    first = client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    ).json()
    second_grant, _ = _consume_grant_for_event(
        client, machine_id, second_event["id"]
    )
    second = client.post(
        receipts_url(machine_id, second_grant["id"]),
        json={
            "evidence_id": second_evidence["id"],
            "action_type": "read",
            "resource": "res/z",
            "outcome": "succeeded",
        },
    ).json()

    # Tamper the FIRST receipt's content: the audit reports the first broken
    # id, even though the second record's stored link still names it.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET resource = :value WHERE id = :id"
            ).bindparams(value="res/TAMPERED", id=first["id"])
        )

    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is False
    assert audit["checked_count"] == 2
    assert audit["broken_receipt_id"] == first["id"]

    # A broken chain link (successor no longer names the real predecessor in
    # the stored value) reports the successor as broken.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET resource = :value WHERE id = :id"
            ).bindparams(value="res/x", id=first["id"])
        )
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET previous_receipt_id = :value WHERE id = :id"
            ).bindparams(value="00000000-0000-0000-0000-000000000000",
                         id=second["id"])
        )
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is False
    assert audit["broken_receipt_id"] == second["id"]


def test_integrity_is_read_only(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )
    before = client.get(integrity_url(machine_id)).content
    with client.app.state.engine.connect() as conn:
        snapshot = conn.execute(
            text(
                "SELECT * FROM authorization_grant_execution_receipts "
                "ORDER BY id"
            )
        ).all()
    for _ in range(3):
        assert client.get(integrity_url(machine_id)).content == before
    with client.app.state.engine.connect() as conn:
        after = conn.execute(
            text(
                "SELECT * FROM authorization_grant_execution_receipts "
                "ORDER BY id"
            )
        ).all()
    assert after == snapshot


def test_integrity_is_machine_isolated(client):
    first = create_machine(client, external_id="machine-1")
    declare(client, first)
    create_rule(client)
    first_event = record_event(client, first).json()
    first_evidence = add_evidence(
        client, first, first_event["id"], fingerprint=_fingerprint("1")
    )
    first_grant, _ = _consume_grant_for_event(client, first, first_event["id"])
    client.post(
        receipts_url(first, first_grant["id"]),
        json={
            "evidence_id": first_evidence["id"],
            "action_type": "read",
            "resource": "res/x",
            "outcome": "succeeded",
        },
    )

    second = create_machine(client, external_id="machine-2")
    # The second machine has no receipts: its own chain stays empty.
    audit = client.get(integrity_url(second)).json()
    assert audit == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
    }
    # Tampering the first machine's receipt never shows in the second's audit.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grant_execution_receipts "
                "SET chain_hash = :value"
            ).bindparams(value="0" * 64)
        )
    assert client.get(integrity_url(second)).json()["valid"] is True


# --------------------------------------------------------------------------- #
# 405 for other methods
# --------------------------------------------------------------------------- #


def test_other_methods_are_405(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    receipt_path = receipts_url(machine_id, grant["id"])
    integrity_path = integrity_url(machine_id)
    for method in ("put", "patch", "delete"):
        assert getattr(client, method)(receipt_path).status_code == 405
        assert getattr(client, method)(integrity_path).status_code == 405
    # GET-only entries reject HEAD; the receipt collection also rejects it.
    for path in (receipt_path, integrity_path):
        assert client.head(path).status_code == 405
    # POST is defined only for the per-grant collection, not the integrity
    # entry or the GET view.
    assert client.post(integrity_path, content=b"").status_code == 405
    # The 405s wrote nothing: the receipt can still be created once.
    assert client.post(
        receipt_path, json=matching_payload(evidence)
    ).status_code == 201
    # And after it exists, non-routed methods still 405 and do not change it.
    before = client.get(receipt_path).content
    for method in ("put", "patch", "delete", "head"):
        assert getattr(client, method)(receipt_path).status_code == 405
    assert client.get(receipt_path).content == before


# --------------------------------------------------------------------------- #
# Restart persistence and old-database migration
# --------------------------------------------------------------------------- #


def test_receipt_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        evidence = add_evidence(first, machine_id, event["id"])
        grant, use = _consume_grant_for_event(first, machine_id, event["id"])
        created = first.post(
            receipts_url(machine_id, grant["id"]),
            json={
                "evidence_id": evidence["id"],
                "action_type": "write",
                "resource": "res/x",
                "outcome": "failed",
            },
        ).json()
        assert created["matches_authorization"] is False

    with TestClient(app) as second:
        fetched = second.get(receipts_url(machine_id, grant["id"]))
        assert fetched.status_code == 200
        assert fetched.json() == created
        # Still exactly one receipt: a repeat after restart is a duplicate.
        repeat = second.post(
            receipts_url(machine_id, grant["id"]),
            json={
                "evidence_id": evidence["id"],
                "action_type": "write",
                "resource": "res/x",
                "outcome": "failed",
            },
        )
        assert repeat.status_code == 409
        assert repeat.json() == {"error": {"code": "duplicate_receipt"}}
        # The chain still verifies after restart.
        audit = second.get(integrity_url(machine_id)).json()
        assert audit == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
        }


def test_old_database_creates_receipt_table_safely(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        evidence = add_evidence(first, machine_id, event["id"])
        grant, use = _consume_grant_for_event(first, machine_id, event["id"])

    # Reproduce a database that predates the receipt feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_grant_execution_receipts")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # Empty chain for the legacy database.
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 0,
            "broken_receipt_id": None,
        }
        # The receipt feature works on the recreated table.
        response = second.post(
            receipts_url(machine_id, grant["id"]),
            json={
                "evidence_id": evidence["id"],
                "action_type": "read",
                "resource": "res/x",
                "outcome": "succeeded",
            },
        )
        assert response.status_code == 201
        assert second.get(integrity_url(machine_id)).json()["valid"] is True


# --------------------------------------------------------------------------- #
# Non-interference with existing views
# --------------------------------------------------------------------------- #


def test_receipts_never_change_existing_grant_views(consumed_grant, client):
    machine_id, event, evidence, grant, use = consumed_grant
    grants_url = f"/machines/{machine_id}/authorization-grants"
    lifecycle_url = (
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/integrity"
    )
    grants_before = client.get(grants_url).content
    lifecycle_before = client.get(lifecycle_url).content
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).content

    client.post(
        receipts_url(machine_id, grant["id"]),
        json=matching_payload(evidence),
    )

    assert client.get(grants_url).content == grants_before
    assert client.get(lifecycle_url).content == lifecycle_before
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).content
        == events_before
    )
    # The grant cannot be consumed or revoked again.
    assert (
        consume(client, machine_id, grant["id"]).status_code == 409
    )
    assert client.post(revoke_url(machine_id, grant["id"])).status_code == 409


def test_independent_grants_have_independent_receipts(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    first_event = record_event(client, machine_id, resource="res/1").json()
    second_event = record_event(client, machine_id, resource="res/2").json()
    first_evidence = add_evidence(
        client, machine_id, first_event["id"], fingerprint=_fingerprint("1")
    )
    second_evidence = add_evidence(
        client, machine_id, second_event["id"], fingerprint=_fingerprint("2")
    )
    first_grant, _ = _consume_grant_for_event(client, machine_id, first_event["id"])
    second_grant, _ = _consume_grant_for_event(
        client, machine_id, second_event["id"]
    )

    first = client.post(
        receipts_url(machine_id, first_grant["id"]),
        json={
            "evidence_id": first_evidence["id"],
            "action_type": "read",
            "resource": "res/1",
            "outcome": "succeeded",
        },
    )
    second = client.post(
        receipts_url(machine_id, second_grant["id"]),
        json={
            "evidence_id": second_evidence["id"],
            "action_type": "read",
            "resource": "res/2",
            "outcome": "failed",
        },
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["previous_receipt_id"] == first.json()["id"]
    assert (
        client.get(receipts_url(machine_id, first_grant["id"])).json()["id"]
        == first.json()["id"]
    )
    assert (
        client.get(receipts_url(machine_id, second_grant["id"])).json()["id"]
        == second.json()["id"]
    )
    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 2,
        "broken_receipt_id": None,
    }
