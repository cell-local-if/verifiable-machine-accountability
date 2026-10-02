"""Tests for execution-completion receipts bound to consumed grant uses.

Two entries sit under the machine path:

    POST /machines/{machine_id}/execution-receipts
    GET  /machines/{machine_id}/execution-receipts/integrity

A receipt closes the accountability loop: it binds one already-consumed
authorization-grant use to its execution result (``succeeded``/``failed``
plus a 64 lowercase-hex result digest) and to the machine's per-machine
tamper-evident receipt chain. The use must belong to the path machine, its
source event must be a committed ``allowed_by_policy`` allow, and the
receipt's action/resource must match the source event verbatim. Each use
gets exactly one receipt; the insert and its chain link commit atomically
in one locked write transaction, so a concurrent burst for one use has one
winner and receipts for different uses cannot fork the chain. Old uses are
never backfilled. These tests cover the success shapes, every validation
and lookup outcome, the 409 authorization/scope/conflict outcomes,
exactly-once under concurrency, chain linkage across receipts, every
read-only integrity anomaly category, machine isolation, restart
persistence, old-database migration, and non-interference with the existing
grant and chain behavior.
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


def issue_and_consume(client, machine_id, event):
    grant = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event["id"], "ttl_seconds": 300},
    ).json()
    use = client.post(
        f"/machines/{machine_id}/authorization-grants/"
        f"{grant['id']}/consume"
    ).json()
    return grant, use


def receipts_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts"


def integrity_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/integrity"


def digest_of(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def receipt_payload(use, action_type="read", resource="res/x",
                    outcome="succeeded", digest=None):
    return {
        "use_id": use["use_id"],
        "action_type": action_type,
        "resource": resource,
        "outcome": outcome,
        "result_digest": digest or digest_of(b"result"),
    }


@pytest.fixture
def consumed_use(client):
    """A machine with one consumed allow grant for read/res/x."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    grant, use = issue_and_consume(client, machine_id, event)
    return machine_id, event, grant, use


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# Success shape and chain linkage
# --------------------------------------------------------------------------- #


def test_create_receipt_success_shape(consumed_use, client):
    machine_id, event, grant, use = consumed_use
    digest = digest_of(b"done")

    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, digest=digest),
    )

    assert response.status_code == 201
    body = response.json()
    assert list(body.keys()) == [
        "id",
        "machine_id",
        "use_id",
        "grant_id",
        "authorization_event_id",
        "occurred_at",
        "previous_receipt_id",
        "content_hash",
        "chain_hash",
    ]
    assert isinstance(body["id"], str) and body["id"]
    assert body["machine_id"] == machine_id
    assert body["use_id"] == use["use_id"]
    assert body["grant_id"] == grant["id"]
    assert body["authorization_event_id"] == event["id"]
    assert body["occurred_at"].endswith("Z")
    assert _parse_z(body["occurred_at"]).tzinfo == timezone.utc
    assert body["previous_receipt_id"] is None
    assert body["content_hash"] == digest_of(
        json.dumps(
            {
                "id": body["id"],
                "machine_id": machine_id,
                "use_id": use["use_id"],
                "grant_id": grant["id"],
                "authorization_event_id": event["id"],
                "action_type": "read",
                "resource": "res/x",
                "outcome": "succeeded",
                "result_digest": digest,
                "occurred_at": body["occurred_at"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert body["chain_hash"] == digest_of(
        f":{body['content_hash']}".encode("utf-8")
    )


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
def test_both_outcomes_are_accepted(consumed_use, client, outcome):
    machine_id, _, _, use = consumed_use
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, outcome=outcome),
    )
    assert response.status_code == 201
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT outcome FROM execution_receipts WHERE use_id = :u").bindparams(
                u=use["use_id"]
            )
        ).scalar_one()
    assert stored == outcome


def test_receipts_link_into_one_chain_per_machine(consumed_use, client):
    machine_id, event, _, use = consumed_use
    # Two more consumable uses from two more allow events.
    event_two = record_event(client, machine_id, resource="res/2").json()
    _, use_two = issue_and_consume(client, machine_id, event_two)
    event_three = record_event(client, machine_id, resource="res/3").json()
    _, use_three = issue_and_consume(client, machine_id, event_three)

    first = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    second = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_two, resource="res/2"),
    ).json()
    third = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_three, resource="res/3", outcome="failed"),
    ).json()

    assert first["previous_receipt_id"] is None
    assert second["previous_receipt_id"] == first["id"]
    assert third["previous_receipt_id"] == second["id"]
    assert second["chain_hash"] == digest_of(
        f"{first['chain_hash']}:{second['content_hash']}".encode("utf-8")
    )
    assert third["chain_hash"] == digest_of(
        f"{second['chain_hash']}:{third['content_hash']}".encode("utf-8")
    )


def test_create_receipt_persists_all_bound_fields(consumed_use, client):
    machine_id, event, grant, use = consumed_use
    digest = digest_of(b"persisted")
    body = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, outcome="failed", digest=digest),
    ).json()

    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, machine_id, use_id, grant_id, "
                "authorization_event_id, action_type, resource, outcome, "
                "result_digest, occurred_at FROM execution_receipts WHERE id = :id"
            ).bindparams(id=body["id"])
        ).one()
    assert row.machine_id == machine_id
    assert row.use_id == use["use_id"]
    assert row.grant_id == grant["id"]
    assert row.authorization_event_id == event["id"]
    assert row.action_type == "read"
    assert row.resource == "res/x"
    assert row.outcome == "failed"
    assert row.result_digest == digest
    assert row.occurred_at == body["occurred_at"]


# --------------------------------------------------------------------------- #
# Request validation: body and query
# --------------------------------------------------------------------------- #


def test_wrong_or_missing_fields_are_invalid_request(consumed_use, client):
    machine_id, _, _, use = consumed_use
    base = receipt_payload(use)
    payloads = [
        {k: v for k, v in base.items() if k != "use_id"},
        {k: v for k, v in base.items() if k != "result_digest"},
        {**base, "extra": 1},
        {},
    ]
    for payload in payloads:
        response = client.post(receipts_url(machine_id), json=payload)
        assert response.status_code == 422
        assert response.json() == {
            "error": {"code": "invalid_execution_receipt_request"}
        }


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_typed_body_is_invalid_request(consumed_use, client, raw):
    machine_id, _, _, _ = consumed_use
    response = client.post(
        receipts_url(machine_id),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_id", 123),
        ("use_id", True),
        ("use_id", None),
        ("use_id", ""),
        ("use_id", "   "),
        ("action_type", 5),
        ("action_type", True),
        ("action_type", ""),
        ("action_type", "  "),
        ("resource", 5),
        ("resource", False),
        ("resource", ""),
        ("resource", " "),
        ("outcome", "succeeded "),
        ("outcome", "SUCCEEDED"),
        ("outcome", "allowed"),
        ("outcome", True),
        ("outcome", None),
        ("result_digest", "a" * 63),
        ("result_digest", "a" * 65),
        ("result_digest", "A" * 64),
        ("result_digest", "g" * 64),
        ("result_digest", 123),
        ("result_digest", True),
        ("result_digest", None),
    ],
)
def test_bad_field_values_are_invalid_request(
    consumed_use, client, field, value
):
    machine_id, _, _, use = consumed_use
    payload = receipt_payload(use)
    payload[field] = value
    response = client.post(receipts_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }


@pytest.mark.parametrize("query", ["?x=1", "?use_id=u", "?=", "?outcome=succeeded"])
def test_any_query_parameter_is_invalid_request_before_lookup(
    consumed_use, client, query
):
    machine_id, _, _, use = consumed_use
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(
        receipts_url(machine_id) + query, json=receipt_payload(use)
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }

    # The format check precedes the machine/use lookups.
    response = client.post(
        receipts_url(missing_machine) + query, json=receipt_payload(use)
    )
    assert response.status_code == 422


def test_body_validation_runs_before_machine_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        receipts_url(missing_machine),
        json={
            "use_id": "anything",
            "action_type": "read",
            "resource": "res/x",
            "outcome": "nope",
            "result_digest": "a" * 64,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == (
        "invalid_execution_receipt_request"
    )


def test_action_and_resource_are_not_trimmed(consumed_use, client):
    machine_id, _, _, use = consumed_use
    # Whitespace-padded scope never matches the source event verbatim: it is
    # a well-formed body (the strings are non-blank) but fails the scope
    # check rather than being silently accepted.
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, resource="res/x "),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}


# --------------------------------------------------------------------------- #
# Lookup outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client, consumed_use):
    _, _, _, use = consumed_use
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        receipts_url(missing_machine), json=receipt_payload(use)
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_use_is_not_found(consumed_use, client):
    machine_id, _, _, _ = consumed_use
    payload = receipt_payload({"use_id": "no-such-use"})
    response = client.post(receipts_url(machine_id), json=payload)
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_use_owned_by_another_machine_is_not_found(consumed_use, client):
    machine_id, _, _, use = consumed_use
    other = create_machine(client, external_id="machine-2")

    response = client.post(
        receipts_url(other), json=receipt_payload(use)
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The foreign machine's attempt neither consumed nor receipted the use:
    # the real owner can still record its one receipt.
    assert (
        client.post(
            receipts_url(machine_id), json=receipt_payload(use)
        ).status_code
        == 201
    )


def test_receipt_for_use_of_another_machine_never_enters_its_chain(
    consumed_use, client
):
    machine_id, _, _, _ = consumed_use
    other = create_machine(client, external_id="machine-2")
    # The second machine has an empty, valid receipt chain.
    audit = client.get(integrity_url(other)).json()
    assert audit == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }


# --------------------------------------------------------------------------- #
# Scope and authorization outcomes
# --------------------------------------------------------------------------- #


def test_action_mismatch_is_execution_scope_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, action_type="write"),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}


def test_resource_mismatch_is_execution_scope_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, resource="res/other"),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}


def test_scope_rejection_writes_nothing(consumed_use, client):
    machine_id, _, _, use = consumed_use
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, action_type="write"),
    )
    assert response.status_code == 409
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one()
    assert count == 0
    # The rejected receipt never consumed the use: the correct one still
    # succeeds exactly once.
    repeat = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert repeat.status_code == 201


def test_source_event_that_is_not_allow_is_authorization_not_allowed(client):
    # A denied event can never have a grant through the public API, but the
    # receipt audit must not trust history: fabricate a consumed use bound to
    # a denied event directly and confirm the receipt refuses it.
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=1)
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()
    assert denied["allowed"] is False

    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_grants (id, machine_id, event_id, "
                "issued_at, expires_at, status, consumed_at, revoked_at) "
                "VALUES (:g, :m, :e, :t, :t2, 'consumed', :t, NULL)"
            ).bindparams(
                g="grant-denied",
                m=machine_id,
                e=denied["id"],
                t="2026-01-01T00:00:00Z",
                t2="2026-01-01T00:05:00Z",
            )
        )
        conn.execute(
            text(
                "INSERT INTO authorization_grant_uses (id, grant_id, "
                "machine_id, event_id, consumed_at) VALUES "
                "('use-denied', 'grant-denied', :m, :e, :t)"
            ).bindparams(m=machine_id, e=denied["id"], t="2026-01-01T00:00:00Z")
        )

    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(
            {"use_id": "use-denied"},
            action_type="read",
            resource="res/d",
        ),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "authorization_not_allowed"}}
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one() == 0


# --------------------------------------------------------------------------- #
# One receipt per use, with concurrency
# --------------------------------------------------------------------------- #


def test_second_receipt_for_same_use_is_receipt_already_exists(
    consumed_use, client
):
    machine_id, _, _, use = consumed_use
    assert (
        client.post(
            receipts_url(machine_id), json=receipt_payload(use)
        ).status_code
        == 201
    )
    response = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use, outcome="failed"),
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "receipt_already_exists"}}


def test_concurrent_receipts_for_one_use_have_exactly_one_winner(
    consumed_use, client
):
    machine_id, _, _, use = consumed_use
    count = 20
    gate = threading.Event()

    def hit():
        gate.wait()
        return client.post(
            receipts_url(machine_id), json=receipt_payload(use)
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
            assert response.json() == {
                "error": {"code": "receipt_already_exists"}
            }

    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id FROM execution_receipts WHERE use_id = :u"
            ).bindparams(u=use["use_id"])
        ).all()
    assert len(rows) == 1


def test_concurrent_receipts_for_different_uses_all_commit_and_chain(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    uses = []
    for index in range(8):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        _, use = issue_and_consume(client, machine_id, event)
        uses.append((use["use_id"], f"res/{index}"))
    gate = threading.Event()

    def hit(item):
        use_id, resource = item
        gate.wait()
        return client.post(
            receipts_url(machine_id),
            json=receipt_payload({"use_id": use_id}, resource=resource),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(hit, item) for item in uses]
        gate.set()
        responses = [f.result() for f in futures]

    assert sorted(r.status_code for r in responses) == [201] * len(uses)
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == len(uses)


def test_failed_receipt_leaves_no_half_record(consumed_use, client):
    # The scope-mismatch rejection happens inside the locked transaction and
    # must leave neither a receipt nor a chain entry.
    machine_id, _, _, use = consumed_use
    assert (
        client.post(
            receipts_url(machine_id),
            json=receipt_payload(use, action_type="write"),
        ).status_code
        == 409
    )
    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }


# --------------------------------------------------------------------------- #
# Read-only integrity: shape, ordering, and isolation
# --------------------------------------------------------------------------- #


def test_empty_chain_is_valid(consumed_use, client):
    machine_id, _, _, _ = consumed_use
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.content == (
        b'{"valid":true,"checked_count":0,'
        b'"broken_receipt_id":null,"anomaly":null}\n'
    )


def test_sound_chain_is_valid_and_byte_identical_on_repeat(
    consumed_use, client
):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    first = client.get(integrity_url(machine_id))
    second = client.get(integrity_url(machine_id))
    assert first.content == second.content
    assert first.json() == {
        "valid": True,
        "checked_count": 1,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def test_integrity_missing_machine_is_not_found(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/"
        "execution-receipts/integrity"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?valid=true"])
def test_integrity_rejects_query_before_machine_lookup(client, query):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.get(
        f"/machines/{missing_machine}/execution-receipts/integrity" + query
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_integrity_rejects_a_body(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        integrity_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "method", ["post", "put", "patch", "delete", "head"]
)
def test_integrity_only_accepts_get(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(integrity_url(machine_id))
    assert response.status_code == 405


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete", "head"])
def test_collection_only_accepts_post(consumed_use, client, method):
    machine_id, _, _, _ = consumed_use
    response = getattr(client, method)(receipts_url(machine_id))
    assert response.status_code == 405


def _tamper(engine, receipt_id, column, value):
    with engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE execution_receipts SET {column} = :v WHERE id = :id"
            ).bindparams(v=value, id=receipt_id)
        )


def _first_receipt_id(engine):
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT id FROM execution_receipts ORDER BY occurred_at, id")
        ).scalar_one()


def test_integrity_detects_digest_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "result_digest", "b" * 64)

    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is False
    assert audit["checked_count"] == 1
    assert audit["broken_receipt_id"] == _first_receipt_id(
        client.app.state.engine
    )
    assert audit["anomaly"] == "digest_mismatch"


def test_integrity_detects_illegal_outcome_as_digest_mismatch(
    consumed_use, client
):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "outcome", "cancelled")
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "digest_mismatch"


def test_integrity_detects_scope_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "resource", "res/tampered")
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "scope_mismatch"


def test_integrity_detects_chain_break(consumed_use, client):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "chain_hash", "f" * 64)
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "chain_break"
    assert audit["valid"] is False


def test_integrity_detects_broken_predecessor_link(consumed_use, client):
    machine_id, _, _, use = consumed_use
    first = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    event_two = record_event(client, machine_id, resource="res/2").json()
    _, use_two = issue_and_consume(client, machine_id, event_two)
    second = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_two, resource="res/2"),
    ).json()

    # Point the second receipt at a non-existent predecessor.
    _tamper(client.app.state.engine, second["id"],
            "previous_receipt_id", "0" * 36)
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "chain_break"
    assert audit["broken_receipt_id"] == second["id"]


def test_integrity_detects_ownership_mismatch_on_missing_use(
    consumed_use, client
):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "use_id", "00000000-0000-0000-0000-000000000000")
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "ownership_mismatch"


def test_integrity_detects_use_mismatch_on_cross_binding(client):
    # Two consumed uses on one machine; point one receipt at the *other*
    # use's id (which exists under the path machine, so ownership passes)
    # while keeping its original grant id — the receipt/use binding
    # disagrees.
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    event_one = record_event(client, machine_id, resource="res/1").json()
    _, use_one = issue_and_consume(client, machine_id, event_one)
    event_two = record_event(client, machine_id, resource="res/2").json()
    _, use_two = issue_and_consume(client, machine_id, event_two)

    receipt = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_one, resource="res/1"),
    ).json()
    _tamper(client.app.state.engine, receipt["id"], "use_id", use_two["use_id"])

    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "use_mismatch"
    assert audit["broken_receipt_id"] == receipt["id"]


def test_integrity_detects_timestamp_unparseable_last(consumed_use, client):
    machine_id, _, _, use = consumed_use
    first = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    event_two = record_event(client, machine_id, resource="res/2").json()
    _, use_two = issue_and_consume(client, machine_id, event_two)
    second = client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_two, resource="res/2"),
    ).json()

    # A corrupted stamp sorts after every parseable record, so the sound
    # second receipt verifies first and the corrupted first receipt is
    # reported last.
    _tamper(client.app.state.engine, first["id"], "occurred_at", "broken-time")
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["anomaly"] == "timestamp_unparseable"
    assert audit["broken_receipt_id"] == first["id"]
    assert audit["checked_count"] == 2


def test_integrity_is_isolated_per_machine(consumed_use, client):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    # Damage the receipt's content hash.
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "content_hash", "c" * 64)

    other = create_machine(client, external_id="machine-2")
    # The other machine's empty chain stays valid despite the damage.
    other_audit = client.get(integrity_url(other)).json()
    assert other_audit["valid"] is True
    assert other_audit["checked_count"] == 0

    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is False
    assert audit["anomaly"] in {"chain_break", "digest_mismatch"}


def test_integrity_is_read_only(consumed_use, client):
    machine_id, _, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    _tamper(client.app.state.engine, _first_receipt_id(client.app.state.engine),
            "result_digest", "d" * 64)
    before = client.get(integrity_url(machine_id)).content
    after = client.get(integrity_url(machine_id)).content
    assert before == after
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT result_digest FROM execution_receipts")
        ).scalar_one()
    assert stored == "d" * 64


# --------------------------------------------------------------------------- #
# Persistence, migration, and non-interference
# --------------------------------------------------------------------------- #


def test_receipts_survive_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        _, use = issue_and_consume(first, machine_id, event)
        receipt = first.post(
            receipts_url(machine_id), json=receipt_payload(use)
        ).json()

    with TestClient(app) as second:
        # The use stays exactly-once after restart.
        repeat = second.post(
            receipts_url(machine_id), json=receipt_payload(use)
        )
        assert repeat.status_code == 409
        assert repeat.json() == {"error": {"code": "receipt_already_exists"}}
        audit = second.get(integrity_url(machine_id)).json()
        assert audit == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
            "anomaly": None,
        }
        with second.app.state.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, content_hash, chain_hash, previous_receipt_id "
                    "FROM execution_receipts WHERE use_id = :u"
                ).bindparams(u=use["use_id"])
            ).one()
        assert row.id == receipt["id"]
        assert row.content_hash == receipt["content_hash"]
        assert row.chain_hash == receipt["chain_hash"]
        assert row.previous_receipt_id is None


def test_old_uses_are_not_backfilled_with_receipts(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'olduses.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        _, use = issue_and_consume(first, machine_id, event)

    # Restart: no receipt is fabricated for the historical consumed use.
    with TestClient(app) as second:
        with second.app.state.engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM execution_receipts")
            ).scalar_one()
        assert count == 0
        audit = second.get(integrity_url(machine_id)).json()
        assert audit["checked_count"] == 0
        # The old use can still receive its one new receipt.
        response = second.post(
            receipts_url(machine_id), json=receipt_payload(use)
        )
        assert response.status_code == 201


def test_old_database_without_receipt_table_works(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        _, use = issue_and_consume(first, machine_id, event)

    # Reproduce a database that predates the receipt feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS execution_receipts")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        response = second.post(
            receipts_url(machine_id), json=receipt_payload(use)
        )
        assert response.status_code == 201
        audit = second.get(integrity_url(machine_id)).json()
        assert audit["valid"] is True
        assert audit["checked_count"] == 1


def test_restart_with_complete_chain_performs_no_rewrites(
    tmp_path, monkeypatch
):
    db_url = f"sqlite:///{tmp_path / 'norewrite.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id, resource_pattern="res/*")
        create_rule(first, resource_pattern="res/*")
        receipts = []
        for index in range(3):
            event = record_event(
                first, machine_id, resource=f"res/{index}"
            ).json()
            _, use = issue_and_consume(first, machine_id, event)
            receipts.append(
                first.post(
                    receipts_url(machine_id),
                    json=receipt_payload(use, resource=f"res/{index}"),
                ).json()
            )

    with TestClient(app) as second:
        with second.app.state.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, content_hash, chain_hash, previous_receipt_id "
                    "FROM execution_receipts ORDER BY occurred_at, id"
                )
            ).all()
    for stored, expected in zip(rows, receipts, strict=True):
        assert stored.id == expected["id"]
        assert stored.content_hash == expected["content_hash"]
        assert stored.chain_hash == expected["chain_hash"]


def test_receipts_never_modify_grant_use_or_event(consumed_use, client):
    machine_id, event, grant, use = consumed_use
    grant_url = f"/machines/{machine_id}/authorization-grants"
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).content
    integrity_event_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).content
    lifecycle_before = client.get(
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/changes",
        params={"limit": "100"},
    ).content

    client.post(receipts_url(machine_id), json=receipt_payload(use))

    grants = client.get(grant_url, params={"limit": "100"}).json()["items"]
    [stored_grant] = [g for g in grants if g["id"] == grant["id"]]
    assert stored_grant["status"] == "consumed"
    assert stored_grant["use_id"] == use["use_id"]
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).content
        == events_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).content
        == integrity_event_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-grant-lifecycle-events/"
            "changes",
            params={"limit": "100"},
        ).content
        == lifecycle_before
    )
    assert event["id"]  # event fixture still resolves
