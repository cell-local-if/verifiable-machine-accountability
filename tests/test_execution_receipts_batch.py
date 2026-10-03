"""Tests for the atomic batch execution-receipt entry.

    POST /machines/{machine_id}/execution-receipts/batch

The batch entry applies the single-receipt binding semantics to one to one
hundred records in one locked write transaction: the body is exactly
``{"records": [...]}`` with the five single-receipt fields per item and no
repeated ``use_id``; any format violation is
``422 invalid_execution_receipt_batch_request`` before any record is read.
Missing or cross-machine machines/uses/grants/events are 404; the records
are then checked in submission order and the first refusal answers 409
``authorization_not_allowed``, ``execution_scope_mismatch``, or
``receipt_already_exists`` — always with no partial receipts. A committed
batch shares one UTC commit instant, links the chain in submission order
with strictly increasing same-instant ids, and serializes with every other
receipt writer so concurrent batches never fork the chain. These tests
cover the success shape and chain linkage, every validation and lookup
outcome, the ordered 409 outcomes, atomicity, exactly-once under
concurrency, restart persistence, and non-interference with the existing
receipt, integrity, and coverage behavior.
"""
import hashlib
import json
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


def batch_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/batch"


def receipts_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts"


def integrity_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/integrity"


def digest_of(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def receipt_item(use, action_type="read", resource="res/x",
                 outcome="succeeded", digest=None):
    return {
        "use_id": use["use_id"],
        "action_type": action_type,
        "resource": resource,
        "outcome": outcome,
        "result_digest": digest or digest_of(b"result"),
    }


def make_consumed_use(client, machine_id, resource="res/x",
                      action_type="read"):
    event = record_event(
        client, machine_id, action_type=action_type, resource=resource
    ).json()
    assert event["allowed"] is True
    grant, use = issue_and_consume(client, machine_id, event)
    return event, grant, use


@pytest.fixture
def machine_with_uses(client):
    """A machine with three consumed allow uses on res/1..res/3."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    uses = []
    for index in range(1, 4):
        event, grant, use = make_consumed_use(
            client, machine_id, resource=f"res/{index}"
        )
        uses.append((event, grant, use))
    return machine_id, uses


def batch_payload(uses, **overrides):
    return {
        "records": [
            receipt_item(use, resource=resource, **overrides)
            for (resource, use) in uses
        ]
    }


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _receipt_count(client):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one()


# --------------------------------------------------------------------------- #
# Success shape, shared commit instant, and chain linkage
# --------------------------------------------------------------------------- #


def test_batch_success_shape_and_shared_commit_instant(
    machine_with_uses, client
):
    machine_id, uses = machine_with_uses
    payload = {
        "records": [
            receipt_item(use, resource=f"res/{index}", outcome="succeeded")
            for index, (_, _, use) in enumerate(uses, start=1)
        ]
    }

    response = client.post(batch_url(machine_id), json=payload)

    assert response.status_code == 201
    body = response.json()
    assert list(body.keys()) == ["records"]
    assert len(body["records"]) == 3
    occurred_at_values = set()
    for index, (record, (event, grant, use)) in enumerate(
        zip(body["records"], uses), start=1
    ):
        assert list(record.keys()) == [
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
        assert isinstance(record["id"], str) and record["id"]
        assert record["machine_id"] == machine_id
        assert record["use_id"] == use["use_id"]
        assert record["grant_id"] == grant["id"]
        assert record["authorization_event_id"] == event["id"]
        assert record["occurred_at"].endswith("Z")
        assert _parse_z(record["occurred_at"]).tzinfo == timezone.utc
        occurred_at_values.add(record["occurred_at"])
        assert record["content_hash"] == digest_of(
            json.dumps(
                {
                    "id": record["id"],
                    "machine_id": machine_id,
                    "use_id": use["use_id"],
                    "grant_id": grant["id"],
                    "authorization_event_id": event["id"],
                    "action_type": "read",
                    "resource": f"res/{index}",
                    "outcome": "succeeded",
                    "result_digest": payload["records"][index - 1][
                        "result_digest"
                    ],
                    "occurred_at": record["occurred_at"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    # The whole batch shares one UTC commit instant.
    assert len(occurred_at_values) == 1


def test_batch_links_chain_in_records_order(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    payload = {
        "records": [
            receipt_item(use, resource=f"res/{index}")
            for index, (_, _, use) in enumerate(uses, start=1)
        ]
    }
    records = client.post(batch_url(machine_id), json=payload).json()["records"]

    first, second, third = records
    assert first["previous_receipt_id"] is None
    assert second["previous_receipt_id"] == first["id"]
    assert third["previous_receipt_id"] == second["id"]
    assert second["chain_hash"] == digest_of(
        f"{first['chain_hash']}:{second['content_hash']}".encode("utf-8")
    )
    assert third["chain_hash"] == digest_of(
        f"{second['chain_hash']}:{third['content_hash']}".encode("utf-8")
    )
    # Same-instant ids sort strictly increasingly in records order.
    assert first["id"] < second["id"] < third["id"]

    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 3,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def test_batch_appends_after_existing_single_receipt(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    single = client.post(
        receipts_url(machine_id), json=receipt_item(uses[0][2], resource="res/1")
    ).json()

    payload = {
        "records": [
            receipt_item(uses[1][2], resource="res/2"),
            receipt_item(uses[2][2], resource="res/3", outcome="failed"),
        ]
    }
    records = client.post(batch_url(machine_id), json=payload).json()["records"]

    assert records[0]["previous_receipt_id"] == single["id"]
    assert records[1]["previous_receipt_id"] == records[0]["id"]
    assert records[0]["chain_hash"] == digest_of(
        f"{single['chain_hash']}:{records[0]['content_hash']}".encode("utf-8")
    )
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 3


def test_single_record_batch(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    response = client.post(
        batch_url(machine_id),
        json={"records": [receipt_item(uses[0][2], resource="res/1")]},
    )
    assert response.status_code == 201
    records = response.json()["records"]
    assert len(records) == 1
    assert records[0]["use_id"] == uses[0][2]["use_id"]
    assert records[0]["previous_receipt_id"] is None


def test_hundred_record_batch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    records = []
    for index in range(100):
        _, _, use = make_consumed_use(
            client, machine_id, resource=f"res/{index}"
        )
        records.append(receipt_item(use, resource=f"res/{index}"))

    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 201
    assert len(response.json()["records"]) == 100
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 100


def test_batch_persists_all_bound_fields(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    digest = digest_of(b"persisted")
    payload = {
        "records": [
            receipt_item(uses[0][2], resource="res/1", digest=digest),
            receipt_item(uses[1][2], resource="res/2", outcome="failed"),
        ]
    }
    body = client.post(batch_url(machine_id), json=payload).json()

    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, machine_id, use_id, grant_id, "
                "authorization_event_id, action_type, resource, outcome, "
                "result_digest, occurred_at FROM execution_receipts "
                "ORDER BY occurred_at, id"
            )
        ).all()
    assert len(rows) == 2
    assert rows[0].id == body["records"][0]["id"]
    assert rows[0].use_id == uses[0][2]["use_id"]
    assert rows[0].grant_id == uses[0][1]["id"]
    assert rows[0].authorization_event_id == uses[0][0]["id"]
    assert rows[0].action_type == "read"
    assert rows[0].resource == "res/1"
    assert rows[0].outcome == "succeeded"
    assert rows[0].result_digest == digest
    assert rows[0].occurred_at == body["records"][0]["occurred_at"]
    assert rows[1].outcome == "failed"
    assert rows[1].resource == "res/2"


# --------------------------------------------------------------------------- #
# Request validation: body, items, duplicates, and query
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"records": []},
        {"records": [{}]},
        {"records": None},
        {"records": "x"},
        {"records": 3},
        {"records": [{"use_id": "u"}]},
        {"records": [{"use_id": "u", "action_type": "read",
                      "resource": "res/x", "outcome": "succeeded",
                      "result_digest": "a" * 64, "extra": 1}]},
        {"records": [{"use_id": "u", "action_type": "read",
                      "resource": "res/x", "outcome": "succeeded"}]},
        {"records": [{"use_id": "u", "action_type": "read",
                      "resource": "res/x", "outcome": "succeeded",
                      "result_digest": "a" * 64}], "extra": 1},
        {"items": []},
        [],
        "records",
        12,
        None,
    ],
)
def test_malformed_batch_body_is_invalid_request(
    machine_with_uses, client, payload
):
    machine_id, _ = machine_with_uses
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


def test_over_hundred_records_is_invalid_request(machine_with_uses, client):
    machine_id, _ = machine_with_uses
    item = {
        "use_id": "u",
        "action_type": "read",
        "resource": "res/x",
        "outcome": "succeeded",
        "result_digest": "a" * 64,
    }
    records = [{**item, "use_id": f"u-{index}"} for index in range(101)]
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_typed_body_is_invalid_request(
    machine_with_uses, client, raw
):
    machine_id, _ = machine_with_uses
    response = client.post(
        batch_url(machine_id),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
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
        ("action_type", ""),
        ("resource", False),
        ("resource", " "),
        ("outcome", "SUCCEEDED"),
        ("outcome", None),
        ("result_digest", "a" * 63),
        ("result_digest", "A" * 64),
        ("result_digest", 123),
    ],
)
def test_bad_item_field_values_are_invalid_request(
    machine_with_uses, client, field, value
):
    machine_id, uses = machine_with_uses
    good = receipt_item(uses[0][2], resource="res/1")
    bad = receipt_item(uses[1][2], resource="res/2")
    bad[field] = value
    response = client.post(
        batch_url(machine_id), json={"records": [good, bad]}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    # The format refusal happens before any record is read or written.
    assert _receipt_count(client) == 0


def test_duplicate_use_id_in_batch_is_invalid_request(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    item = receipt_item(uses[0][2], resource="res/1")
    response = client.post(
        batch_url(machine_id), json={"records": [item, dict(item)]}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    assert _receipt_count(client) == 0


def test_duplicate_use_id_after_trimming_is_invalid_request(
    machine_with_uses, client
):
    machine_id, uses = machine_with_uses
    use_id = uses[0][2]["use_id"]
    first = receipt_item(uses[0][2], resource="res/1")
    second = receipt_item({"use_id": f"  {use_id} "}, resource="res/1")
    response = client.post(
        batch_url(machine_id), json={"records": [first, second]}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


@pytest.mark.parametrize("query", ["?x=1", "?records=r", "?=", "?outcome=succeeded"])
def test_any_query_parameter_is_invalid_request_before_lookup(
    machine_with_uses, client, query
):
    machine_id, uses = machine_with_uses
    payload = {"records": [receipt_item(uses[0][2], resource="res/1")]}
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(batch_url(machine_id) + query, json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }

    # The format check precedes the machine lookup.
    response = client.post(batch_url(missing_machine) + query, json=payload)
    assert response.status_code == 422


def test_body_validation_runs_before_machine_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        batch_url(missing_machine),
        json={"records": [{"use_id": "u", "outcome": "nope"}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == (
        "invalid_execution_receipt_batch_request"
    )


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete", "head"])
def test_batch_only_accepts_post(machine_with_uses, client, method):
    machine_id, _ = machine_with_uses
    response = getattr(client, method)(batch_url(machine_id))
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Lookup outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(machine_with_uses, client):
    _, uses = machine_with_uses
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        batch_url(missing_machine),
        json={"records": [receipt_item(uses[0][2], resource="res/1")]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_use_is_not_found(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    payload = {
        "records": [
            receipt_item(uses[0][2], resource="res/1"),
            receipt_item({"use_id": "no-such-use"}),
        ]
    }
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The missing record rejects the whole batch: no partial receipts.
    assert _receipt_count(client) == 0


def test_use_owned_by_another_machine_is_not_found(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    other = create_machine(client, external_id="machine-2")
    response = client.post(
        batch_url(other),
        json={"records": [receipt_item(uses[0][2], resource="res/1")]},
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The foreign attempt wrote nothing: the real owner can still record.
    assert (
        client.post(
            batch_url(machine_id),
            json={"records": [receipt_item(uses[0][2], resource="res/1")]},
        ).status_code
        == 201
    )


# --------------------------------------------------------------------------- #
# Ordered 409 outcomes and atomicity
# --------------------------------------------------------------------------- #


def _fabricate_denied_use(client, machine_id):
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
    return denied


def test_not_allowed_source_is_authorization_not_allowed(client):
    # A denied event can never have a grant through the public API, but the
    # receipt audit must not trust history: fabricate a consumed use bound to
    # a denied event directly and confirm the batch refuses it.
    machine_id = create_machine(client)
    declare(client, machine_id)
    _fabricate_denied_use(client, machine_id)
    _, _, use = make_consumed_use(client, machine_id, resource="res/ok")
    payload = {
        "records": [
            receipt_item(use, resource="res/ok"),
            receipt_item({"use_id": "use-denied"}, resource="res/d"),
        ]
    }
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "authorization_not_allowed"}}
    # The refusal leaves no partial receipts, not even for the valid record.
    assert _receipt_count(client) == 0


def test_scope_mismatch_rejects_whole_batch(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    payload = {
        "records": [
            receipt_item(uses[0][2], resource="res/1"),
            receipt_item(uses[1][2], resource="res/WRONG"),
        ]
    }
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}
    assert _receipt_count(client) == 0
    # The rejected batch never consumed anything: a corrected batch succeeds.
    payload["records"][1]["resource"] = "res/2"
    assert client.post(batch_url(machine_id), json=payload).status_code == 201


def test_existing_receipt_is_receipt_already_exists(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    client.post(
        receipts_url(machine_id), json=receipt_item(uses[0][2], resource="res/1")
    )
    payload = {
        "records": [
            receipt_item(uses[1][2], resource="res/2"),
            receipt_item(uses[0][2], resource="res/1"),
        ]
    }
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "receipt_already_exists"}}
    # Only the pre-existing single receipt remains.
    assert _receipt_count(client) == 1


def test_first_refusal_in_records_order_decides(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    # Record 1 has a scope mismatch; record 2's use already has a receipt.
    client.post(
        receipts_url(machine_id), json=receipt_item(uses[1][2], resource="res/2")
    )
    payload = {
        "records": [
            receipt_item(uses[0][2], resource="res/WRONG"),
            receipt_item(uses[1][2], resource="res/2"),
        ]
    }
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}

    # Swapping the order surfaces the duplicate-use refusal first.
    payload["records"].reverse()
    response = client.post(batch_url(machine_id), json=payload)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "receipt_already_exists"}}


def test_rejected_batch_leaves_chain_untouched(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    payload = {
        "records": [
            receipt_item(uses[0][2], resource="res/1"),
            receipt_item(uses[1][2], action_type="write", resource="res/2"),
        ]
    }
    assert client.post(batch_url(machine_id), json=payload).status_code == 409
    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }


# --------------------------------------------------------------------------- #
# Concurrency: serialization, exactly-once, no chain fork
# --------------------------------------------------------------------------- #


def test_concurrent_batches_sharing_one_use_have_exactly_one_winner(
    machine_with_uses, client
):
    machine_id, uses = machine_with_uses
    shared = receipt_item(uses[0][2], resource="res/1")
    gate = threading.Event()

    def hit(extra_use, resource):
        gate.wait()
        return client.post(
            batch_url(machine_id),
            json={"records": [shared, receipt_item(extra_use, resource=resource)]},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(hit, uses[1][2], "res/2"),
            pool.submit(hit, uses[2][2], "res/3"),
        ]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [201, 409]
    loser = next(r for r in responses if r.status_code == 409)
    assert loser.json() == {"error": {"code": "receipt_already_exists"}}
    # The winning batch committed both receipts; the loser wrote nothing.
    assert _receipt_count(client) == 2
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 2


def test_concurrent_disjoint_batches_all_commit_without_chain_fork(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    batches = []
    for batch_index in range(4):
        records = []
        for item_index in range(3):
            resource = f"res/{batch_index}-{item_index}"
            _, _, use = make_consumed_use(client, machine_id, resource=resource)
            records.append(receipt_item(use, resource=resource))
        batches.append(records)
    gate = threading.Event()

    def hit(records):
        gate.wait()
        return client.post(batch_url(machine_id), json={"records": records})

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(hit, records) for records in batches]
        gate.set()
        responses = [f.result() for f in futures]

    assert sorted(r.status_code for r in responses) == [201] * len(batches)
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 12


def test_concurrent_batch_and_single_for_one_use_have_one_winner(
    machine_with_uses, client
):
    machine_id, uses = machine_with_uses
    gate = threading.Event()

    def hit_batch():
        gate.wait()
        return client.post(
            batch_url(machine_id),
            json={"records": [receipt_item(uses[0][2], resource="res/1")]},
        )

    def hit_single():
        gate.wait()
        return client.post(
            receipts_url(machine_id),
            json=receipt_item(uses[0][2], resource="res/1"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(hit_batch), pool.submit(hit_single)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [201, 409]
    assert _receipt_count(client) == 1


# --------------------------------------------------------------------------- #
# Persistence and non-interference
# --------------------------------------------------------------------------- #


def test_batch_receipts_survive_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        uses = []
        for index in range(1, 3):
            _, _, use = make_consumed_use(first, machine_id, f"res/{index}")
            uses.append(use)
        payload = {
            "records": [
                receipt_item(uses[0], resource="res/1"),
                receipt_item(uses[1], resource="res/2"),
            ]
        }
        created = first.post(batch_url(machine_id), json=payload).json()

    with TestClient(app) as second:
        # The uses stay exactly-once after restart.
        repeat = second.post(batch_url(machine_id), json=payload)
        assert repeat.status_code == 409
        assert repeat.json() == {"error": {"code": "receipt_already_exists"}}
        audit = second.get(integrity_url(machine_id)).json()
        assert audit == {
            "valid": True,
            "checked_count": 2,
            "broken_receipt_id": None,
            "anomaly": None,
        }
        with second.app.state.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, content_hash, chain_hash FROM "
                    "execution_receipts ORDER BY occurred_at, id"
                )
            ).all()
        assert [row.id for row in rows] == [
            record["id"] for record in created["records"]
        ]
        assert [row.content_hash for row in rows] == [
            record["content_hash"] for record in created["records"]
        ]
        assert [row.chain_hash for row in rows] == [
            record["chain_hash"] for record in created["records"]
        ]


def test_batch_improves_coverage_report(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    before = client.get(
        f"/machines/{machine_id}/execution-receipts/coverage"
    ).json()
    assert before["missing_count"] == 3

    payload = {
        "records": [
            receipt_item(use, resource=f"res/{index}")
            for index, (_, _, use) in enumerate(uses, start=1)
        ]
    }
    client.post(batch_url(machine_id), json=payload)

    after = client.get(
        f"/machines/{machine_id}/execution-receipts/coverage"
    ).json()
    assert after["receipt_count"] == 3
    assert after["covered_count"] == 3
    assert after["missing_count"] == 0
    assert after["valid"] is True


def test_batch_does_not_modify_uses_grants_or_events(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    with client.app.state.engine.connect() as conn:
        uses_before = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_uses")
        ).scalar_one()
        grants_before = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one()
        events_before = conn.execute(
            text("SELECT COUNT(*) FROM authorization_decision_events")
        ).scalar_one()

    payload = {
        "records": [
            receipt_item(use, resource=f"res/{index}")
            for index, (_, _, use) in enumerate(uses, start=1)
        ]
    }
    assert client.post(batch_url(machine_id), json=payload).status_code == 201

    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_uses")
        ).scalar_one() == uses_before
        assert conn.execute(
            text("SELECT COUNT(*) FROM authorization_grants")
        ).scalar_one() == grants_before
        assert conn.execute(
            text("SELECT COUNT(*) FROM authorization_decision_events")
        ).scalar_one() == events_before


def test_single_receipt_still_works_after_batch(machine_with_uses, client):
    machine_id, uses = machine_with_uses
    client.post(
        batch_url(machine_id),
        json={"records": [receipt_item(uses[0][2], resource="res/1")]},
    )
    # The single-create entry links onto the batch-written tail.
    single = client.post(
        receipts_url(machine_id), json=receipt_item(uses[1][2], resource="res/2")
    )
    assert single.status_code == 201
    batch_record = client.post(
        batch_url(machine_id),
        json={"records": [receipt_item(uses[2][2], resource="res/3")]},
    )
    assert batch_record.status_code == 201
    assert batch_record.json()["records"][0]["previous_receipt_id"] == (
        single.json()["id"]
    )
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 3
