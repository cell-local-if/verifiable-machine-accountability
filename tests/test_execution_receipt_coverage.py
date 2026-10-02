"""Tests for the read-only execution-receipt coverage check.

One entry sits under the machine path:

    GET /machines/{machine_id}/execution-receipts/coverage

The coverage check pairs one machine's consumed authorization-grant uses
with its execution-completion receipts by ``use_id`` alone and reports the
consumed uses no receipt covers (``missing_use_ids``) and the receipts
whose ``use_id`` is not one of the machine's consumed uses
(``orphan_receipt_ids``). It is strictly read-only: it never fabricates a
missing receipt, rewrites a record, or re-judges chain, scope, or digest
soundness — those stay with the independent ``integrity`` audit. These
tests cover the success shapes and field order, the missing/orphan
orderings (actual UTC instant, then id, unparseable stamps last), the
422/404/405 request contract, machine isolation, byte-identical repeats,
and non-interference with the receipt chain itself.
"""
import sqlite3
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


def consume_one(client, machine_id):
    """Record one allowed event, issue its grant, and consume it."""
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    return issue_and_consume(client, machine_id, event)


def receipts_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts"


def coverage_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/coverage"


def digest_of(value: bytes) -> str:
    import hashlib

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
    grant, use = consume_one(client, machine_id)
    return machine_id, grant, use


def insert_orphan_receipt(client, machine_id, receipt_id, use_id,
                          occurred_at):
    """Insert a receipt row directly, bypassing every write-time check."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts ("
                "id, machine_id, use_id, grant_id, authorization_event_id,"
                " action_type, resource, outcome, result_digest,"
                " occurred_at, previous_receipt_id, content_hash, chain_hash"
                ") VALUES ("
                ":id, :machine_id, :use_id, :grant_id, :event_id,"
                " 'read', 'res/x', 'succeeded', :digest,"
                " :occurred_at, NULL, :content_hash, :chain_hash)"
            ).bindparams(
                id=receipt_id,
                machine_id=machine_id,
                use_id=use_id,
                grant_id="grant-" + receipt_id,
                event_id="event-" + receipt_id,
                digest=digest_of(b"orphan"),
                occurred_at=occurred_at,
                content_hash=digest_of(b"content" + receipt_id.encode()),
                chain_hash=digest_of(b"chain" + receipt_id.encode()),
            )
        )


def update_use(client, use_id, column, value):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE authorization_grant_uses SET {column} = :v "
                "WHERE id = :id"
            ).bindparams(v=value, id=use_id)
        )


def update_receipt(client, receipt_id, column, value):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE execution_receipts SET {column} = :v WHERE id = :id"
            ).bindparams(v=value, id=receipt_id)
        )


# --------------------------------------------------------------------------- #
# Success shape
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid_with_zero_counts(client):
    machine_id = create_machine(client)
    response = client.get(coverage_url(machine_id))
    assert response.status_code == 200
    assert response.content == (
        b'{"machine_id":"' + machine_id.encode() + b'",'
        b'"consumed_count":0,"receipt_count":0,"covered_count":0,'
        b'"missing_count":0,"missing_use_ids":[],'
        b'"orphan_count":0,"orphan_receipt_ids":[],'
        b'"valid":true}\n'
    )


def test_fully_covered_machine_is_valid(consumed_use, client):
    machine_id, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))

    response = client.get(coverage_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "machine_id",
        "consumed_count",
        "receipt_count",
        "covered_count",
        "missing_count",
        "missing_use_ids",
        "orphan_count",
        "orphan_receipt_ids",
        "valid",
    ]
    assert body == {
        "machine_id": machine_id,
        "consumed_count": 1,
        "receipt_count": 1,
        "covered_count": 1,
        "missing_count": 0,
        "missing_use_ids": [],
        "orphan_count": 0,
        "orphan_receipt_ids": [],
        "valid": True,
    }


def test_counts_are_integers(consumed_use, client):
    machine_id, _, _ = consumed_use
    body = client.get(coverage_url(machine_id)).json()
    for key in (
        "consumed_count",
        "receipt_count",
        "covered_count",
        "missing_count",
        "orphan_count",
    ):
        assert isinstance(body[key], int)
        assert not isinstance(body[key], bool)


def test_missing_use_is_reported_not_backfilled(consumed_use, client):
    machine_id, _, use = consumed_use
    first = client.get(coverage_url(machine_id)).json()
    assert first["consumed_count"] == 1
    assert first["receipt_count"] == 0
    assert first["covered_count"] == 0
    assert first["missing_count"] == 1
    assert first["missing_use_ids"] == [use["use_id"]]
    assert first["orphan_count"] == 0
    assert first["orphan_receipt_ids"] == []
    assert first["valid"] is False

    # The read-only check fabricates nothing.
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one()
    assert count == 0


def test_orphan_receipt_is_reported(consumed_use, client):
    machine_id, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))
    insert_orphan_receipt(
        client, machine_id, "orphan-1", "use-not-of-this-machine",
        "2026-01-01T00:00:00Z",
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body["consumed_count"] == 1
    assert body["receipt_count"] == 2
    assert body["covered_count"] == 1
    assert body["missing_count"] == 0
    assert body["orphan_count"] == 1
    assert body["orphan_receipt_ids"] == ["orphan-1"]
    assert body["valid"] is False


def test_later_legitimate_receipt_improves_coverage(consumed_use, client):
    machine_id, _, use = consumed_use
    assert client.get(coverage_url(machine_id)).json()["valid"] is False

    client.post(receipts_url(machine_id), json=receipt_payload(use))
    body = client.get(coverage_url(machine_id)).json()
    assert body["valid"] is True
    assert body["missing_count"] == 0
    assert body["covered_count"] == 1


# --------------------------------------------------------------------------- #
# Ordering of the reported id lists
# --------------------------------------------------------------------------- #


def test_missing_use_ids_order_by_consumed_instant_then_id(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    uses = [consume_one(client, machine_id)[1] for _ in range(3)]

    # Force distinct, out-of-insertion-order instants; the middle use gets
    # the earliest stamp and the tie breaks by id ascending.
    update_use(client, uses[0]["use_id"], "consumed_at",
               "2026-01-02T00:00:00Z")
    update_use(client, uses[1]["use_id"], "consumed_at",
               "2026-01-01T00:00:00Z")
    update_use(client, uses[2]["use_id"], "consumed_at",
               "2026-01-02T00:00:00Z")

    body = client.get(coverage_url(machine_id)).json()
    assert body["missing_count"] == 3
    assert body["missing_use_ids"] == [
        uses[1]["use_id"],
        min(uses[0]["use_id"], uses[2]["use_id"]),
        max(uses[0]["use_id"], uses[2]["use_id"]),
    ]
    assert body["valid"] is False


def test_missing_use_ids_with_unparseable_stamp_sort_last(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    uses = [consume_one(client, machine_id)[1] for _ in range(2)]

    update_use(client, uses[0]["use_id"], "consumed_at", "not-a-timestamp")
    update_use(client, uses[1]["use_id"], "consumed_at",
               "2026-01-01T00:00:00Z")

    body = client.get(coverage_url(machine_id)).json()
    assert body["missing_use_ids"] == [uses[1]["use_id"], uses[0]["use_id"]]


def test_orphan_receipt_ids_order_by_occurred_instant_then_id(client):
    machine_id = create_machine(client)
    insert_orphan_receipt(
        client, machine_id, "orphan-b", "use-b", "2026-01-02T00:00:00Z"
    )
    insert_orphan_receipt(
        client, machine_id, "orphan-a", "use-a", "2026-01-02T00:00:00Z"
    )
    insert_orphan_receipt(
        client, machine_id, "orphan-c", "use-c", "2026-01-01T00:00:00Z"
    )
    insert_orphan_receipt(
        client, machine_id, "orphan-z", "use-z", "not-a-timestamp"
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body["orphan_count"] == 4
    # Earliest instant first, same-instant ties by id ascending, and the
    # unparseable stamp last.
    assert body["orphan_receipt_ids"] == [
        "orphan-c",
        "orphan-a",
        "orphan-b",
        "orphan-z",
    ]
    assert body["valid"] is False


# --------------------------------------------------------------------------- #
# Independence from the chain audit and from other machines
# --------------------------------------------------------------------------- #


def test_chain_damage_still_counts_as_covered(consumed_use, client):
    machine_id, _, use = consumed_use
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    update_receipt(client, receipt["id"], "content_hash", "0" * 64)

    # Coverage only pairs use_id: the damaged receipt still covers its use,
    # while the independent integrity audit reports the digest anomaly.
    body = client.get(coverage_url(machine_id)).json()
    assert body["valid"] is True
    assert body["covered_count"] == 1

    integrity = client.get(
        f"/machines/{machine_id}/execution-receipts/integrity"
    ).json()
    assert integrity["valid"] is False
    assert integrity["anomaly"] == "chain_break"


def test_other_machines_never_change_the_conclusion(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    _, use = consume_one(client, machine_id)
    client.post(receipts_url(machine_id), json=receipt_payload(use))

    other_id = create_machine(client, external_id="machine-2")
    declare(client, other_id)
    consume_one(client, other_id)  # missing receipt on the other machine
    insert_orphan_receipt(
        client, other_id, "other-orphan", "other-use",
        "2026-01-01T00:00:00Z",
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body == {
        "machine_id": machine_id,
        "consumed_count": 1,
        "receipt_count": 1,
        "covered_count": 1,
        "missing_count": 0,
        "missing_use_ids": [],
        "orphan_count": 0,
        "orphan_receipt_ids": [],
        "valid": True,
    }

    other = client.get(coverage_url(other_id)).json()
    assert other["machine_id"] == other_id
    assert other["missing_count"] == 1
    assert other["orphan_receipt_ids"] == ["other-orphan"]
    assert other["valid"] is False


def test_coverage_is_read_only_and_byte_identical_on_repeat(
    consumed_use, client
):
    machine_id, _, use = consumed_use
    client.post(receipts_url(machine_id), json=receipt_payload(use))

    first = client.get(coverage_url(machine_id))
    second = client.get(coverage_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content
    assert first.content.endswith(b"\n")
    assert b"\n" not in first.content[:-1]

    with client.app.state.engine.connect() as conn:
        receipts = conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one()
        uses = conn.execute(
            text("SELECT COUNT(*) FROM authorization_grant_uses")
        ).scalar_one()
    assert receipts == 1
    assert uses == 1


# --------------------------------------------------------------------------- #
# Request contract: 422 / 404 / 405
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/"
        "execution-receipts/coverage"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize(
    "query", ["?x=1", "?=", "?valid=true", "?a=1&a=2"]
)
def test_coverage_rejects_query_before_machine_lookup(client, query):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.get(
        f"/machines/{missing_machine}/execution-receipts/coverage" + query
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_coverage_rejects_a_body(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        coverage_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "method", ["post", "put", "patch", "delete", "head"]
)
def test_coverage_only_accepts_get(consumed_use, client, method):
    machine_id, _, _ = consumed_use
    response = getattr(client, method)(coverage_url(machine_id))
    assert response.status_code == 405
