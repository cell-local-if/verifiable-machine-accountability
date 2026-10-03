"""Tests for the atomic multi-record execution-receipt batch entry.

    POST /machines/{machine_id}/execution-receipts/batch

The batch entry records one to one hundred execution-completion receipts
for one machine in a single locked write transaction: the request order is
the commit and chain-link order, all records share one UTC commit moment,
and the batch commits as a whole or leaves no trace. These tests cover the
success shape and chain linkage, every 422 format outcome (answered before
any record is read), the 404 lookup outcomes, the three ordered 409
eligibility rejections, atomicity under rejection, exactly-once under
concurrent batches, non-interference with the single-receipt entry and the
read-only audits, and restart persistence.
"""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

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


def coverage_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/coverage"


def digest_of(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record_payload(use, action_type="read", resource="res/x",
                   outcome="succeeded", digest=None):
    return {
        "use_id": use["use_id"],
        "action_type": action_type,
        "resource": resource,
        "outcome": outcome,
        "result_digest": digest or digest_of(b"result"),
    }


@pytest.fixture
def consumed_uses(client):
    """A machine with three consumed allow grants for read/res/{0,1,2}."""
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    prepared = []
    for index in range(3):
        event = record_event(client, machine_id, resource=f"res/{index}").json()
        assert event["allowed"] is True
        grant, use = issue_and_consume(client, machine_id, event)
        prepared.append((event, grant, use))
    return machine_id, prepared


def batch_payload(prepared, **overrides):
    records = []
    for _event, _grant, use in prepared:
        records.append(
            record_payload(use, resource=f"res/{len(records)}", **overrides)
        )
    return {"records": records}


def receipt_count(client):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM execution_receipts")
        ).scalar_one()


# --------------------------------------------------------------------------- #
# Success shape, shared commit moment, and chain linkage
# --------------------------------------------------------------------------- #


def test_batch_success_shape_and_chain(consumed_uses, client):
    machine_id, prepared = consumed_uses

    response = client.post(batch_url(machine_id), json=batch_payload(prepared))

    assert response.status_code == 201
    body = response.json()
    assert list(body.keys()) == ["records"]
    records = body["records"]
    assert len(records) == 3

    occurred_at_values = set()
    for index, (record, (event, grant, use)) in enumerate(
        zip(records, prepared)
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
        assert record["machine_id"] == machine_id
        assert record["use_id"] == use["use_id"]
        assert record["grant_id"] == grant["id"]
        assert record["authorization_event_id"] == event["id"]
        assert record["occurred_at"].endswith("Z")
        occurred_at_values.add(record["occurred_at"])
        # The chain links in request order.
        if index == 0:
            assert record["previous_receipt_id"] is None
        else:
            assert record["previous_receipt_id"] == records[index - 1]["id"]
            assert record["id"] > records[index - 1]["id"]
        # The hashes cover the full content and continue the chain.
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
                    "result_digest": digest_of(b"result"),
                    "occurred_at": record["occurred_at"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        previous_chain_hash = (
            "" if index == 0 else records[index - 1]["chain_hash"]
        )
        assert record["chain_hash"] == digest_of(
            f"{previous_chain_hash}:{record['content_hash']}".encode("utf-8")
        )

    # The whole batch shares one UTC commit moment.
    assert len(occurred_at_values) == 1

    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 3,
        "broken_receipt_id": None,
        "anomaly": None,
    }
    coverage = client.get(coverage_url(machine_id)).json()
    assert coverage["valid"] is True
    assert coverage["covered_count"] == 3


def test_batch_of_one_matches_single_entry_shape(consumed_uses, client):
    machine_id, prepared = consumed_uses
    event, grant, use = prepared[0]

    response = client.post(
        batch_url(machine_id),
        json={"records": [record_payload(use, resource="res/0")]},
    )

    assert response.status_code == 201
    record = response.json()["records"][0]
    assert record["use_id"] == use["use_id"]
    assert record["previous_receipt_id"] is None


def test_batch_continues_chain_after_single_receipt(consumed_uses, client):
    machine_id, prepared = consumed_uses
    first_event, first_grant, first_use = prepared[0]
    single = client.post(
        receipts_url(machine_id),
        json=record_payload(first_use, resource="res/0"),
    )
    assert single.status_code == 201
    tail = single.json()

    response = client.post(
        batch_url(machine_id),
        json={
            "records": [
                record_payload(use, resource=f"res/{index + 1}")
                for index, (_event, _grant, use) in enumerate(prepared[1:])
            ]
        },
    )
    assert response.status_code == 201
    records = response.json()["records"]
    assert len(records) == 2
    assert records[0]["previous_receipt_id"] == tail["id"]
    assert records[1]["previous_receipt_id"] == records[0]["id"]
    assert records[0]["chain_hash"] == digest_of(
        f"{tail['chain_hash']}:{records[0]['content_hash']}".encode("utf-8")
    )

    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 3


def test_batch_max_length_is_accepted(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    records = []
    for index in range(100):
        event = record_event(client, machine_id, resource=f"res/{index}").json()
        _, use = issue_and_consume(client, machine_id, event)
        records.append(record_payload(use, resource=f"res/{index}"))

    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 201
    assert len(response.json()["records"]) == 100
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 100


# --------------------------------------------------------------------------- #
# 422 format outcomes, all answered before any record is read
# --------------------------------------------------------------------------- #


def test_batch_rejects_any_query_parameter(consumed_uses, client):
    machine_id, prepared = consumed_uses
    response = client.post(
        batch_url(machine_id) + "?limit=1", json=batch_payload(prepared)
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    assert receipt_count(client) == 0


def test_batch_rejects_repeated_query_parameter(consumed_uses, client):
    machine_id, prepared = consumed_uses
    response = client.post(
        batch_url(machine_id) + "?a=1&a=2", json=batch_payload(prepared)
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


def test_batch_format_error_precedes_machine_lookup(client):
    # The same malformed request against a non-existent machine is still 422.
    response = client.post(
        batch_url("no-such-machine") + "?x=1", json={"records": []}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    response = client.post(batch_url("no-such-machine"), json={"records": []})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        [],  # non-object body
        "records",  # non-object body
        {},  # missing records
        {"records": [], "extra": 1},  # extra key
        {"items": []},  # wrong key
        {"records": {}},  # records not an array
        {"records": "x"},  # records not an array
        {"records": []},  # empty batch
        {"records": [None]},  # non-object item
        {"records": [{}]},  # empty item
    ],
)
def test_batch_rejects_malformed_body_shapes(consumed_uses, client, body):
    machine_id, _prepared = consumed_uses
    response = client.post(batch_url(machine_id), json=body)
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    assert receipt_count(client) == 0


def test_batch_rejects_non_json_body(consumed_uses, client):
    machine_id, _prepared = consumed_uses
    response = client.post(
        batch_url(machine_id),
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


def test_batch_rejects_overlong_records_array(consumed_uses, client):
    machine_id, prepared = consumed_uses
    one = record_payload(prepared[0][2], resource="res/0")
    records = [dict(one, use_id=f"use-{index}") for index in range(101)]
    response = client.post(
        batch_url(machine_id), json={"records": records}
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.pop("use_id"),  # missing field
        lambda record: record.update({"note": "x"}),  # extra field
        lambda record: record.update({"use_id": "  "}),  # blank use id
        lambda record: record.update({"use_id": 7}),  # non-string use id
        lambda record: record.update({"action_type": ""}),  # blank action
        lambda record: record.update({"resource": 3}),  # non-string resource
        lambda record: record.update({"outcome": "ok"}),  # bad outcome
        lambda record: record.update({"result_digest": "zz"}),  # bad digest
        lambda record: record.update(
            {"result_digest": digest_of(b"x").upper()}  # uppercase digest
        ),
    ],
)
def test_batch_rejects_invalid_item_fields(consumed_uses, client, mutate):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    mutate(records[1])
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    assert receipt_count(client) == 0


def test_batch_rejects_repeated_use_id(consumed_uses, client):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    records[1]["use_id"] = records[0]["use_id"]
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }
    assert receipt_count(client) == 0


def test_batch_rejects_use_id_repeated_after_trimming(consumed_uses, client):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    records[1]["use_id"] = f"  {records[0]['use_id']}  "
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_batch_request"}
    }


# --------------------------------------------------------------------------- #
# 404 lookup outcomes
# --------------------------------------------------------------------------- #


def test_batch_against_missing_machine_is_not_found(consumed_uses, client):
    _machine_id, prepared = consumed_uses
    response = client.post(
        batch_url("no-such-machine"), json=batch_payload(prepared)
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert receipt_count(client) == 0


def test_batch_with_unknown_use_is_not_found(consumed_uses, client):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    records[1]["use_id"] = "no-such-use"
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert receipt_count(client) == 0


def test_batch_with_cross_machine_use_is_not_found(consumed_uses, client):
    machine_id, prepared = consumed_uses
    other_machine = create_machine(client, external_id="machine-2")
    declare(client, other_machine, resource_pattern="res/*")
    other_event = record_event(client, other_machine, resource="res/9").json()
    _, other_use = issue_and_consume(client, other_machine, other_event)

    records = batch_payload(prepared)["records"]
    records[1]["use_id"] = other_use["use_id"]
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert receipt_count(client) == 0


# --------------------------------------------------------------------------- #
# Ordered 409 eligibility rejections; nothing partial is written
# --------------------------------------------------------------------------- #


def _fabricate_denied_use(client, machine_id):
    """A consumed use bound to a denied source event, written directly."""
    # A deny at the same priority as the fixture's allow wins the conflict.
    create_rule(client, resource_pattern="res/d", effect="deny", priority=0)
    denied = record_event(client, machine_id, resource="res/d").json()
    assert denied["allowed"] is False
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_grants (id, machine_id, event_id, "
                "issued_at, expires_at, status, consumed_at, revoked_at) "
                "VALUES ('grant-denied', :m, :e, :t, :t2, 'consumed', :t, NULL)"
            ).bindparams(
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


def test_batch_with_denied_source_is_authorization_not_allowed(
    consumed_uses, client
):
    machine_id, prepared = consumed_uses
    _fabricate_denied_use(client, machine_id)
    records = batch_payload(prepared)["records"]
    records[1] = record_payload(
        {"use_id": "use-denied"}, resource="res/d"
    )
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "authorization_not_allowed"}}
    assert receipt_count(client) == 0


def test_batch_with_scope_mismatch_leaves_no_partial_receipts(
    consumed_uses, client
):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    # The mismatch is on the last record: the earlier valid records must not
    # commit either.
    records[2]["action_type"] = "write"
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}
    assert receipt_count(client) == 0
    audit = client.get(integrity_url(machine_id)).json()
    assert audit == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def test_batch_with_existing_receipt_is_receipt_already_exists(
    consumed_uses, client
):
    machine_id, prepared = consumed_uses
    first_event, first_grant, first_use = prepared[0]
    assert (
        client.post(
            receipts_url(machine_id),
            json=record_payload(first_use, resource="res/0"),
        ).status_code
        == 201
    )

    response = client.post(batch_url(machine_id), json=batch_payload(prepared))
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "receipt_already_exists"}}
    # Only the pre-existing single receipt remains.
    assert receipt_count(client) == 1


def test_batch_first_rejection_in_request_order_wins(consumed_uses, client):
    machine_id, prepared = consumed_uses
    # Record 0 has a scope mismatch; record 1's use gets a receipt first.
    # The earlier record's rejection decides the outcome.
    second_event, second_grant, second_use = prepared[1]
    assert (
        client.post(
            receipts_url(machine_id),
            json=record_payload(second_use, resource="res/1"),
        ).status_code
        == 201
    )
    records = batch_payload(prepared)["records"]
    records[0]["resource"] = "res/other"
    response = client.post(batch_url(machine_id), json={"records": records})
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "execution_scope_mismatch"}}
    assert receipt_count(client) == 1


# --------------------------------------------------------------------------- #
# Concurrency: batches serialize, exactly one winner per use
# --------------------------------------------------------------------------- #


def test_concurrent_batches_sharing_a_use_have_exactly_one_winner(
    consumed_uses, client
):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    count = 8
    gate = threading.Event()

    def hit():
        gate.wait()
        return client.post(batch_url(machine_id), json={"records": records})

    with ThreadPoolExecutor(max_workers=count) as pool:
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
    assert receipt_count(client) == 3
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 3


def test_concurrent_disjoint_batches_all_commit_without_chain_fork(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*")
    batches = []
    for batch_index in range(4):
        records = []
        for item_index in range(3):
            resource = f"res/{batch_index}-{item_index}"
            event = record_event(client, machine_id, resource=resource).json()
            _, use = issue_and_consume(client, machine_id, event)
            records.append(record_payload(use, resource=resource))
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
    assert receipt_count(client) == 12
    audit = client.get(integrity_url(machine_id)).json()
    assert audit["valid"] is True
    assert audit["checked_count"] == 12


# --------------------------------------------------------------------------- #
# Non-interference and persistence
# --------------------------------------------------------------------------- #


def test_batch_rejection_does_not_consume_uses(consumed_uses, client):
    machine_id, prepared = consumed_uses
    records = batch_payload(prepared)["records"]
    records[0]["outcome"] = "bad-outcome"
    assert (
        client.post(batch_url(machine_id), json={"records": records}).status_code
        == 422
    )
    # The rejected batch never touched the uses: the correct batch still
    # succeeds exactly once.
    retry = client.post(batch_url(machine_id), json=batch_payload(prepared))
    assert retry.status_code == 201


def test_batch_records_are_readable_across_restart(consumed_uses, client):
    machine_id, prepared = consumed_uses
    created = client.post(batch_url(machine_id), json=batch_payload(prepared))
    assert created.status_code == 201
    expected = created.json()

    # A fresh app instance over the same database file sees the same rows.
    with TestClient(app) as restarted:
        audit = restarted.get(integrity_url(machine_id)).json()
        assert audit["valid"] is True
        assert audit["checked_count"] == 3
        page = restarted.get(
            f"/machines/{machine_id}/execution-receipts/changes?limit=10"
        ).json()
    assert [record["id"] for record in page["records"]] == [
        record["id"] for record in expected["records"]
    ]
    assert page["records"][0]["occurred_at"].endswith("Z")
