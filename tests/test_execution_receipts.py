"""Tests for the execution completion receipts.

A receipt binds one consumed authorization (the single use record of a
one-time grant) to the execution result it produced. Two entries sit under
the machine path:

    POST /machines/{machine_id}/execution-receipts
    GET  /machines/{machine_id}/execution-receipts/integrity

These tests cover the receipt fields and chaining (one receipt per use,
atomic commit, concurrent single winner), every validation and lookup
outcome, the read-only integrity audit (sound, empty, and each anomaly
category), machine isolation, restart persistence, and old-database
migration that never rewrites old data and never backfills old uses.
"""
import hashlib
import json
import sqlite3
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


def issue(client, machine_id, event_id, ttl_seconds=60):
    return client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume(client, machine_id, grant_id):
    return client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def receipts_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts"


def integrity_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/integrity"


DIGEST = "a" * 64


def complete(client, machine_id, use_id, action_type="read", resource="res/x",
             outcome="succeeded", result_digest=DIGEST):
    return client.post(
        receipts_url(machine_id),
        json={
            "use_id": use_id,
            "action_type": action_type,
            "resource": resource,
            "outcome": outcome,
            "result_digest": result_digest,
        },
    )


@pytest.fixture
def consumed_use(client):
    """A machine + allowed event + issued grant + consumed use."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert event["allowed"] is True
    assert event["reason"] == "allowed_by_policy"
    grant = issue(client, machine_id, event["id"]).json()
    use = consume(client, machine_id, grant["id"]).json()
    return machine_id, event, grant, use


def content_hash_of(record):
    document = json.dumps(
        {
            "id": record["id"],
            "machine_id": record["machine_id"],
            "use_id": record["use_id"],
            "grant_id": record["grant_id"],
            "authorization_event_id": record["authorization_event_id"],
            "action_type": record["action_type"],
            "resource": record["resource"],
            "outcome": record["outcome"],
            "result_digest": record["result_digest"],
            "occurred_at": record["occurred_at"],
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


RECEIPT_FIELDS = [
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


def stored_receipts(client, machine_id):
    """All stored receipt rows of the machine, in chain order."""
    with client.app.state.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM authorization_execution_receipts "
                "WHERE machine_id = :mid ORDER BY occurred_at, id"
            ).bindparams(mid=machine_id)
        ).all()
    return [row._mapping for row in rows]


# --------------------------------------------------------------------------- #
# Successful completion
# --------------------------------------------------------------------------- #


def test_complete_returns_receipt_with_chain_triple(consumed_use, client):
    machine_id, event, grant, use = consumed_use

    response = complete(client, machine_id, use["use_id"])
    assert response.status_code == 201
    receipt = response.json()
    assert list(receipt.keys()) == RECEIPT_FIELDS
    assert receipt["machine_id"] == machine_id
    assert receipt["use_id"] == use["use_id"]
    assert receipt["grant_id"] == grant["id"]
    assert receipt["authorization_event_id"] == event["id"]
    assert receipt["occurred_at"].endswith("Z")
    assert receipt["previous_receipt_id"] is None

    # The stored row carries the full content, and its hashes verify.
    (row,) = stored_receipts(client, machine_id)
    assert row["action_type"] == "read"
    assert row["resource"] == "res/x"
    assert row["outcome"] == "succeeded"
    assert row["result_digest"] == DIGEST
    assert row["occurred_at"] == receipt["occurred_at"]
    assert receipt["content_hash"] == content_hash_of(row)
    assert receipt["chain_hash"] == chain_hash_of("", row["content_hash"])


def test_second_receipt_chains_to_first(consumed_use, client):
    machine_id, event, grant, use = consumed_use
    first = complete(client, machine_id, use["use_id"]).json()

    second_event = record_event(client, machine_id, resource="res/y").json()
    second_grant = issue(client, machine_id, second_event["id"]).json()
    second_use = consume(client, machine_id, second_grant["id"]).json()
    second = complete(
        client, machine_id, second_use["use_id"], resource="res/y",
        outcome="failed",
    ).json()

    assert second["previous_receipt_id"] == first["id"]
    rows = stored_receipts(client, machine_id)
    assert len(rows) == 2
    assert rows[1]["outcome"] == "failed"
    assert second["content_hash"] == content_hash_of(rows[1])
    assert second["chain_hash"] == chain_hash_of(
        first["chain_hash"], rows[1]["content_hash"]
    )


def test_failed_completion_writes_nothing(consumed_use, client):
    machine_id, event, grant, use = consumed_use
    # Scope mismatch writes no receipt.
    response = complete(client, machine_id, use["use_id"], resource="res/zzz")
    assert response.status_code == 409
    assert stored_receipts(client, machine_id) == []


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #


def test_query_parameter_is_rejected(consumed_use, client):
    machine_id, _, _, use = consumed_use
    response = client.post(
        receipts_url(machine_id) + "?x=1",
        json={
            "use_id": use["use_id"],
            "action_type": "read",
            "resource": "res/x",
            "outcome": "succeeded",
            "result_digest": DIGEST,
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }
    # The query check wins over the machine lookup.
    response = client.post(receipts_url("missing-machine") + "?x=1", json={})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }


def test_body_shape_validation(consumed_use, client):
    machine_id, _, _, use = consumed_use
    valid = {
        "use_id": use["use_id"],
        "action_type": "read",
        "resource": "res/x",
        "outcome": "succeeded",
        "result_digest": DIGEST,
    }
    # Missing and extra fields.
    for key in list(valid):
        payload = {k: v for k, v in valid.items() if k != key}
        response = client.post(receipts_url(machine_id), json=payload)
        assert response.status_code == 422, key
        assert response.json() == {
            "error": {"code": "invalid_execution_receipt_request"}
        }
    response = client.post(
        receipts_url(machine_id), json={**valid, "extra": "x"}
    )
    assert response.status_code == 422
    # Non-object and missing bodies.
    for bad in ([], "text", 3, None):
        response = client.post(receipts_url(machine_id), json=bad)
        assert response.status_code == 422
    response = client.post(receipts_url(machine_id))
    assert response.status_code == 422
    # A malformed request against a missing machine is still 422.
    response = client.post(receipts_url("missing-machine"), json=[])
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }


def test_field_value_validation(consumed_use, client):
    machine_id, _, _, use = consumed_use

    def post(**overrides):
        payload = {
            "use_id": use["use_id"],
            "action_type": "read",
            "resource": "res/x",
            "outcome": "succeeded",
            "result_digest": DIGEST,
        }
        payload.update(overrides)
        return client.post(receipts_url(machine_id), json=payload)

    for bad_use_id in ("", "   ", 3, True, None, ["x"]):
        assert post(use_id=bad_use_id).status_code == 422, bad_use_id
    for bad_action in (3, True, None, ["read"]):
        assert post(action_type=bad_action).status_code == 422, bad_action
    for bad_resource in (3, False, None):
        assert post(resource=bad_resource).status_code == 422, bad_resource
    for bad_outcome in ("ok", "SUCCESS", "Succeeded", "", 1, True, None):
        assert post(outcome=bad_outcome).status_code == 422, bad_outcome
    for bad_digest in (
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "",
        3,
        None,
        True,
    ):
        assert post(result_digest=bad_digest).status_code == 422, bad_digest
    # Nothing was written by any rejected attempt.
    assert stored_receipts(client, machine_id) == []
    # The well-formed request still succeeds.
    assert post().status_code == 201
    assert post(outcome="failed").status_code == 409  # already completed


# --------------------------------------------------------------------------- #
# Lookups and conflicts
# --------------------------------------------------------------------------- #


def test_missing_machine_and_use_are_404(consumed_use, client):
    machine_id, _, _, use = consumed_use
    response = complete(client, "missing-machine", use["use_id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    response = complete(client, machine_id, "missing-use")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_cross_machine_use_is_404(consumed_use, client):
    machine_id, _, _, use = consumed_use
    other_id = create_machine(client, external_id="machine-2")
    response = complete(client, other_id, use["use_id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_unconsumed_grant_has_no_use_and_is_404(consumed_use, client):
    machine_id, event, _, _ = consumed_use
    # A second event's grant is issued but never consumed: no use exists.
    second_event = record_event(client, machine_id, resource="res/y").json()
    grant = issue(client, machine_id, second_event["id"]).json()
    response = complete(
        client, machine_id, grant["id"], resource="res/y"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_source_not_allowed_is_409(consumed_use, client):
    machine_id, event, _, use = consumed_use
    # Damage the committed source event so it no longer reads as a policy
    # allow (possible only by direct writes, never through the API).
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_events "
                "SET reason = 'no_matching_policy' WHERE id = :id"
            ).bindparams(id=event["id"])
        )
    response = complete(client, machine_id, use["use_id"])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "authorization_not_allowed"}}
    assert stored_receipts(client, machine_id) == []


def test_scope_mismatch_is_409(consumed_use, client):
    machine_id, _, _, use = consumed_use
    for overrides in (
        {"action_type": "write"},
        {"resource": "res/y"},
        {"action_type": "READ"},
        {"resource": "res/x "},
        {"action_type": "read", "resource": "res/*"},
    ):
        response = complete(client, machine_id, use["use_id"], **overrides)
        assert response.status_code == 409, overrides
        assert response.json() == {
            "error": {"code": "execution_scope_mismatch"}
        }
    assert stored_receipts(client, machine_id) == []


def test_one_receipt_per_use(consumed_use, client):
    machine_id, _, _, use = consumed_use
    assert complete(client, machine_id, use["use_id"]).status_code == 201
    response = complete(client, machine_id, use["use_id"])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "receipt_already_exists"}}
    # A different outcome or digest does not earn a second receipt.
    response = complete(
        client, machine_id, use["use_id"], outcome="failed",
        result_digest="b" * 64,
    )
    assert response.status_code == 409
    assert len(stored_receipts(client, machine_id)) == 1


def test_concurrent_completion_has_one_winner(consumed_use, client):
    machine_id, _, _, use = consumed_use

    barrier = threading.Barrier(8)

    def post_receipt():
        barrier.wait()
        return complete(client, machine_id, use["use_id"]).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: post_receipt(), range(8)))
    assert outcomes.count(201) == 1
    assert outcomes.count(409) == 7
    assert len(stored_receipts(client, machine_id)) == 1


def test_non_post_methods_on_collection_are_405(consumed_use, client):
    machine_id, _, _, _ = consumed_use
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(receipts_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Integrity endpoint
# --------------------------------------------------------------------------- #


def test_integrity_empty_chain(client):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def test_integrity_sound_chain(consumed_use, client):
    machine_id, _, _, use = consumed_use
    complete(client, machine_id, use["use_id"])
    second_event = record_event(client, machine_id, resource="res/y").json()
    grant = issue(client, machine_id, second_event["id"]).json()
    second_use = consume(client, machine_id, grant["id"]).json()
    complete(client, machine_id, second_use["use_id"], resource="res/y")

    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "valid",
        "checked_count",
        "broken_receipt_id",
        "anomaly",
    ]
    assert body == {
        "valid": True,
        "checked_count": 2,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def _tamper(client, sql, **params):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(sql).bindparams(**params))


def test_integrity_timestamp_unparseable(consumed_use, client):
    machine_id, _, _, use = consumed_use
    receipt = complete(client, machine_id, use["use_id"]).json()
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET occurred_at = 'not-a-moment' WHERE id = :id",
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_receipt_id": receipt["id"],
        "anomaly": "timestamp_unparseable",
    }


def test_integrity_chain_break(consumed_use, client):
    machine_id, _, _, use = consumed_use
    complete(client, machine_id, use["use_id"])
    second_event = record_event(client, machine_id, resource="res/y").json()
    grant = issue(client, machine_id, second_event["id"]).json()
    second_use = consume(client, machine_id, grant["id"]).json()
    second = complete(
        client, machine_id, second_use["use_id"], resource="res/y"
    ).json()

    # A forged previous-receipt link breaks the chain on the second record.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET previous_receipt_id = 'bogus' WHERE id = :id",
        id=second["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["broken_receipt_id"] == second["id"]
    assert body["anomaly"] == "chain_break"

    # A forged chain digest breaks it too.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET chain_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=second["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["anomaly"] == "chain_break"
    assert body["broken_receipt_id"] == second["id"]


def test_integrity_ownership_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    other_id = create_machine(client, external_id="machine-2")
    receipt = complete(client, machine_id, use["use_id"]).json()

    # The referenced use is re-owned by another machine.
    _tamper(
        client,
        "UPDATE authorization_grant_uses SET machine_id = :mid "
        "WHERE id = :id",
        mid=other_id,
        id=use["use_id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["broken_receipt_id"] == receipt["id"]
    assert body["anomaly"] == "ownership_mismatch"


def test_integrity_use_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    receipt = complete(client, machine_id, use["use_id"]).json()

    # The receipt names a use that does not exist.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET use_id = 'missing-use' WHERE id = :id",
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["anomaly"] == "use_mismatch"
    assert body["broken_receipt_id"] == receipt["id"]

    # Restore, then break the grant linkage instead.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET use_id = :use_id, grant_id = 'missing-grant' WHERE id = :id",
        use_id=use["use_id"],
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["anomaly"] == "use_mismatch"
    assert body["broken_receipt_id"] == receipt["id"]


def test_integrity_scope_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    receipt = complete(client, machine_id, use["use_id"]).json()
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET resource = 'res/elsewhere' WHERE id = :id",
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["broken_receipt_id"] == receipt["id"]
    assert body["anomaly"] == "scope_mismatch"


def test_integrity_digest_mismatch(consumed_use, client):
    machine_id, _, _, use = consumed_use
    receipt = complete(client, machine_id, use["use_id"]).json()

    # A tampered content field no longer matches the stored digest.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET outcome = 'failed' WHERE id = :id",
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["anomaly"] == "digest_mismatch"
    assert body["broken_receipt_id"] == receipt["id"]

    # A malformed stored result digest is a digest anomaly too.
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET outcome = 'succeeded', result_digest = :digest WHERE id = :id",
        digest="Z" * 64,
        id=receipt["id"],
    )
    body = client.get(integrity_url(machine_id)).json()
    assert body["anomaly"] == "digest_mismatch"


def test_integrity_isolated_per_machine(consumed_use, client):
    machine_id, _, _, use = consumed_use
    other_id = create_machine(client, external_id="machine-2")
    receipt = complete(client, machine_id, use["use_id"]).json()
    _tamper(
        client,
        "UPDATE authorization_execution_receipts "
        "SET content_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=receipt["id"],
    )

    # The other machine's empty chain is unaffected by the damage.
    assert client.get(integrity_url(other_id)).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_receipt_id": None,
        "anomaly": None,
    }
    assert client.get(integrity_url(machine_id)).json()["valid"] is False


def test_integrity_rejects_query_and_body(consumed_use, client):
    machine_id, _, _, _ = consumed_use
    response = client.get(integrity_url(machine_id), params={"limit": 1})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }
    response = client.request(
        "GET",
        integrity_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }
    # The query check wins over the machine lookup.
    response = client.get(integrity_url("missing-machine"), params={"x": "1"})
    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_execution_receipt_request"}
    }


def test_integrity_missing_machine_is_404(client):
    response = client.get(integrity_url("missing-machine"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_non_get_methods_are_405(consumed_use, client):
    machine_id, _, _, _ = consumed_use
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine_id))
        assert response.status_code == 405


def test_integrity_repeat_reads_are_identical(consumed_use, client):
    machine_id, _, _, use = consumed_use
    complete(client, machine_id, use["use_id"])
    first = client.get(integrity_url(machine_id))
    second = client.get(integrity_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content


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
        use = consume(first, machine_id, grant["id"]).json()
        receipt = complete(first, machine_id, use["use_id"]).json()

    with TestClient(app) as second:
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
            "anomaly": None,
        }
        # A duplicate completion is still rejected after the restart.
        response = complete(second, machine_id, use["use_id"])
        assert response.status_code == 409
        assert response.json() == {
            "error": {"code": "receipt_already_exists"}
        }
        (row,) = stored_receipts(second, machine_id)
        assert row["id"] == receipt["id"]


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
        use = consume(first, machine_id, grant["id"]).json()

    # Reproduce a database that predates the receipt feature entirely.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_execution_receipts")
    use_rows = conn.execute(
        "SELECT id, grant_id, machine_id, event_id, consumed_at "
        "FROM authorization_grant_uses"
    ).fetchall()
    grant_rows = conn.execute(
        "SELECT id, machine_id, event_id, issued_at, expires_at, status, "
        "consumed_at, revoked_at FROM authorization_grants"
    ).fetchall()
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # The table is recreated empty: no receipt is fabricated for the
        # historical use, and the old use and grant rows are untouched.
        assert stored_receipts(second, machine_id) == []
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 0,
            "broken_receipt_id": None,
            "anomaly": None,
        }
        with second.app.state.engine.connect() as conn:
            uses = conn.execute(
                text(
                    "SELECT id, grant_id, machine_id, event_id, consumed_at "
                    "FROM authorization_grant_uses"
                )
            ).all()
            grants = conn.execute(
                text(
                    "SELECT id, machine_id, event_id, issued_at, expires_at, "
                    "status, consumed_at, revoked_at FROM authorization_grants"
                )
            ).all()
        assert [tuple(row) for row in uses] == use_rows
        assert [tuple(row) for row in grants] == grant_rows

        # The old use can still be completed on the migrated database, and
        # the new receipt chains as the machine's first.
        response = complete(second, machine_id, use["use_id"])
        assert response.status_code == 201
        receipt = response.json()
        assert receipt["previous_receipt_id"] is None
        assert second.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_receipt_id": None,
            "anomaly": None,
        }
