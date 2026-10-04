"""Tests for the read-only execution-receipt summary entry.

    GET /machines/{machine_id}/execution-receipts/summary

The summary counts one machine's execution-completion receipts and its
consumed authorization grants: receipt rows are bucketed by their stored
``outcome`` (verbatim ``succeeded``/``failed`` against every other value)
and by their stored ``result_digest`` (exactly 64 lowercase hexadecimal
characters against every other value), and consumed uses that no receipt
of the machine names are reported as ``missing_receipt_count``. These
tests cover the empty/full/mixed success shapes, the exact field order and
compact body, damaged stored values counted as invalid (never hidden,
repaired, or folded into success/failure), machine isolation, strict
read-only behavior, the 422/404/405/500 outcomes, and non-interference
with the existing receipt entries.
"""
import hashlib

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


def summary_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/summary"


def coverage_url(machine_id):
    return f"/machines/{machine_id}/execution-receipts/coverage"


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


MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"

SUMMARY_KEYS = [
    "machine_id",
    "total_receipts",
    "succeeded_count",
    "failed_count",
    "invalid_outcome_count",
    "valid_result_digest_count",
    "invalid_result_digest_count",
    "missing_receipt_count",
]


@pytest.fixture
def allowed_machine(client):
    """A machine that can mint and consume allow grants."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    return machine_id


def new_consumed_use(client, machine_id, resource="res/x"):
    event = record_event(client, machine_id, resource=resource).json()
    assert event["allowed"] is True
    _, use = issue_and_consume(client, machine_id, event)
    return use


# Direct-row writers for damage scenarios that the public API cannot
# construct. SQLite here does not enforce the foreign keys, so a use row
# may name placeholder grant/event ids and a receipt row may carry any
# stored outcome or digest text.


def insert_use_row(client, machine_id, use_id, consumed_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_grant_uses (id, grant_id, "
                "machine_id, event_id, consumed_at) VALUES "
                "(:id, :grant, :machine, :event, :consumed_at)"
            ).bindparams(
                id=use_id,
                grant=f"grant-of-{use_id}",
                machine=machine_id,
                event=f"event-of-{use_id}",
                consumed_at=consumed_at,
            )
        )


def insert_receipt_row(client, machine_id, receipt_id, use_id,
                       outcome="succeeded", digest="a" * 64):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts (id, machine_id, use_id, "
                "grant_id, authorization_event_id, action_type, resource, "
                "outcome, result_digest, occurred_at) VALUES "
                "(:id, :machine, :use_id, 'grant-x', 'event-x', 'read', "
                "'res/x', :outcome, :digest, '2026-03-01T00:00:00Z')"
            ).bindparams(
                id=receipt_id,
                machine=machine_id,
                use_id=use_id,
                outcome=outcome,
                digest=digest,
            )
        )


def tamper(engine, receipt_id, column, value):
    with engine.begin() as conn:
        conn.execute(
            text(
                f"UPDATE execution_receipts SET {column} = :v WHERE id = :id"
            ).bindparams(v=value, id=receipt_id)
        )


# --------------------------------------------------------------------------- #
# Success shape
# --------------------------------------------------------------------------- #


def test_empty_machine_reports_all_zeros_with_exact_body(client):
    machine_id = create_machine(client)
    response = client.get(summary_url(machine_id))
    assert response.status_code == 200
    assert list(response.json().keys()) == SUMMARY_KEYS
    assert response.content == (
        b'{"machine_id":"' + machine_id.encode()
        + b'","total_receipts":0,"succeeded_count":0,"failed_count":0,'
        b'"invalid_outcome_count":0,"valid_result_digest_count":0,'
        b'"invalid_result_digest_count":0,"missing_receipt_count":0}\n'
    )


def test_counts_are_integers_when_empty(client):
    machine_id = create_machine(client)
    body = client.get(summary_url(machine_id)).json()
    for key in SUMMARY_KEYS[1:]:
        assert isinstance(body[key], int)
        assert not isinstance(body[key], bool)
        assert body[key] == 0


def test_succeeded_and_failed_receipts_are_counted(allowed_machine, client):
    machine_id = allowed_machine
    use_one = new_consumed_use(client, machine_id, resource="res/1")
    use_two = new_consumed_use(client, machine_id, resource="res/2")
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_one, resource="res/1", outcome="succeeded"),
    )
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(
            use_two, resource="res/2", outcome="failed",
            digest=digest_of(b"other"),
        ),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body == {
        "machine_id": machine_id,
        "total_receipts": 2,
        "succeeded_count": 1,
        "failed_count": 1,
        "invalid_outcome_count": 0,
        "valid_result_digest_count": 2,
        "invalid_result_digest_count": 0,
        "missing_receipt_count": 0,
    }


def test_consumed_uses_without_receipts_are_missing(allowed_machine, client):
    machine_id = allowed_machine
    new_consumed_use(client, machine_id, resource="res/1")
    new_consumed_use(client, machine_id, resource="res/2")

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 0
    assert body["missing_receipt_count"] == 2


def test_mixed_receipts_and_missing_uses(allowed_machine, client):
    machine_id = allowed_machine
    use_one = new_consumed_use(client, machine_id, resource="res/1")
    new_consumed_use(client, machine_id, resource="res/2")
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_one, resource="res/1"),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 1
    assert body["valid_result_digest_count"] == 1
    assert body["missing_receipt_count"] == 1


def test_later_legitimate_receipt_reduces_missing_count(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)

    before = client.get(summary_url(machine_id)).json()
    assert before["missing_receipt_count"] == 1

    response = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert response.status_code == 201

    after = client.get(summary_url(machine_id)).json()
    assert after["total_receipts"] == 1
    assert after["succeeded_count"] == 1
    assert after["missing_receipt_count"] == 0


# --------------------------------------------------------------------------- #
# Damaged stored values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "outcome",
    ["Succeeded", "SUCCEEDED", "success", "ok", "", "succeeded "],
)
def test_non_verbatim_outcomes_are_invalid_not_success_or_failure(
    client, outcome
):
    machine_id = create_machine(client)
    insert_receipt_row(client, machine_id, "r-1", "u-1", outcome=outcome)

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0
    assert body["invalid_outcome_count"] == 1


@pytest.mark.parametrize(
    "digest",
    [
        "A" * 64,          # uppercase hex
        "a" * 63,          # too short
        "a" * 65,          # too long
        "g" * 64,          # non-hex
        "a" * 63 + " ",    # trailing whitespace
        "",                # empty
    ],
)
def test_non_lower_hex_64_digests_are_invalid(client, digest):
    machine_id = create_machine(client)
    insert_receipt_row(client, machine_id, "r-1", "u-1", digest=digest)

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["valid_result_digest_count"] == 0
    assert body["invalid_result_digest_count"] == 1
    # The outcome bucket is unaffected by the damaged digest.
    assert body["succeeded_count"] == 1


def test_damaged_values_are_counted_verbatim_never_repaired(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    engine = client.app.state.engine
    tamper(engine, receipt["id"], "outcome", "SUCCESS")
    tamper(engine, receipt["id"], "result_digest", "F" * 64)

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["invalid_outcome_count"] == 1
    assert body["valid_result_digest_count"] == 0
    assert body["invalid_result_digest_count"] == 1
    assert body["missing_receipt_count"] == 0

    # The stored values are left exactly as tampered — never repaired or
    # recomputed by the read.
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT outcome, result_digest FROM execution_receipts "
                "WHERE id = :id"
            ).bindparams(id=receipt["id"])
        ).one()
    assert row == ("SUCCESS", "F" * 64)


def test_chain_damaged_receipt_still_counts_normally(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    tamper(client.app.state.engine, receipt["id"], "chain_hash", "f" * 64)

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 1
    assert body["valid_result_digest_count"] == 1
    assert body["missing_receipt_count"] == 0

    integrity = client.get(integrity_url(machine_id)).json()
    assert integrity["valid"] is False
    assert integrity["anomaly"] == "chain_break"


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_summary_is_isolated_per_machine(allowed_machine, client):
    machine_a = allowed_machine
    use_a = new_consumed_use(client, machine_a, resource="res/a")
    client.post(
        receipts_url(machine_a),
        json=receipt_payload(use_a, resource="res/a"),
    )

    machine_b = create_machine(client, external_id="machine-2")
    declare(client, machine_b, resource_pattern="res/*")
    new_consumed_use(client, machine_b, resource="res/b")
    insert_receipt_row(
        client, machine_b, "b-damaged", "b-orphan-use", outcome="weird",
        digest="not-hex",
    )

    body_a = client.get(summary_url(machine_a)).json()
    assert body_a == {
        "machine_id": machine_a,
        "total_receipts": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "invalid_outcome_count": 0,
        "valid_result_digest_count": 1,
        "invalid_result_digest_count": 0,
        "missing_receipt_count": 0,
    }

    body_b = client.get(summary_url(machine_b)).json()
    assert body_b == {
        "machine_id": machine_b,
        "total_receipts": 1,
        "succeeded_count": 0,
        "failed_count": 0,
        "invalid_outcome_count": 1,
        "valid_result_digest_count": 0,
        "invalid_result_digest_count": 1,
        "missing_receipt_count": 1,
    }


def test_other_machine_records_do_not_change_empty_summary(
    allowed_machine, client
):
    machine_a = allowed_machine
    use_a = new_consumed_use(client, machine_a)
    insert_receipt_row(
        client, machine_a, "a-damaged", "nobody-use", outcome="???",
        digest="zz",
    )

    machine_b = create_machine(client, external_id="machine-2")
    body_b = client.get(summary_url(machine_b)).json()
    assert body_b == {
        "machine_id": machine_b,
        "total_receipts": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "invalid_outcome_count": 0,
        "valid_result_digest_count": 0,
        "invalid_result_digest_count": 0,
        "missing_receipt_count": 0,
    }


def test_receipt_of_another_machine_does_not_cover_a_use(
    allowed_machine, client
):
    machine_a = allowed_machine
    use_a = new_consumed_use(client, machine_a, resource="res/a")

    machine_b = create_machine(client, external_id="machine-2")
    # A receipt *owned by B* that names A's consumption cannot cover it.
    insert_receipt_row(
        client, machine_b, "b-foreign-receipt", use_a["use_id"],
    )

    body_a = client.get(summary_url(machine_a)).json()
    assert body_a["total_receipts"] == 0
    assert body_a["missing_receipt_count"] == 1


# --------------------------------------------------------------------------- #
# Read-only behavior
# --------------------------------------------------------------------------- #


def test_summary_is_strictly_read_only_and_byte_identical_on_repeat(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    insert_receipt_row(
        client, machine_id, "damaged-1", "elsewhere-use", outcome="odd",
        digest="xx",
    )

    def table_counts():
        with client.app.state.engine.connect() as conn:
            return (
                conn.execute(
                    text("SELECT COUNT(*) FROM authorization_grant_uses")
                ).scalar_one(),
                conn.execute(
                    text("SELECT COUNT(*) FROM execution_receipts")
                ).scalar_one(),
            )

    before_counts = table_counts()
    first = client.get(summary_url(machine_id))
    second = client.get(summary_url(machine_id))
    third = client.get(summary_url(machine_id))
    assert first.content == second.content == third.content
    assert table_counts() == before_counts
    assert first.json()["missing_receipt_count"] == 1

    # The gap is reported, not filled: a later legitimate receipt closes it.
    assert (
        client.post(
            receipts_url(machine_id), json=receipt_payload(use)
        ).status_code
        == 201
    )
    assert client.get(summary_url(machine_id)).json()[
        "missing_receipt_count"
    ] == 0


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?x=1",
        "?valid=true",
        "?=",
        "?x=1&x=2",
        "?limit=1",
        "?machine_id=m",
        "?outcome=succeeded",
    ],
)
def test_any_query_parameter_is_invalid_query_before_lookup(client, query):
    machine_id = create_machine(client)
    response = client.get(summary_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine lookup and all record reads.
    response = client.get(summary_url(MISSING_MACHINE) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_carried_body_is_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        summary_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        summary_url(MISSING_MACHINE),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_valid_query_against_missing_machine_is_not_found(client):
    response = client.get(summary_url(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"total_receipts" not in response.content


@pytest.mark.parametrize(
    "method", ["post", "put", "patch", "delete", "head"]
)
def test_only_get_is_routed(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(summary_url(machine_id))
    assert response.status_code == 405


def test_other_methods_do_not_read_records(client):
    # Drop both record tables: any read attempted on a non-GET method would
    # surface as 500, but method routing must answer 405 first.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_uses"))
        conn.execute(text("DROP TABLE execution_receipts"))
    for method in ("post", "put", "patch", "delete", "head"):
        response = getattr(client, method)(summary_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read failures
# --------------------------------------------------------------------------- #


def test_use_read_failure_is_500_with_no_partial_summary(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_uses"))
    response = client.get(summary_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"missing_receipt_count" not in response.content


def test_receipt_read_failure_is_500_with_no_partial_summary(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))
    response = client.get(summary_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"total_receipts" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_summary(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(summary_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"succeeded_count" not in response.content


# --------------------------------------------------------------------------- #
# Non-interference with existing entries
# --------------------------------------------------------------------------- #


def test_summary_does_not_change_receipt_collection_or_other_reads(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    integrity_before = client.get(integrity_url(machine_id)).content
    coverage_before = client.get(coverage_url(machine_id)).content

    for _ in range(3):
        response = client.get(summary_url(machine_id))
        assert response.status_code == 200

    # The collection still refuses a second receipt exactly once and the
    # other read-only conclusions are unchanged.
    repeat = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "receipt_already_exists"}}
    assert client.get(integrity_url(machine_id)).content == integrity_before
    assert client.get(coverage_url(machine_id)).content == coverage_before
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text(
                "SELECT id FROM execution_receipts WHERE use_id = :u"
            ).bindparams(u=use["use_id"])
        ).scalar_one() == receipt["id"]
