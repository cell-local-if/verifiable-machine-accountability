"""Tests for the read-only execution-receipt summary entry.

    GET /machines/{machine_id}/execution-receipts/summary

The summary rolls one machine's execution-completion receipts up to counts
of the stored results and counts the consumed authorizations that still
carry no receipt: ``outcome`` is counted verbatim (``succeeded`` /
``failed`` / anything else invalid), ``result_digest`` is valid only when
it is exactly 64 lowercase hexadecimal characters, and a consumption with
no receipt contributes one to ``missing_receipt_count``. These tests cover
the empty/full/mixed success shapes and exact compact body, damaged
outcome/digest values, machine isolation, orphan receipts, strict
read-only byte stability, and the 422/404/405/500 outcomes.
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


def digest_of(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


GOOD_DIGEST = digest_of(b"result")


def receipt_payload(use, action_type="read", resource="res/x",
                    outcome="succeeded", digest=None):
    return {
        "use_id": use["use_id"],
        "action_type": action_type,
        "resource": resource,
        "result_digest": digest or GOOD_DIGEST,
        "outcome": outcome,
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


# Direct-row writers for damage scenarios the public API cannot construct.
# SQLite here does not enforce the foreign keys.


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


def insert_receipt_row(
    client,
    machine_id,
    receipt_id,
    use_id,
    *,
    outcome="succeeded",
    digest=GOOD_DIGEST,
    occurred_at="2026-03-01T00:00:00Z",
):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts (id, machine_id, use_id, "
                "grant_id, authorization_event_id, action_type, resource, "
                "outcome, result_digest, occurred_at) VALUES "
                "(:id, :machine, :use_id, 'grant-x', 'event-x', 'read', "
                "'res/x', :outcome, :digest, :occurred_at)"
            ).bindparams(
                id=receipt_id,
                machine=machine_id,
                use_id=use_id,
                outcome=outcome,
                digest=digest,
                occurred_at=occurred_at,
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


def test_empty_machine_reports_all_zero_with_exact_body(client):
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


def test_every_count_is_a_non_negative_integer(client):
    machine_id = create_machine(client)
    body = client.get(summary_url(machine_id)).json()
    for key in SUMMARY_KEYS[1:]:
        assert isinstance(body[key], int)
        assert body[key] >= 0
        assert body[key] is not True and body[key] is not False


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
        json=receipt_payload(use_two, resource="res/2", outcome="failed"),
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


def test_consumed_authorizations_without_receipts_are_missing(
    allowed_machine, client
):
    machine_id = allowed_machine
    new_consumed_use(client, machine_id)
    new_consumed_use(client, machine_id, resource="res/2")

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 0
    assert body["missing_receipt_count"] == 2


def test_mixed_state_counts_each_bucket(allowed_machine, client):
    machine_id = allowed_machine
    covered = new_consumed_use(client, machine_id, resource="res/1")
    new_consumed_use(client, machine_id, resource="res/2")
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(covered, resource="res/1", outcome="failed"),
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 1
    assert body["invalid_outcome_count"] == 0
    assert body["valid_result_digest_count"] == 1
    assert body["invalid_result_digest_count"] == 0
    assert body["missing_receipt_count"] == 1


# --------------------------------------------------------------------------- #
# Damaged / invalid stored values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_outcome",
    ["", "SUCCEEDED", "succeeded ", "cancelled", "unknown", "error"],
)
def test_non_verdict_outcomes_go_to_invalid_outcome(client, bad_outcome):
    machine_id = create_machine(client)
    insert_receipt_row(
        client, machine_id, "receipt-1", "use-x", outcome=bad_outcome
    )
    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0
    assert body["invalid_outcome_count"] == 1
    assert body["valid_result_digest_count"] == 1
    assert body["invalid_result_digest_count"] == 0
    # The receipt names no consumption of this machine, so it is an orphan:
    # counted in the receipt totals but never a missing consumption.
    assert body["missing_receipt_count"] == 0


def test_non_text_outcome_goes_to_invalid_outcome(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts (id, machine_id, use_id, "
                "grant_id, authorization_event_id, action_type, resource, "
                "outcome, result_digest, occurred_at) VALUES "
                "('r-int', :m, 'use-int', 'g', 'e', 'read', 'res/x', "
                "7, :d, '2026-03-01T00:00:01Z')"
            ).bindparams(m=machine_id, d=GOOD_DIGEST)
        )

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0
    assert body["invalid_outcome_count"] == 1


@pytest.mark.parametrize(
    "bad_digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "z" * 64,
        " " + "a" * 63,
    ],
)
def test_non_64_lower_hex_digests_go_to_invalid_digest(client, bad_digest):
    machine_id = create_machine(client)
    insert_receipt_row(
        client, machine_id, "receipt-1", "use-x", digest=bad_digest
    )
    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["valid_result_digest_count"] == 0
    assert body["invalid_result_digest_count"] == 1


def test_non_text_digest_goes_to_invalid_digest(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts (id, machine_id, use_id, "
                "grant_id, authorization_event_id, action_type, resource, "
                "outcome, result_digest, occurred_at) VALUES "
                "('r-int', :m, 'use-x', 'g', 'e', 'read', 'res/x', "
                "'succeeded', 12345, '2026-03-01T00:00:00Z')"
            ).bindparams(m=machine_id)
        )
    body = client.get(summary_url(machine_id)).json()
    assert body["succeeded_count"] == 1
    assert body["invalid_outcome_count"] == 0
    assert body["valid_result_digest_count"] == 0
    assert body["invalid_result_digest_count"] == 1


def test_damaged_values_are_not_hidden_or_refolded(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    tamper(client.app.state.engine, receipt["id"], "outcome", "crashed")
    tamper(
        client.app.state.engine, receipt["id"], "result_digest", "X" * 64
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0
    assert body["invalid_outcome_count"] == 1
    assert body["valid_result_digest_count"] == 0
    assert body["invalid_result_digest_count"] == 1
    assert body["missing_receipt_count"] == 0

    # The damaged row is left exactly as stored.
    with client.app.state.engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT outcome, result_digest FROM execution_receipts "
                "WHERE id = :id"
            ).bindparams(id=receipt["id"])
        ).one()
        assert row == ("crashed", "X" * 64)


def test_outcome_and_digest_buckets_are_independent(client):
    machine_id = create_machine(client)
    # Valid outcome, invalid digest.
    insert_receipt_row(
        client, machine_id, "r-1", "u-1", outcome="succeeded", digest="bad"
    )
    # Invalid outcome, valid digest.
    insert_receipt_row(
        client,
        machine_id,
        "r-2",
        "u-2",
        outcome="cancelled",
        digest=GOOD_DIGEST,
    )
    # Both invalid.
    insert_receipt_row(
        client, machine_id, "r-3", "u-3", outcome="weird", digest=12345
    )
    # Both valid.
    insert_receipt_row(
        client,
        machine_id,
        "r-4",
        "u-4",
        outcome="failed",
        digest=GOOD_DIGEST,
    )

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 4
    assert body["succeeded_count"] == 1
    assert body["failed_count"] == 1
    assert body["invalid_outcome_count"] == 2
    assert body["valid_result_digest_count"] == 2
    assert body["invalid_result_digest_count"] == 2


# --------------------------------------------------------------------------- #
# Missing receipts, orphans, and distinct consumption counting
# --------------------------------------------------------------------------- #


def test_orphan_receipt_counts_in_totals_but_not_missing(client):
    machine_id = create_machine(client)
    insert_use_row(client, machine_id, "use-missing", "2026-03-01T00:00:00Z")
    # A receipt naming no consumption of this machine.
    insert_receipt_row(client, machine_id, "r-orphan", "use-from-elsewhere")

    body = client.get(summary_url(machine_id)).json()
    assert body["total_receipts"] == 1
    assert body["succeeded_count"] == 1
    assert body["missing_receipt_count"] == 1


def test_each_consumption_is_counted_at_most_once(allowed_machine, client):
    machine_id = allowed_machine
    new_consumed_use(client, machine_id, resource="res/1")
    new_consumed_use(client, machine_id, resource="res/2")
    new_consumed_use(client, machine_id, resource="res/3")

    body = client.get(summary_url(machine_id)).json()
    assert body["missing_receipt_count"] == 3
    # Repeated reads do not double count.
    again = client.get(summary_url(machine_id)).json()
    assert again["missing_receipt_count"] == 3


def test_legitimate_receipt_reduces_missing_count(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    assert (
        client.get(summary_url(machine_id)).json()["missing_receipt_count"]
        == 1
    )
    response = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert response.status_code == 201
    after = client.get(summary_url(machine_id)).json()
    assert after["missing_receipt_count"] == 0
    assert after["total_receipts"] == 1


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_summary_is_isolated_per_machine(allowed_machine, client):
    machine_a = allowed_machine
    use_a = new_consumed_use(client, machine_a, resource="res/a")
    # A receipt owned by A with a damaged outcome and digest.
    insert_receipt_row(
        client,
        machine_a,
        "a-damaged",
        "a-orphan-use",
        outcome="bogus",
        digest="nope",
    )

    machine_b = create_machine(client, external_id="machine-2")
    declare(client, machine_b, resource_pattern="res/*")
    use_b = new_consumed_use(client, machine_b, resource="res/b")
    client.post(
        receipts_url(machine_b),
        json=receipt_payload(use_b, resource="res/b"),
    )
    # A receipt owned by B that happens to name A's consumption id.
    insert_receipt_row(
        client, machine_b, "b-foreign", use_a["use_id"], outcome="failed"
    )

    report_a = client.get(summary_url(machine_a)).json()
    assert report_a == {
        "machine_id": machine_a,
        "total_receipts": 1,
        "succeeded_count": 0,
        "failed_count": 0,
        "invalid_outcome_count": 1,
        "valid_result_digest_count": 0,
        "invalid_result_digest_count": 1,
        "missing_receipt_count": 1,
    }

    report_b = client.get(summary_url(machine_b)).json()
    assert report_b["total_receipts"] == 2
    assert report_b["succeeded_count"] == 1
    assert report_b["failed_count"] == 1
    assert report_b["invalid_outcome_count"] == 0
    assert report_b["valid_result_digest_count"] == 2
    assert report_b["invalid_result_digest_count"] == 0
    # B's foreign-named receipt does not cover A's use from B's view, and
    # B's own consumed use is covered by its legitimate receipt.
    assert report_b["missing_receipt_count"] == 0


def test_other_machine_records_do_not_change_empty_summary(
    allowed_machine, client
):
    machine_a = allowed_machine
    new_consumed_use(client, machine_a)
    insert_receipt_row(
        client, machine_a, "a-damaged", "elsewhere", outcome="weird"
    )

    machine_b = create_machine(client, external_id="machine-2")
    report_b = client.get(summary_url(machine_b)).json()
    assert report_b == {
        "machine_id": machine_b,
        "total_receipts": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "invalid_outcome_count": 0,
        "valid_result_digest_count": 0,
        "invalid_result_digest_count": 0,
        "missing_receipt_count": 0,
    }


# --------------------------------------------------------------------------- #
# Read-only behavior and byte stability
# --------------------------------------------------------------------------- #


def test_summary_is_read_only_and_byte_identical(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    insert_receipt_row(
        client,
        machine_id,
        "damaged",
        "orphan-use",
        outcome="weird",
        digest="bad",
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

    before = table_counts()
    first = client.get(summary_url(machine_id))
    second = client.get(summary_url(machine_id))
    third = client.get(summary_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content == third.content
    assert table_counts() == before

    # The un-receipted consumption is still closeable by a real receipt.
    assert (
        client.post(
            receipts_url(machine_id), json=receipt_payload(use)
        ).status_code
        == 201
    )


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
    assert b"missing_receipt_count" not in response.content


# --------------------------------------------------------------------------- #
# Non-interference with existing entries
# --------------------------------------------------------------------------- #


def test_summary_does_not_change_receipt_writes_or_integrity(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    integrity_before = client.get(
        f"/machines/{machine_id}/execution-receipts/integrity"
    ).content
    coverage_before = client.get(
        f"/machines/{machine_id}/execution-receipts/coverage"
    ).content

    for _ in range(3):
        response = client.get(summary_url(machine_id))
        assert response.status_code == 200

    repeat = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "receipt_already_exists"}}
    assert (
        client.get(
            f"/machines/{machine_id}/execution-receipts/integrity"
        ).content
        == integrity_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/execution-receipts/coverage"
        ).content
        == coverage_before
    )
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT id FROM execution_receipts WHERE use_id = :u").bindparams(
                u=use["use_id"]
            )
        ).scalar_one() == receipt["id"]
