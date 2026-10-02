"""Tests for the read-only execution-receipt coverage entry.

    GET /machines/{machine_id}/execution-receipts/coverage

The coverage report matches one machine's authorization-grant consumption
records to its execution-completion receipts by ``use_id`` alone: it reports
consumed uses with no receipt (``missing_use_ids``) and receipts whose
``use_id`` is not one of the machine's consumptions (``orphan_receipt_ids``),
and is ``valid`` exactly when both sets are empty. These tests cover the
empty/full/mixed success shapes, the exact field order and compact body,
instant-then-id ordering (including the exact-second/fractional-second
boundary and unparseable stamps sorting last), machine isolation,
independence from the integrity chain verdict, strict read-only behavior,
improvement after a later legitimate receipt, the 422/404/405/500 outcomes,
and non-interference with the existing receipt entries.
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

COVERAGE_KEYS = [
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


# Direct-row writers for ordering/damage scenarios that the public API
# cannot construct. SQLite here does not enforce the foreign keys, so a use
# row may name placeholder grant/event ids and a receipt row may name a
# use_id absent from the machine's consumption set.


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


def insert_receipt_row(client, machine_id, receipt_id, use_id, occurred_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO execution_receipts (id, machine_id, use_id, "
                "grant_id, authorization_event_id, action_type, resource, "
                "outcome, result_digest, occurred_at) VALUES "
                "(:id, :machine, :use_id, 'grant-x', 'event-x', 'read', "
                "'res/x', 'succeeded', :digest, :occurred_at)"
            ).bindparams(
                id=receipt_id,
                machine=machine_id,
                use_id=use_id,
                digest="a" * 64,
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


def test_empty_machine_is_fully_valid_with_exact_body(client):
    machine_id = create_machine(client)
    response = client.get(coverage_url(machine_id))
    assert response.status_code == 200
    assert list(response.json().keys()) == COVERAGE_KEYS
    assert response.content == (
        b'{"machine_id":"' + machine_id.encode()
        + b'","consumed_count":0,"receipt_count":0,"covered_count":0,'
        b'"missing_count":0,"missing_use_ids":[],"orphan_count":0,'
        b'"orphan_receipt_ids":[],"valid":true}\n'
    )


def test_counts_are_integers_and_lists_are_present_when_empty(client):
    machine_id = create_machine(client)
    body = client.get(coverage_url(machine_id)).json()
    for key in (
        "consumed_count",
        "receipt_count",
        "covered_count",
        "missing_count",
        "orphan_count",
    ):
        assert isinstance(body[key], int)
    assert body["missing_use_ids"] == []
    assert body["orphan_receipt_ids"] == []


def test_every_consumed_use_with_a_receipt_is_valid(allowed_machine, client):
    machine_id = allowed_machine
    use_one = new_consumed_use(client, machine_id, resource="res/1")
    use_two = new_consumed_use(client, machine_id, resource="res/2")
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_one, resource="res/1"),
    )
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_two, resource="res/2"),
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body == {
        "machine_id": machine_id,
        "consumed_count": 2,
        "receipt_count": 2,
        "covered_count": 2,
        "missing_count": 0,
        "missing_use_ids": [],
        "orphan_count": 0,
        "orphan_receipt_ids": [],
        "valid": True,
    }


def test_consumed_uses_without_receipts_are_missing(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)

    body = client.get(coverage_url(machine_id)).json()
    assert body["consumed_count"] == 1
    assert body["receipt_count"] == 0
    assert body["covered_count"] == 0
    assert body["missing_count"] == 1
    assert body["missing_use_ids"] == [use["use_id"]]
    assert body["orphan_count"] == 0
    assert body["orphan_receipt_ids"] == []
    assert body["valid"] is False


def test_mixed_coverage_reports_both_sides(allowed_machine, client):
    machine_id = allowed_machine
    use_one = new_consumed_use(client, machine_id, resource="res/1")
    use_two = new_consumed_use(client, machine_id, resource="res/2")
    use_three = new_consumed_use(client, machine_id, resource="res/3")
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_one, resource="res/1"),
    )
    client.post(
        receipts_url(machine_id),
        json=receipt_payload(use_two, resource="res/2"),
    )
    # A receipt whose use_id is not one of this machine's consumptions.
    insert_receipt_row(
        client, machine_id, "receipt-orphan", "use-from-elsewhere",
        "2026-03-01T00:00:00Z",
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body["consumed_count"] == 3
    assert body["receipt_count"] == 3
    assert body["covered_count"] == 2
    assert body["missing_count"] == 1
    assert body["missing_use_ids"] == [use_three["use_id"]]
    assert body["orphan_count"] == 1
    assert body["orphan_receipt_ids"] == ["receipt-orphan"]
    assert body["valid"] is False


def test_orphan_receipt_is_reported_even_when_no_consumption_exists(
    client,
):
    machine_id = create_machine(client)
    insert_receipt_row(
        client, machine_id, "orphan-1", "unrelated-use",
        "2026-03-01T00:00:00Z",
    )
    body = client.get(coverage_url(machine_id)).json()
    assert body["consumed_count"] == 0
    assert body["receipt_count"] == 1
    assert body["covered_count"] == 0
    assert body["missing_use_ids"] == []
    assert body["orphan_count"] == 1
    assert body["orphan_receipt_ids"] == ["orphan-1"]
    assert body["valid"] is False


def test_later_legitimate_receipt_improves_the_report(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)

    before = client.get(coverage_url(machine_id)).json()
    assert before["valid"] is False
    assert before["missing_use_ids"] == [use["use_id"]]

    response = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert response.status_code == 201

    after = client.get(coverage_url(machine_id)).json()
    assert after["missing_count"] == 0
    assert after["missing_use_ids"] == []
    assert after["covered_count"] == 1
    assert after["valid"] is True


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_missing_use_ids_order_by_consumed_at_instant_then_id(client):
    machine_id = create_machine(client)
    # An exact-second stamp sorts before a fractional stamp of the same
    # second only when compared as instants (lexically '.' precedes 'Z').
    insert_use_row(
        client, machine_id, "use-frac", "2026-03-01T00:00:00.500000Z"
    )
    insert_use_row(
        client, machine_id, "use-exact", "2026-03-01T00:00:00Z"
    )
    insert_use_row(
        client, machine_id, "use-later", "2026-03-01T00:00:01Z"
    )
    # Same instant as use-frac: tie breaks by id.
    insert_use_row(
        client, machine_id, "use-earlyid", "2026-03-01T00:00:00.500000Z"
    )
    # A stamp that no longer parses sorts deterministically last.
    insert_use_row(client, machine_id, "use-broken", "not-a-timestamp")

    body = client.get(coverage_url(machine_id)).json()
    assert body["missing_use_ids"] == [
        "use-exact",
        "use-earlyid",
        "use-frac",
        "use-later",
        "use-broken",
    ]
    assert body["missing_count"] == 5


def test_orphan_receipt_ids_order_by_occurred_at_instant_then_id(client):
    machine_id = create_machine(client)
    insert_receipt_row(
        client, machine_id, "r-frac", "u-x", "2026-03-01T00:00:00.500000Z"
    )
    insert_receipt_row(
        client, machine_id, "r-exact", "u-y", "2026-03-01T00:00:00Z"
    )
    insert_receipt_row(
        client, machine_id, "r-later", "u-z", "2026-03-01T00:00:01Z"
    )
    insert_receipt_row(
        client, machine_id, "r-earlyid", "u-w", "2026-03-01T00:00:00.500000Z"
    )
    insert_receipt_row(
        client, machine_id, "r-broken", "u-v", "broken-time"
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body["orphan_receipt_ids"] == [
        "r-exact",
        "r-earlyid",
        "r-frac",
        "r-later",
        "r-broken",
    ]
    assert body["orphan_count"] == 5


def test_covered_uses_are_excluded_from_the_missing_ordering(client):
    machine_id = create_machine(client)
    insert_use_row(
        client, machine_id, "use-old", "2020-01-01T00:00:00Z"
    )
    insert_use_row(
        client, machine_id, "use-new", "2026-03-01T00:00:00Z"
    )
    # Only the newer use carries a receipt.
    insert_receipt_row(
        client, machine_id, "receipt-new", "use-new",
        "2026-03-02T00:00:00Z",
    )

    body = client.get(coverage_url(machine_id)).json()
    assert body["covered_count"] == 1
    assert body["missing_use_ids"] == ["use-old"]


# --------------------------------------------------------------------------- #
# Machine isolation
# --------------------------------------------------------------------------- #


def test_coverage_is_isolated_per_machine(allowed_machine, client):
    machine_a = allowed_machine
    # A's use carries no receipt owned by A.
    use_a = new_consumed_use(client, machine_a, resource="res/a")

    machine_b = create_machine(client, external_id="machine-2")
    declare(client, machine_b, resource_pattern="res/*")
    use_b = new_consumed_use(client, machine_b, resource="res/b")

    # A receipt *owned by B* that names A's consumption. The database only
    # permits it because A itself has no receipt for that use; coverage is
    # machine scoped, so it cannot cover A and is an orphan for B.
    insert_receipt_row(
        client, machine_b, "b-foreign-receipt", use_a["use_id"],
        "2026-03-01T00:00:00Z",
    )

    report_a = client.get(coverage_url(machine_a)).json()
    assert report_a["consumed_count"] == 1
    assert report_a["receipt_count"] == 0
    assert report_a["covered_count"] == 0
    assert report_a["missing_use_ids"] == [use_a["use_id"]]
    assert report_a["orphan_receipt_ids"] == []
    assert report_a["valid"] is False

    report_b = client.get(coverage_url(machine_b)).json()
    assert report_b["consumed_count"] == 1
    assert report_b["receipt_count"] == 1
    assert report_b["covered_count"] == 0
    assert report_b["missing_use_ids"] == [use_b["use_id"]]
    assert report_b["orphan_receipt_ids"] == ["b-foreign-receipt"]
    assert report_b["valid"] is False


def test_other_machine_records_do_not_change_empty_conclusion(
    allowed_machine, client
):
    machine_a = allowed_machine
    use_a = new_consumed_use(client, machine_a)
    insert_receipt_row(
        client, machine_a, "a-orphan", "nobody-use",
        "2026-03-01T00:00:00Z",
    )

    machine_b = create_machine(client, external_id="machine-2")
    report_b = client.get(coverage_url(machine_b)).json()
    assert report_b == {
        "machine_id": machine_b,
        "consumed_count": 0,
        "receipt_count": 0,
        "covered_count": 0,
        "missing_count": 0,
        "missing_use_ids": [],
        "orphan_count": 0,
        "orphan_receipt_ids": [],
        "valid": True,
    }
    # A remains responsible for its own gap and orphan.
    report_a = client.get(coverage_url(machine_a)).json()
    assert report_a["missing_use_ids"] == [use_a["use_id"]]
    assert report_a["orphan_receipt_ids"] == ["a-orphan"]


# --------------------------------------------------------------------------- #
# Independence from chain / scope / digest integrity
# --------------------------------------------------------------------------- #


def test_chain_damaged_receipt_still_covers_its_use(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    # Break the receipt chain; coverage matches use_id only.
    tamper(client.app.state.engine, receipt["id"], "chain_hash", "f" * 64)

    coverage = client.get(coverage_url(machine_id)).json()
    assert coverage["valid"] is True
    assert coverage["covered_count"] == 1
    assert coverage["missing_use_ids"] == []
    assert coverage["orphan_receipt_ids"] == []

    integrity = client.get(integrity_url(machine_id)).json()
    assert integrity["valid"] is False
    assert integrity["anomaly"] == "chain_break"


def test_scope_damaged_receipt_still_counts_as_covered(allowed_machine, client):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    tamper(
        client.app.state.engine, receipt["id"], "resource", "res/tampered"
    )

    coverage = client.get(coverage_url(machine_id)).json()
    assert coverage["valid"] is True
    assert coverage["covered_count"] == 1
    assert client.get(integrity_url(machine_id)).json()["valid"] is False


# --------------------------------------------------------------------------- #
# Read-only behavior
# --------------------------------------------------------------------------- #


def test_coverage_is_strictly_read_only_and_byte_identical_on_repeat(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    insert_receipt_row(
        client, machine_id, "orphan-1", "elsewhere-use",
        "2026-03-01T00:00:00Z",
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
    first = client.get(coverage_url(machine_id))
    second = client.get(coverage_url(machine_id))
    third = client.get(coverage_url(machine_id))
    assert first.content == second.content == third.content
    assert table_counts() == before_counts

    # The missing use is still exactly one unmodified consumption that a
    # later legitimate receipt can close.
    assert first.json()["missing_use_ids"] == [use["use_id"]]
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
    response = client.get(coverage_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine lookup and all record reads.
    response = client.get(coverage_url(MISSING_MACHINE) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_a_carried_body_is_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET",
        coverage_url(machine_id),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET",
        coverage_url(MISSING_MACHINE),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_valid_query_against_missing_machine_is_not_found(client):
    response = client.get(coverage_url(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"missing_use_ids" not in response.content


@pytest.mark.parametrize(
    "method", ["post", "put", "patch", "delete", "head"]
)
def test_only_get_is_routed(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(coverage_url(machine_id))
    assert response.status_code == 405


def test_other_methods_do_not_read_records(client):
    # Drop both record tables: any read attempted on a non-GET method would
    # surface as 500, but method routing must answer 405 first.
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_uses"))
        conn.execute(text("DROP TABLE execution_receipts"))
    for method in ("post", "put", "patch", "delete", "head"):
        response = getattr(client, method)(coverage_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read failures
# --------------------------------------------------------------------------- #


def test_use_read_failure_is_500_with_no_partial_report(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_grant_uses"))
    response = client.get(coverage_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"missing_use_ids" not in response.content


def test_receipt_read_failure_is_500_with_no_partial_report(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE execution_receipts"))
    response = client.get(coverage_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"orphan_receipt_ids" not in response.content


def test_machine_lookup_failure_is_500_with_no_partial_report(client):
    machine_id = create_machine(client)
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE machines"))
    response = client.get(coverage_url(machine_id))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"valid" not in response.content


# --------------------------------------------------------------------------- #
# Non-interference with existing entries
# --------------------------------------------------------------------------- #


def test_coverage_does_not_change_receipt_collection_or_integrity(
    allowed_machine, client
):
    machine_id = allowed_machine
    use = new_consumed_use(client, machine_id)
    receipt = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    ).json()
    integrity_before = client.get(integrity_url(machine_id)).content

    for _ in range(3):
        report = client.get(coverage_url(machine_id))
        assert report.status_code == 200

    # The collection still refuses a second receipt exactly once and the
    # integrity conclusion is unchanged.
    repeat = client.post(
        receipts_url(machine_id), json=receipt_payload(use)
    )
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "receipt_already_exists"}}
    assert client.get(integrity_url(machine_id)).content == integrity_before
    with client.app.state.engine.connect() as conn:
        assert conn.execute(
            text("SELECT id FROM execution_receipts WHERE use_id = :u").bindparams(
                u=use["use_id"]
            )
        ).scalar_one() == receipt["id"]
