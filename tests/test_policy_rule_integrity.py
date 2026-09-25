"""Tests for the read-only global policy-rule integrity audit.

Covers `GET /policy-rules/integrity`: the `{valid, checked_count,
broken_policy_rule_id}` conclusion over every global rule, the field, format,
and business-identity uniqueness conditions, the (created_at instant, id)
scan order with exact-second records before fractional-second records of the
same second and damaged timestamps sorting last, the raw reporting of a
damaged id, GET-only routing (405), the no-parameter contract (422
``invalid_query`` before any rule is read), strict read-only byte stability,
and persistence across a restart. Also covers the policy-rule list endpoint's
tolerance of a damaged stored ``created_at`` (no crash, damaged record sorts
last with its stored text untouched).
"""
import sqlite3

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


T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"

INTEGRITY_PATH = "/policy-rules/integrity"


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=1,
                    action_type="read", resource_pattern="res/*", effect="allow"):
    """Insert a policy rule directly with fixed field values."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rule_id,
                "action_type": action_type,
                "resource_pattern": resource_pattern,
                "effect": effect,
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at or created_at,
            },
        )


def create_rule(client, action_type="read", resource_pattern="res/*",
                effect="allow", priority=0):
    return client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )


def integrity(client):
    response = client.get(INTEGRITY_PATH)
    assert response.status_code == 200
    return response.json()


# --------------------------------------------------------------------------- #
# Routing and query-string contract
# --------------------------------------------------------------------------- #


def test_only_get_is_accepted(client):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(INTEGRITY_PATH)
        assert response.status_code == 405


@pytest.mark.parametrize("query", ["?unexpected=1", "?effect=allow", "?valid=true"])
def test_any_query_parameter_is_invalid_query(client, query):
    insert_rule_row(client, rid(1), T0)
    response = client.get(f"{INTEGRITY_PATH}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_is_rejected_before_reading_rules(client):
    # The rejection is identical on an empty and a populated table; no rule
    # data is read to produce it.
    empty = client.get(f"{INTEGRITY_PATH}?x=1")
    insert_rule_row(client, rid(1), "not-a-time")
    populated = client.get(f"{INTEGRITY_PATH}?x=1")
    assert empty.status_code == populated.status_code == 422
    assert empty.json() == populated.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Sound tables
# --------------------------------------------------------------------------- #


def test_empty_table_is_valid_with_zero_count(client):
    assert integrity(client) == {
        "valid": True,
        "checked_count": 0,
        "broken_policy_rule_id": None,
    }


def test_sound_rules_are_valid_with_total_count(client):
    create_rule(client, priority=0)
    create_rule(client, action_type="write", priority=1)
    insert_rule_row(client, rid(1), T0, priority=2)

    assert integrity(client) == {
        "valid": True,
        "checked_count": 3,
        "broken_policy_rule_id": None,
    }


def test_fractional_second_timestamps_are_accepted(client):
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00.500000Z")
    assert integrity(client)["valid"] is True


# --------------------------------------------------------------------------- #
# Field and format conditions
# --------------------------------------------------------------------------- #


def test_non_uuid_id_is_broken_and_reported_as_stored(client):
    insert_rule_row(client, "not-a-uuid", T0)
    assert integrity(client) == {
        "valid": False,
        "checked_count": 1,
        "broken_policy_rule_id": "not-a-uuid",
    }


@pytest.mark.parametrize("action_type", ["", "   ", "\t \n"])
def test_blank_action_type_is_broken(client, action_type):
    insert_rule_row(client, rid(1), T0, action_type=action_type)
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


def test_blank_resource_pattern_is_broken(client):
    insert_rule_row(client, rid(1), T0, resource_pattern="  ")
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


@pytest.mark.parametrize("effect", ["ALLOW", "permit", "", "allow "])
def test_effect_outside_allow_deny_is_broken(client, effect):
    insert_rule_row(client, rid(1), T0, effect=effect)
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


@pytest.mark.parametrize("priority", [-1, "abc", 1.5])
def test_non_nonnegative_integer_priority_is_broken(client, priority):
    insert_rule_row(client, rid(1), T0, priority=priority)
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


@pytest.mark.parametrize(
    "created_at",
    [
        "not-a-time",
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form
        "2026-03-01T00:00:00z",           # lowercase suffix
        " 2026-03-01T00:00:00Z",          # surrounding whitespace
        "2026-13-01T00:00:00Z",           # out-of-range month
        "2026-03-01T24:00:00Z",           # out-of-range hour
    ],
)
def test_malformed_created_at_is_broken(client, created_at):
    insert_rule_row(client, rid(1), created_at)
    result = integrity(client)
    assert result["valid"] is False
    assert result["checked_count"] == 1
    assert result["broken_policy_rule_id"] == rid(1)


def test_malformed_updated_at_is_broken(client):
    insert_rule_row(client, rid(1), T0, updated_at="not-a-time")
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


def test_checked_count_covers_all_rules_even_when_broken(client):
    insert_rule_row(client, rid(1), T0, effect="ALLOW")
    insert_rule_row(client, rid(2), T1, priority=2)
    insert_rule_row(client, rid(3), T2, priority=3)
    result = integrity(client)
    assert result["valid"] is False
    assert result["checked_count"] == 3
    assert result["broken_policy_rule_id"] == rid(1)


# --------------------------------------------------------------------------- #
# Business-identity uniqueness
# --------------------------------------------------------------------------- #


def test_duplicate_trimmed_business_identity_flags_first_sorted_record(client):
    # The raw values differ, so the database unique constraint admits both;
    # the trimmed identity (action, resource, priority) is the same.
    insert_rule_row(client, rid(2), T1, action_type=" read ", priority=5)
    insert_rule_row(client, rid(1), T0, action_type="read", priority=5)

    result = integrity(client)
    assert result["valid"] is False
    assert result["checked_count"] == 2
    # The earlier-created duplicate sorts first and is the broken one.
    assert result["broken_policy_rule_id"] == rid(1)


def test_same_text_with_different_priority_is_not_a_duplicate(client):
    insert_rule_row(client, rid(1), T0, priority=1)
    insert_rule_row(client, rid(2), T1, priority=2)
    assert integrity(client)["valid"] is True


def test_same_trimmed_identity_with_different_effect_is_still_duplicate(client):
    # The effect is not part of the business identity.
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=1, action_type=" read ", effect="deny")
    result = integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rid(1)


# --------------------------------------------------------------------------- #
# Scan order
# --------------------------------------------------------------------------- #


def test_first_broken_rule_in_instant_then_id_order_is_reported(client):
    insert_rule_row(client, rid(2), T1, effect="ALLOW", priority=2)
    insert_rule_row(client, rid(1), T0, effect="ALLOW", priority=1)
    assert integrity(client)["broken_policy_rule_id"] == rid(1)


def test_exact_second_broken_rule_precedes_fractional_same_second(client):
    insert_rule_row(
        client, rid(2), "2026-03-01T00:00:00.500000Z", effect="ALLOW", priority=2
    )
    insert_rule_row(client, rid(1), T0, effect="ALLOW", priority=1)
    assert integrity(client)["broken_policy_rule_id"] == rid(1)


def test_damaged_timestamp_sorts_after_every_parseable_rule(client):
    insert_rule_row(client, rid(1), "not-a-time", effect="ALLOW", priority=1)
    insert_rule_row(client, rid(2), T2, effect="ALLOW", priority=2)
    # The parseable T2 rule is broken earlier in the scan order than the
    # damaged record, which sorts last.
    assert integrity(client)["broken_policy_rule_id"] == rid(2)


def test_equal_instant_tie_breaks_by_id(client):
    insert_rule_row(client, rid(20), T0, effect="ALLOW", priority=2)
    insert_rule_row(client, rid(10), T0, effect="ALLOW", priority=1)
    assert integrity(client)["broken_policy_rule_id"] == rid(10)


# --------------------------------------------------------------------------- #
# Read-only behavior, stability, persistence
# --------------------------------------------------------------------------- #


def test_audit_is_read_only_and_byte_stable(client):
    insert_rule_row(client, rid(1), T0, effect="ALLOW")
    insert_rule_row(client, rid(2), "not-a-time", priority=2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    first = client.get(INTEGRITY_PATH)
    second = client.get(INTEGRITY_PATH)
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == after


def test_audit_does_not_change_listing_export_or_authorization(client):
    machine_id = client.post(
        "/machines",
        json={
            "external_id": "machine-1",
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    ).json()["id"]
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
    )
    create_rule(client, effect="allow", priority=0)

    listing_before = client.get("/policy-rules").content
    client.get(INTEGRITY_PATH)
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    export = client.get(
        "/policy-rules/compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z&to_created_at=2100-01-01T00:00:00Z"
    )
    listing_after = client.get("/policy-rules").content

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}
    assert export.status_code == 200
    assert listing_before == listing_after


def test_audit_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        insert_rule_row(first, rid(1), T0, effect="ALLOW")
        expected = first.get(INTEGRITY_PATH).content

    with TestClient(app) as second:
        response = second.get(INTEGRITY_PATH)

    assert response.status_code == 200
    assert response.content == expected
    assert response.json() == {
        "valid": False,
        "checked_count": 1,
        "broken_policy_rule_id": rid(1),
    }


# --------------------------------------------------------------------------- #
# List endpoint tolerance of a damaged stored timestamp
# --------------------------------------------------------------------------- #


def test_list_tolerates_damaged_created_at_and_sorts_it_last(client):
    insert_rule_row(client, rid(1), T1, priority=1)
    insert_rule_row(client, rid(2), "not-a-time", priority=2)
    insert_rule_row(client, rid(3), T0, priority=3)

    response = client.get("/policy-rules")
    assert response.status_code == 200
    rules = response.json()
    assert [rule["id"] for rule in rules] == [rid(3), rid(1), rid(2)]
    # The damaged record keeps its stored text verbatim.
    assert rules[-1]["created_at"] == "not-a-time"

    # The stored value is never repaired by the read.
    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        stored = conn.execute(
            "SELECT created_at FROM policy_rules WHERE id = ?", (rid(2),)
        ).fetchone()[0]
    assert stored == "not-a-time"
