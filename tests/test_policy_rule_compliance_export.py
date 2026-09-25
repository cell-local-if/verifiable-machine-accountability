"""Tests for the read-only global policy-rule time-window compliance export.

Covers `GET /policy-rules/compliance-export`: closed-UTC-window filtering on
each rule's own ``created_at``, ordering by the actual UTC instant then record
id (exact-second rules before fractional-second rules of the same second),
verbatim export of the policy-rule list fields with no normalization, the
``bad_time`` / ``invalid_query`` validation outcomes (rejected before any rule
is read), GET-only routing (405), empty-database results, tolerance of a
damaged stored ``created_at`` (no crash), strict read-only byte stability, and
persistence across a restart.
"""
import json
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


WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"

EXPORT_PATH = "/policy-rules/compliance-export"

RULE_KEYS = {
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
}


def export_url(from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        f"{EXPORT_PATH}?from_created_at={from_created_at}"
        f"&to_created_at={to_created_at}"
    )


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=1,
                    action_type="read", resource_pattern="res/*", effect="allow"):
    """Insert a policy rule directly with a fixed id and timestamp."""
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
                "priority": priority,
                "effect": effect,
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


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    response = client.get(f"{EXPORT_PATH}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form instead of Z
        "2026-03-01T00:00:00z",           # lowercase suffix
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "2026-03-01 00:00:00Z",           # space separator
        "2026-03-01T00:00:00.Z",          # dot without fraction digits
        "garbage",
        "",                               # blank
        "2026-13-01T00:00:00Z",           # bad month
        "2026-02-30T00:00:00Z",           # bad calendar day
        "2026-03-01T24:00:00Z",           # bad hour
        "2026-03-01T00:60:00Z",           # bad minute
        "2026-03-01T00:00:60Z",           # bad second
    ],
)
def test_malformed_from_bound_is_bad_time(client, value):
    response = client.get(export_url(value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "garbage",
        "2026-03-01T00:00:00.123",        # fractional but no Z
        " 2026-03-01T00:00:00Z",
    ],
)
def test_malformed_to_bound_is_bad_time(client, value):
    response = client.get(export_url(T0, value))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_seconds_are_accepted(client):
    insert_rule_row(client, rid(1), T2)
    response = client.get(
        export_url("2026-03-01T00:00:01.250Z", "2026-03-01T00:00:03.750000Z")
    )
    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]


def test_inverted_bounds_are_bad_time(client):
    response = client.get(export_url(T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    insert_rule_row(client, rid(1), T2)
    response = client.get(export_url(T2, T2))
    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    response = client.get(
        f"{EXPORT_PATH}?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_parameter_is_invalid_query_even_when_bounds_also_bad(client):
    # The unknown-parameter rejection takes precedence and never reaches the
    # time-bound check or the rule table.
    response = client.get(
        f"{EXPORT_PATH}?from_created_at=nope&to_created_at={T5}&x=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_without_reading_rules(client):
    # A malformed query is rejected identically on an empty database; there is
    # no 404/200 path that depends on rule contents being present.
    bad_time = client.get(export_url("nope", T5))
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{EXPORT_PATH}?from_created_at={T0}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_only_get_is_accepted(client):
    url = export_url(T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape, windowing, ordering
# --------------------------------------------------------------------------- #


def test_empty_database_returns_complete_empty_result(client):
    response = client.get(export_url(T0, T5))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"from_created_at", "to_created_at", "policy_rules"}
    assert body == {
        "from_created_at": T0,
        "to_created_at": T5,
        "policy_rules": [],
    }


def test_bounds_are_echoed_verbatim(client):
    raw_from = "2026-03-01T00:00:01.250Z"
    raw_to = "2026-03-01T00:00:03.750000Z"
    response = client.get(export_url(raw_from, raw_to))
    assert response.status_code == 200
    body = response.json()
    assert body["from_created_at"] == raw_from
    assert body["to_created_at"] == raw_to


def test_window_is_closed_on_rule_created_at(client):
    insert_rule_row(client, rid(1), T0, priority=1)
    insert_rule_row(client, rid(2), T2, priority=2)
    insert_rule_row(client, rid(3), T4, priority=3)

    body = client.get(export_url(T2, T3)).json()
    assert [rule["id"] for rule in body["policy_rules"]] == [rid(2)]

    # Equal bounds include the boundary rule.
    body = client.get(export_url(T4, T4)).json()
    assert [rule["id"] for rule in body["policy_rules"]] == [rid(3)]

    # An empty window keeps the array rather than omitting the field.
    body = client.get(export_url(T3, T3)).json()
    assert body["policy_rules"] == []


def test_rules_exported_with_exactly_the_list_endpoint_fields(client):
    created = create_rule(client, priority=4).json()
    insert_rule_row(client, rid(1), "2026-03-02T00:00:00Z", priority=5)

    body = client.get(
        export_url("2026-01-01T00:00:00Z", "2026-12-31T00:00:00Z")
    ).json()
    assert len(body["policy_rules"]) == 2
    for rule in body["policy_rules"]:
        assert set(rule.keys()) == RULE_KEYS

    [posted_rule] = [r for r in body["policy_rules"] if r["id"] == created["id"]]
    assert posted_rule == created


def test_stored_values_are_exported_without_normalization(client):
    # Values the write path would never produce must come out byte-for-byte.
    insert_rule_row(
        client,
        rid(1),
        "2026-03-01T00:00:00Z",
        updated_at="2026-04-01T00:00:00Z",
        priority=9,
        action_type="  Read ",
        resource_pattern=" res/* ",
        effect="ALLOW",
    )

    [rule] = client.get(export_url(T0, T5)).json()["policy_rules"]

    assert rule == {
        "id": rid(1),
        "action_type": "  Read ",
        "resource_pattern": " res/* ",
        "effect": "ALLOW",
        "priority": 9,
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
    }


def test_overlapping_business_data_is_exported_verbatim(client):
    # Two rules share action/resource text (differing priority); both survive.
    insert_rule_row(client, rid(1), T1, priority=1)
    insert_rule_row(client, rid(2), T2, priority=2)

    body = client.get(export_url(T0, T5)).json()
    assert [rule["id"] for rule in body["policy_rules"]] == [rid(1), rid(2)]


def test_exact_second_sorts_before_fractional_same_second(client):
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional rule first.
    insert_rule_row(client, rid(2), fractional, priority=2)
    insert_rule_row(client, rid(1), T0, priority=1)

    body = client.get(export_url(T0, T5)).json()
    assert [rule["id"] for rule in body["policy_rules"]] == [rid(1), rid(2)]


def test_rules_ordered_by_created_at_instant_then_id(client):
    insert_rule_row(client, rid(30), "2026-03-01T00:00:03Z", priority=30)
    insert_rule_row(client, rid(21), "2026-03-01T00:00:02Z", priority=21)
    insert_rule_row(client, rid(20), "2026-03-01T00:00:02Z", priority=20)
    insert_rule_row(client, rid(10), "2026-03-01T00:00:01Z", priority=10)
    insert_rule_row(client, rid(1), T0, priority=1)
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00.500000Z", priority=2)

    body = client.get(export_url(T0, T5)).json()
    assert [rule["id"] for rule in body["policy_rules"]] == [
        rid(1),
        rid(2),
        rid(10),
        rid(20),
        rid(21),
        rid(30),
    ]


# --------------------------------------------------------------------------- #
# Damaged stored timestamps
# --------------------------------------------------------------------------- #


def test_damaged_stored_timestamp_does_not_crash_export(client):
    # One sound rule and one rule whose stored created_at no longer parses. The
    # export keeps working, returns the sound in-window rule, and leaves the
    # damaged row's stored text untouched.
    insert_rule_row(client, rid(1), T2, priority=1)
    insert_rule_row(client, rid(2), "not-a-time", priority=2)

    response = client.get(export_url(T0, T5))
    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]

    # The damaged instant sorts after every finite window, even a wide one.
    response = client.get(export_url(*WIDE))
    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]

    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        stored = conn.execute(
            "SELECT created_at FROM policy_rules WHERE id = ?", (rid(2),)
        ).fetchone()[0]
    assert stored == "not-a-time"


def test_damaged_timestamp_response_is_byte_stable(client):
    insert_rule_row(client, rid(1), T2, priority=1)
    insert_rule_row(client, rid(2), "not-a-time", priority=2)

    first = client.get(export_url(T0, T5)).content
    second = client.get(export_url(T0, T5)).content
    assert first == second


# --------------------------------------------------------------------------- #
# Serialization, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_json_with_single_trailing_newline(client):
    insert_rule_row(client, rid(1), T2, priority=1)
    response = client.get(export_url(T0, T5))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")

    payload = response.json()
    expected = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False,
                   separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    assert response.content == expected

    # No floating-point values are produced.
    assert all(
        isinstance(rule["priority"], int) and not isinstance(rule["priority"], bool)
        for rule in payload["policy_rules"]
    )


def test_top_level_field_order_is_stable(client):
    response = client.get(export_url(T0, T5))
    assert response.content.startswith(
        b'{"from_created_at":"2026-03-01T00:00:00Z","to_created_at":'
    )


def test_export_is_read_only_and_byte_stable(client):
    insert_rule_row(client, rid(1), T1, priority=1)
    insert_rule_row(client, rid(2), T2, priority=2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    first = client.get(export_url(T0, T5))
    middle = table_state()
    second = client.get(export_url(T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        create_rule(first, priority=1)
        expected = first.get(
            export_url("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
        ).content

    with TestClient(app) as second:
        response = second.get(
            export_url("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
        )

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["policy_rules"]) == 1


def test_export_does_not_change_policy_listing_or_authorization(client):
    # The export shares the list endpoint's read-only semantics and never
    # participates in authorization evaluation.
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
    client.get(export_url(*WIDE))
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    client.get(export_url(*WIDE))
    listing_after = client.get("/policy-rules").content

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}
    assert listing_before == listing_after
