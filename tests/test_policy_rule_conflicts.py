"""Tests for the read-only global policy-rule conflict/override audit.

Covers `GET /policy-rules/conflicts`: the fixed three-array envelope
(``rules``/``conflicts``/``overrides``), rule details emitted exactly as
stored with a ``relation`` annotation (invalid/unmatched/conflict/override),
detail ordering by the actual ``created_at`` UTC instant then id, the conflict
rule (same action type, intersecting resource patterns, equal priority,
opposite effects), the override rule (same intersecting scope, the smaller
numeric priority covering the larger), the existing ``*`` resource-pattern
intersection semantics, exclusion of invalid rules from matching, id-ascending
stable ordering, ``invalid_query``/405 handling before any rule is read,
500 with no partial analysis on a read failure, empty-database results,
tolerance of damaged records, strict read-only byte stability, and
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


PATH = "/policy-rules/conflicts"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"

RULE_KEYS = {
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
    "relation",
}


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_rule_row(client, rule_id, created_at=T0, *, updated_at=None,
                    priority=1, action_type="read", resource_pattern="res/*",
                    effect="allow"):
    """Insert a policy rule directly, with values exactly as given."""
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
# Parameter validation and method routing
# --------------------------------------------------------------------------- #


def test_any_query_parameter_is_invalid_query(client):
    insert_rule_row(client, rid(1), T0)
    response = client.get(f"{PATH}?effect=allow")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_is_rejected_on_empty_database_without_reading(client):
    # The rejection happens during validation, identically on an empty
    # database: no rule data is ever read.
    response = client.get(f"{PATH}?unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_is_422_even_when_read_would_fail(client):
    # Validation precedes the read: with the table dropped a read would raise,
    # but the extra parameter still wins as 422 invalid_query.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    response = client.get(f"{PATH}?unexpected=1")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    response = getattr(client, method)(PATH)
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and rule details
# --------------------------------------------------------------------------- #


def test_empty_database_returns_three_empty_arrays(client):
    response = client.get(PATH)
    assert response.status_code == 200
    assert list(response.json().keys()) == ["rules", "conflicts", "overrides"]
    assert response.json() == {"rules": [], "conflicts": [], "overrides": []}


def test_rule_details_carry_stored_fields_and_relation(client):
    insert_rule_row(client, rid(2), T1, priority=2, effect="deny",
                    resource_pattern="other/*")
    insert_rule_row(client, rid(1), T0, priority=1)

    body = client.get(PATH).json()
    assert set(body.keys()) == {"rules", "conflicts", "overrides"}
    assert [rule["id"] for rule in body["rules"]] == [rid(1), rid(2)]
    for rule in body["rules"]:
        assert set(rule.keys()) == RULE_KEYS
        assert rule["relation"] == "unmatched"
    assert body["rules"][0]["effect"] == "allow"
    assert body["rules"][1]["effect"] == "deny"
    # Disjoint rules produce no relations.
    assert body["conflicts"] == []
    assert body["overrides"] == []


def test_stored_values_are_detailed_without_normalization(client):
    insert_rule_row(
        client,
        rid(1),
        T0,
        updated_at="2026-04-01T00:00:00Z",
        priority=9,
        action_type="  Read ",
        resource_pattern=" res/* ",
        effect="ALLOW",
    )

    [rule] = client.get(PATH).json()["rules"]
    assert rule == {
        "id": rid(1),
        "action_type": "  Read ",
        "resource_pattern": " res/* ",
        "effect": "ALLOW",
        "priority": 9,
        "created_at": T0,
        "updated_at": "2026-04-01T00:00:00Z",
        "relation": "invalid",
    }


def test_details_ordered_by_created_at_instant_then_id(client):
    # Rows deliberately inserted out of order, with one exact/fractional pair.
    insert_rule_row(client, rid(30), "2026-03-01T00:00:03Z",
                    resource_pattern="p30", priority=30, effect="deny")
    insert_rule_row(client, rid(21), "2026-03-01T00:00:02Z",
                    resource_pattern="p21", priority=20)
    insert_rule_row(client, rid(20), "2026-03-01T00:00:02Z",
                    resource_pattern="p20", priority=20)
    insert_rule_row(client, rid(10), "2026-03-01T00:00:01Z",
                    resource_pattern="p10", priority=10)
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z",
                    resource_pattern="p1", priority=0)
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00.500000Z",
                    resource_pattern="p2", priority=0)

    response = client.get(PATH)

    assert [rule["id"] for rule in response.json()["rules"]] == [
        rid(1),   # exact second sorts before the fractional stamp
        rid(2),   # same wall-clock second, 0.5s later
        rid(10),
        rid(20),  # tie at :02 breaks by id
        rid(21),
        rid(30),
    ]


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #


def test_same_priority_opposite_effect_forms_a_conflict(client):
    insert_rule_row(client, rid(2), T1, priority=3, effect="deny",
                    resource_pattern="res/a*")
    insert_rule_row(client, rid(1), T0, priority=3, effect="allow",
                    resource_pattern="res/*")

    body = client.get(PATH).json()
    assert body["overrides"] == []
    assert body["conflicts"] == [
        {
            "rule_ids": [rid(1), rid(2)],
            "intersection": "res/a*",
            "reason": "same_priority_opposite_effect",
        }
    ]
    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert relations == {rid(1): "conflict", rid(2): "conflict"}


def test_same_priority_same_effect_is_not_a_conflict(client):
    insert_rule_row(client, rid(1), T0, priority=3, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=3, effect="allow",
                    resource_pattern="res/a*")

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == []
    assert {rule["relation"] for rule in body["rules"]} == {"unmatched"}


def test_different_action_types_never_relate(client):
    insert_rule_row(client, rid(1), T0, priority=3, effect="allow",
                    action_type="read")
    insert_rule_row(client, rid(2), T1, priority=3, effect="deny",
                    action_type="write")
    insert_rule_row(client, rid(3), T2, priority=9, effect="deny",
                    action_type="write", resource_pattern="other/*")

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == []
    assert {rule["relation"] for rule in body["rules"]} == {"unmatched"}


def test_disjoint_resource_patterns_never_relate(client):
    insert_rule_row(client, rid(1), T0, priority=3, effect="allow",
                    resource_pattern="res/a*")
    insert_rule_row(client, rid(2), T1, priority=3, effect="deny",
                    resource_pattern="res/b*")
    insert_rule_row(client, rid(3), T2, priority=3, effect="deny",
                    resource_pattern="other/*")

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == []


def test_conflict_group_rule_ids_are_sorted_ascending(client):
    # The lexicographically larger id is inserted first; the group still
    # reports the pair ascending.
    insert_rule_row(client, rid(9), T0, priority=1, effect="deny")
    insert_rule_row(client, rid(4), T1, priority=1, effect="allow",
                    resource_pattern="res/a*")

    [conflict] = client.get(PATH).json()["conflicts"]
    assert conflict["rule_ids"] == [rid(4), rid(9)]


def test_multiple_conflicts_are_sorted_by_rule_id_pair(client):
    insert_rule_row(client, rid(3), T0, priority=1, effect="deny")
    insert_rule_row(client, rid(1), T1, priority=1, effect="allow",
                    resource_pattern="res/a*")
    insert_rule_row(client, rid(2), T2, priority=1, effect="allow",
                    resource_pattern="res/b*")

    body = client.get(PATH).json()
    assert [conflict["rule_ids"] for conflict in body["conflicts"]] == [
        [rid(1), rid(3)],
        [rid(2), rid(3)],
    ]


# --------------------------------------------------------------------------- #
# Resource-pattern intersection semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "first, second, intersection",
    [
        ("res/*", "res/a*", "res/a*"),
        ("*", "res/*", "res/*"),
        ("res/a", "res/*", "res/a"),
        ("res/a", "res/a*", "res/a"),
        ("a*", "*b", "a*b"),
        ("res/*/x", "res/a/*", "res/a/*/x"),
        ("res/*", "*/x", "res/*/x"),
    ],
)
def test_intersecting_patterns(client, first, second, intersection):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow",
                    resource_pattern=first)
    insert_rule_row(client, rid(2), T1, priority=1, effect="deny",
                    resource_pattern=second)

    [conflict] = client.get(PATH).json()["conflicts"]
    assert conflict["intersection"] == intersection


@pytest.mark.parametrize(
    "first, second",
    [
        ("res/a*", "res/b*"),
        ("res/*", "other/*"),
        ("res/a", "res/b"),
        ("res/a", "res/ab"),
        ("a*x", "b*"),
        ("*x", "*y"),
    ],
)
def test_non_intersecting_patterns(client, first, second):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow",
                    resource_pattern=first)
    insert_rule_row(client, rid(2), T1, priority=1, effect="deny",
                    resource_pattern=second)

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == []


# --------------------------------------------------------------------------- #
# Override detection
# --------------------------------------------------------------------------- #


def test_lower_priority_value_overrides(client):
    insert_rule_row(client, rid(1), T0, priority=2, effect="allow",
                    resource_pattern="res/*")
    insert_rule_row(client, rid(2), T1, priority=5, effect="deny",
                    resource_pattern="res/a*")

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == [
        {
            "overriding_rule_id": rid(1),
            "overridden_rule_id": rid(2),
            "intersection": "res/a*",
            "reason": "lower_priority_overrides",
        }
    ]
    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert relations == {rid(1): "override", rid(2): "override"}


def test_override_applies_regardless_of_effect(client):
    # Same effect and different priorities still form a cover relation: the
    # smaller priority value has decisive effect over the intersecting scope.
    insert_rule_row(client, rid(1), T0, priority=7, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=4, effect="allow")

    body = client.get(PATH).json()
    assert body["conflicts"] == []
    assert body["overrides"] == [
        {
            "overriding_rule_id": rid(2),
            "overridden_rule_id": rid(1),
            "intersection": "res/*",
            "reason": "lower_priority_overrides",
        }
    ]


def test_overrides_are_sorted_by_overriding_then_overridden_id(client):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=2, effect="allow")
    insert_rule_row(client, rid(3), T2, priority=3, effect="deny")

    body = client.get(PATH).json()
    assert [
        (entry["overriding_rule_id"], entry["overridden_rule_id"])
        for entry in body["overrides"]
    ] == [
        (rid(1), rid(2)),
        (rid(1), rid(3)),
        (rid(2), rid(3)),
    ]


def test_conflict_marks_outrank_override_marks(client):
    # rid(1) conflicts with rid(2) and also covers rid(3); a conflict mark is
    # stronger than an override mark. rid(3) is only covered.
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow",
                    resource_pattern="res/*")
    insert_rule_row(client, rid(2), T1, priority=1, effect="deny",
                    resource_pattern="res/a*")
    insert_rule_row(client, rid(3), T2, priority=4, effect="allow",
                    resource_pattern="res/b*")

    body = client.get(PATH).json()
    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert relations[rid(1)] == "conflict"
    assert relations[rid(2)] == "conflict"
    assert relations[rid(3)] == "override"


# --------------------------------------------------------------------------- #
# Invalid rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field, value",
    [
        # SQLite TEXT-affinity columns convert inserted numbers to text, so a
        # non-string action type or resource pattern can only be stored as a
        # blob; the details surface it in a deterministic textual form.
        ("action_type", b"read"),
        ("resource_pattern", b"res/*"),
        ("effect", "ALLOW"),
        ("effect", "maybe"),
        ("effect", ""),
        ("priority", -1),
        ("priority", "high"),
        ("priority", 1.5),
    ],
)
def test_invalid_rules_never_conflict_or_override(client, field, value):
    # A valid allow rule and a damaged partner that would otherwise conflict
    # (same action, intersecting pattern, same priority, opposite effect).
    insert_rule_row(client, rid(1), T0, priority=3, effect="allow",
                    resource_pattern="res/a*")
    damaged = {
        "action_type": "read",
        "resource_pattern": "res/*",
        "effect": "deny",
        "priority": 3,
    }
    damaged[field] = value
    insert_rule_row(client, rid(2), T1, **damaged)

    body = client.get(PATH).json()
    assert [rule["relation"] for rule in body["rules"]] == ["unmatched",
                                                            "invalid"]
    assert body["conflicts"] == []
    assert body["overrides"] == []


def test_invalid_rules_still_appear_in_details(client):
    insert_rule_row(client, rid(1), T0, priority=-2, effect="deny")

    body = client.get(PATH).json()
    assert len(body["rules"]) == 1
    assert body["rules"][0]["id"] == rid(1)
    assert body["rules"][0]["priority"] == -2
    assert body["rules"][0]["relation"] == "invalid"


def test_float_priority_never_emits_a_json_number(client):
    # A damaged float priority (SQLite keeps it as REAL) stays invalid and is
    # surfaced as text so the body contains no floating-point value.
    insert_rule_row(client, rid(1), T0, priority=1.5, effect="deny")

    response = client.get(PATH)
    assert response.status_code == 200
    [rule] = response.json()["rules"]
    assert rule["priority"] == "1.5"
    assert rule["relation"] == "invalid"
    # Compact body carries the textualized value, never a bare 1.5 number.
    assert b'"priority":"1.5"' in response.content
    assert b'"priority":1.5' not in response.content


def test_damaged_record_does_not_crash_and_is_not_rewritten(client):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow")
    insert_rule_row(client, rid(2), "not-a-time", priority=1, effect="deny",
                    resource_pattern="res/a*")

    response = client.get(PATH)
    assert response.status_code == 200
    body = response.json()
    # The damaged stamp sorts after the parseable one but its rule still
    # matches and conflicts.
    assert [rule["id"] for rule in body["rules"]] == [rid(1), rid(2)]
    assert body["rules"][1]["created_at"] == "not-a-time"
    assert body["conflicts"] == [
        {
            "rule_ids": [rid(1), rid(2)],
            "intersection": "res/a*",
            "reason": "same_priority_opposite_effect",
        }
    ]

    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        stored = conn.execute(
            "SELECT created_at FROM policy_rules WHERE id = ?", (rid(2),)
        ).fetchone()[0]
    assert stored == "not-a-time"


# --------------------------------------------------------------------------- #
# Serialization, failures, read-only behavior, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_json_with_single_trailing_newline(client):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=1, effect="deny",
                    resource_pattern="res/a*")

    response = client.get(PATH)
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


def test_utf8_values_are_emitted_directly(client):
    insert_rule_row(client, rid(1), T0, action_type="读", resource_pattern="读/*")

    response = client.get(PATH)

    assert "读".encode("utf-8") in response.content


def test_analysis_is_read_only_and_byte_stable(client):
    insert_rule_row(client, rid(1), T0, priority=1, effect="allow")
    insert_rule_row(client, rid(2), T1, priority=1, effect="deny",
                    resource_pattern="res/a*")
    insert_rule_row(client, rid(3), T2, priority=4, effect="allow")

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    first = client.get(PATH)
    middle = table_state()
    second = client.get(PATH)
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_internal_failure_is_500_without_partial_analysis(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    response = client.get(PATH)

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_analysis_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        create_rule(first, priority=1, effect="allow")
        create_rule(first, resource_pattern="res/a*", priority=1, effect="deny")
        expected = first.get(PATH).content

    with TestClient(app) as second:
        response = second.get(PATH)

    assert response.status_code == 200
    assert response.content == expected
    assert len(response.json()["conflicts"]) == 1


def test_analysis_does_not_change_policy_listing_or_authorization(client):
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
        json={"action_type": "read", "resource_pattern": "res/*",
              "enabled": True},
    )
    create_rule(client, effect="allow", priority=0)
    create_rule(client, resource_pattern="res/a*", effect="deny", priority=0)

    listing_before = client.get("/policy-rules").content
    client.get(PATH)
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/a/x"},
    ).json()
    client.get(PATH)
    listing_after = client.get("/policy-rules").content

    # The conflicting deny at the same priority still decides the evaluation;
    # the analysis itself changes nothing.
    assert decision == {"allowed": False, "reason": "denied_by_policy"}
    assert listing_before == listing_after
