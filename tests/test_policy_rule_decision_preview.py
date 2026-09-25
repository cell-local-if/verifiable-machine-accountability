"""Tests for the read-only global policy-rule decision preview.

Covers `POST /policy-rules/decision-preview`: body and query validation
(``invalid_request`` / ``invalid_value`` / ``invalid_query``, all raised
before any rule is read), POST-only routing (405), the candidate/matching
semantics over the global rules only (never machine declarations or
authorization events), the invalid/unmatched/winning/overridden/conflict
relations, the detail ordering (priority, then created_at instant, then id,
damaged stamps last), the allow/deny/no-match decisions with their stable
reasons, the 500 ``internal_error`` failure path with no partial output,
strict read-only byte stability, and persistence across a restart.
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


PREVIEW_PATH = "/policy-rules/decision-preview"

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


def preview(client, action_type="read", resource="res/x", **kwargs):
    return client.post(
        PREVIEW_PATH,
        json={"action_type": action_type, "resource": resource},
        **kwargs,
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


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=1,
                    action_type="read", resource_pattern="res/*", effect="allow"):
    """Insert a policy rule directly with fixed stored values."""
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


def table_state(client):
    with client.app.state.engine.connect() as conn:
        return list(conn.execute(text("SELECT * FROM policy_rules")))


# --------------------------------------------------------------------------- #
# Request validation (never reads rules)
# --------------------------------------------------------------------------- #


def test_unparseable_body_is_invalid_request(client):
    response = client.post(
        PREVIEW_PATH,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize("body", ["[]", '"read"', "42", "null", "true"])
def test_non_object_body_is_invalid_request(client, body):
    response = client.post(
        PREVIEW_PATH, content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action_type": "read"},
        {"resource": "res/x"},
        {"action_type": "read", "resource": "res/x", "extra": 1},
        {"action_type": "read", "resource": "res/x", "effect": "allow"},
        {"action_type": 1, "resource": "res/x"},
        {"action_type": True, "resource": "res/x"},
        {"action_type": None, "resource": "res/x"},
        {"action_type": ["read"], "resource": "res/x"},
        {"action_type": "read", "resource": 2.5},
        {"action_type": "read", "resource": {"x": 1}},
    ],
)
def test_bad_shape_or_type_is_invalid_request(client, payload):
    response = client.post(PREVIEW_PATH, json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "", "resource": "res/x"},
        {"action_type": "   ", "resource": "res/x"},
        {"action_type": "read", "resource": ""},
        {"action_type": "read", "resource": " \t "},
        {"action_type": " ", "resource": " "},
    ],
)
def test_blank_after_strip_is_invalid_value(client, payload):
    response = client.post(PREVIEW_PATH, json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_any_query_parameter_is_invalid_query(client):
    response = client.post(
        f"{PREVIEW_PATH}?unexpected=1",
        json={"action_type": "read", "resource": "res/x"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_takes_precedence_over_bad_body(client):
    response = client.post(
        f"{PREVIEW_PATH}?x=1",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_requests_behave_identically_on_empty_and_nonempty_tables(client):
    # Empty table.
    assert client.post(PREVIEW_PATH, json={}).json() == {
        "error": {"code": "invalid_request"}
    }
    assert client.post(
        PREVIEW_PATH, json={"action_type": " ", "resource": "x"}
    ).json() == {"error": {"code": "invalid_value"}}

    create_rule(client)
    # Non-empty table: same outcomes, and nothing was written.
    before = table_state(client)
    assert client.post(PREVIEW_PATH, json={}).json() == {
        "error": {"code": "invalid_request"}
    }
    assert client.post(
        PREVIEW_PATH, json={"action_type": " ", "resource": "x"}
    ).json() == {"error": {"code": "invalid_value"}}
    assert table_state(client) == before


def test_only_post_is_accepted(client):
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(PREVIEW_PATH)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Empty table and no-match outcomes
# --------------------------------------------------------------------------- #


def test_empty_table_is_no_matching_policy(client):
    response = preview(client)
    assert response.status_code == 200
    assert response.json() == {
        "action_type": "read",
        "resource": "res/x",
        "rules": [],
        "conflicts": [],
        "winning_rules": [],
        "decision": {"allowed": False, "reason": "no_matching_policy"},
    }


def test_action_and_resource_are_stripped_and_echoed(client):
    create_rule(client, action_type="read", resource_pattern="res/*")
    body = preview(client, action_type="  read ", resource=" res/x ").json()
    assert body["action_type"] == "read"
    assert body["resource"] == "res/x"
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}


def test_non_matching_rules_are_unmatched_and_no_candidate_denies(client):
    create_rule(client, action_type="write", resource_pattern="res/*")
    create_rule(client, action_type="read", resource_pattern="other/*")

    body = preview(client, "read", "res/x").json()
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}
    assert body["conflicts"] == []
    assert body["winning_rules"] == []
    assert [rule["relation"] for rule in body["rules"]] == [
        "unmatched",
        "unmatched",
    ]


def test_star_and_literal_segment_matching(client):
    create_rule(client, action_type="read", resource_pattern="res/*/detail")
    create_rule(client, action_type="read", resource_pattern="res/x")

    body = preview(client, "read", "res/x").json()
    relations = {rule["resource_pattern"]: rule["relation"] for rule in body["rules"]}
    assert relations["res/x"] == "winning"
    assert relations["res/*/detail"] == "unmatched"

    body = preview(client, "read", "res/anything/detail").json()
    relations = {rule["resource_pattern"]: rule["relation"] for rule in body["rules"]}
    assert relations["res/*/detail"] == "winning"
    assert relations["res/x"] == "unmatched"


# --------------------------------------------------------------------------- #
# Decisions, relations, conflicts
# --------------------------------------------------------------------------- #


def test_all_allow_lowest_priority_allows(client):
    create_rule(client, effect="allow", priority=0)
    create_rule(client, effect="allow", priority=0, resource_pattern="res/x")

    body = preview(client).json()
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    assert [rule["relation"] for rule in body["rules"]] == ["winning", "winning"]
    assert body["conflicts"] == []
    assert len(body["winning_rules"]) == 2


def test_lowest_priority_deny_denies(client):
    create_rule(client, effect="allow", priority=1)
    create_rule(client, effect="deny", priority=0, resource_pattern="res/y")

    body = preview(client, "read", "res/y").json()
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    relations = {rule["effect"]: rule["relation"] for rule in body["rules"]}
    assert relations == {"allow": "overridden", "deny": "winning"}
    [winner] = body["winning_rules"]
    assert winner["effect"] == "deny"
    assert winner["priority"] == 0
    assert set(winner) == {"id", "effect", "priority", "created_at"}


def test_higher_priority_candidates_are_overridden(client):
    create_rule(client, effect="deny", priority=5)
    create_rule(client, effect="allow", priority=2, resource_pattern="res/x")
    create_rule(client, effect="allow", priority=9, resource_pattern="res*")

    body = preview(client).json()
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    by_priority = {rule["priority"]: rule["relation"] for rule in body["rules"]}
    assert by_priority == {2: "winning", 5: "overridden", 9: "overridden"}
    assert body["conflicts"] == []
    [winner] = body["winning_rules"]
    assert winner["priority"] == 2


def test_mixed_lowest_priority_is_deny_with_conflict_pairs(client):
    allow1 = create_rule(client, effect="allow", priority=0,
                         resource_pattern="res/x*").json()["id"]
    deny1 = create_rule(client, effect="deny", priority=0,
                        resource_pattern="res/*").json()["id"]
    allow2 = create_rule(client, effect="allow", priority=0,
                         resource_pattern="res/x").json()["id"]
    create_rule(client, effect="allow", priority=3, resource_pattern="*")

    body = preview(client).json()
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    assert body["winning_rules"] == []

    by_id = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert by_id[allow1] == "conflict"
    assert by_id[allow2] == "conflict"
    assert by_id[deny1] == "conflict"

    expected_pairs = sorted(
        sorted(pair) for pair in ((allow1, deny1), (allow2, deny1))
    )
    assert [entry["rule_ids"] for entry in body["conflicts"]] == expected_pairs


def test_same_priority_all_deny_has_no_conflict(client):
    create_rule(client, effect="deny", priority=0)
    create_rule(client, effect="deny", priority=0, resource_pattern="res/x")

    body = preview(client).json()
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    assert body["conflicts"] == []
    assert [rule["relation"] for rule in body["rules"]] == ["winning", "winning"]
    assert len(body["winning_rules"]) == 2


# --------------------------------------------------------------------------- #
# Invalid stored rules
# --------------------------------------------------------------------------- #


def test_invalid_rules_never_participate(client):
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z", effect="ALLOW")
    insert_rule_row(client, rid(2), "2026-03-01T00:00:01Z", priority=-1)
    insert_rule_row(client, rid(3), "2026-03-01T00:00:02Z", effect="deny",
                    priority=0)

    body = preview(client).json()
    by_id = {rule["id"]: rule for rule in body["rules"]}
    assert by_id[rid(1)]["relation"] == "invalid"
    assert by_id[rid(2)]["relation"] == "invalid"
    assert by_id[rid(3)]["relation"] == "winning"
    # Stored values are surfaced exactly as stored.
    assert by_id[rid(1)]["effect"] == "ALLOW"
    assert by_id[rid(2)]["priority"] == -1
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}


def test_table_of_only_invalid_rules_is_no_matching_policy(client):
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z", effect="maybe")
    insert_rule_row(client, rid(2), "2026-03-01T00:00:01Z", priority=-3)

    body = preview(client).json()
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}
    assert body["conflicts"] == []
    assert body["winning_rules"] == []
    assert [rule["relation"] for rule in body["rules"]] == ["invalid", "invalid"]


def test_invalid_rule_with_matching_shape_stays_invalid(client):
    # A non-string stored action type (SQLite dynamic typing) never matches.
    # Insert through the raw driver: SQLAlchemy's String type would coerce an
    # integer to text on the way in, but a blob survives the round trip.
    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        conn.execute(
            "INSERT INTO policy_rules "
            "(id, action_type, resource_pattern, effect, priority, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid(1), b"\xff\xfe", "res/*", "deny", 0,
             "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        )
    body = preview(client).json()
    [rule] = body["rules"]
    assert rule["relation"] == "invalid"
    # The blob is surfaced in a deterministic textual form, never crashing
    # the response and never emitting a non-JSON value.
    assert rule["action_type"] == "��"
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}


def test_damaged_priority_shapes_are_invalid_and_never_emitted_as_floats(client):
    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        conn.execute(
            "INSERT INTO policy_rules "
            "(id, action_type, resource_pattern, effect, priority, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid(1), "read", "res/*", "deny", "high",
             "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO policy_rules "
            "(id, action_type, resource_pattern, effect, priority, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid(2), "read", "res/?", "allow", 2.5,
             "2026-03-01T00:00:01Z", "2026-03-01T00:00:01Z"),
        )
    body = preview(client).json()
    assert [rule["relation"] for rule in body["rules"]] == [
        "invalid",
        "invalid",
    ]
    # Damaged priorities are surfaced textually; the body carries no float.
    by_id = {rule["id"]: rule["priority"] for rule in body["rules"]}
    assert by_id == {rid(1): "high", rid(2): "2.5"}
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}


# --------------------------------------------------------------------------- #
# Detail ordering
# --------------------------------------------------------------------------- #


def test_details_ordered_by_priority_then_instant_then_id(client):
    insert_rule_row(client, rid(30), "2026-03-01T00:00:03Z", priority=30,
                    action_type="zzz")
    insert_rule_row(client, rid(21), "2026-03-01T00:00:02Z", priority=20,
                    action_type="zzz", resource_pattern="res/a")
    insert_rule_row(client, rid(20), "2026-03-01T00:00:02Z", priority=20,
                    action_type="zzz", resource_pattern="res/b")
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z", priority=10,
                    action_type="zzz", resource_pattern="res/c")
    # Fractional stamp of the same second sorts after the exact second.
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00.500000Z", priority=10,
                    action_type="zzz", resource_pattern="res/d")

    body = preview(client).json()
    assert [rule["id"] for rule in body["rules"]] == [
        rid(1),
        rid(2),
        rid(20),
        rid(21),
        rid(30),
    ]


def test_damaged_timestamp_sorts_last_and_never_crashes(client):
    # The damaged stamp sorts after the parseable stamp of the same priority
    # even though its id is smaller, and still before any higher priority.
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00Z", priority=0)
    insert_rule_row(client, rid(1), "not-a-time", priority=0,
                    resource_pattern="res/y")
    insert_rule_row(client, rid(3), "2026-03-01T00:00:01Z", priority=1)

    body = preview(client).json()
    assert [rule["id"] for rule in body["rules"]] == [rid(2), rid(1), rid(3)]
    # The damaged row keeps its stored text untouched.
    assert body["rules"][1]["created_at"] == "not-a-time"

    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        stored = conn.execute(
            "SELECT created_at FROM policy_rules WHERE id = ?", (rid(1),)
        ).fetchone()[0]
    assert stored == "not-a-time"


# --------------------------------------------------------------------------- #
# Envelope shape and serialization
# --------------------------------------------------------------------------- #


def test_top_level_field_order_and_rule_keys(client):
    create_rule(client)
    response = preview(client)
    assert response.status_code == 200
    assert list(response.json().keys()) == [
        "action_type",
        "resource",
        "rules",
        "conflicts",
        "winning_rules",
        "decision",
    ]
    [rule] = response.json()["rules"]
    assert set(rule.keys()) == RULE_KEYS
    assert list(rule.keys())[-1] == "relation"


def test_body_is_compact_json_with_single_trailing_newline(client):
    create_rule(client)
    response = preview(client)

    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")

    expected = (
        json.dumps(
            response.json(), ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert response.content == expected


def test_preview_is_read_only_and_byte_stable(client):
    create_rule(client, effect="allow", priority=0)
    create_rule(client, effect="deny", priority=1, resource_pattern="res/x")

    before = table_state(client)
    first = preview(client)
    middle = table_state(client)
    second = preview(client)
    after = table_state(client)

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_preview_ignores_machines_declarations_and_events(client):
    # The preview reads only the global rules: a machine with no enabling
    # declaration still previews an allow, and no decision event is written.
    machine_id = client.post(
        "/machines",
        json={
            "external_id": "machine-1",
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    ).json()["id"]
    create_rule(client, effect="allow", priority=0)

    body = preview(client).json()
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    assert client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json() == []

    # And the preview never changes the real evaluation outcome.
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    assert decision == {"allowed": False, "reason": "no_enabled_declaration"}


def test_preview_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        create_rule(first, effect="deny", priority=0)
        expected = preview(first).content

    with TestClient(app) as second:
        response = preview(second)

    assert response.status_code == 200
    assert response.content == expected
    assert response.json()["decision"] == {
        "allowed": False,
        "reason": "denied_by_policy",
    }


# --------------------------------------------------------------------------- #
# Internal failure
# --------------------------------------------------------------------------- #


def test_internal_failure_is_500_with_no_partial_preview(client, monkeypatch):
    create_rule(client)

    def boom(*args, **kwargs):
        raise RuntimeError("storage gone")

    monkeypatch.setattr(
        "accountability.app._decision_preview_body", boom
    )
    response = preview(client)
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}

    # The failure wrote nothing and the preview works again afterwards.
    monkeypatch.undo()
    assert preview(client).status_code == 200
