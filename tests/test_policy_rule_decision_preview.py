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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def preview(client, action="read", resource="res/x", **kwargs):
    return client.post(
        "/policy-rules/decision-preview",
        json={"action": action, "resource": resource},
        **kwargs,
    )


def create_rule(
    client,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=0,
):
    return client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )


def insert_rule_row(
    client,
    rule_id,
    created_at,
    *,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=1,
    updated_at=None,
):
    """Insert a policy rule directly, bypassing write-path validation."""
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


STAMP = "2026-03-01T00:00:00Z"


# --- empty table and response shape --------------------------------------


def test_empty_table_previews_deny_no_matching_policy(client):
    response = preview(client)

    assert response.status_code == 200
    assert response.json() == {
        "action": "read",
        "resource": "res/x",
        "rules": [],
        "conflicts": [],
        "winners": [],
        "decision": {"allowed": False, "reason": "no_matching_policy"},
    }


def test_response_is_compact_utf8_json_with_single_newline(client):
    response = preview(client, action="读", resource="res/x")

    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    # Compact: no JSON whitespace separators.
    assert b'": ' not in response.content
    assert b", " not in response.content
    # UTF-8 is emitted directly, never ASCII-escaped.
    assert "读".encode("utf-8") in response.content


def test_top_level_field_order_is_stable(client):
    assert list(preview(client).json().keys()) == [
        "action",
        "resource",
        "rules",
        "conflicts",
        "winners",
        "decision",
    ]
    assert list(preview(client).json()["decision"].keys()) == ["allowed", "reason"]


def test_repeated_calls_are_byte_identical(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=2)

    first = preview(client).content
    second = preview(client).content
    third = preview(client).content

    assert first == second == third


# --- request validation ---------------------------------------------------


def test_query_parameter_is_invalid_query_against_empty_table(client):
    response = client.post(
        "/policy-rules/decision-preview?x=1",
        json={"action": "read", "resource": "res/x"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_check_precedes_body_parsing(client):
    response = client.post(
        "/policy-rules/decision-preview?x=1",
        content="not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "{",
        "[1, 2]",
        '"read"',
        "123",
        "null",
        "true",
    ],
)
def test_unparseable_or_non_object_body_is_invalid_request(client, content):
    response = client.post(
        "/policy-rules/decision-preview",
        content=content,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"action": "read"},
        {"resource": "res/x"},
        {"action": "read", "resource": "res/x", "extra": 1},
        {"action": "read", "resource": "res/x", "action_type": "read"},
        {"action": 5, "resource": "res/x"},
        {"action": "read", "resource": 9},
        {"action": None, "resource": "res/x"},
        {"action": True, "resource": "res/x"},
        {"action": ["read"], "resource": "res/x"},
        {"action": "read", "resource": {"x": 1}},
    ],
)
def test_missing_extra_or_wrong_type_fields_are_invalid_request(client, payload):
    response = client.post("/policy-rules/decision-preview", json=payload)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


def test_type_check_precedes_blank_value_check(client):
    # resource is the wrong type even though action is blank; the structural
    # invalid_request wins over the value-domain invalid_value.
    response = client.post(
        "/policy-rules/decision-preview",
        json={"action": "   ", "resource": 7},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_request"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "   ", "resource": "res/x"},
        {"action": "read", "resource": ""},
        {"action": "\t\n", "resource": "res/x"},
        {"action": "read", "resource": "  "},
    ],
)
def test_blank_after_trim_is_invalid_value(client, payload):
    response = client.post("/policy-rules/decision-preview", json=payload)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_illegal_requests_keep_their_error_with_rules_present(client):
    create_rule(client)

    assert client.post(
        "/policy-rules/decision-preview?x=1",
        json={"action": "read", "resource": "res/x"},
    ).status_code == 422
    assert (
        client.post(
            "/policy-rules/decision-preview",
            content="garbage",
            headers={"content-type": "application/json"},
        ).json()
        == {"error": {"code": "invalid_request"}}
    )
    assert (
        client.post(
            "/policy-rules/decision-preview",
            json={"action": " ", "resource": "res/x"},
        ).json()
        == {"error": {"code": "invalid_value"}}
    )


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_return_405(client, method):
    response = getattr(client, method)("/policy-rules/decision-preview")

    assert response.status_code == 405


# --- decision semantics ---------------------------------------------------


def test_action_and_resource_are_trimmed(client):
    create_rule(client)

    response = preview(client, action="  read\t", resource=" res/x ")

    body = response.json()
    assert body["action"] == "read"
    assert body["resource"] == "res/x"
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    assert body["winners"] and all(w["id"] for w in body["winners"])


def test_all_allow_minimum_tier_allows(client):
    first = create_rule(client, priority=0).json()
    second = create_rule(client, resource_pattern="res/x", priority=0).json()
    create_rule(client, effect="deny", priority=4)

    body = preview(client).json()

    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    assert [rule["relation"] for rule in body["rules"]] == [
        "winner",
        "winner",
        "overridden",
    ]
    assert sorted(w["id"] for w in body["winners"]) == sorted(
        [first["id"], second["id"]]
    )
    assert body["conflicts"] == []
    for winner in body["winners"]:
        assert set(winner) == {"id", "effect", "priority", "created_at"}


def test_deny_at_minimum_tier_denies(client):
    allow = create_rule(client, effect="allow", priority=2).json()
    deny = create_rule(client, effect="deny", priority=1).json()

    body = preview(client).json()

    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert relations[deny["id"]] == "winner"
    assert relations[allow["id"]] == "overridden"
    assert [w["id"] for w in body["winners"]] == [deny["id"]]
    assert body["conflicts"] == []


def test_overridden_rules_keep_their_stored_identity(client):
    create_rule(client, effect="allow", priority=0)
    overridden = create_rule(client, effect="deny", priority=3).json()

    [rule] = [
        rule
        for rule in preview(client).json()["rules"]
        if rule["id"] == overridden["id"]
    ]

    assert rule["relation"] == "overridden"
    assert rule["id"] == overridden["id"]
    assert rule["effect"] == "deny"
    assert rule["priority"] == 3
    assert rule["created_at"] == overridden["created_at"]


def test_mixed_minimum_tier_is_conflict_and_denies(client):
    allow = create_rule(client, effect="allow", resource_pattern="*", priority=1).json()
    deny_a = create_rule(client, effect="deny", resource_pattern="res/x", priority=1).json()
    deny_b = create_rule(client, effect="deny", resource_pattern="res/*", priority=1).json()
    create_rule(client, effect="allow", priority=5)

    body = preview(client).json()

    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    assert body["winners"] == []
    # The mixed tier is reported as all id-sorted pairs, pair list ordered by
    # first then second id.
    conflict_ids = sorted([allow["id"], deny_a["id"], deny_b["id"]])
    assert body["conflicts"] == [
        [conflict_ids[0], conflict_ids[1]],
        [conflict_ids[0], conflict_ids[2]],
        [conflict_ids[1], conflict_ids[2]],
    ]
    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    for rule_id in conflict_ids:
        assert relations[rule_id] == "conflict"
    assert all(value != "winner" for value in relations.values())


def test_unmatched_action_and_pattern(client):
    other_action = create_rule(client, action_type="write").json()
    other_resource = create_rule(client, resource_pattern="other/*").json()
    matching = create_rule(client, priority=0).json()

    relations = {rule["id"]: rule["relation"] for rule in preview(client).json()["rules"]}

    assert relations[other_action["id"]] == "unmatched"
    assert relations[other_resource["id"]] == "unmatched"
    assert relations[matching["id"]] == "winner"


def test_no_candidate_even_with_unmatched_rules(client):
    create_rule(client, action_type="write")
    create_rule(client, resource_pattern="other/*")

    body = preview(client).json()

    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}
    assert body["winners"] == []
    assert body["conflicts"] == []
    assert {rule["relation"] for rule in body["rules"]} == {"unmatched"}


def test_star_matches_any_string(client):
    create_rule(client, resource_pattern="*", priority=0)

    assert preview(client, resource="a/b/c").json()["decision"] == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }


def test_pattern_segments_are_literal(client):
    # "?" and "." are literals, not regex wildcards.
    create_rule(client, resource_pattern="res/?", priority=0)

    assert preview(client, resource="res/x").json()["decision"] == {
        "allowed": False,
        "reason": "no_matching_policy",
    }
    assert preview(client, resource="res/?").json()["decision"]["allowed"] is True


# --- invalid stored rules -------------------------------------------------


def test_invalid_effect_is_marked_invalid_and_ignored(client):
    insert_rule_row(client, rid(1), STAMP, effect="ALLOW", priority=0)

    body = preview(client).json()

    assert body["rules"][0]["relation"] == "invalid"
    assert body["rules"][0]["effect"] == "ALLOW"
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}
    assert body["winners"] == []
    assert body["conflicts"] == []


def test_invalid_priority_is_marked_invalid_and_ignored(client):
    insert_rule_row(client, rid(1), STAMP, effect="deny", priority="x")
    create_rule(client, effect="allow", priority=9)

    relations = {rule["id"]: rule["relation"] for rule in preview(client).json()["rules"]}

    assert relations[rid(1)] == "invalid"
    # The invalid deny at nominal priority "x" cannot shadow the numeric allow.
    assert preview(client).json()["decision"] == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }


def test_negative_priority_is_invalid(client):
    insert_rule_row(client, rid(1), STAMP, effect="deny", priority=-3)

    body = preview(client).json()

    assert body["rules"][0]["relation"] == "invalid"
    assert body["rules"][0]["priority"] == -3
    assert body["decision"]["reason"] == "no_matching_policy"


def test_invalid_rules_never_conflict_or_win_but_are_listed(client):
    insert_rule_row(
        client, rid(1), STAMP, action_type="read", resource_pattern="*",
        effect="maybe", priority=0,
    )
    insert_rule_row(client, rid(2), STAMP, effect="deny", priority="bad")
    good = create_rule(client, effect="allow", priority=0).json()

    body = preview(client).json()

    relations = {rule["id"]: rule["relation"] for rule in body["rules"]}
    assert relations[rid(1)] == "invalid"
    assert relations[rid(2)] == "invalid"
    assert relations[good["id"]] == "winner"
    assert body["conflicts"] == []
    assert body["decision"]["allowed"] is True


# --- detail ordering ------------------------------------------------------


def test_details_ordered_by_priority_then_instant_then_id(client):
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

    ids = [rule["id"] for rule in preview(client).json()["rules"]]

    assert ids == [rid(1), rid(2), rid(10), rid(20), rid(21), rid(30)]


def test_damaged_created_at_sorts_after_parseable_within_priority(client):
    insert_rule_row(client, rid(2), "2026-03-01T00:00:05Z",
                    resource_pattern="p2", priority=0)
    insert_rule_row(client, rid(3), "not-a-timestamp",
                    resource_pattern="p3", priority=0)
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00Z",
                    resource_pattern="p1", priority=0)

    ids = [rule["id"] for rule in preview(client).json()["rules"]]

    assert ids == [rid(1), rid(2), rid(3)]
    assert preview(client).json()["rules"][-1]["created_at"] == "not-a-timestamp"


def test_each_detail_keeps_exactly_the_stored_fields_plus_relation(client):
    created = create_rule(client, priority=2).json()

    [detail] = preview(client).json()["rules"]

    assert set(detail) == {
        "id",
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
        "relation",
    }
    for key in (
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
    ):
        assert detail[key] == created[key]


# --- read-only, isolation, persistence ------------------------------------


def test_preview_is_read_only(client):
    create_rule(client)
    rules_before = client.get("/policy-rules").content
    chain_before = client.get("/policy-rules/chain").content

    preview(client)
    preview(client)

    assert client.get("/policy-rules").content == rules_before
    assert client.get("/policy-rules/chain").content == chain_before


def test_preview_ignores_machines_and_declarations(client):
    # With a matching rule the preview allows even though no machine or
    # behavior declaration exists.
    create_rule(client, priority=0)

    assert preview(client).json()["decision"]["allowed"] is True


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


def test_internal_failure_is_500_without_partial_preview(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    response = preview(client)

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
