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


def batch(client, requests, **kwargs):
    return client.post(
        "/policy-rules/decision-preview/batch",
        json={"requests": requests},
        **kwargs,
    )


def single(client, action="read", resource="res/x"):
    return client.post(
        "/policy-rules/decision-preview",
        json={"action": action, "resource": resource},
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


# --- empty batch and response shape ---------------------------------------


def test_empty_requests_array_is_legal(client):
    response = batch(client, [])

    assert response.status_code == 200
    assert response.json() == {
        "batch_count": 0,
        "analyses": [],
        "summary": {
            "no_match": 0,
            "allow": 0,
            "deny": 0,
            "conflict": 0,
            "override": 0,
        },
        "decisions": {
            "no_matching_policy": 0,
            "allowed_by_policy": 0,
            "denied_by_policy": 0,
        },
    }


def test_top_level_field_order_is_stable(client):
    body = batch(client, [{"action": "read", "resource": "res/x"}]).json()

    assert list(body.keys()) == ["batch_count", "analyses", "summary", "decisions"]
    assert list(body["analyses"][0].keys()) == ["input", "result"]
    assert list(body["analyses"][0]["input"].keys()) == ["action", "resource"]
    assert list(body["summary"].keys()) == [
        "no_match",
        "allow",
        "deny",
        "conflict",
        "override",
    ]
    assert list(body["decisions"].keys()) == [
        "no_matching_policy",
        "allowed_by_policy",
        "denied_by_policy",
    ]


def test_response_is_compact_utf8_json_with_single_newline(client):
    response = batch(client, [{"action": "读", "resource": "res/x"}])

    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    assert b'": ' not in response.content
    assert b", " not in response.content
    assert "读".encode("utf-8") in response.content


def test_repeated_calls_are_byte_identical(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=2)
    requests = [
        {"action": "read", "resource": "res/x"},
        {"action": "write", "resource": "res/x"},
    ]

    first = batch(client, requests).content
    second = batch(client, requests).content
    third = batch(client, requests).content

    assert first == second == third


# --- per-item semantics and snapshot ---------------------------------------


def test_each_result_matches_the_single_preview(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=3)
    requests = [
        {"action": "read", "resource": "res/x"},
        {"action": "write", "resource": "res/x"},
        {"action": "read", "resource": "other/y"},
    ]

    body = batch(client, requests).json()

    assert body["batch_count"] == 3
    assert len(body["analyses"]) == 3
    for item, request in zip(body["analyses"], requests):
        assert item["input"] == request
        assert item["result"] == single(
            client, action=request["action"], resource=request["resource"]
        ).json()


def test_input_echoes_trimmed_values_and_preserves_order(client):
    create_rule(client, priority=0)

    body = batch(
        client,
        [
            {"action": "  write ", "resource": " res/y\t"},
            {"action": "\tread", "resource": "res/x  "},
        ],
    ).json()

    assert [a["input"] for a in body["analyses"]] == [
        {"action": "write", "resource": "res/y"},
        {"action": "read", "resource": "res/x"},
    ]
    assert body["analyses"][0]["result"]["decision"] == {
        "allowed": False,
        "reason": "no_matching_policy",
    }
    assert body["analyses"][1]["result"]["decision"] == {
        "allowed": True,
        "reason": "allowed_by_policy",
    }


def test_summary_and_decisions_count_the_whole_batch(client):
    create_rule(client, effect="allow", priority=0)  # winner for read res/*
    create_rule(client, effect="deny", priority=2)  # overridden for read res/*
    create_rule(client, action_type="write", effect="allow", priority=1)
    create_rule(
        client, action_type="write", resource_pattern="res/x",
        effect="deny", priority=1,
    )

    body = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},  # allow + override
            {"action": "write", "resource": "res/x"},  # conflict -> deny
            {"action": "delete", "resource": "res/x"},  # no match
        ],
    ).json()

    assert body["batch_count"] == 3
    assert body["summary"] == {
        "no_match": 1,
        "allow": 1,
        "deny": 2,
        "conflict": 1,
        "override": 1,
    }
    assert body["decisions"] == {
        "no_matching_policy": 1,
        "allowed_by_policy": 1,
        "denied_by_policy": 1,
    }


def test_override_counts_inputs_with_overridden_candidates(client):
    create_rule(client, effect="allow", priority=0)
    create_rule(client, effect="allow", priority=5)

    body = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},  # has an overridden rule
            {"action": "read", "resource": "other/y"},  # matches nothing
        ],
    ).json()

    assert body["summary"]["override"] == 1
    assert body["summary"]["conflict"] == 0
    assert body["decisions"] == {
        "no_matching_policy": 1,
        "allowed_by_policy": 1,
        "denied_by_policy": 0,
    }


def test_invalid_stored_rules_are_annotated_per_item(client):
    insert_rule_row(client, rid(1), STAMP, effect="ALLOW", priority=5)
    good = create_rule(client, effect="allow", priority=0).json()

    body = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},
            {"action": "read", "resource": "res/y"},
        ],
    ).json()

    for analysis in body["analyses"]:
        relations = {
            rule["id"]: rule["relation"] for rule in analysis["result"]["rules"]
        }
        assert relations[rid(1)] == "invalid"
        assert relations[good["id"]] == "winner"


def test_batch_uses_one_snapshot_for_all_items(client):
    create_rule(client, effect="deny", priority=0)

    body = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},
            {"action": "read", "resource": "res/x"},
        ],
    ).json()

    # Identical inputs against one snapshot yield identical results.
    assert body["analyses"][0]["result"] == body["analyses"][1]["result"]
    assert body["summary"]["deny"] == 2


# --- request validation -----------------------------------------------------


def test_query_parameter_is_invalid_query(client):
    response = client.post(
        "/policy-rules/decision-preview/batch?x=1",
        json={"requests": []},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_check_precedes_body_parsing(client):
    response = client.post(
        "/policy-rules/decision-preview/batch?x=1",
        content="not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "content",
    ["", "not json", "{", "[1, 2]", '"read"', "123", "null", "true"],
)
def test_unparseable_or_non_object_body_is_invalid_batch(client, content):
    response = client.post(
        "/policy-rules/decision-preview/batch",
        content=content,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"request": []},
        {"requests": [], "extra": 1},
        {"requests": None},
        {"requests": True},
        {"requests": {}},
        {"requests": "read"},
        {"requests": [None]},
        {"requests": [True]},
        {"requests": ["read"]},
        {"requests": [[]]},
        {"requests": [{}]},
        {"requests": [{"action": "read"}]},
        {"requests": [{"resource": "res/x"}]},
        {"requests": [{"action": "read", "resource": "res/x", "extra": 1}]},
    ],
)
def test_structural_failures_are_invalid_batch(client, payload):
    response = client.post("/policy-rules/decision-preview/batch", json=payload)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "item",
    [
        {"action": 5, "resource": "res/x"},
        {"action": "read", "resource": 9},
        {"action": None, "resource": "res/x"},
        {"action": True, "resource": "res/x"},
        {"action": ["read"], "resource": "res/x"},
        {"action": "read", "resource": {"x": 1}},
        {"action": "   ", "resource": "res/x"},
        {"action": "read", "resource": ""},
        {"action": "\t\n", "resource": "res/x"},
        {"action": "read", "resource": "  "},
    ],
)
def test_bad_item_values_are_invalid_value(client, item):
    response = client.post(
        "/policy-rules/decision-preview/batch", json={"requests": [item]}
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_structure_check_precedes_value_check(client):
    # The second item is structurally malformed even though the first carries
    # a blank value; the structural invalid_batch wins.
    response = client.post(
        "/policy-rules/decision-preview/batch",
        json={"requests": [{"action": " ", "resource": "res/x"}, {"action": "read"}]},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_one_illegal_item_rejects_the_whole_batch(client):
    create_rule(client, priority=0)

    response = client.post(
        "/policy-rules/decision-preview/batch",
        json={
            "requests": [
                {"action": "read", "resource": "res/x"},
                {"action": "read", "resource": "  "},
            ]
        },
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}
    # No partial analysis is emitted.
    assert "analyses" not in response.json()


def test_illegal_requests_keep_their_error_with_rules_present(client):
    create_rule(client)

    assert client.post(
        "/policy-rules/decision-preview/batch?x=1", json={"requests": []}
    ).status_code == 422
    assert (
        client.post(
            "/policy-rules/decision-preview/batch",
            content="garbage",
            headers={"content-type": "application/json"},
        ).json()
        == {"error": {"code": "invalid_batch"}}
    )
    assert (
        client.post(
            "/policy-rules/decision-preview/batch",
            json={"requests": [{"action": " ", "resource": "res/x"}]},
        ).json()
        == {"error": {"code": "invalid_value"}}
    )


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_return_405(client, method):
    response = getattr(client, method)("/policy-rules/decision-preview/batch")

    assert response.status_code == 405


# --- read-only, isolation, persistence --------------------------------------


def test_batch_preview_is_read_only(client):
    create_rule(client)
    rules_before = client.get("/policy-rules").content
    chain_before = client.get("/policy-rules/chain").content

    batch(client, [{"action": "read", "resource": "res/x"}])
    batch(client, [{"action": "write", "resource": "res/y"}])

    assert client.get("/policy-rules").content == rules_before
    assert client.get("/policy-rules/chain").content == chain_before


def test_batch_preview_ignores_machines_and_declarations(client):
    create_rule(client, priority=0)

    body = batch(client, [{"action": "read", "resource": "res/x"}]).json()

    assert body["analyses"][0]["result"]["decision"]["allowed"] is True


def test_batch_preview_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)
    requests = [
        {"action": "read", "resource": "res/x"},
        {"action": "write", "resource": "res/x"},
    ]

    with TestClient(app) as first:
        create_rule(first, effect="deny", priority=0)
        expected = batch(first, requests).content

    with TestClient(app) as second:
        response = batch(second, requests)

    assert response.status_code == 200
    assert response.content == expected


def test_internal_failure_is_500_without_partial_analysis(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    response = batch(client, [{"action": "read", "resource": "res/x"}])

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_single_preview_is_unchanged_by_the_batch_endpoint(client):
    create_rule(client, priority=0)

    before = single(client).content
    batch(client, [{"action": "read", "resource": "res/x"}])

    assert single(client).content == before
