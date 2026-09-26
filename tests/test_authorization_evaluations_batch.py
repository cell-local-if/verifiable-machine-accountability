"""Tests for machine-level read-only batch authorization evaluation.

Covers `POST /machines/{machine_id}/authorization-evaluations/batch`:

- success: 200 with a position-aligned ``results`` array, the empty batch
  against an existing machine, fixed field order, trimmed input echo, and all
  five single-evaluation reason categories counted by ``summary`` and
  ``decisions``;
- snapshot semantics: the whole batch equals the single evaluations against
  one read-time snapshot, suspended machines short-circuit every item without
  consulting declarations or rules, and another machine's declarations never
  enter the result;
- validation: ``invalid_query`` first (even against a missing machine and
  before the body is parsed), ``invalid_batch`` for body/requests shape,
  ``invalid_value`` for item/field shape and blank values, ``not_found`` for
  a machine that is missing only once the body is legal, and ``405`` for
  non-POST methods;
- read-only and stable: no decision events or other rows are written, repeat
  calls are byte-identical, and results persist across restarts.
"""

import pytest
from fastapi.testclient import TestClient

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def batch_path(machine_id):
    return f"/machines/{machine_id}/authorization-evaluations/batch"


MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def create_rule(
    client,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=0,
):
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


def declare(
    client,
    machine_id,
    action_type="read",
    resource_pattern="res/*",
    enabled=True,
):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201


def suspend(client, machine_id):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": "suspended"}
    )
    assert response.status_code == 200


def batch(client, machine_id, items, **kwargs):
    return client.post(
        batch_path(machine_id), json={"requests": items}, **kwargs
    )


def item(action_type="read", resource="res/x"):
    return {"action_type": action_type, "resource": resource}


def single(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": action_type, "resource": resource},
    )


# --- empty batch and response shape ----------------------------------------


def test_empty_requests_array_is_legal_and_returns_complete_empty_result(client):
    machine_id = create_machine(client)

    response = batch(client, machine_id, [])

    assert response.status_code == 200
    assert response.json() == {
        "batch_count": 0,
        "results": [],
        "summary": {
            "machine_suspended": 0,
            "no_enabled_declaration": 0,
            "no_matching_policy": 0,
            "denied_by_policy": 0,
            "allowed_by_policy": 0,
        },
        "decisions": {"allowed": 0, "denied": 0},
    }


def test_empty_library_can_be_evaluated(client):
    machine_id = create_machine(client)

    response = batch(client, machine_id, [item()])

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [
        {
            "action_type": "read",
            "resource": "res/x",
            "allowed": False,
            "reason": "no_enabled_declaration",
        }
    ]


def test_top_level_and_item_field_order_is_stable(client):
    machine_id = create_machine(client)

    body = batch(client, machine_id, [item()]).json()

    assert list(body.keys()) == ["batch_count", "results", "summary", "decisions"]
    assert list(body["results"][0].keys()) == [
        "action_type",
        "resource",
        "allowed",
        "reason",
    ]
    assert list(body["summary"].keys()) == [
        "machine_suspended",
        "no_enabled_declaration",
        "no_matching_policy",
        "denied_by_policy",
        "allowed_by_policy",
    ]
    assert list(body["decisions"].keys()) == ["allowed", "denied"]


def test_body_is_compact_utf8_json_with_one_trailing_newline(client):
    machine_id = create_machine(client)

    response = batch(client, machine_id, [])
    raw = response.content

    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b": " not in raw
    assert b", " not in raw
    assert response.headers["content-type"] == "application/json"


def test_non_ascii_action_and_resource_are_echoed_as_utf8(client):
    machine_id = create_machine(client)
    declare(client, machine_id, action_type="读", resource_pattern="目录/*")
    create_rule(client, action_type="读", resource_pattern="目录/*")

    response = batch(
        client, machine_id, [item(action_type="读", resource="目录/a")]
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["action_type"] == "读"
    assert result["resource"] == "目录/a"
    assert result["allowed"] is True
    assert result["reason"] == "allowed_by_policy"
    assert "读".encode() in response.content


# --- decision semantics ------------------------------------------------------


def test_results_follow_input_order_and_echo_trimmed_pair(client):
    machine_id = create_machine(client)
    declare(client, machine_id, action_type="read", resource_pattern="res/*")
    create_rule(client, action_type="read", resource_pattern="res/*")
    requests_ = [
        item(action_type=" write ", resource="\tres/y\n"),
        item(action_type="read", resource=" res/x "),
        item(action_type="read", resource="res/zzz"),
    ]

    body = batch(client, machine_id, requests_).json()

    assert [result["action_type"] for result in body["results"]] == [
        "write",
        "read",
        "read",
    ]
    assert [result["resource"] for result in body["results"]] == [
        "res/y",
        "res/x",
        "res/zzz",
    ]
    assert body["batch_count"] == 3


def test_batch_matches_single_evaluation_for_every_item(client):
    machine_id = create_machine(client)
    declare(client, machine_id, action_type="read", resource_pattern="res/*")
    declare(client, machine_id, action_type="write", resource_pattern="res/*")
    create_rule(client, action_type="read", resource_pattern="res/*", effect="allow")
    create_rule(
        client, action_type="write", resource_pattern="res/*",
        effect="deny", priority=0,
    )
    create_rule(
        client, action_type="read", resource_pattern="res/secret*",
        effect="deny", priority=0,
    )
    create_rule(
        client, action_type="admin", resource_pattern="*", effect="allow",
    )
    requests_ = [
        item("read", "res/x"),
        item("read", "res/secret/1"),
        item("write", "res/x"),
        item("admin", "anything"),
        item("read", "other"),
    ]

    body = batch(client, machine_id, requests_).json()

    for submitted, result in zip(requests_, body["results"]):
        single_body = single(
            client, machine_id, submitted["action_type"], submitted["resource"]
        ).json()
        assert result["allowed"] == single_body["allowed"]
        assert result["reason"] == single_body["reason"]


def test_summary_and_decisions_count_all_five_categories(client):
    machine_id = create_machine(client)
    # Declarations: read on res/* and on * (enabled), write on res/* but
    # disabled.
    declare(client, machine_id, "read", "res/*", enabled=True)
    declare(client, machine_id, "read", "*", enabled=True)
    declare(client, machine_id, "write", "res/*", enabled=False)
    # Rules: allow read on res/* at priority 1, deny read on res/secret* at
    # priority 0, allow write on res/* at priority 0 (write still lacks an
    # enabled declaration), no rule for the "delete" action.
    create_rule(client, "read", "res/*", "allow", 1)
    create_rule(client, "read", "res/secret*", "deny", 0)
    create_rule(client, "write", "res/*", "allow", 0)
    requests_ = [
        item("read", "res/a"),          # allowed_by_policy
        item("read", "res/secret/x"),   # denied_by_policy
        item("write", "res/a"),         # no_enabled_declaration (disabled)
        item("delete", "res/a"),        # no_enabled_declaration
        item("read", "elsewhere/1"),    # no_matching_policy
    ]

    body = batch(client, machine_id, requests_).json()

    reasons = [result["reason"] for result in body["results"]]
    assert reasons == [
        "allowed_by_policy",
        "denied_by_policy",
        "no_enabled_declaration",
        "no_enabled_declaration",
        "no_matching_policy",
    ]
    assert body["summary"] == {
        "machine_suspended": 0,
        "no_enabled_declaration": 2,
        "no_matching_policy": 1,
        "denied_by_policy": 1,
        "allowed_by_policy": 1,
    }
    assert body["decisions"] == {"allowed": 1, "denied": 4}
    assert (
        body["decisions"]["allowed"] + body["decisions"]["denied"]
        == body["batch_count"]
    )
    assert sum(body["summary"].values()) == body["batch_count"]


def test_suspended_machine_denies_every_item_without_reading_declarations(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*", enabled=True)
    create_rule(client, "read", "res/*", "allow", 0)
    suspend(client, machine_id)

    body = batch(
        client,
        machine_id,
        [item("read", "res/a"), item("write", "anything"), item()],
    ).json()

    assert body["batch_count"] == 3
    assert body["results"] == [
        {
            "action_type": "read",
            "resource": "res/a",
            "allowed": False,
            "reason": "machine_suspended",
        },
        {
            "action_type": "write",
            "resource": "anything",
            "allowed": False,
            "reason": "machine_suspended",
        },
        {
            "action_type": "read",
            "resource": "res/x",
            "allowed": False,
            "reason": "machine_suspended",
        },
    ]
    assert body["summary"]["machine_suspended"] == 3
    assert body["decisions"] == {"allowed": 0, "denied": 3}


def test_suspended_machine_without_any_declaration_or_rule_is_denied(client):
    machine_id = create_machine(client)
    suspend(client, machine_id)

    body = batch(client, machine_id, [item("read", "res/a")]).json()

    assert body["results"][0]["reason"] == "machine_suspended"


def test_another_machines_declarations_never_enter_the_result(client):
    machine_id = create_machine(client, "machine-1")
    other_id = create_machine(client, "machine-2")
    declare(client, other_id, "read", "res/*", enabled=True)
    create_rule(client, "read", "res/*", "allow", 0)

    body = batch(client, machine_id, [item("read", "res/a")]).json()

    assert body["results"][0] == {
        "action_type": "read",
        "resource": "res/a",
        "allowed": False,
        "reason": "no_enabled_declaration",
    }


def test_deny_wins_over_allow_at_the_lowest_priority(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 2)
    create_rule(client, "read", "res/*", "deny", 1)

    result = batch(client, machine_id, [item("read", "res/a")]).json()["results"][0]

    assert result["allowed"] is False
    assert result["reason"] == "denied_by_policy"


def test_no_input_can_change_a_later_inputs_result(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 0)
    requests_ = [item("read", "res/a"), item("read", "res/b")]

    first = batch(client, machine_id, [requests_[0]]).json()
    second = batch(client, machine_id, [requests_[1]]).json()
    together = batch(client, machine_id, requests_).json()

    assert together["results"][0] == first["results"][0]
    assert together["results"][1] == second["results"][0]


# --- stability, read-only, persistence --------------------------------------


def test_repeated_calls_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 0)
    create_rule(client, "read", "res/secret*", "deny", 0)
    requests_ = [
        item("read", "res/a"),
        item("read", "res/secret/b"),
        item("write", "res/a"),
    ]

    first = batch(client, machine_id, requests_).content
    second = batch(client, machine_id, requests_).content
    third = batch(client, machine_id, requests_).content

    assert first == second == third


def test_batch_is_read_only_and_writes_no_decision_events(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 0)
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    assert client.get(events_url).json() == []
    declarations_before = client.get(
        f"/machines/{machine_id}/behavior-declarations"
    ).content
    rules_before = client.get("/policy-rules").content

    batch(client, machine_id, [item("read", "res/a"), item("write", "res/b")])

    assert client.get(events_url).json() == []
    assert (
        client.get(f"/machines/{machine_id}/behavior-declarations").content
        == declarations_before
    )
    assert client.get("/policy-rules").content == rules_before
    assert (
        client.get(f"/machines/{machine_id}").json()["status"] == "active"
    )


def test_batch_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)
    requests_ = [item("read", "res/a"), item("write", "res/b")]
    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id, "read", "res/*")
        create_rule(first, "read", "res/*", "allow", 0)
        expected = batch(first, machine_id, requests_).content

    with TestClient(app) as second:
        response = batch(second, machine_id, requests_)

    assert response.status_code == 200
    assert response.content == expected


def test_single_evaluation_is_unchanged_by_the_batch_endpoint(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 0)

    before = single(client, machine_id).content
    batch(client, machine_id, [item("read", "res/a"), item("write", "res/b")])
    after = single(client, machine_id).content

    assert before == after


# --- query validation --------------------------------------------------------


def test_unknown_query_parameter_is_invalid_query_with_a_valid_body(client):
    machine_id = create_machine(client)

    response = client.post(
        f"{batch_path(machine_id)}?unexpected=1", json={"requests": []}
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_precedes_body_parsing(client):
    machine_id = create_machine(client)

    response = client.post(
        f"{batch_path(machine_id)}?x=1",
        content="{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_precedes_machine_lookup(client):
    response = client.post(
        f"{batch_path(MISSING_ID)}?x=1", json={"requests": []}
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --- batch/body validation ---------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "{not json",
        "[]",
        '"a string"',
        "123",
        "true",
        "null",
    ],
)
def test_missing_or_non_object_body_is_invalid_batch(client, content):
    machine_id = create_machine(client)
    kwargs = (
        {"content": content, "headers": {"content-type": "application/json"}}
        if content is not None
        else {}
    )

    response = client.post(batch_path(machine_id), **kwargs)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_missing_requests_key_is_invalid_batch(client):
    machine_id = create_machine(client)

    response = client.post(batch_path(machine_id), json={})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_extra_top_level_field_is_invalid_batch(client):
    machine_id = create_machine(client)

    response = client.post(
        batch_path(machine_id),
        json={"requests": [], "extra": 1},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize("value", ["x", 1, 1.5, True, False, None, {}])
def test_requests_that_is_not_an_array_is_invalid_batch(client, value):
    machine_id = create_machine(client)

    response = client.post(batch_path(machine_id), json={"requests": value})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "bad_item",
    [
        "not-an-object",
        123,
        1.5,
        True,
        False,
        None,
        ["read", "res/x"],
        {},
        {"action_type": "read"},
        {"resource": "res/x"},
        {"action_type": "read", "resource": "res/x", "extra": 1},
    ],
)
def test_item_shape_failures_are_invalid_value(client, bad_item):
    machine_id = create_machine(client)

    response = batch(client, machine_id, [item(), bad_item])

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


@pytest.mark.parametrize(
    "bad_item",
    [
        {"action_type": 1, "resource": "res/x"},
        {"action_type": 1.5, "resource": "res/x"},
        {"action_type": True, "resource": "res/x"},
        {"action_type": None, "resource": "res/x"},
        {"action_type": ["read"], "resource": "res/x"},
        {"action_type": {"a": 1}, "resource": "res/x"},
        {"action_type": "read", "resource": 1},
        {"action_type": "read", "resource": 1.5},
        {"action_type": "read", "resource": True},
        {"action_type": "read", "resource": None},
        {"action_type": "read", "resource": ["res/x"]},
        {"action_type": "read", "resource": {"r": 1}},
        {"action_type": "   ", "resource": "res/x"},
        {"action_type": "\t\n", "resource": "res/x"},
        {"action_type": "read", "resource": ""},
        {"action_type": "read", "resource": "   "},
    ],
)
def test_item_value_failures_are_invalid_value(client, bad_item):
    machine_id = create_machine(client)

    response = batch(client, machine_id, [bad_item])

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_one_illegal_item_rejects_the_whole_batch_with_no_results(client):
    machine_id = create_machine(client)
    declare(client, machine_id, "read", "res/*")
    create_rule(client, "read", "res/*", "allow", 0)

    response = batch(
        client,
        machine_id,
        [item("read", "res/a"), {"action_type": "read", "resource": "   "}],
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_invalid_value_precedes_machine_lookup(client):
    response = batch(client, MISSING_ID, [item(), {"action_type": "x"}])

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_invalid_batch_precedes_machine_lookup(client):
    response = client.post(batch_path(MISSING_ID), json={"requests": "nope"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


# --- machine lookup and method handling --------------------------------------


def test_missing_machine_is_404_for_a_legal_non_empty_batch(client):
    response = batch(client, MISSING_ID, [item()])

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_machine_is_404_for_an_empty_batch(client):
    response = batch(client, MISSING_ID, [])

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_return_405(client, method):
    machine_id = create_machine(client)

    response = getattr(client, method)(batch_path(machine_id))

    assert response.status_code == 405
