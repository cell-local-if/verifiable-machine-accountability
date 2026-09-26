"""Tests for machine-level batch authorization evaluation.

Covers `POST /machines/{machine_id}/authorization-evaluations/batch`:

- success: 200 with a position-aligned ``results`` array, the fixed
  ``{batch_count, results, summary, decisions}`` top-level shape, the empty
  batch, and the five reason categories across suspended and active machines;
- semantics: suspended machines short-circuit every item to
  ``machine_suspended`` without consulting declarations or rules, active
  machines reuse the single-evaluation declaration/policy semantics, and the
  whole batch shares one snapshot;
- validation: ``invalid_query`` for query parameters, ``invalid_batch`` for
  body shape, ``invalid_value`` for item shape/types/blank values — every
  422 precedes the machine lookup;
- lifecycle: ``404 not_found`` after validation, POST-only ``405``, compact
  newline-terminated byte-identical bodies, and no writes of any kind.
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


MISSING_ID = "00000000-0000-0000-0000-000000000000"


def batch_path(machine_id):
    return f"/machines/{machine_id}/authorization-evaluations/batch"


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


def suspend(client, machine_id):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": "suspended"}
    )
    assert response.status_code == 200


def declare(
    client, machine_id, action_type="read", resource_pattern="res/*", enabled=True
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


def item(action_type="read", resource="res/x"):
    return {"action_type": action_type, "resource": resource}


def batch(client, machine_id, requests, **kwargs):
    return client.post(batch_path(machine_id), json={"requests": requests}, **kwargs)


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #


def test_empty_batch_returns_complete_empty_result(client):
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
        "decisions": {"allow": 0, "deny": 0},
    }


def test_top_level_key_order_and_trailing_newline(client):
    machine_id = create_machine(client)
    response = batch(client, machine_id, [])
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert body.endswith("\n") and not body.endswith("\n\n")
    assert list(response.json()) == ["batch_count", "results", "summary", "decisions"]
    assert list(response.json()["summary"]) == [
        "machine_suspended",
        "no_enabled_declaration",
        "no_matching_policy",
        "denied_by_policy",
        "allowed_by_policy",
    ]
    assert list(response.json()["decisions"]) == ["allow", "deny"]
    # Compact serialization: no spaces after separators.
    assert ',"results"' in body


def test_results_align_with_input_positions(client):
    machine_id = create_machine(client)
    declare(client, machine_id)  # read on res/*
    create_rule(client)  # allow read on res/* at priority 0
    create_rule(client, action_type="write", effect="deny", priority=0)
    declare(client, machine_id, action_type="write", resource_pattern="res/*")

    requests = [
        item("read", "res/x"),  # allowed_by_policy
        item("read", "other/y"),  # no_enabled_declaration
        item("write", "res/x"),  # denied_by_policy
        item("delete", "res/x"),  # no_enabled_declaration
    ]
    response = batch(client, machine_id, requests)
    assert response.status_code == 200
    payload = response.json()
    assert payload["batch_count"] == 4
    assert [r["reason"] for r in payload["results"]] == [
        "allowed_by_policy",
        "no_enabled_declaration",
        "denied_by_policy",
        "no_enabled_declaration",
    ]
    assert [r["allowed"] for r in payload["results"]] == [True, False, False, False]
    assert payload["results"][0] == {
        "action_type": "read",
        "resource": "res/x",
        "allowed": True,
        "reason": "allowed_by_policy",
    }
    assert payload["summary"] == {
        "machine_suspended": 0,
        "no_enabled_declaration": 2,
        "no_matching_policy": 0,
        "denied_by_policy": 1,
        "allowed_by_policy": 1,
    }
    assert payload["decisions"] == {"allow": 1, "deny": 3}


def test_no_matching_policy_category(client):
    machine_id = create_machine(client)
    declare(client, machine_id)  # declaration but no rule at all
    response = batch(client, machine_id, [item("read", "res/x")])
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["reason"] == "no_matching_policy"
    assert payload["summary"]["no_matching_policy"] == 1
    assert payload["decisions"] == {"allow": 0, "deny": 1}


def test_suspended_machine_short_circuits_every_item(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    suspend(client, machine_id)

    requests = [item("read", "res/x"), item("write", "res/y"), item("delete", "z")]
    response = batch(client, machine_id, requests)
    assert response.status_code == 200
    payload = response.json()
    assert payload["batch_count"] == 3
    for result in payload["results"]:
        assert result["allowed"] is False
        assert result["reason"] == "machine_suspended"
    assert payload["summary"] == {
        "machine_suspended": 3,
        "no_enabled_declaration": 0,
        "no_matching_policy": 0,
        "denied_by_policy": 0,
        "allowed_by_policy": 0,
    }
    assert payload["decisions"] == {"allow": 0, "deny": 3}


def test_values_are_trimmed_before_evaluation_and_echo(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    response = batch(client, machine_id, [item("  read  ", "\tres/x\n")])
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result == {
        "action_type": "read",
        "resource": "res/x",
        "allowed": True,
        "reason": "allowed_by_policy",
    }


def test_disabled_declaration_does_not_enable(client):
    machine_id = create_machine(client)
    declare(client, machine_id, enabled=False)
    create_rule(client)
    response = batch(client, machine_id, [item()])
    assert response.json()["results"][0]["reason"] == "no_enabled_declaration"


def test_deny_wins_over_same_priority_allow(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="allow", priority=1)
    create_rule(client, effect="deny", priority=1, resource_pattern="res/x")
    response = batch(client, machine_id, [item()])
    assert response.json()["results"][0]["reason"] == "denied_by_policy"


def test_lower_priority_decides(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="deny", priority=5)
    create_rule(client, effect="allow", priority=2)
    response = batch(client, machine_id, [item()])
    assert response.json()["results"][0]["reason"] == "allowed_by_policy"


def test_batch_matches_single_evaluation_results(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    requests = [item("read", "res/x"), item("read", "nope"), item("write", "res/x")]
    payload = batch(client, machine_id, requests).json()
    for request, result in zip(requests, payload["results"]):
        single = client.post(
            f"/machines/{machine_id}/authorization-evaluations", json=request
        ).json()
        assert result["allowed"] == single["allowed"]
        assert result["reason"] == single["reason"]


def test_other_machine_data_does_not_leak(client):
    machine_id = create_machine(client, "machine-1")
    other_id = create_machine(client, "machine-2")
    declare(client, other_id)  # only the other machine declares read
    create_rule(client)
    response = batch(client, machine_id, [item("read", "res/x")])
    assert response.json()["results"][0]["reason"] == "no_enabled_declaration"


def test_repeat_calls_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    requests = [item(), item("read", "nope")]
    first = batch(client, machine_id, requests).content
    second = batch(client, machine_id, requests).content
    assert first == second


def test_batch_writes_nothing(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    batch(client, machine_id, [item(), item()])
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert events == []


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("machine_id", [MISSING_ID])
def test_unknown_query_parameter_rejected_before_body(client, machine_id):
    response = client.post(
        batch_path(machine_id) + "?extra=1", json={"requests": []}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_query_parameter_rejected_with_invalid_body(client):
    # The query check precedes body parsing entirely.
    response = client.post(
        batch_path(MISSING_ID) + "?extra=1", content=b"not json"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "content",
    [None, b"", b"not json", b"[1,2]", b'"text"', b"42", b"null"],
)
def test_invalid_body_shape_is_invalid_batch(client, content):
    kwargs = {"content": content} if content is not None else {}
    response = client.post(batch_path(MISSING_ID), **kwargs)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"requests": None},
        {"requests": "read"},
        {"requests": 42},
        {"requests": True},
        {"requests": {"action_type": "read"}},
    ],
)
def test_invalid_requests_field_is_invalid_batch(client, payload):
    response = client.post(batch_path(MISSING_ID), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "requests",
    [
        ["read"],
        [None],
        [[]],
        [{}],
        [{"action_type": "read"}],
        [{"resource": "res/x"}],
        [{"action_type": 1, "resource": "res/x"}],
        [{"action_type": "read", "resource": None}],
        [{"action_type": True, "resource": "res/x"}],
        [{"action_type": "read", "resource": False}],
        [{"action_type": "", "resource": "res/x"}],
        [{"action_type": "read", "resource": "   "}],
        [{"action_type": " \t\n ", "resource": "res/x"}],
    ],
)
def test_invalid_item_is_invalid_value(client, requests):
    response = batch(client, MISSING_ID, requests)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_one_invalid_item_rejects_the_whole_batch(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    requests = [item(), {"action_type": " ", "resource": "res/x"}, item()]
    response = batch(client, machine_id, requests)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_validation_precedes_machine_lookup(client):
    # A 422 against a non-existent machine, never a 404.
    response = batch(client, MISSING_ID, [{"action_type": "", "resource": "x"}])
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_missing_machine_is_404_after_validation(client):
    response = batch(client, MISSING_ID, [item()])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_machine_empty_batch_is_404(client):
    response = batch(client, MISSING_ID, [])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_return_405(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(batch_path(machine_id))
    assert response.status_code == 405


def test_read_failure_returns_500_without_partial_analysis(client, monkeypatch):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    from accountability import authorization

    def boom(*args, **kwargs):
        raise RuntimeError("read failed")

    monkeypatch.setattr(authorization, "decide", boom)
    response = batch(client, machine_id, [item(), item()])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_empty_database_machine_evaluates(client):
    # A machine with no declarations and no rules still evaluates.
    machine_id = create_machine(client)
    response = batch(client, machine_id, [item()])
    assert response.status_code == 200
    assert response.json()["results"][0]["reason"] == "no_enabled_declaration"
