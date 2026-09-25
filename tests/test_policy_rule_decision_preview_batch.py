"""Tests for read-only batch policy decision preview.

Covers `POST /policy-rules/decision-preview/batch`:

- success: 200 with ``batch_count``, position-aligned ``analyses``
  (``input`` plus a single-shaped ``result``), ``summary`` counts, and
  ``decisions`` counts; the empty batch short-circuit shape;
- semantics: each result is byte-identical to the single preview for the
  same (action, resource), all items share one rule snapshot, and
  ``conflict``/``override`` count inputs rather than rules;
- validation: ``invalid_query`` for query parameters, ``invalid_batch`` for
  body/item shape failures, ``invalid_value`` for non-string or
  empty-after-trim action/resource — every 422 precedes the rule read and an
  illegal item rejects the whole batch with no partial analysis;
- lifecycle: POST-only ``405``, ``500 internal_error`` on a rule-read
  failure, read-only behavior, byte-identical repeats, and stability across
  restarts.
"""
import json

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


BATCH_PATH = "/policy-rules/decision-preview/batch"
SINGLE_PATH = "/policy-rules/decision-preview"


def batch(client, requests_):
    return client.post(BATCH_PATH, json={"requests": requests_})


def single(client, action, resource):
    return client.post(
        SINGLE_PATH, json={"action": action, "resource": resource}
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


# --- empty batch and response shape ---------------------------------------


def test_empty_batch_returns_200_empty_result_against_empty_table(client):
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
            "allowed_by_policy": 0,
            "denied_by_policy": 0,
            "no_matching_policy": 0,
        },
    }


def test_empty_category_counts_are_json_integer_zero(client):
    content = batch(client, []).content

    # Zero categories are present as integer 0, never null or omitted.
    body = content.decode("utf-8")
    assert '"summary":{"no_match":0,"allow":0,"deny":0,"conflict":0,"override":0}' in body
    assert '"decisions":{"allowed_by_policy":0,"denied_by_policy":0,"no_matching_policy":0}' in body


def test_top_level_field_order_is_stable(client):
    assert list(batch(client, []).json().keys()) == [
        "batch_count",
        "analyses",
        "summary",
        "decisions",
    ]


def test_summary_and_decisions_key_order_is_stable(client):
    body = batch(client, []).json()
    assert list(body["summary"].keys()) == [
        "no_match",
        "allow",
        "deny",
        "conflict",
        "override",
    ]
    assert list(body["decisions"].keys()) == [
        "allowed_by_policy",
        "denied_by_policy",
        "no_matching_policy",
    ]


def test_response_is_compact_utf8_json_with_single_newline(client):
    response = batch(client, [{"action": "读", "resource": "res/x"}])

    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    # Compact: no JSON whitespace separators.
    assert b'": ' not in response.content
    assert b", " not in response.content
    # UTF-8 is emitted directly, never ASCII-escaped.
    assert "读".encode("utf-8") in response.content


# --- alignment, input echo, single-preview equivalence --------------------


def test_batch_count_matches_requests_and_analyses_are_position_aligned(client):
    requests_ = [
        {"action": "write", "resource": "res/a"},
        {"action": " read ", "resource": "\tres/x\n"},
        {"action": "delete", "resource": "other/9"},
    ]

    body = batch(client, requests_).json()

    assert body["batch_count"] == 3
    assert len(body["analyses"]) == 3
    assert body["analyses"][0]["input"] == {"action": "write", "resource": "res/a"}
    # The input echoes the trimmed values, exactly as the single preview
    # echoes them in its action/resource fields.
    assert body["analyses"][1]["input"] == {"action": "read", "resource": "res/x"}
    assert body["analyses"][2]["input"] == {"action": "delete", "resource": "other/9"}
    assert [set(analysis) for analysis in body["analyses"]] == [
        {"input", "result"}
    ] * 3
    assert list(body["analyses"][0]["input"].keys()) == ["action", "resource"]


def test_each_result_has_the_single_preview_shape(client):
    create_rule(client, priority=0)

    [analysis] = batch(client, [{"action": "read", "resource": "res/x"}]).json()[
        "analyses"
    ]

    assert list(analysis["result"].keys()) == [
        "action",
        "resource",
        "rules",
        "conflicts",
        "winners",
        "decision",
    ]


def test_each_result_equals_the_single_preview_payload(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=4)
    create_rule(
        client, action_type="write", resource_pattern="*", effect="allow", priority=1
    )
    create_rule(
        client, action_type="write", resource_pattern="res/x",
        effect="deny", priority=1,
    )
    create_rule(client, action_type="write", effect="allow", priority=7)

    requests_ = [
        {"action": "read", "resource": "res/x"},
        {"action": "write", "resource": "res/x"},
        {"action": "write", "resource": "res/y"},
        {"action": "delete", "resource": "res/x"},
    ]

    analyses = batch(client, requests_).json()["analyses"]

    for submitted, analysis in zip(requests_, analyses, strict=True):
        expected = single(
            client, submitted["action"], submitted["resource"]
        ).json()
        assert analysis["result"] == expected


def test_identical_inputs_produce_identical_results(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=2)

    body = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},
            {"action": "read", "resource": "res/x"},
        ],
    ).json()

    first, second = body["analyses"]
    assert first["result"] == second["result"]
    assert body["batch_count"] == 2


def test_one_input_cannot_change_the_result_of_another(client):
    create_rule(client, action_type="read", resource_pattern="res/*", priority=0)

    # The same request previews identically whether or not other (even
    # unmatchable, blank-looking, or differently shaped) inputs sit beside it
    # in the batch: every item is decided against one shared snapshot.
    alone = batch(client, [{"action": "read", "resource": "res/x"}]).content
    mixed = batch(
        client,
        [
            {"action": "write", "resource": "zzz"},
            {"action": "read", "resource": "res/x"},
            {"action": "admin", "resource": "everything"},
        ],
    ).content

    mixed_body = json.loads(mixed)
    assert mixed_body["analyses"][1]["result"] == json.loads(alone)["analyses"][0]["result"]


# --- summary and decision counts ------------------------------------------


def test_summary_counts_inputs_across_all_five_categories(client):
    create_rule(
        client, action_type="read", resource_pattern="res/*",
        effect="allow", priority=0,
    )
    create_rule(
        client, action_type="read", resource_pattern="res/x",
        effect="deny", priority=2,
    )
    create_rule(
        client, action_type="write", resource_pattern="*",
        effect="allow", priority=1,
    )
    create_rule(
        client, action_type="write", resource_pattern="res/x",
        effect="deny", priority=1,
    )

    requests_ = [
        {"action": "read", "resource": "res/x"},    # allow, one overridden
        {"action": "write", "resource": "res/x"},   # conflict deny
        {"action": "read", "resource": "other/1"},  # no match
        {"action": "write", "resource": "res/y"},   # plain allow
        {"action": "delete", "resource": "res/x"},  # no match
    ]

    body = batch(client, requests_).json()

    assert body["batch_count"] == 5
    assert body["summary"] == {
        "no_match": 2,
        "allow": 2,
        "deny": 1,
        "conflict": 1,
        "override": 1,
    }
    assert body["decisions"] == {
        "allowed_by_policy": 2,
        "denied_by_policy": 1,
        "no_matching_policy": 2,
    }
    assert [a["result"]["decision"] for a in body["analyses"]] == [
        {"allowed": True, "reason": "allowed_by_policy"},
        {"allowed": False, "reason": "denied_by_policy"},
        {"allowed": False, "reason": "no_matching_policy"},
        {"allowed": True, "reason": "allowed_by_policy"},
        {"allowed": False, "reason": "no_matching_policy"},
    ]


def test_conflict_counts_inputs_not_pairs_and_override_counts_inputs(client):
    # Three rules in the same mixed minimum tier produce three conflict pairs
    # for one input, but the input counts once.
    create_rule(client, action_type="write", resource_pattern="*",
                effect="allow", priority=1)
    create_rule(client, action_type="write", resource_pattern="res/*",
                effect="deny", priority=1)
    create_rule(client, action_type="write", resource_pattern="res/a",
                effect="deny", priority=1)
    create_rule(client, action_type="write", resource_pattern="res/a",
                effect="allow", priority=5)

    body = batch(
        client,
        [
            {"action": "write", "resource": "res/a"},  # conflict + override
            {"action": "write", "resource": "res/a"},  # same, counted again
        ],
    ).json()

    first_rules = body["analyses"][0]["result"]
    assert len(first_rules["conflicts"]) == 3
    assert any(
        rule["relation"] == "overridden" for rule in first_rules["rules"]
    )
    assert body["summary"]["conflict"] == 2
    assert body["summary"]["override"] == 2
    assert body["summary"]["deny"] == 2
    assert body["decisions"]["denied_by_policy"] == 2


def test_plain_deny_input_counts_deny_without_conflict_or_override(client):
    create_rule(client, effect="deny", priority=0)
    create_rule(
        client, action_type="read", resource_pattern="res/other",
        effect="allow", priority=4,
    )

    body = batch(client, [{"action": "read", "resource": "res/x"}]).json()

    assert body["summary"] == {
        "no_match": 0,
        "allow": 0,
        "deny": 1,
        "conflict": 0,
        "override": 0,
    }


# --- validation: query -----------------------------------------------------


def test_query_parameter_is_invalid_query(client):
    response = client.post(
        f"{BATCH_PATH}?x=1", json={"requests": []}
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_check_precedes_body_parsing(client):
    response = client.post(
        f"{BATCH_PATH}?x=1",
        content="not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --- validation: invalid_batch --------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "{",
        "[1, 2]",
        '"requests"',
        "123",
        "null",
        "true",
    ],
)
def test_unparseable_or_non_object_body_is_invalid_batch(client, content):
    response = client.post(
        BATCH_PATH,
        content=content,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"responses": []},
        {"requests": [], "extra": 1},
        {"requests": None},
        {"requests": "x"},
        {"requests": {}},
        {"requests": 1},
        {"requests": True},
        {"requests": [None]},
        {"requests": [1]},
        {"requests": ["x"]},
        {"requests": [True]},
        {"requests": [[]]},
        {"requests": [{}]},
        {"requests": [{"action": "read"}]},
        {"requests": [{"resource": "res/x"}]},
        {"requests": [{"action": "read", "resource": "res/x", "extra": 1}]},
        {"requests": [
            {"action": "read", "resource": "res/x", "action_type": "read"}
        ]},
        {"requests": [
            {"action": "read", "resource": "res/x"},
            "not-an-object",
        ]},
    ],
)
def test_bad_body_or_item_shape_is_invalid_batch(client, payload):
    response = client.post(BATCH_PATH, json=payload)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_any_illegal_item_rejects_the_whole_batch_without_partial_analysis(
    client,
):
    create_rule(client, priority=0)

    response = batch(
        client,
        [
            {"action": "read", "resource": "res/x"},  # would be fine
            {"action": "read"},                        # illegal shape
        ],
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}
    assert b"analyses" not in response.content


# --- validation: invalid_value --------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [5, 5.0, None, True, False, ["read"], {"x": 1}],
)
def test_non_string_action_or_resource_is_invalid_value(client, bad_value):
    for payload in (
        {"requests": [{"action": bad_value, "resource": "res/x"}]},
        {"requests": [{"action": "read", "resource": bad_value}]},
    ):
        response = client.post(BATCH_PATH, json=payload)

        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_value"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"requests": [{"action": "   ", "resource": "res/x"}]},
        {"requests": [{"action": "read", "resource": ""}]},
        {"requests": [{"action": "\t\n", "resource": "res/x"}]},
        {"requests": [{"action": "read", "resource": "  "}]},
        {"requests": [
            {"action": "read", "resource": "res/x"},
            {"action": " ", "resource": "res/x"},
        ]},
    ],
)
def test_blank_after_trim_is_invalid_value(client, payload):
    response = client.post(BATCH_PATH, json=payload)

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_item_shape_check_precedes_value_domain_check(client):
    # A non-object item is a structural invalid_batch even though a string
    # inside it would have been a value problem; a well-shaped item with a
    # non-string field is invalid_value.
    shape = client.post(BATCH_PATH, json={"requests": [["read", "res/x"]]})
    assert shape.json() == {"error": {"code": "invalid_batch"}}

    value = client.post(
        BATCH_PATH,
        json={"requests": [{"action": 7, "resource": "res/x"}]},
    )
    assert value.json() == {"error": {"code": "invalid_value"}}


def test_validation_runs_before_rules_are_read(client):
    # Every illegal request is rejected identically after the table is gone:
    # validation never reaches the rule read.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    assert client.post(
        f"{BATCH_PATH}?x=1", json={"requests": []}
    ).json() == {"error": {"code": "invalid_query"}}
    assert client.post(
        BATCH_PATH,
        content="garbage",
        headers={"content-type": "application/json"},
    ).json() == {"error": {"code": "invalid_batch"}}
    assert batch(
        client, [{"action": "read", "resource": None}]
    ).json() == {"error": {"code": "invalid_value"}}
    assert batch(
        client, [{"action": " ", "resource": "res/x"}]
    ).json() == {"error": {"code": "invalid_value"}}


def test_empty_batch_is_valid_even_with_a_later_illegal_item(client):
    # Sanity for the all-or-nothing rule: only genuinely present items count.
    assert batch(client, []).status_code == 200
    assert batch(
        client, [{"action": "read", "resource": None}]
    ).status_code == 422


# --- methods, read failure, read-only, stability --------------------------


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_non_post_methods_return_405(client, method):
    response = getattr(client, method)(BATCH_PATH)

    assert response.status_code == 405


def test_internal_failure_is_500_without_partial_analysis(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE policy_rules"))

    response = batch(client, [{"action": "read", "resource": "res/x"}])

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"analyses" not in response.content


def test_batch_is_read_only(client):
    create_rule(client, priority=0)
    rules_before = client.get("/policy-rules").content
    chain_before = client.get("/policy-rules/chain").content

    batch(
        client,
        [
            {"action": "read", "resource": "res/x"},
            {"action": "write", "resource": "res/y"},
        ],
    )
    batch(client, [])

    assert client.get("/policy-rules").content == rules_before
    assert client.get("/policy-rules/chain").content == chain_before


def test_repeated_calls_are_byte_identical(client):
    create_rule(client, priority=0)
    create_rule(client, effect="deny", priority=2)
    payload = {
        "requests": [
            {"action": "read", "resource": "res/x"},
            {"action": "read", "resource": "other"},
        ]
    }

    first = client.post(BATCH_PATH, json=payload).content
    second = client.post(BATCH_PATH, json=payload).content
    third = client.post(BATCH_PATH, json=payload).content

    assert first == second == third


def test_batch_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)
    payload = {"requests": [{"action": "read", "resource": "res/x"}]}

    with TestClient(app) as first:
        create_rule(first, effect="deny", priority=0)
        expected = first.post(BATCH_PATH, json=payload).content

    with TestClient(app) as second:
        response = second.post(BATCH_PATH, json=payload)

    assert response.status_code == 200
    assert response.content == expected
