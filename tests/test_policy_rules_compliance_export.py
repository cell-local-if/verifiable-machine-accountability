"""Tests for the read-only global policy rule compliance export.

Covers `GET /policy-rules/compliance-export`: the strict query validation
(``invalid_query`` for unknown parameters, ``bad_time`` for missing, blank,
offset, whitespace, missing-``Z``, out-of-range, or inverted bounds — all
before any rule is read), GET-only ``405`` routing, closed-UTC-window
filtering on each rule's own ``created_at``, ordering by the actual UTC
instant then record id (exact-second before fractional-second), verbatim
export of stored values without filtering, repair, or normalization, the
always-present ``policy_rules`` array (empty on an empty window or empty
database), the verbatim echo of the request bounds, the fixed-field-order
compact newline-terminated UTF-8 body, strict read-only byte stability, no
interaction with authorization evaluation, and persistence across a restart.
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


WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"


def export_url(from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        "/policy-rules/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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


def insert_rule_row(
    client,
    rule_id,
    created_at,
    *,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=None,
    updated_at=None,
):
    """Insert a policy rule directly with a fixed id, timestamp, and values."""
    if priority is None:
        # The table is unique on (action_type, resource_pattern, priority);
        # derive a distinct default priority from the id's numeric suffix.
        priority = int(rule_id.rsplit("-", 1)[1]) + 1000
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
                "updated_at": updated_at if updated_at is not None else created_at,
            },
        )


def test_empty_database_returns_full_empty_envelope(client):
    response = client.get(export_url())

    assert response.status_code == 200
    assert response.json() == {
        "from_created_at": WIDE[0],
        "to_created_at": WIDE[1],
        "policy_rules": [],
    }


def test_bounds_are_echoed_verbatim(client):
    response = client.get(
        export_url("2026-03-01T00:00:00.500000Z", "2026-03-02T00:00:00Z")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["from_created_at"] == "2026-03-01T00:00:00.500000Z"
    assert body["to_created_at"] == "2026-03-02T00:00:00Z"
    assert body["policy_rules"] == []


def test_export_includes_only_rules_inside_the_closed_window(client):
    insert_rule_row(client, rid(1), T0)
    insert_rule_row(client, rid(2), T1)
    insert_rule_row(client, rid(3), T2)
    insert_rule_row(client, rid(4), T3)

    response = client.get(export_url(T1, T2))

    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [
        rid(2),
        rid(3),
    ]


def test_window_bounds_are_inclusive(client):
    insert_rule_row(client, rid(1), T1)
    insert_rule_row(client, rid(2), T2)

    response = client.get(export_url(T1, T2))

    assert [rule["id"] for rule in response.json()["policy_rules"]] == [
        rid(1),
        rid(2),
    ]


def test_equal_bounds_form_a_valid_instant_window(client):
    insert_rule_row(client, rid(1), T1)
    insert_rule_row(client, rid(2), T2)

    response = client.get(export_url(T1, T1))

    assert response.status_code == 200
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]


def test_fractional_bound_instant_is_respected(client):
    insert_rule_row(client, rid(1), "2026-03-01T00:00:00.500000Z")
    insert_rule_row(client, rid(2), T1)

    # A bound with a fractional part excludes the exact-second record at T1.
    response = client.get(export_url(T0, "2026-03-01T00:00:00.750000Z"))

    assert [rule["id"] for rule in response.json()["policy_rules"]] == [rid(1)]


def test_empty_window_returns_empty_array_not_omission(client):
    insert_rule_row(client, rid(1), T0)

    response = client.get(export_url(T1, T2))

    assert response.status_code == 200
    assert response.json() == {
        "from_created_at": T1,
        "to_created_at": T2,
        "policy_rules": [],
    }


def test_records_ordered_by_created_at_instant_then_id(client):
    insert_rule_row(client, rid(30), T3)
    insert_rule_row(client, rid(21), T2)
    insert_rule_row(client, rid(20), T2)
    insert_rule_row(client, rid(1), T0)
    insert_rule_row(client, rid(2), "2026-03-01T00:00:00.500000Z")

    response = client.get(export_url())

    assert [rule["id"] for rule in response.json()["policy_rules"]] == [
        rid(1),   # exact second sorts before the fractional stamp
        rid(2),   # same wall-clock second, 0.5s later
        rid(20),  # tie at T2 breaks by id
        rid(21),
        rid(30),
    ]


def test_items_have_exactly_the_listing_fields(client):
    created = create_rule(client, priority=4)

    [rule] = client.get(export_url()).json()["policy_rules"]

    assert list(rule.keys()) == [
        "id",
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
    ]
    assert rule == created


def test_stored_values_are_exported_verbatim_without_normalization(client):
    # Values the write path would never produce are echoed exactly as stored.
    insert_rule_row(
        client,
        rid(1),
        T0,
        action_type="  Read ",
        resource_pattern=" res/* ",
        effect="ALLOW",
        priority=9,
        updated_at="2026-04-01T00:00:00Z",
    )

    [rule] = client.get(export_url()).json()["policy_rules"]

    assert rule == {
        "id": rid(1),
        "action_type": "  Read ",
        "resource_pattern": " res/* ",
        "effect": "ALLOW",
        "priority": 9,
        "created_at": T0,
        "updated_at": "2026-04-01T00:00:00Z",
    }


def test_response_body_is_compact_json_ending_in_one_newline(client):
    insert_rule_row(client, rid(1), T0)

    response = client.get(export_url())

    assert response.headers["content-type"] == "application/json"
    raw = response.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Compact separators: no spaces after ':' or ','.
    assert b": " not in raw and b", " not in raw
    assert json.loads(raw.decode("utf-8"))["policy_rules"][0]["id"] == rid(1)


def test_non_ascii_values_stay_utf8_unescaped(client):
    insert_rule_row(client, rid(1), T0, action_type="读取", resource_pattern="资源/*")

    raw = client.get(export_url()).content

    assert "读取".encode("utf-8") in raw
    assert b"\\u" not in raw


@pytest.mark.parametrize(
    "query",
    [
        "",  # both bounds missing
        "?from_created_at=2026-03-01T00:00:00Z",  # to_created_at missing
        "?to_created_at=2026-03-01T00:00:00Z",  # from_created_at missing
        "?from_created_at=&to_created_at=2026-03-01T00:00:00Z",  # blank
        "?from_created_at=2026-03-01&to_created_at=2026-03-02T00:00:00Z",
        "?from_created_at=2026-03-01T00:00:00&to_created_at=2026-03-02T00:00:00Z",  # no Z
        "?from_created_at=2026-03-01T00:00:00+00:00&to_created_at=2026-03-02T00:00:00Z",  # offset
        "?from_created_at=%202026-03-01T00:00:00Z&to_created_at=2026-03-02T00:00:00Z",  # whitespace
        "?from_created_at=2026-13-01T00:00:00Z&to_created_at=2026-03-02T00:00:00Z",  # month 13
        "?from_created_at=2026-02-30T00:00:00Z&to_created_at=2026-03-02T00:00:00Z",  # Feb 30
        "?from_created_at=2026-03-01T24:00:00Z&to_created_at=2026-03-02T00:00:00Z",  # hour 24
        "?from_created_at=2026-03-02T00:00:00Z&to_created_at=2026-03-01T00:00:00Z",  # inverted
    ],
)
def test_invalid_time_bounds_return_422_bad_time(client, query):
    insert_rule_row(client, rid(1), T0)

    response = client.get(f"/policy-rules/compliance-export{query}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_unknown_query_parameter_returns_422_invalid_query(client):
    insert_rule_row(client, rid(1), T0)

    response = client.get(export_url() + "&limit=10")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_precedes_any_rule_read(client):
    # An invalid query is rejected even though rules exist to export.
    insert_rule_row(client, rid(1), T0)

    response = client.get("/policy-rules/compliance-export")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_return_405(client, method):
    insert_rule_row(client, rid(1), T0)

    response = getattr(client, method)(
        "/policy-rules/compliance-export",
        params={"from_created_at": WIDE[0], "to_created_at": WIDE[1]},
    )

    assert response.status_code == 405
    # The stored rule is untouched by the rejected call.
    assert [rule["id"] for rule in client.get("/policy-rules").json()] == [rid(1)]


def test_export_is_read_only_and_byte_identical_on_repeat_calls(client):
    insert_rule_row(client, rid(1), T0)
    insert_rule_row(client, rid(2), T1)

    first = client.get(export_url()).content
    second = client.get(export_url()).content
    third = client.get(export_url()).content

    assert first == second == third
    # The export never modifies the rules: the listing is unchanged.
    assert [rule["id"] for rule in client.get("/policy-rules").json()] == [
        rid(1),
        rid(2),
    ]


def test_export_does_not_change_authorization_evaluation(client):
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
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )
    create_rule(client, effect="allow", priority=0)

    client.get(export_url())
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    client.get(export_url())

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}


def test_export_reads_rules_persisted_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        insert_rule_row(first, rid(1), T0)
        insert_rule_row(first, rid(2), T1)
        expected = first.get(export_url()).json()

    with TestClient(app) as second:
        response = second.get(export_url())

    assert response.status_code == 200
    assert response.json() == expected
    assert [rule["id"] for rule in response.json()["policy_rules"]] == [
        rid(1),
        rid(2),
    ]
