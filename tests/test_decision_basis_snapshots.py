"""Tests for the immutable authorization decision-basis snapshot.

Every new authorization decision event commits an immutable snapshot of the
basis the decision used — machine state, declaration scope, and policy
candidate relations — in the same locked write as the event. The snapshot is
served read-only at the event's ``decision-basis`` sub-entry, byte-identical
on repeat queries and across restarts, strictly isolated to the path machine;
events predating the feature report ``snapshot_not_found`` rather than a
basis fabricated from current data.
"""
import json
import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"
RFC3339_Z_RE = "Z"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "restart.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{path}")
    return path


def create_machine(client, external_id="machine-1"):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "Machine One",
            "public_key": "key-secret-material",
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


def declare(client, machine_id, action_type="read", resource_pattern="res/*", enabled=True):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201
    return response.json()


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def basis_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{event_id}/decision-basis"
    )


def get_basis(client, machine_id, event):
    response = client.get(basis_url(machine_id, event["id"]))
    assert response.status_code == 200
    return response


# --------------------------------------------------------------------------- #
# Snapshot content per decision reason
# --------------------------------------------------------------------------- #


def test_suspended_event_basis_records_unread_inputs(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})

    event = record_event(client, machine_id).json()
    response = get_basis(client, machine_id, event)
    body = json.loads(response.content)

    assert list(body.keys()) == [
        "event_summary",
        "status_basis",
        "declaration_basis",
        "policy_candidates",
        "decision",
    ]
    summary = body["event_summary"]
    assert list(summary.keys()) == [
        "id",
        "machine_id",
        "action_type",
        "resource",
        "allowed",
        "reason",
        "created_at",
        "captured_at",
    ]
    assert summary["id"] == event["id"]
    assert summary["machine_id"] == machine_id
    assert summary["allowed"] is False
    assert summary["reason"] == "machine_suspended"
    assert summary["created_at"] == event["created_at"]
    assert summary["captured_at"].endswith(RFC3339_Z_RE)

    assert body["status_basis"] == {
        "machine_id": machine_id,
        "status": "suspended",
        "declarations_read": False,
        "policies_read": False,
    }
    assert body["declaration_basis"] == []
    assert body["policy_candidates"] == []
    assert body["decision"] == {"allowed": False, "reason": "machine_suspended"}


def test_active_machine_basis_records_participating_declarations(client):
    machine_id = create_machine(client)
    matching = declare(client, machine_id, resource_pattern="res/*")
    non_matching = declare(client, machine_id, resource_pattern="other/*")
    disabled = declare(client, machine_id, resource_pattern="res/disabled/*", enabled=False)
    create_rule(client)

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    assert body["status_basis"]["status"] == "active"
    assert body["status_basis"]["declarations_read"] is True
    assert body["status_basis"]["policies_read"] is True

    entries = body["declaration_basis"]
    by_id = {entry["id"]: entry for entry in entries}
    assert matching["id"] in by_id and non_matching["id"] in by_id
    assert disabled["id"] not in by_id  # disabled declarations did not participate
    assert by_id[matching["id"]]["matched"] is True
    assert by_id[non_matching["id"]]["matched"] is False
    for entry in entries:
        assert list(entry.keys()) == [
            "id",
            "machine_id",
            "action_type",
            "resource_pattern",
            "enabled",
            "created_at",
            "updated_at",
            "matched",
        ]
        assert entry["enabled"] is True
        assert entry["machine_id"] == machine_id
    # Stable (created_at, id) order.
    stamps = [(e["created_at"], e["id"]) for e in entries]
    assert stamps == sorted(stamps)


def test_no_enabled_declaration_records_policy_as_unread(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="other/*")
    create_rule(client)  # a matching policy exists but must not be consulted

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    assert event["reason"] == "no_enabled_declaration"
    assert body["status_basis"] == {
        "machine_id": machine_id,
        "status": "active",
        "declarations_read": True,
        "policies_read": False,
    }
    assert body["declaration_basis"][0]["matched"] is False
    assert body["policy_candidates"] == []
    assert body["decision"] == {
        "allowed": False,
        "reason": "no_enabled_declaration",
    }


def test_no_matching_policy_basis_marks_read_rules_not_adopted(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    rule = create_rule(client, resource_pattern="other/*")

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    assert body["status_basis"]["policies_read"] is True
    assert len(body["policy_candidates"]) == 1
    candidate = body["policy_candidates"][0]
    assert candidate["id"] == rule["id"]
    assert candidate["relation"] == "not_adopted"
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}


def test_winner_allow_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    rule = create_rule(client, effect="allow", priority=0)

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    candidates = body["policy_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert list(candidate.keys()) == [
        "id",
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
        "relation",
    ]
    assert candidate["id"] == rule["id"]
    assert candidate["relation"] == "winner"
    assert candidate["priority"] == 0
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}


def test_winner_deny_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="deny", priority=0)

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    assert body["policy_candidates"][0]["relation"] == "winner"
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}


def test_candidate_relations_winner_overridden_conflict_not_adopted(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    # Decisive tier 0 mixes allow and deny over the requested resource with
    # distinct patterns -> conflict; a tier-5 allow is overridden; an action
    # rule whose scope misses the resource is not adopted. (Rules keep the
    # global (action_type, resource_pattern, priority) uniqueness constraint,
    # so same-tier rules must differ in pattern.)
    allow_0 = create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    deny_0 = create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    miss_1 = create_rule(client, resource_pattern="other/*", effect="allow", priority=1)
    over_5 = create_rule(client, resource_pattern="res/*", effect="allow", priority=5)

    event = record_event(client, machine_id, resource="res/x").json()
    body = json.loads(get_basis(client, machine_id, event).content)

    assert event["allowed"] is False
    assert event["reason"] == "denied_by_policy"
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}

    ordered = [(c["id"], c["relation"], c["priority"]) for c in body["policy_candidates"]]
    by_relation = {c["id"]: c["relation"] for c in body["policy_candidates"]}
    assert by_relation[allow_0["id"]] == "conflict"
    assert by_relation[deny_0["id"]] == "conflict"
    assert by_relation[over_5["id"]] == "overridden"
    assert by_relation[miss_1["id"]] == "not_adopted"
    # Priority-first stable order: tier 0 conflicts, the priority-1
    # non-adopted rule, then the priority-5 overridden rule.
    assert [item[1] for item in ordered] == [
        "conflict",
        "conflict",
        "not_adopted",
        "overridden",
    ]
    priorities = [item[2] for item in ordered]
    assert priorities == sorted(priorities)


def test_overridden_higher_priority_cannot_change_result(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    create_rule(client, resource_pattern="res/*", effect="deny", priority=5)

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)

    relations = {c["effect"]: c["relation"] for c in body["policy_candidates"]}
    assert relations["allow"] == "winner"
    assert relations["deny"] == "overridden"
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}


@pytest.mark.parametrize(
    "setup",
    [
        "suspended",
        "no_declaration",
        "no_policy",
        "allow",
        "deny",
        "conflict",
    ],
)
def test_basis_decision_always_matches_committed_event(client, setup):
    machine_id = create_machine(client)
    if setup == "suspended":
        declare(client, machine_id)
        create_rule(client)
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})
    elif setup == "no_declaration":
        # Active machine with no matching enabled declaration (a policy may
        # exist but is never consulted).
        create_rule(client)
    elif setup == "no_policy":
        declare(client, machine_id)
    elif setup == "allow":
        declare(client, machine_id)
        create_rule(client, effect="allow", priority=0)
    elif setup == "deny":
        declare(client, machine_id)
        create_rule(client, effect="deny", priority=0)
    elif setup == "conflict":
        declare(client, machine_id)
        create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
        create_rule(client, resource_pattern="res/x", effect="deny", priority=0)

    event = record_event(client, machine_id).json()
    body = json.loads(get_basis(client, machine_id, event).content)
    assert body["decision"]["allowed"] == event["allowed"]
    assert body["decision"]["reason"] == event["reason"]
    assert body["event_summary"]["allowed"] == event["allowed"]
    assert body["event_summary"]["reason"] == event["reason"]


# --------------------------------------------------------------------------- #
# Encoding, immutability, isolation, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_json_with_single_newline(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    raw = get_basis(client, machine_id, event).content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b", " not in raw and b": " not in raw
    raw.decode("utf-8")  # valid UTF-8
    # Compact re-encoding of the parsed document reproduces the stored bytes.
    parsed = json.loads(raw)
    assert json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n" == raw
    # No floating-point syntax: priorities and booleans stay JSON ints/bools.
    assert b"0.0" not in raw and b"-0.0" not in raw
    assert b"NaN" not in raw and b"Infinity" not in raw


def test_utf8_content_round_trips_byte_identically(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id, resource="res/カフェ/Ω").json()

    first = get_basis(client, machine_id, event).content
    second = get_basis(client, machine_id, event).content
    assert first == second
    assert "res/カフェ/Ω".encode("utf-8") in first


def test_repeat_queries_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="allow", priority=0)
    create_rule(client, effect="deny", priority=3)
    event = record_event(client, machine_id).json()

    responses = [
        client.get(basis_url(machine_id, event["id"])).content for _ in range(5)
    ]
    assert all(chunk == responses[0] for chunk in responses)


def test_snapshot_is_immutable_when_policy_then_changes(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client, effect="allow", priority=0)
    first_event = record_event(client, machine_id).json()
    first_bytes = client.get(basis_url(machine_id, first_event["id"])).content

    # A later same-priority deny changes subsequent decisions, but the earlier
    # snapshot must remain exactly what it committed.
    create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    second_event = record_event(client, machine_id).json()

    assert client.get(basis_url(machine_id, first_event["id"])).content == first_bytes
    second_body = json.loads(
        client.get(basis_url(machine_id, second_event["id"])).content
    )
    assert second_body["decision"] == {"allowed": False, "reason": "denied_by_policy"}
    first_body = json.loads(first_bytes)
    assert first_body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}


def test_basis_persists_across_restart(db_path):
    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id).json()
        first_bytes = first.get(basis_url(machine_id, event["id"])).content

    with TestClient(app) as second:
        response = second.get(basis_url(machine_id, event["id"]))

    assert response.status_code == 200
    assert response.content == first_bytes


def test_snapshot_is_isolated_to_path_machine(client):
    one = create_machine(client, "machine-1")
    two = create_machine(client, "machine-2")
    declare(client, one)
    declare(client, two)
    create_rule(client)
    event_one = record_event(client, one).json()
    record_event(client, two)

    # Asking machine two's path for machine one's event is an ownership
    # mismatch: 404 not_found with no basis document.
    response = client.get(basis_url(two, event_one["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"event_summary" not in response.content


def test_basis_exposes_no_key_material(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    raw = get_basis(client, machine_id, event).content
    assert b"key-secret-material" not in raw
    assert b"public_key" not in raw


def test_evaluation_creates_no_snapshot(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    )
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_decision_basis")
        ).scalar_one()
    assert count == 0


# --------------------------------------------------------------------------- #
# Legacy events and old databases
# --------------------------------------------------------------------------- #


def test_historical_event_without_snapshot_is_snapshot_not_found(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    assert get_basis(client, machine_id, event).status_code == 200

    # Simulate a pre-feature event: its committed row exists but no snapshot
    # row was ever written with it.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :eid"
            ),
            {"eid": event["id"]},
        )

    response = client.get(basis_url(machine_id, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "snapshot_not_found"}}
    assert b"event_summary" not in response.content

    # Current declarations/rules must not be used to fabricate the old basis.
    create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    again = client.get(basis_url(machine_id, event["id"]))
    assert again.status_code == 404
    assert again.json() == {"error": {"code": "snapshot_not_found"}}


def test_old_database_recreates_table_safely_on_startup(db_path):
    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        old_event = record_event(first, machine_id).json()
        # Simulate a database from before the feature: drop the basis table.
        with first.app.state.engine.begin() as conn:
            conn.execute(text("DROP TABLE authorization_decision_basis"))

    # Restarting over the old database safely creates the table; the old event
    # has no historical snapshot, but new events commit and serve one.
    with TestClient(app) as second:
        old_response = second.get(basis_url(machine_id, old_event["id"]))
        assert old_response.status_code == 404
        assert old_response.json() == {"error": {"code": "snapshot_not_found"}}

        new_event = record_event(second, machine_id).json()
        new_response = second.get(basis_url(machine_id, new_event["id"]))
        assert new_response.status_code == 200
        new_body = json.loads(new_response.content)
        assert new_body["event_summary"]["id"] == new_event["id"]


# --------------------------------------------------------------------------- #
# Validation, routing, and read-failure ordering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    ["?x=1", "?limit=10", "?x=1&x=2", "?=", "?foo"],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = client.get(basis_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_get_with_a_body_is_invalid_query(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = client.request(
        "GET",
        basis_url(machine_id, event["id"]),
        content=b'{"unexpected": true}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_any_lookup(client):
    response = client.get(basis_url(MISSING_ID, MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_any_lookup(client):
    response = client.request(
        "GET",
        basis_url(MISSING_ID, MISSING_ID),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_business_reads(client):
    # With the business tables dropped, a read would 500; validation wins.
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
        conn.execute(text("DROP TABLE authorization_decision_events"))
    response = client.get(basis_url(machine_id, event["id"]) + "?x=1")
    assert response.status_code == 422
    response = client.request(
        "GET",
        basis_url(machine_id, event["id"]),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = getattr(client, method)(basis_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"event_summary" not in response.content


def test_method_routing_does_not_read_snapshot_or_event(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))
        conn.execute(text("DROP TABLE authorization_decision_events"))
    for method in ("head", "post", "put", "patch", "delete"):
        assert (
            getattr(client, method)(basis_url(machine_id, event["id"])).status_code
            == 405
        )


def test_missing_machine_is_404(client):
    response = client.get(basis_url(MISSING_ID, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_event_is_404(client):
    machine_id = create_machine(client)
    response = client.get(basis_url(machine_id, MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_read_failure_returns_internal_error_without_partial_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))

    response = client.get(basis_url(machine_id, event["id"]))
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content


# --------------------------------------------------------------------------- #
# Event creation compatibility and atomicity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource": "r"},
        {"action_type": "a"},
        {},
        {"action_type": 1, "resource": "r"},
        {"action_type": None, "resource": "r"},
    ],
)
def test_event_creation_body_validation_unchanged(client, payload):
    machine_id = create_machine(client)
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events", json=payload
    )
    assert response.status_code == 422
    with client.app.state.engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT COUNT(*) FROM authorization_decision_basis")
            ).scalar_one()
            == 0
        )


def test_event_creation_missing_machine_unchanged(client):
    response = record_event(client, MISSING_ID)
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_every_committed_event_has_exactly_one_snapshot(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    for index in range(5):
        assert record_event(client, machine_id, resource=f"res/{index}").status_code == 201
    with client.app.state.engine.connect() as conn:
        events = conn.execute(
            text(
                "SELECT id, allowed, reason FROM authorization_decision_events "
                "WHERE machine_id = :m ORDER BY created_at, id"
            ),
            {"m": machine_id},
        ).all()
        rows = conn.execute(
            text(
                "SELECT event_id, basis_json FROM authorization_decision_basis "
                "WHERE machine_id = :m ORDER BY captured_at, event_id"
            ),
            {"m": machine_id},
        ).all()
    assert len(rows) == len(events) == 5
    for event_row, basis_row in zip(events, rows, strict=True):
        assert basis_row.event_id == event_row.id
        document = json.loads(basis_row.basis_json)
        assert document["decision"]["allowed"] == bool(event_row.allowed)
        assert document["decision"]["reason"] == event_row.reason


def test_concurrent_events_commit_snapshots_atomically(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    errors = []

    def burst():
        try:
            for index in range(10):
                response = record_event(client, machine_id, resource=f"res/{index}")
                if response.status_code != 201:
                    errors.append(response.status_code)
        except Exception as exc:  # pragma: no cover - failure surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=burst) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with client.app.state.engine.connect() as conn:
        event_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_decision_events "
                "WHERE machine_id = :m"
            ),
            {"m": machine_id},
        ).scalar_one()
        basis_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_decision_basis "
                "WHERE machine_id = :m"
            ),
            {"m": machine_id},
        ).scalar_one()
        orphaned = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_decision_basis b "
                "LEFT JOIN authorization_decision_events e ON e.id = b.event_id "
                "WHERE e.id IS NULL"
            )
        ).scalar_one()
    assert event_count == basis_count == 60
    assert orphaned == 0


def test_suspended_then_active_events_each_keep_own_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})
    suspended_event = record_event(client, machine_id).json()
    client.post(f"/machines/{machine_id}/status", json={"status": "active"})
    active_event = record_event(client, machine_id).json()

    suspended = json.loads(
        client.get(basis_url(machine_id, suspended_event["id"])).content
    )
    active = json.loads(client.get(basis_url(machine_id, active_event["id"])).content)

    assert suspended["status_basis"]["status"] == "suspended"
    assert suspended["status_basis"]["declarations_read"] is False
    assert suspended["decision"]["reason"] == "machine_suspended"
    assert active["status_basis"]["status"] == "active"
    assert active["status_basis"]["declarations_read"] is True
    assert active["decision"]["reason"] == "allowed_by_policy"
