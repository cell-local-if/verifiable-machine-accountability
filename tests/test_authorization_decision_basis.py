"""Tests for the immutable authorization decision-basis snapshot.

Every new authorization decision event commits an immutable snapshot of the
basis the decision used — machine status, declaration matching scope, policy
candidate relations, result, reason, and capture moment — in the same locked
write transaction as the event. The snapshot is read back through

    GET /machines/{machine_id}/authorization-decision-events/{event_id}/decision-basis

as five groups in a fixed order, compact UTF-8 JSON terminated by one
newline, byte-identical on repeat queries and across restarts, strictly
isolated to the path machine. These tests cover the five decision shapes,
candidate winner/overridden/conflict/unmatched semantics, validation and
routing outcomes (422/404/405/500), the legacy ``snapshot_not_found`` case,
restart persistence, machine isolation, and event/snapshot atomicity under a
forced snapshot-write failure and under concurrency.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability import decision_basis


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


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


def create_rule(client, action_type="read", resource_pattern="res/*",
                effect="allow", priority=0):
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


def declare(client, machine_id, action_type="read", resource_pattern="res/*",
            enabled=True):
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


def get_basis(client, machine_id, event_id):
    return client.get(basis_url(machine_id, event_id))


_GROUPS = [
    "event_summary",
    "status_basis",
    "declaration_basis",
    "policy_candidates",
    "decision",
]


# --------------------------------------------------------------------------- #
# Shape and contents
# --------------------------------------------------------------------------- #


def test_basis_has_five_groups_in_fixed_order_for_allowed_event(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    event = record_event(client, machine_id).json()
    response = get_basis(client, machine_id, event["id"])

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == _GROUPS
    assert isinstance(body["event_summary"], dict)
    assert isinstance(body["decision"], dict)


def test_event_summary_and_decision_match_the_committed_event(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    event = record_event(client, machine_id, resource="res/abc").json()
    body = get_basis(client, machine_id, event["id"]).json()

    summary = body["event_summary"]
    assert summary == {
        "id": event["id"],
        "machine_id": machine_id,
        "action_type": "read",
        "resource": "res/abc",
        "allowed": True,
        "reason": "allowed_by_policy",
        "created_at": event["created_at"],
        "previous_event_id": event["previous_event_id"],
        "content_hash": event["content_hash"],
        "chain_hash": event["chain_hash"],
    }
    assert body["decision"] == {"allowed": True, "reason": "allowed_by_policy"}
    assert (
        body["decision"]["allowed"] == event["allowed"]
        and body["decision"]["reason"] == event["reason"]
    )


def test_active_allowed_basis_records_status_capture_moment_and_reads(client):
    machine_id = create_machine(client)
    declaration = declare(client, machine_id)
    create_rule(client)

    event = record_event(client, machine_id).json()
    body = get_basis(client, machine_id, event["id"]).json()

    status_basis = body["status_basis"]
    assert status_basis["machine_id"] == machine_id
    assert status_basis["status"] == "active"
    assert status_basis["captured_at"] == event["created_at"]
    assert status_basis["declarations_read"] is True
    assert status_basis["policies_read"] is True

    assert body["declaration_basis"]["read"] is True
    items = body["declaration_basis"]["declarations"]
    assert len(items) == 1
    assert set(items[0]) == {
        "id",
        "action_type",
        "resource_pattern",
        "enabled",
        "created_at",
        "updated_at",
        "matched",
    }
    assert items[0]["id"] == declaration["id"]
    assert items[0]["matched"] is True


def test_candidate_winner_and_overridden_follow_priority_semantics(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    deny = create_rule(client, resource_pattern="res/x", effect="deny", priority=1)
    create_rule(client, resource_pattern="res/*", effect="allow", priority=5)

    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "denied_by_policy"
    body = get_basis(client, machine_id, event["id"]).json()

    candidates = body["policy_candidates"]["candidates"]
    by_relation = {}
    for candidate in candidates:
        assert set(candidate) == {
            "id",
            "action_type",
            "resource_pattern",
            "effect",
            "priority",
            "created_at",
            "updated_at",
            "relation",
        }
        by_relation.setdefault(candidate["relation"], []).append(candidate)

    assert [c["id"] for c in by_relation["winner"]] == [deny["id"]]
    assert len(by_relation["overridden"]) == 1
    assert by_relation["overridden"][0]["priority"] == 5
    # Priority ascending: the priority-1 winner is listed first.
    assert [c["priority"] for c in candidates] == sorted(
        c["priority"] for c in candidates
    )
    assert [c["id"] for c in body["policy_candidates"]["winners"]] == [deny["id"]]
    assert body["policy_candidates"]["winners"][0] == {
        "id": deny["id"],
        "effect": "deny",
        "priority": 1,
        "created_at": deny["created_at"],
    }
    assert body["policy_candidates"]["conflicts"] == []


def test_unmatched_rules_are_candidates_marked_unused(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    # Same action but a pattern the requested resource does not match.
    create_rule(client, resource_pattern="other/*", effect="deny", priority=0)
    # A different action never participates and never enters the basis.
    create_rule(
        client, action_type="write", resource_pattern="res/*",
        effect="deny", priority=0,
    )

    event = record_event(client, machine_id, resource="res/x").json()
    body = get_basis(client, machine_id, event["id"]).json()

    candidates = body["policy_candidates"]["candidates"]
    relations = {c["resource_pattern"]: c["relation"] for c in candidates}
    assert relations == {"res/*": "winner", "other/*": "unmatched"}
    assert all(c["action_type"] == "read" for c in candidates)


def test_same_priority_mixed_tier_is_conflict_and_denies(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    allow_rule = create_rule(
        client, resource_pattern="res/*", effect="allow", priority=2
    )
    deny_rule = create_rule(
        client, resource_pattern="res/x", effect="deny", priority=2
    )

    event = record_event(client, machine_id, resource="res/x").json()
    assert event["allowed"] is False
    assert event["reason"] == "denied_by_policy"
    body = get_basis(client, machine_id, event["id"]).json()

    relations = sorted(c["relation"] for c in body["policy_candidates"]["candidates"])
    assert relations == ["conflict", "conflict"]
    conflicts = body["policy_candidates"]["conflicts"]
    assert conflicts == [sorted([allow_rule["id"], deny_rule["id"]])]
    assert body["policy_candidates"]["winners"] == []
    assert body["decision"] == {"allowed": False, "reason": "denied_by_policy"}


def test_suspended_basis_records_no_declaration_or_policy_read(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )

    event = record_event(client, machine_id).json()
    assert event["reason"] == "machine_suspended"
    body = get_basis(client, machine_id, event["id"]).json()

    assert body["status_basis"]["status"] == "suspended"
    assert body["status_basis"]["declarations_read"] is False
    assert body["status_basis"]["policies_read"] is False
    assert body["declaration_basis"] == {"read": False, "declarations": []}
    assert body["policy_candidates"] == {
        "read": False,
        "candidates": [],
        "winners": [],
        "conflicts": [],
    }
    assert body["decision"] == {"allowed": False, "reason": "machine_suspended"}


def test_no_enabled_declaration_basis_reads_declarations_but_not_policy(client):
    machine_id = create_machine(client)
    # A disabled declaration does not participate and is not snapshotted.
    declare(client, machine_id, enabled=False)
    create_rule(client)

    event = record_event(client, machine_id).json()
    assert event["reason"] == "no_enabled_declaration"
    body = get_basis(client, machine_id, event["id"]).json()

    assert body["status_basis"] == {
        "machine_id": machine_id,
        "status": "active",
        "captured_at": event["created_at"],
        "declarations_read": True,
        "policies_read": False,
    }
    assert body["declaration_basis"] == {"read": True, "declarations": []}
    assert body["policy_candidates"]["read"] is False
    assert body["policy_candidates"]["candidates"] == []
    assert body["decision"] == {
        "allowed": False,
        "reason": "no_enabled_declaration",
    }


def test_no_matching_policy_basis_reads_empty_candidate_set(client):
    machine_id = create_machine(client)
    declaration = declare(client, machine_id, resource_pattern="res/*")

    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "no_matching_policy"
    body = get_basis(client, machine_id, event["id"]).json()

    assert body["status_basis"]["policies_read"] is True
    assert body["policy_candidates"] == {
        "read": True,
        "candidates": [],
        "winners": [],
        "conflicts": [],
    }
    items = body["declaration_basis"]["declarations"]
    assert len(items) == 1 and items[0]["id"] == declaration["id"]
    assert items[0]["matched"] is True
    assert body["decision"] == {"allowed": False, "reason": "no_matching_policy"}


def test_non_matching_enabled_declaration_is_recorded_as_unmatched(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="docs/*")
    create_rule(client)

    event = record_event(client, machine_id, resource="res/x").json()
    assert event["reason"] == "no_enabled_declaration"
    items = get_basis(client, machine_id, event["id"]).json()[
        "declaration_basis"
    ]["declarations"]
    assert len(items) == 1
    assert items[0]["resource_pattern"] == "docs/*"
    assert items[0]["matched"] is False


# --------------------------------------------------------------------------- #
# Encoding: compact, single newline, no floats, byte-identical repeats
# --------------------------------------------------------------------------- #


def _assert_no_floats(value):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        raise AssertionError(f"floating-point value in snapshot: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_floats(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_floats(item)
    elif not isinstance(value, (int, str)):
        raise AssertionError(f"unexpected value type: {type(value)!r}")


def test_body_is_compact_json_with_single_newline_and_no_floats(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    response = get_basis(client, machine_id, event["id"])
    raw = response.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    # Compact separators: no whitespace around JSON punctuation in a payload
    # that carries no free-text spaces of its own.
    assert b": " not in raw and b", " not in raw
    assert b"-0.0" not in raw and b"NaN" not in raw and b"Infinity" not in raw
    _assert_no_floats(response.json())


def test_repeated_queries_are_byte_identical(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    first = get_basis(client, machine_id, event["id"]).content
    for _ in range(3):
        assert get_basis(client, machine_id, event["id"]).content == first


def test_snapshot_does_not_change_when_rules_change_later(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    first_event = record_event(client, machine_id, resource="res/x").json()
    first_bytes = get_basis(client, machine_id, first_event["id"]).content

    # A later deny changes subsequent decisions but never the stored basis.
    create_rule(client, resource_pattern="res/x", effect="deny", priority=0)
    second_event = record_event(client, machine_id, resource="res/x").json()
    assert second_event["reason"] == "denied_by_policy"

    assert get_basis(client, machine_id, first_event["id"]).content == first_bytes
    first_body = __import__("json").loads(first_bytes)
    assert first_body["decision"]["reason"] == "allowed_by_policy"


# --------------------------------------------------------------------------- #
# Validation, routing, and lookup outcomes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", ["?x=1", "?limit=1", "?=", "?foo", "?x=1&x=2"])
def test_any_query_parameter_is_invalid_query_before_lookup(client, query):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(basis_url(machine_id, event["id"]) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Validation precedes the machine lookup: an invalid query against a
    # missing machine and missing event is still 422, never 404.
    assert (
        client.get(basis_url(missing_machine, event["id"]) + query).status_code == 422
    )


def test_request_body_is_invalid_query_before_lookup(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    url = basis_url(machine_id, event["id"])

    response = client.request(
        "GET", url, content=b"{}", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    missing = "00000000-0000-0000-0000-000000000000"
    response = client.request(
        "GET",
        basis_url(missing, "anything"),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_zero_content_length_get_is_accepted(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = client.request("GET", basis_url(machine_id, event["id"]), content=b"")
    assert response.status_code == 200


@pytest.mark.parametrize("method", ["head", "post", "put", "patch", "delete"])
def test_only_get_is_routed(client, method):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    response = getattr(client, method)(basis_url(machine_id, event["id"]))
    assert response.status_code == 405
    assert b"event_summary" not in response.content


def test_non_get_methods_do_not_read_snapshot(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()
    # With every table the query could read dropped, only routing is in play.
    with client.app.state.engine.begin() as conn:
        for table in (
            "authorization_decision_basis",
            "authorization_decision_events",
        ):
            conn.execute(text(f"DROP TABLE {table}"))
    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)(basis_url(machine_id, event["id"]))
        assert response.status_code == 405


def test_missing_machine_event_or_ownership_returns_not_found(client):
    machine_id = create_machine(client)
    other = create_machine(client, external_id="machine-2")
    event = record_event(client, machine_id).json()
    missing = "00000000-0000-0000-0000-000000000000"

    assert client.get(basis_url(missing, event["id"])).status_code == 404
    assert client.get(basis_url(machine_id, "no-such-event")).status_code == 404
    # An existing event owned by another machine is indistinguishable from a
    # missing event on this path.
    response = client.get(basis_url(other, event["id"]))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_legacy_event_without_snapshot_is_snapshot_not_found(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    # Simulate an event committed before the feature existed: its row is
    # present, its immutable basis is not. The query must not fabricate one
    # from current data.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM authorization_decision_basis WHERE event_id = :id"
            ).bindparams(id=event["id"])
        )

    response = get_basis(client, machine_id, event["id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "snapshot_not_found"}}


def test_read_failure_returns_internal_error_without_partial_basis(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id).json()

    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_basis"))

    response = get_basis(client, machine_id, event["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"event_summary" not in response.content


# --------------------------------------------------------------------------- #
# Isolation, persistence, and atomicity
# --------------------------------------------------------------------------- #


def test_snapshots_are_strictly_isolated_per_machine(client):
    one = create_machine(client, external_id="machine-1")
    two = create_machine(client, external_id="machine-2")
    for machine_id in (one, two):
        declare(client, machine_id, resource_pattern="*")
    create_rule(client, resource_pattern="*", effect="allow", priority=0)

    event_one = record_event(client, one, resource="a").json()
    event_two = record_event(client, two, resource="b").json()

    body_one = get_basis(client, one, event_one["id"]).json()
    body_two = get_basis(client, two, event_two["id"]).json()
    assert body_one["event_summary"]["machine_id"] == one
    assert body_two["event_summary"]["machine_id"] == two
    assert body_one["event_summary"]["resource"] == "a"
    assert body_two["event_summary"]["resource"] == "b"
    # Guessing the other machine's event id on this path is a 404, never a
    # leaked snapshot.
    assert get_basis(client, two, event_one["id"]).status_code == 404
    assert get_basis(client, one, event_two["id"]).status_code == 404


def test_snapshots_persist_byte_identically_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        event = record_event(first, machine_id, resource="res/persist").json()
        first_bytes = get_basis(first, machine_id, event["id"]).content

    with TestClient(app) as second:
        response = get_basis(second, machine_id, event["id"])

    assert response.status_code == 200
    assert response.content == first_bytes


def test_old_database_startup_recreates_snapshot_table_safely(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        legacy_event = record_event(first, machine_id).json()

    # Remove only the basis table, reproducing a database that predates it.
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE authorization_decision_basis")
    conn.commit()
    conn.close()

    with TestClient(app) as second:
        # The old event still has no historical snapshot.
        response = get_basis(second, machine_id, legacy_event["id"])
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "snapshot_not_found"
        # New events on the migrated database capture and read normally.
        new_event = record_event(second, machine_id, resource="res/new").json()
        assert new_event["reason"] == "allowed_by_policy"
        assert get_basis(second, machine_id, new_event["id"]).status_code == 200


def test_snapshot_insert_failure_leaves_neither_event_nor_snapshot(
    client, monkeypatch
):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    def failing_insert(conn, row):
        raise RuntimeError("simulated snapshot write failure")

    monkeypatch.setattr(decision_basis, "insert", failing_insert)

    with pytest.raises(RuntimeError, match="simulated snapshot write failure"):
        record_event(client, machine_id)

    # The joint transaction rolled back: no event and no snapshot remain.
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert events == []
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM authorization_decision_basis")
        ).scalar_one()
    assert count == 0


def test_every_event_in_a_concurrent_burst_has_one_matching_basis(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client, resource_pattern="res/*", effect="allow", priority=0)
    create_rule(client, resource_pattern="res/deny", effect="deny", priority=0)
    count = 30

    gate = threading.Event()

    def hit(index):
        gate.wait()
        resource = "res/deny" if index % 4 == 0 else f"res/{index}"
        return record_event(client, machine_id, resource=resource)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit, i) for i in range(count)]
        gate.set()
        events = [f.result().json() for f in futures]

    assert len({e["id"] for e in events}) == count
    # Exactly one snapshot per event, and every basis repeats the committed
    # result of its own event.
    for event in events:
        response = get_basis(client, machine_id, event["id"])
        assert response.status_code == 200
        body = response.json()
        assert body["event_summary"]["id"] == event["id"]
        assert body["decision"]["allowed"] == event["allowed"]
        assert body["decision"]["reason"] == event["reason"]
        assert body["status_basis"]["status"] == "active"

    with client.app.state.engine.connect() as conn:
        basis_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM authorization_decision_basis "
                "WHERE machine_id = :id"
            ).bindparams(id=machine_id)
        ).scalar_one()
    assert basis_count == count


def test_burst_racing_suspension_records_each_basis_at_its_serial_status(client):
    machine_id = create_machine(client)
    declare(client, machine_id, resource_pattern="res/*")
    create_rule(client)
    count = 30

    gate = threading.Event()

    def event(index):
        gate.wait()
        return record_event(client, machine_id, resource=f"res/{index}")

    def suspend():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        event_futures = [pool.submit(event, i) for i in range(count)]
        suspend_future = pool.submit(suspend)
        gate.set()
        events = [f.result().json() for f in event_futures]
        assert suspend_future.result().status_code == 200

    # Each snapshot's status/reads and decision are mutually consistent with
    # the one definite serial order the burst committed in.
    statuses = []
    for event in events:
        body = get_basis(client, machine_id, event["id"]).json()
        status = body["status_basis"]["status"]
        statuses.append(status)
        if status == "suspended":
            assert event["reason"] == "machine_suspended"
            assert body["status_basis"]["declarations_read"] is False
            assert body["status_basis"]["policies_read"] is False
            assert body["declaration_basis"]["declarations"] == []
        else:
            assert event["reason"] == "allowed_by_policy"
            assert body["status_basis"]["declarations_read"] is True
        assert body["decision"]["reason"] == event["reason"]

    # At least the serialized tail was suspended.
    assert "suspended" in statuses


def test_get_query_never_writes(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    event = record_event(client, machine_id).json()

    def count_basis():
        with client.app.state.engine.connect() as conn:
            return conn.execute(
                text("SELECT COUNT(*) FROM authorization_decision_basis")
            ).scalar_one()

    before = count_basis()
    for _ in range(3):
        get_basis(client, machine_id, event["id"])
    # Also read the missing-snapshot and not-found paths.
    get_basis(client, machine_id, "missing-event")
    get_basis(client, create_machine(client, "machine-3"), event["id"])
    assert count_basis() == before
