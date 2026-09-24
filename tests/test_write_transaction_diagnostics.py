"""Tests for the read-only joint-write transaction diagnostics.

Covers `GET /machines/{machine_id}/diag`: one diagnostic per status-change
or decision-event attempt, terminal commit/rollback records with stable
failure categories and ordered lock-wait/retry flags, closed-UTC-window
filtering in `(at, tid)` order, the `bad_time` / `invalid_query` /
`not_found` outcomes, strict read-only stability, per-machine isolation, the
per-record and top-level event-chain checks, and crash-residual recovery by
evidence across a restart.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from accountability.app import app
from accountability import chain


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


WIDE = ("2000-01-01T00:00:00Z", "2099-01-01T00:00:00Z")


def diag_url(machine_id, from_=WIDE[0], to=WIDE[1]):
    return f"/machines/{machine_id}/diag?from={from_}&to={to}"


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


def allow_read(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )


def record_event(client, machine_id, resource="res/1"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )


def records_for(client, machine_id):
    return client.get(diag_url(machine_id)).json()["records"]


RECORD_KEYS = {
    "tid",
    "at",
    "phase",
    "op",
    "fail",
    "flags",
    "status",
    "event",
    "count",
    "check",
}
CHECK_KEYS = {"valid", "checked_count", "broken_event_id"}


def test_success_envelope_shape_and_machine_404(client):
    machine_id = create_machine(client)
    response = client.get(diag_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"id", "from", "to", "records", "check"}
    assert body["id"] == machine_id
    assert body["from"] == WIDE[0]
    assert body["to"] == WIDE[1]
    assert body["records"] == []
    assert body["check"] == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }

    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(diag_url(missing)).status_code == 404
    assert client.get(diag_url(missing)).json() == {"error": {"code": "not_found"}}


def test_committed_event_attempt_record(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event_response = record_event(client, machine_id)
    assert event_response.status_code == 201
    event = event_response.json()

    records = records_for(client, machine_id)
    assert len(records) == 1
    record = records[0]
    assert set(record.keys()) == RECORD_KEYS
    assert record["op"] == "event"
    assert record["phase"] == "started-commit"
    assert record["fail"] == "none"
    assert record["flags"] == []
    assert record["status"] == "active"
    assert record["event"] == event["id"]
    assert record["count"] == 1
    assert record["at"].endswith("Z")
    assert set(record["check"].keys()) == CHECK_KEYS
    assert record["check"] == {
        "valid": True,
        "checked_count": 1,
        "broken_event_id": None,
    }


def test_committed_change_attempt_record(client):
    machine_id = create_machine(client)
    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )

    records = records_for(client, machine_id)
    assert len(records) == 1
    record = records[0]
    assert record["op"] == "change"
    assert record["phase"] == "started-commit"
    assert record["fail"] == "none"
    assert record["flags"] == []
    assert record["status"] == "suspended"
    # A status change creates no event: event is null and count stays 0.
    assert record["event"] is None
    assert record["count"] == 0
    assert record["check"] == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_same_target_status_loser_is_rollback_race(client):
    machine_id = create_machine(client)
    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )
    loser = client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})
    assert loser.status_code == 409
    assert loser.json() == {"error": {"code": "invalid_status_transition"}}

    records = records_for(client, machine_id)
    assert len(records) == 2
    winner, loser_record = records
    assert winner["phase"] == "started-commit"
    assert winner["fail"] == "none"
    assert loser_record["phase"] == "started-rollback"
    assert loser_record["fail"] == "race"
    assert loser_record["op"] == "change"
    assert loser_record["event"] is None
    assert loser_record["status"] == "suspended"


def test_missing_machine_event_and_change_attempts_are_rollback_other(client):
    missing = "00000000-0000-0000-0000-000000000000"
    event_response = client.post(
        f"/machines/{missing}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/x"},
    )
    assert event_response.status_code == 404
    change_response = client.post(
        f"/machines/{missing}/status", json={"status": "suspended"}
    )
    assert change_response.status_code == 404

    # The machine does not exist, so neither record is visible under it; the
    # rows still exist, owned by the attempted (missing) machine id only.
    assert client.get(diag_url(missing)).status_code == 404
    with client.app.state.engine.connect() as conn:
        rows = list(
            conn.execute(
                text(
                    "SELECT op, phase, fail, status, event, count "
                    "FROM write_transaction_diagnostics ORDER BY at, id"
                )
            )
        )
    assert len(rows) == 2
    for row in rows:
        assert row.phase == "started-rollback"
        assert row.fail == "other"
        assert row.status == ""
        assert row.event is None
        assert row.count == 0


def test_records_ordered_by_at_then_tid_and_window_is_closed(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    record_event(client, machine_id, "res/1")
    client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})
    record_event(client, machine_id, "res/2")

    all_records = records_for(client, machine_id)
    assert [r["op"] for r in all_records] == ["event", "change", "event"]

    def at_of(record):
        return record["at"]

    # Ordered by the actual UTC instant of at, then tid.
    keys = [(r["at"], r["tid"]) for r in all_records]
    assert keys == sorted(keys)

    first_at, last_at = at_of(all_records[0]), at_of(all_records[-1])
    # Closed interval: equal bounds include the boundary record.
    first_only = client.get(diag_url(machine_id, first_at, first_at)).json()["records"]
    assert [r["tid"] for r in first_only] == [all_records[0]["tid"]]
    # A window ending just before the last at excludes it.
    head = client.get(
        diag_url(machine_id, WIDE[0], first_at)
    ).json()["records"]
    assert len(head) == 1
    assert head[0]["tid"] == all_records[0]["tid"]
    assert last_at > first_at


def test_records_are_isolated_per_machine(client):
    one = create_machine(client, "m-1")
    two = create_machine(client, "m-2")
    allow_read(client, one)
    allow_read(client, two)
    record_event(client, one)
    record_event(client, two)
    record_event(client, two)

    one_records = records_for(client, one)
    two_records = records_for(client, two)
    assert len(one_records) == 1
    assert len(two_records) == 2
    assert all(r["op"] == "event" for r in one_records + two_records)


def test_query_is_read_only_and_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    record_event(client, machine_id)

    def table_rows():
        with client.app.state.engine.connect() as conn:
            return list(
                conn.execute(
                    text("SELECT id, at, phase, fail, flags FROM "
                         "write_transaction_diagnostics ORDER BY at, id")
                )
            )

    before = table_rows()
    first = client.get(diag_url(machine_id)).json()
    middle = table_rows()
    second = client.get(diag_url(machine_id)).json()
    after = table_rows()

    assert first == second
    assert before == middle == after


def test_top_level_check_matches_integrity_endpoint(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    for i in range(3):
        record_event(client, machine_id, f"res/{i}")

    diag = client.get(diag_url(machine_id)).json()
    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert diag["check"] == integrity
    assert diag["check"]["checked_count"] == 3


# --- query validation -------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from=2020-01-01T00:00:00Z",
        "?to=2099-01-01T00:00:00Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(f"/machines/{machine_id}/diag{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2020-01-01T00:00:00",            # missing Z
        "2020-01-01T00:00:00+00:00",      # offset form
        "2020-01-01T00:00:00z",           # lowercase suffix
        " 2020-01-01T00:00:00Z",          # leading whitespace
        "2020-01-01T00:00:00Z ",          # trailing whitespace
        "2020-01-01 00:00:00Z",           # space separator
        "garbage",
        "2020-13-01T00:00:00Z",           # bad month
        "2020-02-30T00:00:00Z",           # bad day
        "2020-01-01T24:00:00Z",           # bad hour
        "",                               # blank
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/diag?from={value}&to=2099-01-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_and_equal_bounds(client):
    machine_id = create_machine(client)
    inverted = client.get(
        f"/machines/{machine_id}/diag"
        "?from=2099-01-01T00:00:00Z&to=2020-01-01T00:00:00Z"
    )
    assert inverted.status_code == 422
    assert inverted.json() == {"error": {"code": "bad_time"}}

    equal = client.get(
        f"/machines/{machine_id}/diag"
        "?from=2030-01-01T00:00:00Z&to=2030-01-01T00:00:00Z"
    )
    assert equal.status_code == 200


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/diag"
        "?from=2020-01-01T00:00:00Z&to=2099-01-01T00:00:00Z&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_precedence_over_missing_machine(client):
    missing = "00000000-0000-0000-0000-000000000000"
    bad_time = client.get(f"/machines/{missing}/diag?from=nope&to=2099-01-01T00:00:00Z")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}
    unknown = client.get(
        f"/machines/{missing}/diag"
        "?from=2020-01-01T00:00:00Z&to=2099-01-01T00:00:00Z&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


# --- lock-wait / retry flags -------------------------------------------------


def test_lock_wait_flag_recorded_when_lock_contended(client, monkeypatch):
    machine_id = create_machine(client)
    allow_read(client, machine_id)

    hold = threading.Event()
    entered = threading.Event()
    from accountability import authorization

    real_decide = authorization.decide

    def stall(executor, mid, status, action_type, resource):
        if status == "active":
            entered.set()
            assert hold.wait(timeout=10)
        return real_decide(executor, mid, status, action_type, resource)

    monkeypatch.setattr(authorization, "decide", stall)

    def first_event():
        return record_event(client, machine_id, "res/1")

    def second_event():
        assert entered.wait(timeout=10)
        return record_event(client, machine_id, "res/2")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first_event)
        second_future = pool.submit(second_event)
        assert entered.wait(timeout=10)
        # Let the second attempt queue on the write lock for a visible wait,
        # then release the holder.
        threading.Event().wait(0.15)
        hold.set()
        assert first_future.result().status_code == 201
        assert second_future.result().status_code == 201

    records = records_for(client, machine_id)
    assert len(records) == 2
    waiter = next(r for r in records if "lock_wait" in r["flags"])
    assert waiter["phase"] == "started-commit"
    assert waiter["fail"] == "none"
    # lock_wait always precedes retry when both are present.
    for record in records:
        assert record["flags"] in ([], ["lock_wait"], ["retry"], ["lock_wait", "retry"])
        if record["flags"] == ["lock_wait", "retry"]:
            assert record["flags"].index("lock_wait") < record["flags"].index("retry")


def test_retry_flag_recorded_and_merged_into_one_record(client, monkeypatch):
    machine_id = create_machine(client)
    allow_read(client, machine_id)

    real_locked = chain._locked_connection
    calls = {"n": 0}

    def flaky_locked(engine, on_lock_acquired=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OperationalError(
                "BEGIN IMMEDIATE", {}, Exception("database is locked")
            )
        return real_locked(engine, on_lock_acquired=on_lock_acquired)

    monkeypatch.setattr(chain, "_locked_connection", flaky_locked)

    response = record_event(client, machine_id)
    assert response.status_code == 201

    records = records_for(client, machine_id)
    assert len(records) == 1  # the retry never produces a second diagnostic
    record = records[0]
    assert record["flags"] == ["retry"]
    assert record["phase"] == "started-commit"
    assert record["fail"] == "none"
    assert record["event"] == response.json()["id"]
    assert record["count"] == 1


def test_io_failure_is_rollback_io(client, monkeypatch):
    machine_id = create_machine(client)
    allow_read(client, machine_id)

    def broken_locked(engine, on_lock_acquired=None):
        raise OperationalError("COMMIT", {}, Exception("disk I/O error"))

    monkeypatch.setattr(chain, "_locked_connection", broken_locked)

    with pytest.raises(OperationalError):
        chain.append_decision_event(
            client.app.state.engine,
            machine_id=machine_id,
            action_type="read",
            resource="res/x",
        )

    records = records_for(client, machine_id)
    assert len(records) == 1
    assert records[0]["phase"] == "started-rollback"
    assert records[0]["fail"] == "io"
    assert records[0]["op"] == "event"
    assert records[0]["event"] is None
    # The failed attempt wrote no business event.
    assert records[0]["count"] == 0


def test_unexpected_failure_is_rollback_other(client, monkeypatch):
    machine_id = create_machine(client)
    allow_read(client, machine_id)

    def boom(engine, on_lock_acquired=None):
        raise RuntimeError("process crashed")

    monkeypatch.setattr(chain, "_locked_connection", boom)

    with pytest.raises(RuntimeError):
        chain.append_decision_event(
            client.app.state.engine,
            machine_id=machine_id,
            action_type="read",
            resource="res/x",
        )

    records = records_for(client, machine_id)
    assert len(records) == 1
    assert records[0]["phase"] == "started-rollback"
    assert records[0]["fail"] == "other"


# --- same-target concurrency: exactly one commit ----------------------------


def test_same_target_concurrency_has_one_commit_and_complete_diagnostics(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    gate = threading.Event()

    def suspend():
        gate.wait()
        return client.post(
            f"/machines/{machine_id}/status", json={"status": "suspended"}
        )

    def event():
        gate.wait()
        return record_event(client, machine_id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        suspend_futures = [pool.submit(suspend) for _ in range(4)]
        event_future = pool.submit(event)
        gate.set()
        status_responses = [f.result() for f in suspend_futures]
        event_response = event_future.result()

    assert sum(r.status_code == 200 for r in status_responses) == 1
    assert event_response.status_code == 201

    records = records_for(client, machine_id)
    assert len(records) == 5
    assert sum(r["phase"] == "started-commit" and r["op"] == "change" for r in records) == 1
    assert sum(r["phase"] == "started-rollback" and r["fail"] == "race" for r in records) == 3
    event_records = [r for r in records if r["op"] == "event"]
    assert len(event_records) == 1
    assert event_records[0]["phase"] == "started-commit"
    assert event_records[0]["event"] == event_response.json()["id"]
    # Every attempt is exactly one well-formed record, no partials.
    for record in records:
        assert record["phase"] in ("started-commit", "started-rollback")
        assert set(record.keys()) == RECORD_KEYS
        if record["phase"] == "started-commit":
            assert record["fail"] == "none"


# --- crash recovery across a restart -----------------------------------------


def _insert_started_marker(client, *, tid, machine_id, op, started_at, status="active"):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO write_transaction_diagnostics "
                "(id, machine_id, op, at, phase, fail, flags, status, event, "
                " count, check_valid, check_checked_count, check_broken_event_id, "
                " started_at) "
                "VALUES (:id, :machine_id, :op, :started_at, 'started', 'none', "
                " '[]', :status, NULL, 0, 1, 0, NULL, :started_at)"
            ),
            {
                "id": tid,
                "machine_id": machine_id,
                "op": op,
                "started_at": started_at,
                "status": status,
            },
        )


def test_crash_residual_event_committed_is_recovered_from_evidence(client):
    """Real crash window: the joint transaction committed the event but the
    process died before finalizing its own marker, so no finalized diagnostic
    references the event. Recovery classifies the residual as a commit from
    the event evidence.
    """
    machine_id = create_machine(client)
    event_id = "bbbbbbbb-0000-0000-0000-000000000002"
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, created_at) "
                "VALUES (:id, :mid, 'read', 'res/x', 1, 'allowed_by_policy', :at)"
            ),
            {"id": event_id, "mid": machine_id, "at": "2026-03-01T00:00:00Z"},
        )
    chain.backfill_chains(client.app.state.engine)

    # The attempt marker started before the committed event was durable.
    _insert_started_marker(
        client,
        tid="11111111-1111-1111-1111-111111111111",
        machine_id=machine_id,
        op="event",
        started_at="2000-01-01T00:00:00Z",
    )

    with TestClient(app) as restarted:
        body = restarted.get(diag_url(machine_id)).json()

    assert len(body["records"]) == 1
    recovered = body["records"][0]
    assert recovered["tid"] == "11111111-1111-1111-1111-111111111111"
    assert recovered["phase"] == "started-commit"
    assert recovered["fail"] == "none"
    assert recovered["op"] == "event"
    assert recovered["event"] == event_id
    assert recovered["count"] == 1
    assert recovered["at"].endswith("Z")
    assert recovered["check"]["valid"] is True
    assert body["check"]["checked_count"] == 1


def test_crash_residual_without_effect_is_recovered_as_rollback(
    client, tmp_path
):
    machine_id = create_machine(client)
    # No events; a future-dated event marker can have no effect after it.
    _insert_started_marker(
        client,
        tid="22222222-2222-2222-2222-222222222222",
        machine_id=machine_id,
        op="event",
        started_at="2099-01-01T00:00:00Z",
    )
    # A change marker that started after the machine's last update cannot
    # have committed the status update.
    _insert_started_marker(
        client,
        tid="33333333-3333-3333-3333-333333333333",
        machine_id=machine_id,
        op="change",
        started_at="2099-01-02T00:00:00Z",
    )

    with TestClient(app) as restarted:
        body = restarted.get(diag_url(machine_id)).json()

    by_tid = {r["tid"]: r for r in body["records"]}
    for tid in (
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ):
        assert by_tid[tid]["phase"] == "started-rollback"
        assert by_tid[tid]["fail"] == "other"
        assert by_tid[tid]["event"] is None
        assert by_tid[tid]["flags"] == []


def test_crash_residual_committed_change_is_recovered(client):
    machine_id = create_machine(client)
    assert (
        client.post(f"/machines/{machine_id}/status", json={"status": "suspended"}).status_code
        == 200
    )
    # Marker started before the committed update: updated_at is evidence the
    # change landed.
    _insert_started_marker(
        client,
        tid="44444444-4444-4444-4444-444444444444",
        machine_id=machine_id,
        op="change",
        started_at="2000-01-01T00:00:00Z",
        status="suspended",
    )

    with TestClient(app) as restarted:
        body = restarted.get(diag_url(machine_id)).json()

    recovered = next(
        r for r in body["records"] if r["tid"] == "44444444-4444-4444-4444-444444444444"
    )
    assert recovered["phase"] == "started-commit"
    assert recovered["fail"] == "none"
    assert recovered["op"] == "change"
    assert recovered["status"] == "suspended"
    assert recovered["event"] is None


def test_two_residual_event_markers_share_one_committed_event(client):
    """Only one joint event attempt can produce a given committed event.

    Two crash residuals (no finalized commit diagnostic exists) with one
    committed event written before the crash: the earliest-starting marker
    claims the event (commit) and the other is recovered as a crash rollback,
    so the event id is attributed exactly once.
    """
    machine_id = create_machine(client)

    # Write the committed event directly, as the crashed process left it,
    # with no finalized diagnostic referencing it; backfill makes the chain
    # sound exactly as startup would.
    event_id = "aaaaaaaa-0000-0000-0000-000000000001"
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, created_at) "
                "VALUES (:id, :mid, 'read', 'res/x', 1, 'allowed_by_policy', :at)"
            ),
            {
                "id": event_id,
                "mid": machine_id,
                "at": "2026-03-01T00:00:00Z",
            },
        )
    chain.backfill_chains(client.app.state.engine)

    _insert_started_marker(
        client,
        tid="66666666-6666-6666-6666-666666666666",
        machine_id=machine_id,
        op="event",
        started_at="2000-01-01T00:00:00Z",
    )
    _insert_started_marker(
        client,
        tid="77777777-7777-7777-7777-777777777777",
        machine_id=machine_id,
        op="event",
        started_at="2000-01-02T00:00:00Z",
    )

    with TestClient(app) as restarted:
        body = restarted.get(diag_url(machine_id)).json()

    by_tid = {r["tid"]: r for r in body["records"]}
    claimant = by_tid["66666666-6666-6666-6666-666666666666"]
    loser = by_tid["77777777-7777-7777-7777-777777777777"]
    assert claimant["phase"] == "started-commit"
    assert claimant["fail"] == "none"
    assert claimant["event"] == event_id
    assert claimant["count"] == 1
    assert loser["phase"] == "started-rollback"
    assert loser["fail"] == "other"
    assert loser["event"] is None
    # The one committed event is attributed exactly once.
    assert sum(1 for r in body["records"] if r["event"] == event_id) == 1


def test_residual_cannot_reclaim_event_already_attributed(client):
    """A crash residual never re-attributes an event a finalized live commit
    diagnostic already accounts for: the residual is a crash rollback and the
    existing attribution stays unique.
    """
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    event_response = record_event(client, machine_id)
    live_event_id = event_response.json()["id"]

    _insert_started_marker(
        client,
        tid="88888888-8888-8888-8888-888888888888",
        machine_id=machine_id,
        op="event",
        started_at="2000-01-01T00:00:00Z",
    )

    with TestClient(app) as restarted:
        body = restarted.get(diag_url(machine_id)).json()

    by_tid = {r["tid"]: r for r in body["records"]}
    residual = by_tid["88888888-8888-8888-8888-888888888888"]
    assert residual["phase"] == "started-rollback"
    assert residual["fail"] == "other"
    assert residual["event"] is None
    assert sum(1 for r in body["records"] if r["event"] == live_event_id) == 1


def test_recovery_is_idempotent_across_repeated_restarts(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    record_event(client, machine_id)
    _insert_started_marker(
        client,
        tid="55555555-5555-5555-5555-555555555555",
        machine_id=machine_id,
        op="event",
        started_at="2000-01-01T00:00:00Z",
    )

    with TestClient(app) as first:
        first_body = first.get(diag_url(machine_id)).json()
    with TestClient(app) as second:
        second_body = second.get(diag_url(machine_id)).json()

    assert first_body == second_body
    assert all(r["phase"] != "started" for r in second_body["records"])


def test_empty_database_serves_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}"
    )
    with TestClient(app) as fresh:
        machine_id = create_machine(fresh)
        body = fresh.get(diag_url(machine_id)).json()
        assert body["records"] == []
        assert body["check"] == {
            "valid": True,
            "checked_count": 0,
            "broken_event_id": None,
        }
        assert fresh.get("/health").json() == {"status": "ok"}
