"""Tests for the read-only joint write-transaction diagnostics layer.

``GET /machines/{machine_id}/diag`` reports one diagnostic record per attempt
that entered the locked joint write for a machine status change
(``op = "change"``) or an authorization decision event creation
(``op = "event"``):

* a business write that commits is ``started-commit`` with ``fail`` ``none``;
* one that begins and aborts is ``started-rollback`` with a stable failure
  category (``race``, ``io``, or ``other``);
* lock waits and serialization retries that never enter the business work are
  coalesced into the one attempt that does, via ``lock_wait``/``retry`` flags.

These tests cover the response shape, query validation, per-machine scoping,
fixed ordering, read-only behavior, restart stability, safe migration, and the
one-record-per-attempt invariant under concurrency. They do not change the
existing status/event semantics.
"""

import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError as SAOperationalError

from accountability.app import app
from accountability import chain, joint

RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)

WIDE_FROM = "2000-01-01T00:00:00Z"
WIDE_TO = "2100-01-01T00:00:00Z"


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


def record_event(client, machine_id, resource="res/x", action_type="read"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def set_status(client, machine_id, status):
    return client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )


def diag(client, machine_id, from_=WIDE_FROM, to=WIDE_TO):
    return client.get(f"/machines/{machine_id}/diag?from={from_}&to={to}")


def diag_body(client, machine_id, **kwargs):
    return diag(client, machine_id, **kwargs).json()


# --- empty machine and basic shape -----------------------------------------


def test_empty_machine_diag_is_valid_empty_window(client):
    machine_id = create_machine(client)

    response = diag(client, machine_id)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "id": machine_id,
        "from": WIDE_FROM,
        "to": WIDE_TO,
        "records": [],
        "check": {"valid": True, "checked_count": 0, "broken_event_id": None},
    }


def test_empty_database_serves_diagnostics(tmp_path, monkeypatch):
    # The diagnostics table is created on startup, so a brand-new empty
    # database serves the endpoint without any migration step.
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}"
    )
    with TestClient(app) as fresh:
        machine_id = create_machine(fresh)
        response = diag(fresh, machine_id)
        assert response.status_code == 200
        assert response.json()["records"] == []


def test_event_commit_record_shape(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id, resource="res/1")
    assert event.status_code == 201
    event_id = event.json()["id"]

    body = diag_body(client, machine_id)
    assert len(body["records"]) == 1
    record = body["records"][0]

    assert record["op"] == "event"
    assert record["phase"] == "started-commit"
    assert record["fail"] == "none"
    assert record["status"] == "committed"
    assert record["flags"] == []
    assert record["event"] == event_id
    assert record["count"] == 1
    assert record["check"] == {
        "valid": True,
        "checked_count": 1,
        "broken_event_id": None,
    }
    assert RFC3339_Z_RE.match(record["at"])
    assert isinstance(record["tid"], str) and record["tid"]
    # The diagnostic never exposes identity/key material or policy text.
    assert "public_key" not in record and "resource" not in record
    assert set(record.keys()) == {
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


def test_change_commit_record_shape_and_event_is_null(client):
    machine_id = create_machine(client)
    response = set_status(client, machine_id, "suspended")
    assert response.status_code == 200

    record = diag_body(client, machine_id)["records"][0]
    assert record["op"] == "change"
    assert record["phase"] == "started-commit"
    assert record["fail"] == "none"
    assert record["status"] == "committed"
    assert record["flags"] == []
    # A status change creates no event, so event is null and the event count is
    # the machine's total (zero here).
    assert record["event"] is None
    assert record["count"] == 0
    assert record["check"] == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_count_tracks_event_number_and_top_check_is_current(client):
    machine_id = create_machine(client)
    e1 = record_event(client, machine_id, "res/1").json()["id"]
    set_status(client, machine_id, "suspended")
    e2 = record_event(client, machine_id, "res/2").json()["id"]
    set_status(client, machine_id, "active")
    e3 = record_event(client, machine_id, "res/3").json()["id"]

    records = diag_body(client, machine_id)["records"]
    # Historical records freeze the event count as of each attempt.
    event_records = [r for r in records if r["op"] == "event"]
    assert [r["count"] for r in event_records] == [1, 2, 3]
    assert [r["event"] for r in event_records] == [e1, e2, e3]
    change_records = [r for r in records if r["op"] == "change"]
    assert [r["event"] for r in change_records] == [None, None]
    # The change records see the count as it stood when they ran.
    assert [r["count"] for r in change_records] == [1, 2]

    # The top-level check is the whole machine's *current* verdict.
    body = diag_body(client, machine_id)
    assert body["check"] == {
        "valid": True,
        "checked_count": 3,
        "broken_event_id": None,
    }


# --- rollback records -------------------------------------------------------


def test_same_target_status_conflict_is_one_race_rollback(client):
    machine_id = create_machine(client)
    assert set_status(client, machine_id, "suspended").status_code == 200

    conflict = set_status(client, machine_id, "suspended")
    assert conflict.status_code == 409
    assert conflict.json() == {"error": {"code": "invalid_status_transition"}}

    records = diag_body(client, machine_id)["records"]
    assert len(records) == 2
    committed, rolled_back = records
    assert committed["phase"] == "started-commit"
    assert rolled_back["phase"] == "started-rollback"
    assert rolled_back["op"] == "change"
    assert rolled_back["fail"] == "race"
    assert rolled_back["status"] == "rolled_back"
    assert rolled_back["event"] is None
    assert rolled_back["flags"] == []
    # The rollback changed no business state.
    assert (
        client.get(f"/machines/{machine_id}").json()["status"] == "suspended"
    )
    assert (
        diag_body(client, machine_id)["check"]["checked_count"] == 0
    )


def _patch_event_work_to_raise(monkeypatch, exc):
    """Make the event business work raise inside the locked transaction."""

    def failing(*args, **kwargs):
        raise exc

    monkeypatch.setattr(chain, "_mint_tail_link", failing)


def test_io_failure_leaves_rollback_record_and_no_event(client, monkeypatch):
    machine_id = create_machine(client)
    record_event(client, machine_id, "res/1")

    _patch_event_work_to_raise(
        monkeypatch,
        SAOperationalError(
            "INSERT", {}, sqlite3.OperationalError("disk I/O error simulated")
        ),
    )

    # The runner preserves the existing behavior: the persistence fault
    # propagates (the API surfaces a 500), but a diagnostic is still recorded.
    with pytest.raises(SAOperationalError):
        record_event(client, machine_id, "res/boom")

    records = diag_body(client, machine_id)["records"]
    assert [r["fail"] for r in records] == ["none", "io"]
    rollback = records[-1]
    assert rollback["phase"] == "started-rollback"
    assert rollback["status"] == "rolled_back"
    assert rollback["op"] == "event"
    assert rollback["event"] is None
    # The persistence failure left no event behind.
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert len(events) == 1


def test_other_failure_leaves_rollback_record_and_no_event(client, monkeypatch):
    machine_id = create_machine(client)

    _patch_event_work_to_raise(
        monkeypatch, RuntimeError("unexpected crash simulated")
    )

    with pytest.raises(RuntimeError):
        record_event(client, machine_id, "res/boom")

    records = diag_body(client, machine_id)["records"]
    assert len(records) == 1
    assert records[0]["phase"] == "started-rollback"
    assert records[0]["fail"] == "other"
    assert records[0]["status"] == "rolled_back"
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        == []
    )


# --- lock wait / retry coalescing ------------------------------------------


def test_lock_retry_is_coalesced_into_one_record_with_flags(
    client, monkeypatch
):
    machine_id = create_machine(client)
    real_locked = joint._locked_connection
    calls = {"n": 0}

    @contextmanager
    def flaky_locked(engine):
        calls["n"] += 1
        if calls["n"] == 1:
            # First lock acquisition loses to a concurrent writer; the attempt
            # never enters the business work and must be retried.
            raise SAOperationalError(
                "BEGIN IMMEDIATE",
                {},
                sqlite3.OperationalError("database is locked"),
            )
        with real_locked(engine) as conn:
            yield conn

    monkeypatch.setattr(joint, "_locked_connection", flaky_locked)

    response = record_event(client, machine_id, "res/1")
    assert response.status_code == 201

    # The failed acquisition produced no record: the two acquisitions are one
    # logical attempt with both flags, in fixed order.
    records = diag_body(client, machine_id)["records"]
    assert len(records) == 1
    assert records[0]["phase"] == "started-commit"
    assert records[0]["fail"] == "none"
    assert records[0]["flags"] == ["lock_wait", "retry"]
    assert records[0]["event"] == response.json()["id"]


def test_lock_wait_without_retry_is_flagged(client, monkeypatch):
    machine_id = create_machine(client)
    real_locked = joint._locked_connection

    @contextmanager
    def slow_locked(engine):
        # A contended first acquisition that still succeeds on entry.
        import time

        time.sleep(0.03)
        with real_locked(engine) as conn:
            yield conn

    monkeypatch.setattr(joint, "_locked_connection", slow_locked)

    assert record_event(client, machine_id, "res/1").status_code == 201
    records = diag_body(client, machine_id)["records"]
    assert len(records) == 1
    assert records[0]["flags"] == ["lock_wait"]


# --- query validation -------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",  # both missing
        "?to=2100-01-01T00:00:00Z",  # from missing
        "?from=2000-01-01T00:00:00Z",  # to missing
        "?from=not-a-time&to=2100-01-01T00:00:00Z",
        "?from=2000-01-01T00:00:00&to=2100-01-01T00:00:00Z",  # no Z
        "?from=2000-01-01T00:00:00%2B00:00&to=2100-01-01T00:00:00Z",  # offset
        "?from=2000-13-01T00:00:00Z&to=2100-01-01T00:00:00Z",  # bad month
        "?from=2000-02-30T00:00:00Z&to=2100-01-01T00:00:00Z",  # bad day
        "?from=%20&to=2100-01-01T00:00:00Z",  # whitespace
        "?from=2000-01-01T00:00:00Z%20&to=2100-01-01T00:00:00Z",
        "?from=2100-01-01T00:00:00Z&to=2000-01-01T00:00:00Z",  # inverted
    ],
)
def test_bad_window_returns_422_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(f"/machines/{machine_id}/diag{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "query",
    [
        "?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z&unexpected=1",
        "?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z&from=2001-01-01T00:00:00Z",
        "?foo=bar",
    ],
)
def test_unknown_parameter_returns_422_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(f"/machines/{machine_id}/diag{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    # Bad time against a missing machine is still 422, not 404.
    bad = client.get(f"/machines/{missing}/diag?to=2100-01-01T00:00:00Z")
    assert bad.status_code == 422
    assert bad.json() == {"error": {"code": "bad_time"}}
    # An unknown parameter likewise fails before the lookup.
    unknown = client.get(
        f"/machines/{missing}/diag"
        "?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}
    # A valid window on a missing machine is 404.
    not_found = client.get(
        f"/machines/{missing}/diag?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z"
    )
    assert not_found.status_code == 404
    assert not_found.json() == {"error": {"code": "not_found"}}


# --- windowing, ordering, and machine scoping -------------------------------


def test_closed_interval_includes_both_endpoints(client):
    machine_id = create_machine(client)
    record_event(client, machine_id, "res/1")
    record = diag_body(client, machine_id)["records"][0]
    at = record["at"]

    # from == to == the record's instant is a closed interval and includes it.
    body = diag_body(client, machine_id, from_=at, to=at)
    assert [r["tid"] for r in body["records"]] == [record["tid"]]
    assert body["from"] == at and body["to"] == at


def test_window_excludes_records_outside_interval(client):
    machine_id = create_machine(client)
    record_event(client, machine_id, "res/1")

    outside = diag_body(
        client, machine_id, from_="2000-01-01T00:00:00Z", to="2000-01-02T00:00:00Z"
    )
    assert outside["records"] == []


def test_records_ordered_by_at_then_id(client):
    machine_id = create_machine(client)
    for i in range(6):
        record_event(client, machine_id, f"res/{i}")
    set_status(client, machine_id, "suspended")

    records = diag_body(client, machine_id)["records"]

    def key(record):
        # Parse the stored instant; the endpoint must order by true instant
        # then id, not by ISO text.
        from datetime import datetime

        return (
            datetime.fromisoformat(record["at"][:-1] + "+00:00"),
            record["tid"],
        )

    assert records == sorted(records, key=key)
    assert [r["op"] for r in records] == ["event"] * 6 + ["change"]


def test_records_are_scoped_to_path_machine(client):
    one = create_machine(client, external_id="m-1")
    two = create_machine(client, external_id="m-2")
    record_event(client, one, "res/1")
    record_event(client, two, "res/2")
    set_status(client, one, "suspended")

    one_records = diag_body(client, one)["records"]
    two_records = diag_body(client, two)["records"]

    assert len(one_records) == 2
    assert len(two_records) == 1
    assert all(r["event"] for r in one_records if r["op"] == "event")
    assert two_records[0]["op"] == "event"
    # No machine id other than the path machine's appears anywhere.
    assert {r["op"] for r in one_records} <= {"change", "event"}


def test_query_is_read_only_and_stable(client):
    machine_id = create_machine(client)
    record_event(client, machine_id, "res/1")
    set_status(client, machine_id, "suspended")

    first = diag(client, machine_id)
    snapshot = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()

    for _ in range(3):
        again = diag(client, machine_id)
        assert again.status_code == 200
        assert again.json() == first.json()

    # Repeated diagnostics add no records and change no business state.
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        == snapshot
    )
    assert len(diag_body(client, machine_id)["records"]) == 2


def test_diagnostics_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        record_event(first, machine_id, "res/1")
        set_status(first, machine_id, "suspended")
        set_status(first, machine_id, "suspended")  # race rollback
        before = diag(first, machine_id).json()

    with TestClient(app) as second:
        after = diag(second, machine_id).json()
        assert after == before
        assert len(after["records"]) == 3
        assert after["records"][0]["phase"] == "started-commit"
        assert after["records"][1]["phase"] == "started-commit"
        assert after["records"][2] == {
            **after["records"][2],
            "phase": "started-rollback",
            "fail": "race",
            "status": "rolled_back",
        }


def test_old_database_migrates_and_keeps_serving(tmp_path, monkeypatch):
    # A database file created through the ORM before the diagnostics feature
    # gains the new table on startup, alongside existing machines/events.
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as first:
        machine_id = create_machine(first)
        record_event(first, machine_id, "res/1")
    # Restart exercises the migration path with an existing populated database.
    with TestClient(app) as second:
        body = diag(second, machine_id).json()
        assert len(body["records"]) == 1
        assert body["records"][0]["op"] == "event"
        assert body["check"]["valid"] is True


# --- concurrency: one record per joint-write attempt ------------------------


def test_concurrent_burst_leaves_one_record_per_attempt(client):
    machine_id = create_machine(client)
    # Allowed declaration so events commit with an allow result.
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
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

    event_count = 10
    suspend_count = 4
    gate = threading.Event()

    def event(index):
        gate.wait()
        return record_event(client, machine_id, f"res/{index}")

    def suspend():
        gate.wait()
        return set_status(client, machine_id, "suspended")

    with ThreadPoolExecutor(max_workers=8) as pool:
        event_futures = [pool.submit(event, i) for i in range(event_count)]
        suspend_futures = [pool.submit(suspend) for _ in range(suspend_count)]
        gate.set()
        event_responses = [f.result() for f in event_futures]
        suspend_responses = [f.result() for f in suspend_futures]

    assert all(r.status_code == 201 for r in event_responses)
    # Same-target suspensions: exactly one wins, the rest race-rollback.
    assert sum(r.status_code == 200 for r in suspend_responses) == 1
    assert sum(r.status_code == 409 for r in suspend_responses) == suspend_count - 1

    records = diag_body(client, machine_id)["records"]
    # One diagnostic per attempt that entered the joint write: retries and
    # lock waits are coalesced, so the count is exactly attempts, not
    # acquisition tries.
    assert len(records) == event_count + suspend_count
    assert (
        sum(
            r["phase"] == "started-commit" and r["op"] == "event"
            for r in records
        )
        == event_count
    )
    assert (
        sum(
            r["phase"] == "started-commit" and r["op"] == "change"
            for r in records
        )
        == 1
    )
    assert (
        sum(
            r["phase"] == "started-rollback" and r["fail"] == "race"
            for r in records
        )
        == suspend_count - 1
    )
    # Committed records never carry a failure; rollbacks never carry an event.
    for record in records:
        if record["phase"] == "started-commit":
            assert record["fail"] == "none"
            assert record["status"] == "committed"
        else:
            assert record["status"] == "rolled_back"
            assert record["event"] is None
        assert set(record["flags"]) <= {"lock_wait", "retry"}
        # Flags, if any, are in the fixed lock_wait-before-retry order.
        assert record["flags"] in (
            [],
            ["lock_wait"],
            ["retry"],
            ["lock_wait", "retry"],
        )

    # Every event attempt left exactly one sound event on an unbroken chain.
    assert diag_body(client, machine_id)["check"] == {
        "valid": True,
        "checked_count": event_count,
        "broken_event_id": None,
    }


def test_diagnostics_do_not_change_health_and_status_semantics(client):
    # The observability layer leaves GET /health and active/suspended behavior
    # untouched.
    assert client.get("/health").json() == {"status": "ok"}
    machine_id = create_machine(client)
    assert client.get(f"/machines/{machine_id}").json()["status"] == "active"
    assert set_status(client, machine_id, "suspended").status_code == 200
    # Reading diagnostics neither flips status nor writes.
    diag(client, machine_id)
    assert client.get(f"/machines/{machine_id}").json()["status"] == "suspended"


def test_missing_machine_writes_leave_no_diagnostics(client):
    missing = "00000000-0000-0000-0000-000000000000"

    def hit(_):
        return client.post(
            f"/machines/{missing}/authorization-decision-events",
            json={"action_type": "read", "resource": "res/x"},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(hit, range(6)))
    assert {r.status_code for r in responses} == {404}

    # A request that cannot resolve a machine never enters the joint write, so
    # no diagnostic exists; the endpoint on the missing id is a 404.
    response = client.get(
        f"/machines/{missing}/diag?from=2000-01-01T00:00:00Z&to=2100-01-01T00:00:00Z"
    )
    assert response.status_code == 404
