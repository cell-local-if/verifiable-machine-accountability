"""Tests for machine-level batch privacy access registration.

Covers `POST /machines/{machine_id}/privacy-accesses/batch`:

- success: 200 with a position-aligned ``results`` array; an empty batch is
  legal and returns ``[]``; ``success`` items carry the full new record while
  ``duplicate_access`` items echo only the five submitted fields;
- duplicate identity (access time, window, result — never the hit count) is
  detected against already-stored rows and against earlier items in the same
  batch, with the first occurrence winning and duplicates never aborting the
  other items;
- the whole batch commits in one locked write transaction serialized against
  single registrations, and a persistence failure rolls every row back with
  500 ``internal_error``;
- validation precedence: ``invalid_query`` (query params) and the full body
  validation (``invalid_batch`` / ``bad_time`` / ``invalid_value``) complete
  before the machine is looked up; afterwards a missing machine is 404 and
  other methods are 405;
- batch rows persist across restarts and keep the per-machine privacy access
  hash chain valid.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from accountability.app import app
from accountability import privacy_chain


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"
WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")


def batch_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/batch"


def accesses_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def export_url(machine_id):
    return (
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={WIDE[0]}&to_accessed_at={WIDE[1]}"
    )


def integrity_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/integrity"


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


def item(
    *,
    accessed_at=T0,
    window_start=T1,
    window_end=T2,
    result="success",
    matches_count=3,
):
    return {
        "accessed_at": accessed_at,
        "window_start": window_start,
        "window_end": window_end,
        "result": result,
        "matches_count": matches_count,
    }


def batch(*items):
    return {"privacy_accesses": list(items)}


SUBMITTED_KEYS = {
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
}
RECORD_KEYS = SUBMITTED_KEYS | {"id", "machine_id"}


# --------------------------------------------------------------------------- #
# Success and result shapes
# --------------------------------------------------------------------------- #


def test_empty_batch_returns_200_empty_results(client):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch())
    assert response.status_code == 200
    assert response.json() == {"results": []}
    # An empty batch registers nothing.
    assert client.get(export_url(machine_id)).json()["privacy_accesses"] == []


def test_single_item_success_returns_full_record(client):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(item()))
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    entry = results[0]
    assert set(entry.keys()) == RECORD_KEYS | {"outcome"}
    assert entry["outcome"] == "success"
    assert entry["machine_id"] == machine_id
    assert entry["accessed_at"] == T0
    assert entry["window_start"] == T1
    assert entry["window_end"] == T2
    assert entry["result"] == "success"
    assert entry["matches_count"] == 3
    import uuid

    uuid.UUID(entry["id"])


def test_results_follow_request_array_positions(client):
    machine_id = create_machine(client)
    payload = batch(
        item(accessed_at=T0),
        item(accessed_at=T1),
        item(accessed_at=T2),
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["accessed_at"] for r in results] == [T0, T1, T2]
    assert [r["outcome"] for r in results] == ["success"] * 3
    assert [r["matches_count"] for r in results] == [3, 3, 3]


def test_duplicate_item_echoes_only_submitted_fields(client):
    machine_id = create_machine(client)
    first = client.post(batch_path(machine_id), json=batch(item())).json()["results"][0]

    response = client.post(batch_path(machine_id), json=batch(item(matches_count=99)))
    assert response.status_code == 200
    entry = response.json()["results"][0]
    assert entry["outcome"] == "duplicate_access"
    assert set(entry.keys()) == SUBMITTED_KEYS | {"outcome"}
    # The submitted fields are echoed verbatim, including the (ignored) count.
    assert entry["accessed_at"] == T0
    assert entry["window_start"] == T1
    assert entry["window_end"] == T2
    assert entry["result"] == "success"
    assert entry["matches_count"] == 99
    assert "id" not in entry and "machine_id" not in entry

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 1
    assert rows[0]["id"] == first["id"]
    assert rows[0]["matches_count"] == 3


def test_in_batch_duplicates_first_wins_later_duplicate(client):
    machine_id = create_machine(client)
    payload = batch(
        item(accessed_at=T0, matches_count=1),
        item(accessed_at=T1),
        item(accessed_at=T0, matches_count=77),  # same identity as item 0
        item(accessed_at=T2),
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["outcome"] for r in results] == [
        "success",
        "success",
        "duplicate_access",
        "success",
    ]
    # Position alignment, including the duplicate in the middle.
    assert [r["accessed_at"] for r in results] == [T0, T1, T0, T2]
    assert results[2]["matches_count"] == 77

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 3
    # The first occurrence kept its own hit count.
    assert sorted(r["matches_count"] for r in rows) == [1, 3, 3]


def test_duplicate_detects_stored_row_even_with_different_count(client):
    machine_id = create_machine(client)
    single = client.post(accesses_path(machine_id), json=item(matches_count=5))
    assert single.status_code == 201

    response = client.post(batch_path(machine_id), json=batch(item(matches_count=120)))
    assert response.status_code == 200
    assert response.json()["results"][0]["outcome"] == "duplicate_access"
    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 1
    assert rows[0]["matches_count"] == 5


def test_batch_and_single_registration_see_each_others_duplicates(client):
    machine_id = create_machine(client)
    # A batch row is a duplicate for the single endpoint (409).
    assert (
        client.post(batch_path(machine_id), json=batch(item())).status_code == 200
    )
    assert client.post(accesses_path(machine_id), json=item()).status_code == 409
    # And a single row is a duplicate inside a later batch.
    other = item(accessed_at=T4)
    assert client.post(accesses_path(machine_id), json=other).status_code == 201
    response = client.post(batch_path(machine_id), json=batch(other))
    assert response.json()["results"][0]["outcome"] == "duplicate_access"
    assert len(client.get(export_url(machine_id)).json()["privacy_accesses"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"accessed_at": T3},
        {"window_start": T0},
        {"window_end": T4},
        {"result": "failed", "matches_count": 0},
    ],
)
def test_different_time_window_or_result_is_distinct(client, change):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id), json=batch(item(), item(**change))
    )
    assert response.status_code == 200
    assert [r["outcome"] for r in response.json()["results"]] == [
        "success",
        "success",
    ]
    assert len(client.get(export_url(machine_id)).json()["privacy_accesses"]) == 2


def test_failed_item_must_carry_zero_matches(client):
    machine_id = create_machine(client)
    ok = client.post(
        batch_path(machine_id), json=batch(item(result="failed", matches_count=0))
    )
    assert ok.status_code == 200
    entry = ok.json()["results"][0]
    assert entry["outcome"] == "success"
    assert entry["result"] == "failed"
    assert entry["matches_count"] == 0


def test_equal_window_bounds_are_accepted(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id), json=batch(item(window_start=T2, window_end=T2))
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["outcome"] == "success"


def test_extra_fields_are_ignored_and_never_echoed(client):
    machine_id = create_machine(client)
    payload = batch()
    payload["extra_top"] = "nope"
    entry = item()
    entry["party"] = "the-secret-responsible-party"
    entry["public_key"] = "the-secret-material"
    payload["privacy_accesses"].append(entry)
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    body = response.text
    assert "the-secret-responsible-party" not in body
    assert "the-secret-material" not in body
    assert "extra_top" not in body


def test_batch_keeps_chain_valid_including_out_of_order_timestamps(client):
    machine_id = create_machine(client)
    # Deliberately non-chronological submitted order.
    payload = batch(
        item(accessed_at=T3),
        item(accessed_at=T0),
        item(accessed_at="2026-03-01T00:00:00.500Z"),
        item(accessed_at=T1),
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    integrity = client.get(integrity_path(machine_id)).json()
    assert integrity["valid"] is True
    assert integrity["checked_count"] == 4
    assert integrity["broken_access_id"] is None


def test_batch_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        response = first.post(
            batch_path(machine_id),
            json=batch(item(accessed_at=T0), item(accessed_at=T1)),
        )
        assert response.status_code == 200
        expected = response.json()

    with TestClient(app) as second:
        rows = second.get(export_url(machine_id)).json()["privacy_accesses"]
        assert len(rows) == 2
        integrity = second.get(integrity_path(machine_id)).json()
        assert integrity["valid"] is True

    assert [r["accessed_at"] for r in expected["results"]] == [T0, T1]
    assert {r["id"] for r in expected["results"]} == {r["id"] for r in rows}


# --------------------------------------------------------------------------- #
# Body validation: invalid_batch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-an-object",
        123,
        True,
        {},
        {"privacy_accesses": None},
        {"privacy_accesses": "x"},
        {"privacy_accesses": 1},
        {"privacy_accesses": {}},
        {"privacy_accesses": True},
        # item is not an object
        {"privacy_accesses": [None]},
        {"privacy_accesses": ["x"]},
        {"privacy_accesses": [1]},
        {"privacy_accesses": [[]]},
        {"privacy_accesses": [True]},
        # missing field inside an otherwise good batch
        {"privacy_accesses": [
            {"window_start": T1, "window_end": T2, "result": "success",
             "matches_count": 1}]},
        {"privacy_accesses": [
            {"accessed_at": T0, "window_end": T2, "result": "success",
             "matches_count": 1}]},
        {"privacy_accesses": [
            {"accessed_at": T0, "window_start": T1, "result": "success",
             "matches_count": 1}]},
        {"privacy_accesses": [
            {"accessed_at": T0, "window_start": T1, "window_end": T2,
             "matches_count": 1}]},
        {"privacy_accesses": [
            {"accessed_at": T0, "window_start": T1, "window_end": T2,
             "result": "success"}]},
        # wrong JSON types for business fields
        {"privacy_accesses": [item(accessed_at=123)]},
        {"privacy_accesses": [item(window_start=None)]},
        {"privacy_accesses": [item(window_end=[])]},
        {"privacy_accesses": [item(result=1)]},
        {"privacy_accesses": [item(matches_count="2")]},
        {"privacy_accesses": [item(matches_count=1.0)]},
        {"privacy_accesses": [item(matches_count=True)]},
        {"privacy_accesses": [item(matches_count=None)]},
    ],
)
def test_invalid_batch_is_422(client, payload):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_unparseable_or_empty_body_is_invalid_batch(client):
    machine_id = create_machine(client)
    for raw in (b"", b"{", b"not json", b"[1, 2]"):
        response = client.post(
            batch_path(machine_id),
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_batch"}}


# --------------------------------------------------------------------------- #
# Body validation: bad_time / invalid_value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_item",
    [
        item(accessed_at="2026-03-01T00:00:00"),
        item(accessed_at="2026-03-01T00:00:00+00:00"),
        item(accessed_at="2026-03-01T00:00:00z"),
        item(accessed_at=" 2026-03-01T00:00:00Z"),
        item(accessed_at="2026-03-01T00:00:00Z "),
        item(accessed_at="2026-13-01T00:00:00Z"),
        item(accessed_at="2026-02-30T00:00:00Z"),
        item(accessed_at="2026-03-01T24:00:00Z"),
        item(window_start="garbage"),
        item(window_end="2026-03-01T00:00:02z"),
        item(window_end="2026-02-30T00:00:02Z"),
        item(window_start=T2, window_end=T1),
    ],
)
def test_bad_time_is_422(client, bad_item):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(bad_item))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "bad_item",
    [
        item(result="ok"),
        item(result="SUCCESS"),
        item(matches_count=-1),
        item(result="failed", matches_count=5),
    ],
)
def test_invalid_value_is_422(client, bad_item):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(bad_item))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


def test_entire_batch_is_validated_before_anything_registers(client):
    machine_id = create_machine(client)
    # The third item is invalid; even though the first two are valid nothing
    # may be registered.
    payload = batch(
        item(accessed_at=T0),
        item(accessed_at=T1),
        item(result="weird"),
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}
    assert client.get(export_url(machine_id)).json()["privacy_accesses"] == []


# --------------------------------------------------------------------------- #
# Query validation, machine lookup, method routing
# --------------------------------------------------------------------------- #


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id) + "?unexpected=1", json=batch(item())
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_and_body_validation_run_before_machine_lookup(client):
    # Unknown query against a missing machine: invalid_query, no writes.
    response = client.post(
        batch_path(MISSING_ID) + "?x=1", json=batch(item())
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # Invalid body against a missing machine: the body's own error code.
    response = client.post(batch_path(MISSING_ID), json=batch(item(result="bad")))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}

    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM privacy_accesses WHERE machine_id = :m"),
            {"m": MISSING_ID},
        ).scalar_one()
    assert count == 0


def test_empty_batch_against_missing_machine_is_404(client):
    # Validation passes, then the machine lookup fails — even with no items.
    response = client.post(batch_path(MISSING_ID), json=batch())
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_valid_batch_for_missing_machine_is_404_and_writes_nothing(client):
    response = client.post(batch_path(MISSING_ID), json=batch(item()))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM privacy_accesses WHERE machine_id = :m"),
            {"m": MISSING_ID},
        ).scalar_one()
    assert count == 0


def test_only_post_is_accepted_on_batch_path(client):
    machine_id = create_machine(client)
    url = batch_path(machine_id)
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Atomicity and serialization with single registrations
# --------------------------------------------------------------------------- #


def test_persistence_failure_returns_500_and_leaves_no_partial_records(
    client, monkeypatch
):
    machine_id = create_machine(client)
    # Break the chain relink after the rows have been inserted, inside the
    # same locked transaction. The raised OperationalError must roll the whole
    # batch back and surface as a 500 internal_error.
    def _boom(conn, rows):
        raise OperationalError(None, None, RuntimeError("simulated io failure"))

    monkeypatch.setattr(privacy_chain, "_apply_updates", _boom)
    response = client.post(
        batch_path(machine_id), json=batch(item(accessed_at=T0), item(accessed_at=T1))
    )
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert client.get(export_url(machine_id)).json()["privacy_accesses"] == []


def test_concurrent_batches_and_singles_serialize_without_loss(client):
    machine_id = create_machine(client)
    barrier = threading.Barrier(8)

    def post_single(index):
        barrier.wait()
        return client.post(accesses_path(machine_id), json=item(accessed_at=ts(index)))

    def post_batch(start):
        barrier.wait()
        return client.post(
            batch_path(machine_id),
            json=batch(
                item(accessed_at=ts(start)),
                item(accessed_at=ts(start + 1)),
            ),
        )

    def ts(index):
        return f"2026-04-01T00:00:{index:02d}Z"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for i in range(4):
            futures.append(pool.submit(post_single, i))        # seconds 0..3
        for i, start in enumerate((10, 20, 30, 40)):
            futures.append(pool.submit(post_batch, start))     # 10,11 20,21 ...
        statuses = [future.result().status_code for future in futures]

    assert statuses.count(201) == 4
    assert statuses.count(200) == 4
    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 12
    integrity = client.get(integrity_path(machine_id)).json()
    assert integrity["valid"] is True
    assert integrity["checked_count"] == 12


def test_concurrent_same_identity_has_exactly_one_winner(client):
    machine_id = create_machine(client)
    attempts = 12
    barrier = threading.Barrier(attempts)

    def race(index):
        barrier.wait()
        # Half come through the single endpoint, half through one-item batches.
        if index % 2 == 0:
            return client.post(accesses_path(machine_id), json=item())
        return client.post(batch_path(machine_id), json=batch(item()))

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        futures = [pool.submit(race, i) for i in range(attempts)]
        responses = [future.result() for future in futures]

    def is_winner(response):
        if response.status_code == 201:
            return True
        if response.status_code == 200:
            return response.json()["results"][0]["outcome"] == "success"
        return False

    winners = [r for r in responses if is_winner(r)]
    assert len(winners) == 1
    # Singles lose with 409; batch losers get 200 + duplicate_access.
    for response in responses:
        assert response.status_code in (200, 201, 409)
        if response.status_code == 200 and not is_winner(response):
            assert response.json()["results"][0]["outcome"] == "duplicate_access"

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 1
    assert client.get(integrity_path(machine_id)).json()["valid"] is True
