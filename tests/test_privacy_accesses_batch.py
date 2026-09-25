"""Tests for machine-level batch privacy access registration.

Covers `POST /machines/{machine_id}/privacy-accesses/batch`:

- success: 200 with a position-aligned ``results`` array, the empty batch
  short-circuit, full single-record shape for ``success`` outcomes and only
  echoed submitted fields for ``duplicate_access`` outcomes;
- duplication: pre-existing and intra-batch repeats never add rows and never
  abort the other items, the hit count stays out of the identity, batches
  interoperate with the single-registration path in both directions;
- validation: ``invalid_batch`` for body/item shape and field types,
  ``bad_time`` for timestamps and inverted windows, ``invalid_value`` for
  result/count domains, ``invalid_query`` for query parameters — every 422
  precedes the machine lookup;
- lifecycle: ``404 not_found`` after validation, POST-only ``405``,
  ``500 internal_error`` with no partial records on a persistence failure,
  hash-chain integrity after a batch, and persistence across restarts.
"""
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


MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def batch_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/batch"


def single_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def export_url(machine_id, from_accessed_at=T0, to_accessed_at=T5):
    return (
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={from_accessed_at}&to_accessed_at={to_accessed_at}"
    )


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


def batch(*items_):
    return {"privacy_accesses": list(items_)}


SUBMITTED_KEYS = {
    "outcome",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
}

RECORD_KEYS = SUBMITTED_KEYS | {"id", "machine_id"}


# --------------------------------------------------------------------------- #
# Success
# --------------------------------------------------------------------------- #


def test_batch_registers_every_item_in_order_with_200(client):
    machine_id = create_machine(client)
    payload = batch(
        item(accessed_at=T0, matches_count=1),
        item(accessed_at=T3, result="failed", matches_count=0),
        item(accessed_at=T4, matches_count=9, window_start=T0, window_end=T5),
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3

    for index, (submitted, result) in enumerate(zip(payload["privacy_accesses"], results)):
        assert result["outcome"] == "success"
        assert set(result.keys()) == RECORD_KEYS
        assert result["machine_id"] == machine_id
        assert result["accessed_at"] == submitted["accessed_at"]
        assert result["window_start"] == submitted["window_start"]
        assert result["window_end"] == submitted["window_end"]
        assert result["result"] == submitted["result"]
        assert result["matches_count"] == submitted["matches_count"]
        import uuid

        uuid.UUID(result["id"])
        # Results follow the request array positions, not time ordering.
        assert result["accessed_at"] == payload["privacy_accesses"][index]["accessed_at"]


def test_empty_batch_returns_200_empty_results_without_lookup(client):
    # Legal even against a non-existent machine: nothing to register.
    response = client.post(batch_path(MISSING_ID), json=batch())
    assert response.status_code == 200
    assert response.json() == {"results": []}

    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch())
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_batch_persists_records_and_keeps_chain_valid(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id),
        json=batch(
            item(accessed_at=T4, matches_count=4),
            item(accessed_at=T0, matches_count=0),
            item(accessed_at=T2, result="failed", matches_count=0),
        ),
    )
    assert response.status_code == 200

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert [r["accessed_at"] for r in rows] == [T0, T2, T4]
    assert [r["matches_count"] for r in rows] == [0, 0, 4]

    integrity = client.get(
        f"/machines/{machine_id}/privacy-accesses/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 3, "broken_access_id": None}


def test_equal_window_bounds_are_accepted(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id), json=batch(item(window_start=T2, window_end=T2))
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["outcome"] == "success"


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


def test_intra_batch_duplicates_first_wins_rest_are_duplicates(client):
    machine_id = create_machine(client)
    payload = batch(
        item(accessed_at=T0, matches_count=3),
        item(accessed_at=T0, matches_count=99),  # same identity, different count
        item(accessed_at=T1, matches_count=5),
        item(accessed_at=T1, matches_count=5),   # exact repeat
    )
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["outcome"] for r in results] == [
        "success",
        "duplicate_access",
        "success",
        "duplicate_access",
    ]

    duplicate = results[1]
    # Duplicates only echo the submitted fields — no id or machine id.
    assert set(duplicate.keys()) == SUBMITTED_KEYS
    assert duplicate["matches_count"] == 99
    assert duplicate["accessed_at"] == T0

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 2
    # The first item's record is the one kept, including its hit count.
    t0_row = next(r for r in rows if r["accessed_at"] == T0)
    assert t0_row["matches_count"] == 3
    t1_row = next(r for r in rows if r["accessed_at"] == T1)
    assert t1_row["matches_count"] == 5


def test_batch_detects_records_registered_by_single_endpoint(client):
    machine_id = create_machine(client)
    assert client.post(single_path(machine_id), json=item()).status_code == 201

    response = client.post(
        batch_path(machine_id),
        json=batch(item(matches_count=99), item(accessed_at=T4)),
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["outcome"] for r in results] == ["duplicate_access", "success"]
    assert set(results[0].keys()) == SUBMITTED_KEYS

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 2
    assert next(r for r in rows if r["accessed_at"] == T0)["matches_count"] == 3


def test_single_endpoint_detects_records_registered_by_batch(client):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(item()))
    assert response.status_code == 200

    repeat = client.post(single_path(machine_id), json=item(matches_count=8))
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "duplicate_access"}}

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 1
    assert rows[0]["matches_count"] == 3


@pytest.mark.parametrize(
    "change",
    [
        {"accessed_at": T3},
        {"window_start": T0},
        {"window_end": T4},
        {"result": "failed", "matches_count": 0},
    ],
)
def test_different_identity_fields_register_separately(client, change):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id), json=batch(item(), item(**change))
    )
    assert response.status_code == 200
    assert [r["outcome"] for r in response.json()["results"]] == [
        "success",
        "success",
    ]
    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 2


def test_duplicate_is_scoped_per_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    response = client.post(
        batch_path(machine_one), json=batch(item(), item())
    )
    assert [r["outcome"] for r in response.json()["results"]] == [
        "success",
        "duplicate_access",
    ]
    # The identical tuple on another machine is a fresh access.
    response = client.post(batch_path(machine_two), json=batch(item()))
    assert response.json()["results"][0]["outcome"] == "success"


def test_extra_fields_are_never_stored_or_echoed(client):
    machine_id = create_machine(client)
    payload = batch()
    secret = {"party": "secret-party", "public_key": "secret-key"}
    first = item()
    first.update(secret)
    payload["privacy_accesses"].append(first)
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 200
    assert "secret-party" not in response.text
    assert "secret-key" not in response.text
    with client.app.state.engine.connect() as conn:
        raw = conn.execute(
            text("SELECT * FROM privacy_accesses WHERE machine_id = :m"),
            {"m": machine_id},
        )
        assert set(raw.keys()) == {
            "id",
            "machine_id",
            "accessed_at",
            "window_start",
            "window_end",
            "result",
            "matches_count",
            "previous_access_id",
            "content_hash",
            "chain_hash",
        }


# --------------------------------------------------------------------------- #
# invalid_batch
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
        {"privacy_accesses": {}},
        {"privacy_accesses": [1]},
        {"privacy_accesses": ["x"]},
        {"privacy_accesses": [None]},
        batch(item(), "not-an-object"),
        # missing each field in turn
        batch({"window_start": T1, "window_end": T2, "result": "success",
               "matches_count": 1}),
        batch({"accessed_at": T0, "window_end": T2, "result": "success",
               "matches_count": 1}),
        batch({"accessed_at": T0, "window_start": T1, "result": "success",
               "matches_count": 1}),
        batch({"accessed_at": T0, "window_start": T1, "window_end": T2,
               "matches_count": 1}),
        batch({"accessed_at": T0, "window_start": T1, "window_end": T2,
               "result": "success"}),
        # wrong business field types
        batch(item(accessed_at=123)),
        batch(item(window_start=None)),
        batch(item(window_end=["x"])),
        batch(item(result=1)),
        batch(item(result=None)),
        batch(item(matches_count=1.0)),
        batch(item(matches_count=True)),
        batch(item(matches_count="2")),
        batch(item(matches_count=None)),
    ],
)
def test_invalid_batch_is_422(client, payload):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_malformed_json_body_is_invalid_batch(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id),
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}


def test_invalid_batch_precedes_machine_lookup_and_writes_nothing(client):
    response = client.post(batch_path(MISSING_ID), json=batch(item(result=1)))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_batch"}}
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM privacy_accesses WHERE machine_id = :m"),
            {"m": MISSING_ID},
        ).scalar_one()
    assert count == 0


# --------------------------------------------------------------------------- #
# bad_time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_item",
    [
        item(accessed_at="2026-03-01T00:00:00"),          # missing Z
        item(accessed_at="2026-03-01T00:00:00+00:00"),    # offset
        item(accessed_at="2026-03-01T00:00:00z"),         # lowercase z
        item(accessed_at=" 2026-03-01T00:00:00Z"),        # leading space
        item(accessed_at="2026-03-01T00:00:00Z "),        # trailing space
        item(accessed_at="2026-13-01T00:00:00Z"),         # bad month
        item(accessed_at="2026-02-30T00:00:00Z"),         # bad day
        item(accessed_at="2026-03-01T24:00:00Z"),         # bad hour
        item(window_start="garbage"),
        item(window_end="2026-03-01T00:00:02+00:00"),
        item(window_start=T2, window_end=T1),             # inverted window
    ],
)
def test_bad_time_is_422(client, bad_item):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(bad_item))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_bad_time_on_any_item_rejects_the_whole_batch(client):
    machine_id = create_machine(client)
    response = client.post(
        batch_path(machine_id),
        json=batch(item(accessed_at=T0), item(accessed_at="nope")),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}
    # Validation is all-or-nothing: the valid first item was not written.
    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert rows == []


# --------------------------------------------------------------------------- #
# invalid_value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_item",
    [
        item(result="ok"),
        item(result="SUCCESS"),
        item(result=""),
        item(matches_count=-1),
        item(result="failed", matches_count=5),
    ],
)
def test_invalid_value_is_422(client, bad_item):
    machine_id = create_machine(client)
    response = client.post(batch_path(machine_id), json=batch(bad_item))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_value"}}


# --------------------------------------------------------------------------- #
# Query validation, lookup, methods
# --------------------------------------------------------------------------- #


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.post(f"{batch_path(machine_id)}?unexpected=1", json=batch())
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_precedes_body_and_machine_lookup(client):
    response = client.post(
        f"{batch_path(MISSING_ID)}?x=1",
        json=batch(item(result=1)),
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_404_and_writes_nothing(client):
    response = client.post(batch_path(MISSING_ID), json=batch(item(), item(accessed_at=T4)))
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


def test_persistence_failure_is_500_and_leaves_no_partial_records(
    client, monkeypatch
):
    machine_id = create_machine(client)

    # Fail at chain linking, after both rows have been inserted inside the
    # single batch write transaction: the rollback must remove both.
    def boom(*args, **kwargs):
        raise RuntimeError("storage gone")

    monkeypatch.setattr("accountability.privacy_chain._apply_updates", boom)

    response = client.post(
        batch_path(machine_id), json=batch(item(), item(accessed_at=T4))
    )
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert rows == []


def test_batch_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        response = first.post(
            batch_path(machine_id),
            json=batch(
                item(accessed_at=T0, matches_count=2),
                item(accessed_at=T3, result="failed", matches_count=0),
                item(accessed_at=T0, matches_count=7),  # intra-batch duplicate
            ),
        )
        assert response.status_code == 200
        outcomes = [r["outcome"] for r in response.json()["results"]]
        assert outcomes == ["success", "success", "duplicate_access"]

    with TestClient(app) as second:
        rows = second.get(export_url(machine_id)).json()["privacy_accesses"]
        integrity = second.get(
            f"/machines/{machine_id}/privacy-accesses/integrity"
        ).json()

    assert [r["accessed_at"] for r in rows] == [T0, T3]
    assert integrity["valid"] is True
    assert integrity["checked_count"] == 2
