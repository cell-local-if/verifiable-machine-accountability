"""Tests for machine-level privacy access registration and read-only query.

Covers `POST /machines/{machine_id}/privacy-accesses` and
`GET /machines/{machine_id}/privacy-accesses/compliance-export`:

- registration: 201 shape, strict object/field validation (422) before any
  path lookup, ``404 not_found`` for a valid body against a missing machine,
  ``409 duplicate_access`` for a repeated (time, window, result) tuple that
  writes nothing, POST-only ``405``, no responsibility/key rawtext stored or
  echoed, and persistence across a restart;
- query: strict query validation (``bad_time`` / ``invalid_query``) before any
  machine or access read, ``404 not_found`` with no access data, GET-only
  ``405``, closed UTC-window filtering on each record's own ``accessed_at``,
  ordering by the actual UTC instant then record id (exact-second before
  fractional-second), strict machine isolation, read-only byte stability, and
  persistence across a restart.
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

WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def accesses_path(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def export_url(machine_id, from_accessed_at=WIDE[0], to_accessed_at=WIDE[1]):
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


def good_body(
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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_access_row(
    client,
    machine_id,
    access_id,
    accessed_at,
    *,
    window_start=T1,
    window_end=T2,
    result="success",
    matches_count=1,
):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO privacy_accesses "
                "(id, machine_id, accessed_at, window_start, window_end, "
                "result, matches_count) VALUES "
                "(:id, :machine_id, :accessed_at, :window_start, :window_end, "
                ":result, :matches_count)"
            ),
            {
                "id": access_id,
                "machine_id": machine_id,
                "accessed_at": accessed_at,
                "window_start": window_start,
                "window_end": window_end,
                "result": result,
                "matches_count": matches_count,
            },
        )


ACCESS_KEYS = {
    "id",
    "machine_id",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_create_access_returns_201_with_record(client):
    machine_id = create_machine(client)
    response = client.post(accesses_path(machine_id), json=good_body())
    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == ACCESS_KEYS
    assert body["machine_id"] == machine_id
    assert body["accessed_at"] == T0
    assert body["window_start"] == T1
    assert body["window_end"] == T2
    assert body["result"] == "success"
    assert body["matches_count"] == 3
    # A fresh UUID identifier is assigned.
    import uuid

    uuid.UUID(body["id"])


def test_timestamps_are_echoed_verbatim_including_fractional_seconds(client):
    machine_id = create_machine(client)
    body = good_body(
        accessed_at="2026-03-01T00:00:00.250Z",
        window_start="2026-03-01T00:00:01.500000Z",
        window_end="2026-03-01T00:00:02.750Z",
    )
    response = client.post(accesses_path(machine_id), json=body)
    assert response.status_code == 201
    record = response.json()
    assert record["accessed_at"] == "2026-03-01T00:00:00.250Z"
    assert record["window_start"] == "2026-03-01T00:00:01.500000Z"
    assert record["window_end"] == "2026-03-01T00:00:02.750Z"


def test_failed_access_is_stored_with_zero_matches(client):
    machine_id = create_machine(client)
    response = client.post(
        accesses_path(machine_id), json=good_body(result="failed", matches_count=0)
    )
    assert response.status_code == 201
    record = response.json()
    assert record["result"] == "failed"
    assert record["matches_count"] == 0

    exported = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(exported) == 1
    assert exported[0]["result"] == "failed"
    assert exported[0]["matches_count"] == 0


def test_zero_matches_success_is_allowed(client):
    machine_id = create_machine(client)
    response = client.post(
        accesses_path(machine_id), json=good_body(matches_count=0)
    )
    assert response.status_code == 201
    assert response.json()["matches_count"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-an-object",
        123,
        {},
        # missing each field in turn
        {"window_start": T1, "window_end": T2, "result": "success",
         "matches_count": 1},
        {"accessed_at": T0, "window_end": T2, "result": "success",
         "matches_count": 1},
        {"accessed_at": T0, "window_start": T1, "result": "success",
         "matches_count": 1},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "matches_count": 1},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success"},
        # invalid access time
        {"accessed_at": "2026-03-01T00:00:00", "window_start": T1,
         "window_end": T2, "result": "success", "matches_count": 1},
        {"accessed_at": "2026-03-01T00:00:00+00:00", "window_start": T1,
         "window_end": T2, "result": "success", "matches_count": 1},
        {"accessed_at": " 2026-03-01T00:00:00Z", "window_start": T1,
         "window_end": T2, "result": "success", "matches_count": 1},
        {"accessed_at": "2026-13-01T00:00:00Z", "window_start": T1,
         "window_end": T2, "result": "success", "matches_count": 1},
        # invalid window bounds
        {"accessed_at": T0, "window_start": "garbage", "window_end": T2,
         "result": "success", "matches_count": 1},
        {"accessed_at": T0, "window_start": T1,
         "window_end": "2026-03-01T00:00:02z", "result": "success",
         "matches_count": 1},
        {"accessed_at": T0, "window_start": T1,
         "window_end": "2026-02-30T00:00:02Z", "result": "success",
         "matches_count": 1},
        # inverted window
        {"accessed_at": T0, "window_start": T2, "window_end": T1,
         "result": "success", "matches_count": 1},
        # invalid result
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "ok", "matches_count": 1},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "SUCCESS", "matches_count": 1},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": None, "matches_count": 1},
        # invalid matches count
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success", "matches_count": -1},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success", "matches_count": True},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success", "matches_count": 1.0},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success", "matches_count": "2"},
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "success", "matches_count": None},
        # a failed access must carry zero matches
        {"accessed_at": T0, "window_start": T1, "window_end": T2,
         "result": "failed", "matches_count": 5},
    ],
)
def test_invalid_payload_is_422(client, payload):
    machine_id = create_machine(client)
    response = client.post(accesses_path(machine_id), json=payload)
    assert response.status_code == 422


def test_equal_window_bounds_are_accepted(client):
    machine_id = create_machine(client)
    response = client.post(
        accesses_path(machine_id), json=good_body(window_start=T2, window_end=T2)
    )
    assert response.status_code == 201


def test_invalid_payload_is_422_before_machine_lookup_and_writes_nothing(client):
    response = client.post(
        accesses_path(MISSING_ID),
        json=good_body(result="weird"),
    )
    assert response.status_code == 422
    # No row may be left behind, even for the (non-existent) path machine.
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM privacy_accesses WHERE machine_id = :m"),
            {"m": MISSING_ID},
        ).scalar_one()
    assert count == 0


def test_valid_payload_for_missing_machine_is_404_without_data(client):
    response = client.post(accesses_path(MISSING_ID), json=good_body())
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_duplicate_registration_is_409_and_writes_nothing(client):
    machine_id = create_machine(client)
    first = client.post(accesses_path(machine_id), json=good_body())
    assert first.status_code == 201

    duplicate = client.post(accesses_path(machine_id), json=good_body())
    assert duplicate.status_code == 409
    assert duplicate.json() == {"error": {"code": "duplicate_access"}}

    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 1
    assert rows[0]["id"] == first.json()["id"]
    assert rows[0]["matches_count"] == 3


def test_duplicate_detects_even_with_different_matches_count(client):
    machine_id = create_machine(client)
    assert (
        client.post(accesses_path(machine_id), json=good_body(matches_count=3)).status_code
        == 201
    )
    # Same (time, window, result), different hit count: still the same access.
    response = client.post(
        accesses_path(machine_id), json=good_body(matches_count=99)
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_access"}}
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
def test_different_time_window_or_result_is_a_distinct_record(client, change):
    machine_id = create_machine(client)
    assert client.post(accesses_path(machine_id), json=good_body()).status_code == 201
    response = client.post(accesses_path(machine_id), json=good_body(**change))
    assert response.status_code == 201
    rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert len(rows) == 2


def test_duplicate_is_scoped_per_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    assert client.post(accesses_path(machine_one), json=good_body()).status_code == 201
    # The identical tuple on another machine is a separate access.
    response = client.post(accesses_path(machine_two), json=good_body())
    assert response.status_code == 201
    for machine_id in (machine_one, machine_two):
        rows = client.get(export_url(machine_id)).json()["privacy_accesses"]
        assert len(rows) == 1
        assert rows[0]["machine_id"] == machine_id


def test_extra_responsibility_or_key_fields_are_never_stored_or_echoed(client):
    machine_id = create_machine(client)
    secret_party = "the-secret-responsible-party"
    secret_key = "the-secret-material"
    payload = good_body()
    payload["party"] = secret_party
    payload["responsible_party"] = f"  {secret_party}  "
    payload["public_key"] = secret_key
    response = client.post(accesses_path(machine_id), json=payload)
    assert response.status_code == 201
    assert secret_party not in response.text
    assert secret_key not in response.text

    # The stored row has only the seven audit columns, no responsibility data.
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
        }
        exported = client.get(export_url(machine_id)).content.decode()
    assert secret_party not in exported
    assert secret_key not in exported


def test_only_post_is_accepted_on_registration_path(client):
    machine_id = create_machine(client)
    url = accesses_path(machine_id)
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


def test_registration_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        response = first.post(accesses_path(machine_id), json=good_body())
        assert response.status_code == 201
        expected = response.json()

    with TestClient(app) as second:
        rows = second.get(export_url(machine_id)).json()["privacy_accesses"]

    assert len(rows) == 1
    assert rows[0] == expected


# --------------------------------------------------------------------------- #
# Query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_accessed_at=2026-03-01T00:00:00Z",
        "?to_accessed_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    base = f"/machines/{machine_id}/privacy-accesses/compliance-export"
    response = client.get(f"{base}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form
        "2026-03-01T00:00:00z",           # lowercase suffix
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "2026-03-01 00:00:00Z",           # space separator
        "garbage",
        "",                               # blank
        "2026-13-01T00:00:00Z",           # bad month
        "2026-02-30T00:00:00Z",           # bad day
        "2026-03-01T24:00:00Z",           # bad hour
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={value}&to_accessed_at={T5}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_second_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at=2026-03-01T00:00:01.250Z"
        f"&to_accessed_at=2026-03-01T00:00:03.750000Z"
    )
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["privacy_accesses"]] == [rid(1)]


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted_on_query(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["privacy_accesses"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={T0}&to_accessed_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = f"/machines/{MISSING_ID}/privacy-accesses/compliance-export"
    bad_time = client.get(f"{base}?from_accessed_at=nope&to_accessed_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(
        f"{base}?from_accessed_at={T0}&to_accessed_at={T5}&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_query_missing_machine_returns_404_without_data(client):
    response = client.get(export_url(MISSING_ID, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted_on_export_path(client):
    machine_id = create_machine(client)
    url = (
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={T0}&to_accessed_at={T5}"
    )
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Query result shape, windowing, ordering, isolation
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_window(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_accessed_at",
        "to_accessed_at",
        "privacy_accesses",
    }
    assert body["machine_id"] == machine_id
    assert body["from_accessed_at"] == WIDE[0]
    assert body["to_accessed_at"] == WIDE[1]
    assert body["privacy_accesses"] == []


def test_window_is_closed_on_accessed_at(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T0)
    insert_access_row(client, machine_id, rid(2), T2)
    insert_access_row(client, machine_id, rid(3), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(2)]

    # Equal bounds include the boundary access.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["privacy_accesses"] == []


def test_window_uses_access_time_not_export_window(client):
    machine_id = create_machine(client)
    # Accessed at T4, but the recorded desensitized export window is T0..T1:
    # membership must follow accessed_at, not the stored window bounds.
    insert_access_row(
        client, machine_id, rid(1), T4, window_start=T0, window_end=T1
    )
    body = client.get(export_url(machine_id, T0, T1)).json()
    assert body["privacy_accesses"] == []
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(1)]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_access_row(client, machine_id, rid(2), fractional)
    insert_access_row(client, machine_id, rid(1), T0)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(1), rid(2)]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    # Same access instant but distinct recorded windows (so the records are not
    # duplicates); ordering at the shared instant falls back to record id.
    insert_access_row(
        client, machine_id, rid(30), T2, window_start=T0, window_end=T1
    )
    insert_access_row(
        client, machine_id, rid(20), T2, window_start=T3, window_end=T4
    )

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(20), rid(30)]


def test_records_carry_exact_fields_and_echoed_windows(client):
    machine_id = create_machine(client)
    insert_access_row(
        client,
        machine_id,
        rid(7),
        "2026-03-01T00:00:00.250Z",
        window_start="2026-03-02T00:00:00Z",
        window_end="2026-03-03T00:00:00Z",
        result="failed",
        matches_count=0,
    )
    record = client.get(export_url(machine_id)).json()["privacy_accesses"][0]
    assert set(record.keys()) == ACCESS_KEYS
    assert record == {
        "id": rid(7),
        "machine_id": machine_id,
        "accessed_at": "2026-03-01T00:00:00.250Z",
        "window_start": "2026-03-02T00:00:00Z",
        "window_end": "2026-03-03T00:00:00Z",
        "result": "failed",
        "matches_count": 0,
    }


def test_other_machine_accesses_are_never_returned(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), T1)
    insert_access_row(client, machine_two, rid(2), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(1)]

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_query_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T1, matches_count=4)
    insert_access_row(client, machine_id, rid(2), T3, result="failed",
                      matches_count=0)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM privacy_accesses")))

    before = table_state()
    first = client.get(export_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(export_url(machine_id, T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_empty_database_exports_empty(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json()["privacy_accesses"] == []


def test_query_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_access_row(first, machine_id, rid(1), T1)
        insert_access_row(first, machine_id, rid(2), T3, result="failed",
                          matches_count=0)
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    rows = response.json()["privacy_accesses"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
