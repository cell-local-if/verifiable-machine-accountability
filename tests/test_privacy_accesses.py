"""Tests for machine-level privacy access registration and read-only query.

Covers `POST /machines/{machine_id}/privacy-accesses` (registration) and
`GET /machines/{machine_id}/privacy-accesses/compliance-export` (the
read-only query): strict object body validation as 422 before any machine
lookup (missing fields, malformed/offset/whitespace timestamps, inverted
window, unknown result, non-integer/negative hit count, extra fields such as
a raw responsible party), `404 not_found`, `409 duplicate_access` without a
write, failed accesses recorded with zero hits, GET-only/POST-only `405`
routing, closed-UTC-window filtering on each record's own `accessed_at`,
ordering by the actual UTC instant then record id, machine isolation,
strict read-only byte stability, and persistence across a restart.
"""
import uuid

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
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"

MISSING = "00000000-0000-0000-0000-000000000000"


def access_url(machine_id):
    return f"/machines/{machine_id}/privacy-accesses"


def export_url(machine_id, from_accessed_at=WIDE[0], to_accessed_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={from_accessed_at}&to_accessed_at={to_accessed_at}"
    )


def make_body(**overrides):
    body = {
        "accessed_at": T1,
        "window_start": T0,
        "window_end": T5,
        "result": "success",
        "hit_count": 2,
    }
    body.update(overrides)
    return body


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


def register(client, machine_id, **overrides):
    return client.post(access_url(machine_id), json=make_body(**overrides))


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_access_row(client, machine_id, access_id, accessed_at, *,
                      window_start=T0, window_end=T5, result="success",
                      hit_count=1):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO privacy_accesses "
                "(id, machine_id, accessed_at, window_start, window_end, "
                "result, hit_count) VALUES "
                "(:id, :machine_id, :accessed_at, :window_start, :window_end, "
                ":result, :hit_count)"
            ),
            {
                "id": access_id,
                "machine_id": machine_id,
                "accessed_at": accessed_at,
                "window_start": window_start,
                "window_end": window_end,
                "result": result,
                "hit_count": hit_count,
            },
        )


ACCESS_KEYS = {
    "id",
    "machine_id",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "hit_count",
}


# --------------------------------------------------------------------------- #
# Registration success
# --------------------------------------------------------------------------- #


def test_successful_registration_returns_201_with_record(client):
    machine_id = create_machine(client)
    response = register(client, machine_id)

    assert response.status_code == 201
    record = response.json()
    assert set(record.keys()) == ACCESS_KEYS
    assert record["machine_id"] == machine_id
    assert record["accessed_at"] == T1
    assert record["window_start"] == T0
    assert record["window_end"] == T5
    assert record["result"] == "success"
    assert record["hit_count"] == 2
    # The id is a fresh UUID v4.
    parsed = uuid.UUID(record["id"])
    assert parsed.version == 4


def test_extra_fields_are_ignored_and_never_stored_or_echoed(client):
    # The entry point never accepts responsible-party raw text: an extra
    # ``party`` (or any other field) is not a validation error, but it is
    # ignored on write and absent from every response.
    machine_id = create_machine(client)
    payload = make_body(party="alice", role="lead", public_key="secret-key")
    response = client.post(access_url(machine_id), json=payload)
    assert response.status_code == 201
    record = response.json()
    assert set(record.keys()) == ACCESS_KEYS
    assert "alice" not in response.text

    exported = client.get(export_url(machine_id))
    assert "alice" not in exported.text
    assert "secret-key" not in exported.text
    with client.app.state.engine.connect() as conn:
        columns = [
            row[1]
            for row in conn.execute(text("PRAGMA table_info(privacy_accesses)"))
        ]
    assert "party" not in columns and "role" not in columns


def test_success_with_zero_hits_is_allowed(client):
    machine_id = create_machine(client)
    response = register(client, machine_id, hit_count=0)
    assert response.status_code == 201
    assert response.json()["hit_count"] == 0


def test_equal_window_bounds_are_allowed(client):
    machine_id = create_machine(client)
    response = register(
        client, machine_id, window_start=T2, window_end=T2, hit_count=0
    )
    assert response.status_code == 201


def test_failed_access_with_zero_hits_is_recorded(client):
    machine_id = create_machine(client)
    response = register(client, machine_id, result="failed", hit_count=0)
    assert response.status_code == 201
    record = response.json()
    assert record["result"] == "failed"
    assert record["hit_count"] == 0


def test_failed_access_with_nonzero_hits_is_422(client):
    machine_id = create_machine(client)
    response = register(client, machine_id, result="failed", hit_count=1)
    assert response.status_code == 422
    # No record was written despite the valid coordinates.
    body = client.get(export_url(machine_id)).json()
    assert body["privacy_accesses"] == []


def test_registration_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        response = register(first, machine_id, hit_count=4)
        assert response.status_code == 201
        expected_id = response.json()["id"]

    with TestClient(app) as second:
        body = second.get(export_url(machine_id)).json()

    records = body["privacy_accesses"]
    assert len(records) == 1
    assert records[0]["id"] == expected_id
    assert records[0]["hit_count"] == 4


# --------------------------------------------------------------------------- #
# Body validation (422 before the machine lookup)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        None,
        ["accessed_at"],
        "accessed_at",
        make_body(accessed_at="2026-03-01T00:00:01"),       # missing Z
        make_body(accessed_at="2026-03-01T00:00:01+00:00"), # offset form
        make_body(accessed_at="2026-03-01T00:00:01z"),      # lowercase suffix
        make_body(accessed_at=" 2026-03-01T00:00:01Z"),     # leading space
        make_body(accessed_at="2026-03-01T00:00:01Z "),     # trailing space
        make_body(accessed_at="garbage"),
        make_body(accessed_at=""),
        make_body(accessed_at="2026-13-01T00:00:01Z"),      # bad month
        make_body(accessed_at="2026-02-30T00:00:01Z"),      # bad day
        make_body(accessed_at="2026-03-01T24:00:01Z"),      # bad hour
        make_body(accessed_at=123),
        make_body(accessed_at=True),
        make_body(window_start="2026-03-01T00:00:00"),
        make_body(window_end="2026-03-01T00:00:05+00:00"),
        make_body(window_start=T5, window_end=T0),          # inverted window
        make_body(result="ok"),
        make_body(result=""),
        make_body(result="Success"),
        make_body(result=1),
        make_body(hit_count=-1),
        make_body(hit_count=True),
        make_body(hit_count=2.0),
        make_body(hit_count="2"),
        make_body(hit_count=None),
        {"accessed_at": T1, "window_start": T0, "window_end": T5,
         "result": "success"},  # missing hit_count
        {"accessed_at": T1, "window_start": T0, "window_end": T5,
         "hit_count": 1},       # missing result
        {"accessed_at": T1, "window_start": T0, "result": "success",
         "hit_count": 1},       # missing window_end
    ],
)
def test_invalid_body_is_422_before_lookup(client, payload):
    machine_id = create_machine(client)
    response = client.post(access_url(machine_id), json=payload)
    assert response.status_code == 422
    # The validation failure leaves no record behind.
    assert client.get(export_url(machine_id)).json()["privacy_accesses"] == []


def test_invalid_body_is_422_even_for_missing_machine(client):
    response = client.post(access_url(MISSING), json=make_body(result="nope"))
    assert response.status_code == 422
    response = client.post(access_url(MISSING), json={})
    assert response.status_code == 422


def test_invalid_body_against_missing_machine_writes_nothing(client):
    client.post(access_url(MISSING), json=make_body(accessed_at="bad"))
    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM privacy_accesses")
        ).scalar_one()
    assert count == 0


def test_only_post_is_accepted_on_collection(client):
    machine_id = create_machine(client)
    for method in ("get", "put", "patch", "delete"):
        response = getattr(client, method)(access_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Missing machine and duplicate handling
# --------------------------------------------------------------------------- #


def test_valid_body_against_missing_machine_is_404_without_data(client):
    response = client.post(access_url(MISSING), json=make_body())
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_duplicate_access_returns_409_and_writes_nothing(client):
    machine_id = create_machine(client)
    first = register(client, machine_id)
    assert first.status_code == 201

    second = register(client, machine_id)
    assert second.status_code == 409
    assert second.json() == {"error": {"code": "duplicate_access"}}

    records = client.get(export_url(machine_id)).json()["privacy_accesses"]
    assert [r["id"] for r in records] == [first.json()["id"]]


def test_duplicate_detection_ignores_hit_count(client):
    machine_id = create_machine(client)
    assert register(client, machine_id, hit_count=2).status_code == 201
    # Same access time, window, and result but a different reported count is
    # still the same access.
    duplicate = register(client, machine_id, hit_count=99)
    assert duplicate.status_code == 409


def test_different_result_at_same_coordinates_is_distinct(client):
    machine_id = create_machine(client)
    assert register(client, machine_id, result="success", hit_count=1).status_code == 201
    failed = register(client, machine_id, result="failed", hit_count=0)
    assert failed.status_code == 201
    assert len(client.get(export_url(machine_id)).json()["privacy_accesses"]) == 2


def test_same_coordinates_under_different_machines_are_both_registered(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    assert register(client, machine_one).status_code == 201
    assert register(client, machine_two).status_code == 201

    assert (
        len(client.get(export_url(machine_one)).json()["privacy_accesses"]) == 1
    )
    assert (
        len(client.get(export_url(machine_two)).json()["privacy_accesses"]) == 1
    )


# --------------------------------------------------------------------------- #
# Export parameter validation
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
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export{query}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "2026-03-01T00:00:00z",
        " 2026-03-01T00:00:00Z",
        "2026-03-01T00:00:00Z ",
        "2026-03-01 00:00:00Z",
        "garbage",
        "",
        "2026-13-01T00:00:00Z",
        "2026-02-30T00:00:00Z",
        "2026-03-01T24:00:00Z",
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_seconds_and_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2)
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.250Z",
            "2026-03-01T00:00:03.750000Z",
        )
    )
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["privacy_accesses"]] == [rid(1)]

    response = client.get(export_url(machine_id, T2, T2))
    assert [r["id"] for r in response.json()["privacy_accesses"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={T0}&to_accessed_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_or_access_lookup(client):
    bad_time = client.get(
        f"/machines/{MISSING}/privacy-accesses/compliance-export"
        f"?from_accessed_at=nope&to_accessed_at={T5}"
    )
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(
        f"/machines/{MISSING}/privacy-accesses/compliance-export"
        f"?from_accessed_at={T0}&to_accessed_at={T5}&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_access_data(client):
    response = client.get(export_url(MISSING, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_export_only_accepts_get(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope, windowing, ordering, isolation
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_accesses(client):
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


def test_record_exported_with_exact_fields(client):
    machine_id = create_machine(client)
    created = register(
        client,
        machine_id,
        accessed_at=T2,
        window_start=T0,
        window_end=T5,
        result="success",
        hit_count=7,
    ).json()

    exported = client.get(export_url(machine_id)).json()["privacy_accesses"][0]
    assert exported == {
        "id": created["id"],
        "machine_id": machine_id,
        "accessed_at": T2,
        "window_start": T0,
        "window_end": T5,
        "result": "success",
        "hit_count": 7,
    }
    assert set(exported.keys()) == ACCESS_KEYS
    # No responsibility or key material appears anywhere in the response.
    assert "party" not in exported
    assert "role" not in exported
    assert "public_key" not in exported


def test_window_is_closed_on_accessed_at(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T0)
    insert_access_row(client, machine_id, rid(2), T2)
    insert_access_row(client, machine_id, rid(3), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(2)]

    # Equal bounds include the boundary record.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["privacy_accesses"] == []


def test_window_uses_access_time_not_window_bounds(client):
    machine_id = create_machine(client)
    # Accessed at T4, but its desensitized export window covers T0..T5.
    insert_access_row(client, machine_id, rid(1), T4,
                      window_start=T0, window_end=T5)
    # The query window [T0, T2] does not include T4, so the record is out
    # even though the stored export window overlaps the query window.
    body = client.get(export_url(machine_id, T0, T2)).json()
    assert body["privacy_accesses"] == []

    body = client.get(export_url(machine_id, T3, T5)).json()
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
    insert_access_row(client, machine_id, rid(30), T2, result="success",
                      hit_count=1)
    insert_access_row(client, machine_id, rid(20), T2, result="failed",
                      hit_count=0)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(20), rid(30)]


def test_other_machine_accesses_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), T1)
    insert_access_row(client, machine_two, rid(2), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(1)]
    assert all(r["machine_id"] == machine_one
               for r in body["privacy_accesses"])

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [r["id"] for r in body["privacy_accesses"]] == [rid(2)]


def test_failed_access_exported_with_zero_hits(client):
    machine_id = create_machine(client)
    assert register(client, machine_id, accessed_at=T3,
                    result="failed", hit_count=0).status_code == 201
    records = client.get(export_url(machine_id, T0, T5)).json()[
        "privacy_accesses"
    ]
    assert len(records) == 1
    assert records[0]["result"] == "failed"
    assert records[0]["hit_count"] == 0


# --------------------------------------------------------------------------- #
# Read-only, empty database, restart
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    register(client, machine_id, accessed_at=T1, hit_count=3)
    register(client, machine_id, accessed_at=T2, result="failed", hit_count=0)

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


def test_empty_database_works_and_exports_empty(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json()["privacy_accesses"] == []


def test_records_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_access_row(first, machine_id, rid(1), T1, hit_count=5)
        insert_access_row(first, machine_id, rid(2), T3, result="failed",
                          hit_count=0)
        expected = first.get(export_url(machine_id, T0, T5)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.content == expected
    records = response.json()["privacy_accesses"]
    assert [(r["id"], r["result"], r["hit_count"]) for r in records] == [
        (rid(1), "success", 5),
        (rid(2), "failed", 0),
    ]
