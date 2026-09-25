"""Tests for the machine-level privacy access fixed-bucket summary.

Covers `GET /machines/{machine_id}/privacy-accesses/summary/buckets`:

- 900-second UTC quarter-hour buckets, left-closed right-open membership,
  per-bucket success/failed/matches totals with ``matches_count`` summed
  independently, only non-empty buckets in ascending order, and the exact
  response key order;
- strict query validation (``bad_time`` / ``invalid_query``) completed before
  any machine or access read, ``404 not_found`` with no bucket data for a
  machine missing after validation, GET-only ``405``;
- read-only byte stability and persistence across a restart.
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

# Quarter-hour aligned instants and instants inside the buckets around them.
B0 = "2026-03-01T00:00:00Z"   # bucket [00:00:00, 00:15:00)
B0_IN = "2026-03-01T00:07:30Z"
B0_FRAC = "2026-03-01T00:14:59.999Z"
B1 = "2026-03-01T00:15:00Z"   # bucket [00:15:00, 00:30:00)
B1_IN = "2026-03-01T00:15:00.500Z"
B2 = "2026-03-01T00:30:00Z"   # bucket [00:30:00, 00:45:00)
B3 = "2026-03-01T00:45:00Z"   # bucket [00:45:00, 01:00:00)


def buckets_url(machine_id, from_accessed_at=WIDE[0], to_accessed_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/privacy-accesses/summary/buckets"
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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_access_row(
    client,
    machine_id,
    access_id,
    accessed_at,
    *,
    window_start=B0,
    window_end=B1,
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


# --------------------------------------------------------------------------- #
# Response shape and bucketing
# --------------------------------------------------------------------------- #


def test_empty_window_returns_complete_envelope_with_no_buckets(client):
    machine_id = create_machine(client)
    response = client.get(buckets_url(machine_id, B0, B3))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "from_accessed_at": B0,
        "to_accessed_at": B3,
        "bucket_width_seconds": 900,
        "summary_buckets": [],
    }


def test_bounds_are_echoed_verbatim_including_fractional_seconds(client):
    machine_id = create_machine(client)
    raw_from = "2026-03-01T00:00:01.250Z"
    raw_to = "2026-03-01T00:44:03.750000Z"
    body = client.get(buckets_url(machine_id, raw_from, raw_to)).json()
    assert body["from_accessed_at"] == raw_from
    assert body["to_accessed_at"] == raw_to


def test_records_group_into_quarter_hour_buckets_in_order(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0_IN, result="success",
                      matches_count=2)
    insert_access_row(client, machine_id, rid(2), B0_FRAC, result="failed",
                      matches_count=0)
    insert_access_row(client, machine_id, rid(3), B2, result="success",
                      matches_count=5)
    insert_access_row(client, machine_id, rid(4), B1_IN, result="success",
                      matches_count=3)

    body = client.get(buckets_url(machine_id)).json()
    assert body["bucket_width_seconds"] == 900
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 1,
            "failed_count": 1,
            "matches_count": 2,
        },
        {
            "bucket_start": "2026-03-01T00:15:00Z",
            "bucket_end": "2026-03-01T00:30:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 3,
        },
        {
            "bucket_start": "2026-03-01T00:30:00Z",
            "bucket_end": "2026-03-01T00:45:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 5,
        },
    ]


def test_bucket_interval_is_left_closed_right_open(client):
    machine_id = create_machine(client)
    # Exactly on a boundary belongs to the bucket starting there, never to
    # the one ending there.
    insert_access_row(client, machine_id, rid(1), B1, result="success",
                      matches_count=4)

    body = client.get(buckets_url(machine_id)).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:15:00Z",
            "bucket_end": "2026-03-01T00:30:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 4,
        }
    ]


def test_request_window_stays_closed_while_buckets_are_half_open(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0, result="success",
                      matches_count=1)
    insert_access_row(client, machine_id, rid(2), B1, result="failed",
                      matches_count=0)
    insert_access_row(client, machine_id, rid(3), B2, result="success",
                      matches_count=7)

    # The closed request window includes the records on both bounds; each
    # still lands in its own half-open bucket.
    body = client.get(buckets_url(machine_id, B0, B2)).json()
    assert [bucket["bucket_start"] for bucket in body["summary_buckets"]] == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:15:00Z",
        "2026-03-01T00:30:00Z",
    ]

    # Narrowing the window past a bound drops that bound's bucket entirely.
    body = client.get(buckets_url(machine_id, B0, B1)).json()
    assert [bucket["bucket_start"] for bucket in body["summary_buckets"]] == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:15:00Z",
    ]


def test_matches_total_is_an_independent_sum_within_each_bucket(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0, result="success",
                      matches_count=0)
    insert_access_row(client, machine_id, rid(2), B0_IN, result="success",
                      matches_count=7)
    insert_access_row(client, machine_id, rid(3), B0_FRAC, result="failed",
                      matches_count=0)

    body = client.get(buckets_url(machine_id)).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 2,
            "failed_count": 1,
            "matches_count": 7,
        }
    ]


def test_buckets_spanning_midnight_align_to_utc_quarter_hours(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T23:59:59Z",
                      result="success", matches_count=1)
    insert_access_row(client, machine_id, rid(2), "2026-03-02T00:00:00Z",
                      result="success", matches_count=2)

    body = client.get(buckets_url(machine_id)).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T23:45:00Z",
            "bucket_end": "2026-03-02T00:00:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 1,
        },
        {
            "bucket_start": "2026-03-02T00:00:00Z",
            "bucket_end": "2026-03-02T00:15:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 2,
        },
    ]


def test_other_machine_accesses_never_enter_buckets(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), B0_IN, result="success",
                      matches_count=2)
    insert_access_row(client, machine_two, rid(2), B0_IN, result="success",
                      matches_count=9)
    insert_access_row(client, machine_two, rid(3), B1, result="failed",
                      matches_count=0)

    one = client.get(buckets_url(machine_one)).json()
    assert one["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 2,
        }
    ]

    two = client.get(buckets_url(machine_two)).json()
    assert [bucket["bucket_start"] for bucket in two["summary_buckets"]] == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:15:00Z",
    ]


def test_response_key_order_is_machine_bounds_width_then_buckets(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0_IN, matches_count=3)
    raw = client.get(buckets_url(machine_id)).content.decode()

    positions = [
        raw.index('"machine_id"'),
        raw.index('"from_accessed_at"'),
        raw.index('"to_accessed_at"'),
        raw.index('"bucket_width_seconds"'),
        raw.index('"summary_buckets"'),
    ]
    assert positions == sorted(positions)

    bucket_positions = [
        raw.index('"bucket_start"'),
        raw.index('"bucket_end"'),
        raw.index('"success_count"'),
        raw.index('"failed_count"'),
        raw.index('"matches_count"'),
    ]
    assert bucket_positions == sorted(bucket_positions)


# --------------------------------------------------------------------------- #
# Query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_accessed_at=2026-03-01T00:00:00Z",
        "?to_accessed_at=2026-03-01T00:45:00Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/summary/buckets{query}"
    )
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
        f"/machines/{machine_id}/privacy-accesses/summary/buckets"
        f"?from_accessed_at={value}&to_accessed_at={B3}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(buckets_url(machine_id, B3, B0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0_IN, matches_count=6)
    response = client.get(buckets_url(machine_id, B0_IN, B0_IN))
    assert response.status_code == 200
    assert response.json()["summary_buckets"][0]["matches_count"] == 6


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/summary/buckets"
        f"?from_accessed_at={B0}&to_accessed_at={B3}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = f"/machines/{MISSING_ID}/privacy-accesses/summary/buckets"

    bad_time = client.get(f"{base}?from_accessed_at=nope&to_accessed_at={B3}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_accessed_at={B0}&to_accessed_at={B3}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_bucket_data(client):
    response = client.get(buckets_url(MISSING_ID, B0, B3))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted_on_buckets_path(client):
    machine_id = create_machine(client)
    url = buckets_url(machine_id, B0, B3)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_buckets_are_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), B0_IN, result="success",
                      matches_count=4)
    insert_access_row(client, machine_id, rid(2), B1, result="failed",
                      matches_count=0)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM privacy_accesses")))

    before = table_state()
    first = client.get(buckets_url(machine_id))
    middle = table_state()
    second = client.get(buckets_url(machine_id))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_buckets_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_access_row(first, machine_id, rid(1), B0_IN, result="success",
                          matches_count=4)
        insert_access_row(first, machine_id, rid(2), B1, result="failed",
                          matches_count=0)
        expected = first.get(buckets_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(buckets_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    assert response.json() == {
        "machine_id": machine_id,
        "from_accessed_at": WIDE[0],
        "to_accessed_at": WIDE[1],
        "bucket_width_seconds": 900,
        "summary_buckets": [
            {
                "bucket_start": "2026-03-01T00:00:00Z",
                "bucket_end": "2026-03-01T00:15:00Z",
                "success_count": 1,
                "failed_count": 0,
                "matches_count": 4,
            },
            {
                "bucket_start": "2026-03-01T00:15:00Z",
                "bucket_end": "2026-03-01T00:30:00Z",
                "success_count": 0,
                "failed_count": 1,
                "matches_count": 0,
            },
        ],
    }
