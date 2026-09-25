"""Tests for the machine-level fixed-width privacy access bucket summary.

Covers `GET /machines/{machine_id}/privacy-accesses/summary/buckets`:

- fixed 900-second UTC quarter-hour bucketing, half-open ``[start, end)``
  edges aligned to ``:00``/``:15``/``:30``/``:45`` regardless of the request
  window, only non-empty buckets returned in ascending order, with
  ``success_count``/``failed_count``/``matches_count`` totals per bucket and
  machine isolation;
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

T0 = "2026-03-01T00:00:00Z"
T5 = "2026-03-01T00:00:05Z"


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
    window_start=T0,
    window_end=T5,
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


def test_empty_window_returns_empty_buckets(client):
    machine_id = create_machine(client)
    response = client.get(buckets_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "from_accessed_at": T0,
        "to_accessed_at": T5,
        "bucket_width_seconds": 900,
        "summary_buckets": [],
    }


def test_bounds_are_echoed_verbatim_including_fractional_seconds(client):
    machine_id = create_machine(client)
    raw_from = "2026-03-01T00:00:01.250Z"
    raw_to = "2026-03-01T01:00:03.750000Z"
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:14:59.9Z")
    response = client.get(buckets_url(machine_id, raw_from, raw_to))
    assert response.status_code == 200
    body = response.json()
    assert body["from_accessed_at"] == raw_from
    assert body["to_accessed_at"] == raw_to
    assert body["bucket_width_seconds"] == 900


def test_records_are_grouped_into_quarter_hour_buckets(client):
    machine_id = create_machine(client)
    # Bucket [00:00, 00:15): two successes (2, 5 hits), one failed.
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:00:00Z",
                      result="success", matches_count=2)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:07:30.5Z",
                      result="success", matches_count=5)
    insert_access_row(client, machine_id, rid(3), "2026-03-01T00:14:59.999Z",
                      result="failed", matches_count=0)
    # Bucket [00:15, 00:30): one success with 10 hits, edge instant included.
    insert_access_row(client, machine_id, rid(4), "2026-03-01T00:15:00Z",
                      result="success", matches_count=10)
    # Bucket [01:30, 01:45): one failed.
    insert_access_row(client, machine_id, rid(5), "2026-03-01T01:44:59Z",
                      result="failed", matches_count=0)

    body = client.get(buckets_url(machine_id)).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 2,
            "failed_count": 1,
            "matches_count": 7,
        },
        {
            "bucket_start": "2026-03-01T00:15:00Z",
            "bucket_end": "2026-03-01T00:30:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 10,
        },
        {
            "bucket_start": "2026-03-01T01:30:00Z",
            "bucket_end": "2026-03-01T01:45:00Z",
            "success_count": 0,
            "failed_count": 1,
            "matches_count": 0,
        },
    ]


def test_bucket_edges_align_to_all_four_quarter_hours(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:30:00Z")
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:45:00Z")

    body = client.get(buckets_url(machine_id)).json()
    starts = [bucket["bucket_start"] for bucket in body["summary_buckets"]]
    assert starts == [
        "2026-03-01T00:30:00Z",
        "2026-03-01T00:45:00Z",
    ]
    assert body["summary_buckets"][0]["bucket_end"] == "2026-03-01T00:45:00Z"
    assert body["summary_buckets"][1]["bucket_end"] == "2026-03-01T01:00:00Z"


def test_bucket_is_half_open_on_the_edge(client):
    machine_id = create_machine(client)
    # The 00:15:00 edge instant belongs to the [00:15, 00:30) bucket, not the
    # earlier one.
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:14:59Z",
                      matches_count=3)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:15:00Z",
                      matches_count=8)

    body = client.get(buckets_url(machine_id)).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 3,
        },
        {
            "bucket_start": "2026-03-01T00:15:00Z",
            "bucket_end": "2026-03-01T00:30:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 8,
        },
    ]


def test_bucket_edges_are_independent_of_request_window(client):
    machine_id = create_machine(client)
    # A request window whose own bounds sit inside a bucket never shifts the
    # edges.
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:07:00Z",
                      matches_count=4)

    body = client.get(
        buckets_url(machine_id, "2026-03-01T00:05:00Z", "2026-03-01T00:10:00Z")
    ).json()
    assert body["summary_buckets"] == [
        {
            "bucket_start": "2026-03-01T00:00:00Z",
            "bucket_end": "2026-03-01T00:15:00Z",
            "success_count": 1,
            "failed_count": 0,
            "matches_count": 4,
        },
    ]


def test_request_window_is_closed_but_buckets_stay_fixed(client):
    machine_id = create_machine(client)
    # Two records on the two different bucket edges; a closed window equal to
    # the first instant keeps only the first record.
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:15:00Z",
                      matches_count=4)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:30:00Z",
                      matches_count=9)

    body = client.get(
        buckets_url(machine_id, "2026-03-01T00:15:00Z", "2026-03-01T00:15:00Z")
    ).json()
    assert len(body["summary_buckets"]) == 1
    assert body["summary_buckets"][0]["bucket_start"] == "2026-03-01T00:15:00Z"
    assert body["summary_buckets"][0]["matches_count"] == 4


def test_only_buckets_with_records_are_returned(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:44:00Z",
                      matches_count=1)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T02:00:00Z",
                      matches_count=2)

    body = client.get(buckets_url(machine_id)).json()
    assert [b["bucket_start"] for b in body["summary_buckets"]] == [
        "2026-03-01T00:30:00Z",
        "2026-03-01T02:00:00Z",
    ]


def test_buckets_are_ordered_ascending_across_hours(client):
    machine_id = create_machine(client)
    for n, instant in enumerate(
        (
            "2026-03-01T03:14:59Z",
            "2026-03-01T00:00:00Z",
            "2026-03-01T01:59:59Z",
            "2026-03-01T00:00:30Z",
        ),
        start=1,
    ):
        insert_access_row(client, machine_id, rid(n), instant, matches_count=1)

    body = client.get(buckets_url(machine_id)).json()
    starts = [b["bucket_start"] for b in body["summary_buckets"]]
    assert starts == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T01:45:00Z",
        "2026-03-01T03:00:00Z",
    ]


def test_matches_count_is_summed_independently_per_bucket(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:01:00Z",
                      result="success", matches_count=0)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:02:00Z",
                      result="success", matches_count=7)
    insert_access_row(client, machine_id, rid(3), "2026-03-01T00:03:00Z",
                      result="failed", matches_count=0)

    bucket = client.get(buckets_url(machine_id)).json()["summary_buckets"][0]
    assert bucket["success_count"] == 2
    assert bucket["failed_count"] == 1
    assert bucket["matches_count"] == 7


def test_other_machine_accesses_are_never_bucketed(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), "2026-03-01T00:01:00Z",
                      result="success", matches_count=2)
    insert_access_row(client, machine_two, rid(2), "2026-03-01T00:01:00Z",
                      result="success", matches_count=9)
    insert_access_row(client, machine_two, rid(3), "2026-03-01T00:16:00Z",
                      result="failed", matches_count=0)

    one = client.get(buckets_url(machine_one)).json()
    assert len(one["summary_buckets"]) == 1
    assert one["summary_buckets"][0]["success_count"] == 1
    assert one["summary_buckets"][0]["matches_count"] == 2

    two = client.get(buckets_url(machine_two)).json()
    assert [b["bucket_start"] for b in two["summary_buckets"]] == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:15:00Z",
    ]
    assert two["summary_buckets"][0]["matches_count"] == 9
    assert two["summary_buckets"][1]["failed_count"] == 1


def test_bucket_totals_match_summary_totals(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:01:00Z",
                      result="success", matches_count=2)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:20:00Z",
                      result="failed", matches_count=0)
    insert_access_row(client, machine_id, rid(3), "2026-03-01T01:05:00Z",
                      result="success", matches_count=11)

    summary = client.get(
        f"/machines/{machine_id}/privacy-accesses/summary"
        f"?from_accessed_at={WIDE[0]}&to_accessed_at={WIDE[1]}"
    ).json()
    buckets = client.get(buckets_url(machine_id)).json()["summary_buckets"]
    assert sum(b["success_count"] for b in buckets) == summary["success_count"]
    assert sum(b["failed_count"] for b in buckets) == summary["failed_count"]
    assert sum(b["matches_count"] for b in buckets) == summary["matches_count"]


def test_response_key_order(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:01:00Z")
    raw = client.get(buckets_url(machine_id)).content.decode()

    envelope_positions = [
        raw.index('"machine_id"'),
        raw.index('"from_accessed_at"'),
        raw.index('"to_accessed_at"'),
        raw.index('"bucket_width_seconds"'),
        raw.index('"summary_buckets"'),
    ]
    assert envelope_positions == sorted(envelope_positions)

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
        "?to_accessed_at=2026-03-01T00:00:05Z",
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
        f"?from_accessed_at={value}&to_accessed_at={T5}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(buckets_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:15:00Z",
                      matches_count=6)
    response = client.get(
        buckets_url(machine_id, "2026-03-01T00:15:00Z", "2026-03-01T00:15:00Z")
    )
    assert response.status_code == 200
    assert response.json()["summary_buckets"][0]["matches_count"] == 6


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/summary/buckets"
        f"?from_accessed_at={T0}&to_accessed_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = f"/machines/{MISSING_ID}/privacy-accesses/summary/buckets"

    bad_time = client.get(f"{base}?from_accessed_at=nope&to_accessed_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_accessed_at={T0}&to_accessed_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_bucket_data(client):
    response = client.get(buckets_url(MISSING_ID, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"summary_buckets" not in response.content


def test_only_get_is_accepted_on_buckets_path(client):
    machine_id = create_machine(client)
    url = buckets_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_bucket_summary_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), "2026-03-01T00:01:00Z",
                      result="success", matches_count=4)
    insert_access_row(client, machine_id, rid(2), "2026-03-01T00:20:00Z",
                      result="failed", matches_count=0)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM privacy_accesses")))

    before = table_state()
    first = client.get(buckets_url(machine_id, T0, "2026-03-01T01:00:00Z"))
    middle = table_state()
    second = client.get(buckets_url(machine_id, T0, "2026-03-01T01:00:00Z"))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_bucket_summary_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_access_row(first, machine_id, rid(1), "2026-03-01T00:01:00Z",
                          result="success", matches_count=4)
        insert_access_row(first, machine_id, rid(2), "2026-03-01T00:20:00Z",
                          result="failed", matches_count=0)
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
