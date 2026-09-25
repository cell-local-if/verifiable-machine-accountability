"""Tests for the machine-level privacy access aggregate summary.

Covers `GET /machines/{machine_id}/privacy-accesses/summary`:

- success/failure/matches totals over the closed ``accessed_at`` window, with
  ``matches_count`` summed independently of the result, an empty window still
  carrying all three zero totals, strict machine isolation, and the exact
  response key order (``success_count``, ``failed_count``,
  ``matches_count``);
- strict query validation (``bad_time`` / ``invalid_query``) completed before
  any machine or access read, ``404 not_found`` with no summary data for a
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
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def summary_url(machine_id, from_accessed_at=WIDE[0], to_accessed_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/privacy-accesses/summary"
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


# --------------------------------------------------------------------------- #
# Response shape and totals
# --------------------------------------------------------------------------- #


def test_empty_window_returns_complete_zero_summary(client):
    machine_id = create_machine(client)
    response = client.get(summary_url(machine_id, T0, T5))
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "machine_id": machine_id,
        "from_accessed_at": T0,
        "to_accessed_at": T5,
        "success_count": 0,
        "failed_count": 0,
        "matches_count": 0,
    }


def test_bounds_are_echoed_verbatim_including_fractional_seconds(client):
    machine_id = create_machine(client)
    raw_from = "2026-03-01T00:00:01.250Z"
    raw_to = "2026-03-01T00:00:03.750000Z"
    response = client.get(summary_url(machine_id, raw_from, raw_to))
    assert response.status_code == 200
    body = response.json()
    assert body["from_accessed_at"] == raw_from
    assert body["to_accessed_at"] == raw_to


def test_totals_count_only_in_window_results_and_sum_matches(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T0, result="success",
                      matches_count=2)
    insert_access_row(client, machine_id, rid(2), T1, result="success",
                      matches_count=5)
    insert_access_row(client, machine_id, rid(3), T2, result="failed",
                      matches_count=0)
    insert_access_row(client, machine_id, rid(4), T4, result="success",
                      matches_count=10)
    insert_access_row(client, machine_id, rid(5), T5, result="failed",
                      matches_count=0)

    body = client.get(summary_url(machine_id, T1, T4)).json()
    # T1, T2 and T4 are inside the closed window; T0 and T5 stay out.
    assert body["success_count"] == 2
    assert body["failed_count"] == 1
    assert body["matches_count"] == 15


def test_matches_total_is_an_independent_sum(client):
    machine_id = create_machine(client)
    # A success may itself hit zero records; it still counts as a success while
    # contributing nothing to the hit total.
    insert_access_row(client, machine_id, rid(1), T1, result="success",
                      matches_count=0)
    insert_access_row(client, machine_id, rid(2), T2, result="success",
                      matches_count=7)
    insert_access_row(client, machine_id, rid(3), T3, result="failed",
                      matches_count=0)

    body = client.get(summary_url(machine_id, T0, T5)).json()
    assert body["success_count"] == 2
    assert body["failed_count"] == 1
    assert body["matches_count"] == 7


def test_window_is_closed_on_accessed_at(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2, result="success",
                      matches_count=4)
    insert_access_row(client, machine_id, rid(2), T4, result="failed",
                      matches_count=0)

    # Records exactly on the bounds are included.
    body = client.get(summary_url(machine_id, T2, T2)).json()
    assert body == {
        "machine_id": machine_id,
        "from_accessed_at": T2,
        "to_accessed_at": T2,
        "success_count": 1,
        "failed_count": 0,
        "matches_count": 4,
    }

    # A window between accesses totals nothing but keeps the full envelope.
    body = client.get(summary_url(machine_id, T3, T3)).json()
    assert body["success_count"] == 0
    assert body["failed_count"] == 0
    assert body["matches_count"] == 0


def test_window_follows_access_time_not_recorded_window(client):
    machine_id = create_machine(client)
    # Accessed at T4 while the recorded desensitized window is T0..T1.
    insert_access_row(
        client, machine_id, rid(1), T4, window_start=T0, window_end=T1,
        result="success", matches_count=3,
    )
    assert client.get(summary_url(machine_id, T0, T1)).json()[
        "success_count"
    ] == 0
    body = client.get(summary_url(machine_id, T4, T4)).json()
    assert body["success_count"] == 1
    assert body["matches_count"] == 3


def test_other_machine_accesses_are_never_totaled(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_access_row(client, machine_one, rid(1), T1, result="success",
                      matches_count=2)
    insert_access_row(client, machine_two, rid(2), T1, result="success",
                      matches_count=9)
    insert_access_row(client, machine_two, rid(3), T2, result="failed",
                      matches_count=0)

    one = client.get(summary_url(machine_one, T0, T5)).json()
    assert one["success_count"] == 1
    assert one["failed_count"] == 0
    assert one["matches_count"] == 2

    two = client.get(summary_url(machine_two, T0, T5)).json()
    assert two["success_count"] == 1
    assert two["failed_count"] == 1
    assert two["matches_count"] == 9


def test_response_key_order_is_machine_bounds_then_three_counts(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T1, matches_count=3)
    raw = client.get(summary_url(machine_id, T0, T5)).content.decode()

    positions = [
        raw.index('"machine_id"'),
        raw.index('"from_accessed_at"'),
        raw.index('"to_accessed_at"'),
        raw.index('"success_count"'),
        raw.index('"failed_count"'),
        raw.index('"matches_count"'),
    ]
    assert positions == sorted(positions)


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
        f"/machines/{machine_id}/privacy-accesses/summary{query}"
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
        f"/machines/{machine_id}/privacy-accesses/summary"
        f"?from_accessed_at={value}&to_accessed_at={T5}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(summary_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T2, matches_count=6)
    response = client.get(summary_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert response.json()["matches_count"] == 6


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/privacy-accesses/summary"
        f"?from_accessed_at={T0}&to_accessed_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_runs_before_machine_lookup(client):
    base = f"/machines/{MISSING_ID}/privacy-accesses/summary"

    bad_time = client.get(f"{base}?from_accessed_at=nope&to_accessed_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_accessed_at={T0}&to_accessed_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_summary_data(client):
    response = client.get(summary_url(MISSING_ID, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted_on_summary_path(client):
    machine_id = create_machine(client)
    url = summary_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_summary_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_access_row(client, machine_id, rid(1), T1, result="success",
                      matches_count=4)
    insert_access_row(client, machine_id, rid(2), T3, result="failed",
                      matches_count=0)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM privacy_accesses")))

    before = table_state()
    first = client.get(summary_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(summary_url(machine_id, T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_summary_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_access_row(first, machine_id, rid(1), T1, result="success",
                          matches_count=4)
        insert_access_row(first, machine_id, rid(2), T3, result="failed",
                          matches_count=0)
        expected = first.get(summary_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(summary_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    assert response.json() == {
        "machine_id": machine_id,
        "from_accessed_at": WIDE[0],
        "to_accessed_at": WIDE[1],
        "success_count": 1,
        "failed_count": 1,
        "matches_count": 4,
    }
