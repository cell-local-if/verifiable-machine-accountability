"""Tests for the machine-level privacy access aggregate summary.

Covers `GET /machines/{machine_id}/privacy-accesses/summary`:

- success shape: machine id, verbatim time bounds, and the
  ``success_count``/``failed_count``/``matches_count`` aggregates over the
  closed ``accessed_at`` window, computed from the stored values;
- strict query validation (``bad_time`` / ``invalid_query``) before any
  machine or access read, ``404 not_found`` with no summary data, GET-only
  ``405``;
- closed-interval membership (both bounds inclusive, equal bounds allowed),
  strict machine isolation, zero-filled aggregates for an empty window,
  read-only byte stability, and persistence across a restart.
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


def register_access(
    client,
    machine_id,
    *,
    accessed_at=T1,
    window_start=T1,
    window_end=T2,
    result="success",
    matches_count=3,
):
    response = client.post(
        f"/machines/{machine_id}/privacy-accesses",
        json={
            "accessed_at": accessed_at,
            "window_start": window_start,
            "window_end": window_end,
            "result": result,
            "matches_count": matches_count,
        },
    )
    assert response.status_code == 201
    return response.json()


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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


SUMMARY_KEYS = {
    "machine_id",
    "from_accessed_at",
    "to_accessed_at",
    "success_count",
    "failed_count",
    "matches_count",
}


# --------------------------------------------------------------------------- #
# success shape and aggregation
# --------------------------------------------------------------------------- #


def test_summary_aggregates_success_failed_and_matches(client):
    machine_id = create_machine(client)
    register_access(client, machine_id, accessed_at=T1, matches_count=3)
    register_access(client, machine_id, accessed_at=T2, matches_count=5)
    register_access(
        client, machine_id, accessed_at=T3, result="failed", matches_count=0
    )

    response = client.get(summary_url(machine_id))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == SUMMARY_KEYS
    assert body["machine_id"] == machine_id
    assert body["from_accessed_at"] == WIDE[0]
    assert body["to_accessed_at"] == WIDE[1]
    assert body["success_count"] == 2
    assert body["failed_count"] == 1
    assert body["matches_count"] == 8


def test_summary_echoes_bounds_verbatim(client):
    machine_id = create_machine(client)

    body = client.get(
        summary_url(machine_id, "2026-03-01T00:00:01.500Z", T4)
    ).json()

    assert body["from_accessed_at"] == "2026-03-01T00:00:01.500Z"
    assert body["to_accessed_at"] == T4


def test_summary_window_is_closed_on_both_bounds(client):
    machine_id = create_machine(client)
    register_access(client, machine_id, accessed_at=T0, matches_count=1)
    register_access(client, machine_id, accessed_at=T1, matches_count=2)
    register_access(client, machine_id, accessed_at=T2, matches_count=4)
    register_access(client, machine_id, accessed_at=T3, matches_count=8)

    body = client.get(summary_url(machine_id, T1, T2)).json()

    assert body["success_count"] == 2
    assert body["failed_count"] == 0
    assert body["matches_count"] == 6


def test_summary_equal_bounds_allowed(client):
    machine_id = create_machine(client)
    register_access(client, machine_id, accessed_at=T1, matches_count=7)

    response = client.get(summary_url(machine_id, T1, T1))

    assert response.status_code == 200
    body = response.json()
    assert body["success_count"] == 1
    assert body["failed_count"] == 0
    assert body["matches_count"] == 7


def test_summary_compares_actual_instants_across_fractional_seconds(client):
    machine_id = create_machine(client)
    # A fractional-second stamp of the same second is a later instant than
    # the exact-second stamp; both fall inside the closed window.
    register_access(client, machine_id, accessed_at=T1, matches_count=1)
    insert_access_row(
        client, machine_id, rid(1), "2026-03-01T00:00:01.500Z", matches_count=2
    )

    body = client.get(summary_url(machine_id, T1, T1)).json()
    assert body["success_count"] == 1
    assert body["matches_count"] == 1

    body = client.get(summary_url(machine_id, T1, "2026-03-01T00:00:01.500Z")).json()
    assert body["success_count"] == 2
    assert body["matches_count"] == 3


def test_summary_empty_window_returns_zero_counts(client):
    machine_id = create_machine(client)
    register_access(client, machine_id, accessed_at=T1, matches_count=3)

    response = client.get(summary_url(machine_id, T2, T3))

    assert response.status_code == 200
    body = response.json()
    assert set(body) == SUMMARY_KEYS
    assert body["machine_id"] == machine_id
    assert body["from_accessed_at"] == T2
    assert body["to_accessed_at"] == T3
    assert body["success_count"] == 0
    assert body["failed_count"] == 0
    assert body["matches_count"] == 0


def test_summary_machine_with_no_accesses_returns_zero_counts(client):
    machine_id = create_machine(client)

    body = client.get(summary_url(machine_id)).json()

    assert body["success_count"] == 0
    assert body["failed_count"] == 0
    assert body["matches_count"] == 0


def test_summary_never_counts_other_machines(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    register_access(client, machine_one, accessed_at=T1, matches_count=3)
    register_access(
        client, machine_two, accessed_at=T1, result="failed", matches_count=0
    )
    register_access(client, machine_two, accessed_at=T2, matches_count=9)

    body = client.get(summary_url(machine_one)).json()
    assert body["success_count"] == 1
    assert body["failed_count"] == 0
    assert body["matches_count"] == 3

    body = client.get(summary_url(machine_two)).json()
    assert body["success_count"] == 1
    assert body["failed_count"] == 1
    assert body["matches_count"] == 9


def test_summary_counts_stored_values_as_stored(client):
    machine_id = create_machine(client)
    # A result outside success/failed counts in neither bucket, but its
    # stored hit count still joins the matches sum.
    insert_access_row(
        client, machine_id, rid(1), T1, result="other", matches_count=4
    )
    register_access(client, machine_id, accessed_at=T2, matches_count=3)

    body = client.get(summary_url(machine_id)).json()
    assert body["success_count"] == 1
    assert body["failed_count"] == 0
    assert body["matches_count"] == 7


# --------------------------------------------------------------------------- #
# query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",  # both bounds missing
        "?from_accessed_at=2026-03-01T00:00:00Z",  # to missing
        "?to_accessed_at=2026-03-01T00:00:00Z",  # from missing
        "?from_accessed_at=&to_accessed_at=2026-03-01T00:00:00Z",  # blank
        "?from_accessed_at=%20&to_accessed_at=2026-03-01T00:00:00Z",  # whitespace
        "?from_accessed_at=2026-03-01T00:00:00%2B00:00"
        "&to_accessed_at=2026-03-01T00:00:00Z",  # offset form
        "?from_accessed_at=2026-03-01T00:00:00"
        "&to_accessed_at=2026-03-01T00:00:00Z",  # missing Z
        "?from_accessed_at=2026-13-01T00:00:00Z"
        "&to_accessed_at=2026-03-01T00:00:00Z",  # illegal month
        "?from_accessed_at=2026-02-30T00:00:00Z"
        "&to_accessed_at=2026-03-01T00:00:00Z",  # illegal day
        "?from_accessed_at=2026-03-01T24:00:00Z"
        "&to_accessed_at=2026-03-02T00:00:00Z",  # illegal hour
        "?from_accessed_at=2026-03-02T00:00:00Z"
        "&to_accessed_at=2026-03-01T00:00:00Z",  # inverted bounds
    ],
)
def test_summary_bad_time_rejected(client, query):
    machine_id = create_machine(client)

    response = client.get(f"/machines/{machine_id}/privacy-accesses/summary{query}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_summary_unknown_parameter_rejected(client):
    machine_id = create_machine(client)

    response = client.get(f"{summary_url(machine_id)}&extra=1")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_summary_validation_runs_before_machine_lookup(client):
    response = client.get(f"{summary_url(MISSING_ID)}&extra=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        f"/machines/{MISSING_ID}/privacy-accesses/summary"
        "?from_accessed_at=not-a-time&to_accessed_at=2026-03-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_summary_missing_machine_returns_404_without_summary(client):
    response = client.get(summary_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_summary_rejects_non_get_methods(client):
    machine_id = create_machine(client)
    url = summary_url(machine_id)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405, method


# --------------------------------------------------------------------------- #
# read-only behaviour and persistence
# --------------------------------------------------------------------------- #


def test_summary_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    register_access(client, machine_id, accessed_at=T1, matches_count=3)
    register_access(
        client, machine_id, accessed_at=T2, result="failed", matches_count=0
    )

    first = client.get(summary_url(machine_id))
    second = client.get(summary_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content

    # The reads changed nothing: the registered records are intact.
    exported = client.get(
        f"/machines/{machine_id}/privacy-accesses/compliance-export"
        f"?from_accessed_at={WIDE[0]}&to_accessed_at={WIDE[1]}"
    ).json()
    assert len(exported["privacy_accesses"]) == 2


def test_summary_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as first:
        machine_id = create_machine(first)
        register_access(first, machine_id, accessed_at=T1, matches_count=3)
        register_access(
            first, machine_id, accessed_at=T2, result="failed", matches_count=0
        )
        expected = first.get(summary_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(summary_url(machine_id))
        assert response.status_code == 200
        assert response.content == expected
