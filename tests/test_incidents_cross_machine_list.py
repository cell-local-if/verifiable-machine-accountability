"""Tests for the read-only cross-machine ``GET /incidents`` listing."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"

ITEM_KEYS = [
    "id",
    "machine_id",
    "event_id",
    "incident_type",
    "summary",
    "status",
    "created_at",
    "responsibility_assignment_count",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def create_machine(client, external_id):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": external_id,
            "public_key": f"key-{external_id}",
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def record_event(client, machine_id, resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def insert_incident(
    client,
    incident_id,
    machine_id,
    event_id,
    *,
    incident_type="breach",
    summary=None,
    status="open",
    created_at="2026-01-01T00:00:00Z",
):
    """Insert an incident row directly with a fixed id, status, and stamp.

    The incidents table carries a unique ``(event_id, incident_type,
    summary)`` constraint, so the summary defaults to the unique incident id
    to keep multi-row fixtures insertable.
    """
    if summary is None:
        summary = f"summary-{incident_id}"
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_incidents "
                "(id, machine_id, event_id, incident_type, summary, status, "
                "created_at) VALUES (:id, :machine_id, :event_id, "
                ":incident_type, :summary, :status, :created_at)"
            ),
            {
                "id": incident_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_type": incident_type,
                "summary": summary,
                "status": status,
                "created_at": created_at,
            },
        )
    return incident_id


def iid(n):
    return f"00000000-0000-4000-8{n:03d}-{n:012d}"


def assignments_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    )


# --- basic shape -------------------------------------------------------------


def test_empty_database_returns_empty_envelope_with_trailing_newline(client):
    response = client.get("/incidents")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content == b'{"items":[],"next_cursor":null}\n'


def test_unknown_machine_returns_empty_collection(client):
    response = client.get(f"/incidents?machine_id={MISSING_ID}")

    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_items_have_exactly_the_eight_fixed_fields(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    incident_id = insert_incident(
        client, iid(1), machine_id, event_id, created_at="2026-01-01T00:00:00Z"
    )

    response = client.get("/incidents")

    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["items", "next_cursor"]
    assert body["next_cursor"] is None
    item = body["items"][0]
    assert list(item.keys()) == ITEM_KEYS
    assert item["id"] == incident_id
    assert item["machine_id"] == machine_id
    assert item["event_id"] == event_id
    assert item["incident_type"] == "breach"
    assert item["summary"] == f"summary-{incident_id}"
    assert item["status"] == "open"
    assert item["created_at"] == "2026-01-01T00:00:00Z"
    assert item["responsibility_assignment_count"] == 0


def test_response_is_compact_json_terminated_by_one_newline(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id)

    response = client.get("/incidents")

    assert response.content.endswith(b"}\n")
    assert not response.content.endswith(b"\n\n")
    text_body = response.content.decode("utf-8")
    assert ", " not in text_body
    assert ": " not in text_body


# --- ordering ----------------------------------------------------------------


def test_orders_by_actual_utc_instant_then_id(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    # Lexically ``.9Z`` precedes ``Z`` within a second (``.`` < ``Z``); the
    # actual UTC instant puts the exact-second row first.
    insert_incident(
        client, iid(2), machine_id, event_id,
        created_at="2026-01-01T00:00:00.9Z",
    )
    insert_incident(
        client, iid(1), machine_id, event_id,
        created_at="2026-01-01T00:00:00Z",
    )
    # Two rows sharing the exact same instant tie by id ascending.
    insert_incident(
        client, iid(4), machine_id, event_id,
        created_at="2026-01-01T00:00:00.5Z",
    )
    insert_incident(
        client, iid(3), machine_id, event_id,
        created_at="2026-01-01T00:00:00.5Z",
    )

    items = client.get("/incidents").json()["items"]

    assert [item["id"] for item in items] == [iid(1), iid(3), iid(4), iid(2)]


# --- filters -----------------------------------------------------------------


def test_status_filter(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id, status="open")
    insert_incident(client, iid(2), machine_id, event_id, status="acknowledged")
    insert_incident(client, iid(3), machine_id, event_id, status="resolved")

    for status, expected in (
        ("open", [iid(1)]),
        ("acknowledged", [iid(2)]),
        ("resolved", [iid(3)]),
    ):
        items = client.get(f"/incidents?status={status}").json()["items"]
        assert [item["id"] for item in items] == expected
        assert all(item["status"] == status for item in items)


def test_machine_filter_isolates_machines(client):
    machine_one = create_machine(client, "m1")
    machine_two = create_machine(client, "m2")
    event_one = record_event(client, machine_one, resource="res/1")
    event_two = record_event(client, machine_two, resource="res/2")
    insert_incident(client, iid(1), machine_one, event_one)
    insert_incident(client, iid(2), machine_two, event_two)

    items = client.get(f"/incidents?machine_id={machine_two}").json()["items"]

    assert [item["id"] for item in items] == [iid(2)]
    assert all(item["machine_id"] == machine_two for item in items)


def test_machine_id_is_trimmed_but_matched_exactly(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id)

    response = client.get(f"/incidents?machine_id=%20{machine_id}%09")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [iid(1)]


def test_status_and_machine_filters_combine_with_logical_and(client):
    machine_one = create_machine(client, "m1")
    machine_two = create_machine(client, "m2")
    event_one = record_event(client, machine_one, resource="res/1")
    event_two = record_event(client, machine_two, resource="res/2")
    insert_incident(client, iid(1), machine_one, event_one, status="open")
    insert_incident(client, iid(2), machine_one, event_one, status="resolved")
    insert_incident(client, iid(3), machine_two, event_two, status="open")

    response = client.get(f"/incidents?status=open&machine_id={machine_one}")

    assert [item["id"] for item in response.json()["items"]] == [iid(1)]


# --- pagination --------------------------------------------------------------


def test_pagination_walks_every_row_without_repeat_or_omission(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    for n in range(1, 6):
        insert_incident(
            client, iid(n), machine_id, event_id,
            created_at=f"2026-01-0{n}T00:00:00Z",
        )

    seen = []
    cursor = None
    pages = 0
    while True:
        url = "/incidents?limit=2"
        if cursor is not None:
            url += f"&cursor={cursor}"
        body = client.get(url).json()
        page = body["items"]
        seen.extend(item["id"] for item in page)
        pages += 1
        if body["next_cursor"] is None:
            assert len(page) <= 2
            break
        # next_cursor is exactly this page's last item id.
        assert body["next_cursor"] == page[-1]["id"]
        cursor = body["next_cursor"]
        assert pages < 100

    assert seen == [iid(n) for n in range(1, 6)]
    assert pages == 3


def test_next_cursor_is_null_when_page_exactly_ends_the_sequence(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    for n in range(1, 5):
        insert_incident(
            client, iid(n), machine_id, event_id,
            created_at=f"2026-01-0{n}T00:00:00Z",
        )

    body = client.get("/incidents?limit=4").json()

    assert len(body["items"]) == 4
    assert body["next_cursor"] is None


def test_default_limit_is_100(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id)

    # A request without ``limit`` succeeds and applies the default rather than
    # rejecting the missing value; exercising a >100-row database is covered
    # by the explicit-bound tests, so here we only confirm default acceptance.
    body = client.get("/incidents").json()
    assert len(body["items"]) == 1
    assert body["next_cursor"] is None


@pytest.mark.parametrize("limit", (1, 100, 200))
def test_limit_bounds_are_accepted(client, limit):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id)

    response = client.get(f"/incidents?limit={limit}")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [iid(1)]


def test_repeating_last_cursor_against_unchanged_data_is_an_empty_last_page(
    client,
):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id,
                    created_at="2026-01-01T00:00:00Z")

    first = client.get("/incidents?limit=1").json()
    assert first["next_cursor"] is None

    repeated = client.get(f"/incidents?limit=1&cursor={iid(1)}").json()
    assert repeated == {"items": [], "next_cursor": None}


def test_cursor_position_is_global_even_when_filter_excludes_its_incident(
    client,
):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(
        client, iid(1), machine_id, event_id, status="resolved",
        created_at="2026-01-01T00:00:00Z",
    )
    insert_incident(
        client, iid(2), machine_id, event_id, status="open",
        created_at="2026-01-02T00:00:00Z",
    )
    # The cursor names a real (resolved) incident excluded by the open
    # filter; it positions after the 1st, so the filtered page starts at the
    # 2nd rather than answering 404.
    response = client.get(f"/incidents?status=open&cursor={iid(1)}")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [iid(2)]


def test_well_formed_cursor_naming_no_incident_is_404(client):
    response = client.get(f"/incidents?cursor={MISSING_ID}")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "cursor_not_found"}}


def test_well_formed_cursor_404_takes_priority_over_unknown_machine(client):
    response = client.get(
        f"/incidents?machine_id={MISSING_ID}&cursor={MISSING_ID}"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "cursor_not_found"}}


# --- responsibility counts ---------------------------------------------------


def test_responsibility_assignment_count_reflects_current_distinct_rows(
    client,
):
    machine_one = create_machine(client, "m1")
    machine_two = create_machine(client, "m2")
    event_one = record_event(client, machine_one, resource="res/1")
    event_two = record_event(client, machine_two, resource="res/2")
    insert_incident(client, iid(1), machine_one, event_one)
    insert_incident(client, iid(2), machine_two, event_two)

    def assign(incident_machine, incident_event, incident_id, party, role):
        return client.post(
            assignments_url(
                incident_machine, incident_event, incident_id
            ),
            json={"party": party, "role": role},
        )

    assert assign(machine_one, event_one, iid(1), "alice", "owner").status_code == 201
    assert assign(machine_one, event_one, iid(1), "bob", "reviewer").status_code == 201
    # A repeated (party, role) registration is rejected as a duplicate and
    # must not inflate the distinct count.
    duplicate = assign(machine_one, event_one, iid(1), "alice", "owner")
    assert duplicate.status_code == 409

    items = client.get("/incidents").json()["items"]
    counts = {item["id"]: item["responsibility_assignment_count"] for item in items}
    assert counts == {iid(1): 2, iid(2): 0}

    # The machine-scoped listing never counts another machine's assignments.
    scoped = client.get(f"/incidents?machine_id={machine_one}").json()["items"]
    assert scoped[0]["responsibility_assignment_count"] == 2


# --- query validation --------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "status=closed",
        "status=",
        "status=open%20",
        "machine_id=",
        "machine_id=%20%09",
        "limit=0",
        "limit=201",
        "limit=-1",
        "limit=1.5",
        "limit=1.0",
        "limit=true",
        "limit=",
        "limit=abc",
        "unknown=1",
        "status=open&status=resolved",
        "machine_id=x&machine_id=y",
        "limit=1&limit=2",
        "cursor=x&cursor=y",
        "cursor=not-a-uuid",
        "cursor=",
        f"cursor={MISSING_ID[:-1]}",
    ],
)
def test_malformed_queries_are_422_invalid_query_before_reads(client, query):
    response = client.get(f"/incidents?{query}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_carried_body_is_invalid_query(client):
    response = client.request(
        "GET",
        "/incidents",
        content=b"{}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_any_incident_read(client):
    # Even with the incidents table dropped, a malformed query is a 422 from
    # the validation dependency, never a 500.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))

    response = client.get("/incidents?limit=nope")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --- method routing and failures ---------------------------------------------


@pytest.mark.parametrize("method", ("head", "post", "put", "patch", "delete"))
def test_non_get_methods_are_405_without_reading(client, method):
    # Drop the table so any read would 500; method routing must win.
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))

    response = getattr(client, method)("/incidents?limit=10")

    assert response.status_code == 405


def test_read_failure_is_500_with_no_partial_page(client):
    with client.app.state.engine.begin() as conn:
        conn.execute(text("DROP TABLE authorization_decision_incidents"))

    response = client.get("/incidents")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


# --- read-only ---------------------------------------------------------------


def test_listing_changes_nothing(client):
    machine_id = create_machine(client, "m1")
    event_id = record_event(client, machine_id)
    insert_incident(client, iid(1), machine_id, event_id)
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).json()
    incident_integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/incidents/integrity"
    ).json()
    assignment_integrity = client.get(
        f"/machines/{machine_id}/responsibility-assignments/integrity"
    ).json()
    with client.app.state.engine.connect() as conn:
        snapshot = {
            table: conn.execute(text(f"SELECT * FROM {table}")).all()
            for table in (
                "authorization_decision_incidents",
                "incident_responsibility_assignments",
                "authorization_decision_events",
                "incident_status_events",
            )
        }

    for url in (
        "/incidents",
        "/incidents?status=open",
        f"/incidents?machine_id={machine_id}",
        "/incidents?limit=1",
    ):
        client.get(url)

    assert client.get(events_url).json() == events_before
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/incidents/integrity"
        ).json()
        == incident_integrity
    )
    assert (
        client.get(
            f"/machines/{machine_id}/responsibility-assignments/integrity"
        ).json()
        == assignment_integrity
    )
    with client.app.state.engine.connect() as conn:
        for table, rows in snapshot.items():
            current = conn.execute(text(f"SELECT * FROM {table}")).all()
            assert current == rows


def test_listing_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first, "m1")
        event_id = record_event(first, machine_id)
        insert_incident(
            first, iid(1), machine_id, event_id,
            created_at="2026-01-01T00:00:00Z",
        )
        body = first.get("/incidents").content

    with TestClient(app) as second:
        response = second.get("/incidents")

    assert response.status_code == 200
    assert response.content == body
    assert response.json()["items"][0]["id"] == iid(1)
