"""Tests for the read-only cross-machine incident listing.

The entry is::

    GET /incidents

It is strictly read-only: it never creates, updates, closes, backfills, or
deletes an incident, a responsibility assignment, an event, an evidence
record, a hash chain, or an export. These tests cover the fixed
``{items, next_cursor}`` envelope and eight-field item shape, the derived
``responsibility_assignment_count``, the ``status``/``machine_id`` filters
combining with logical AND, ordering by the actual UTC instant of
``created_at`` then id, keyset pagination by incident id, every 422
``invalid_query`` validation case, 404 ``cursor_not_found`` for a
well-shaped cursor naming no stored incident, method routing, the compact
newline-terminated body, and non-interference with the existing incident
and responsibility endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def record_event(client, machine_id, resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def incidents_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        "/incidents"
    )


def create_incident(client, machine_id, event_id, summary="something happened"):
    response = client.post(
        incidents_url(machine_id, event_id),
        json={"incident_type": "breach", "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def assign(client, machine_id, event_id, incident_id, party="team-a", role="owner"):
    response = client.post(
        f"{incidents_url(machine_id, event_id)}/{incident_id}"
        "/responsibility-assignments",
        json={"party": party, "role": role},
    )
    assert response.status_code == 201
    return response.json()


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"{incidents_url(machine_id, event_id)}/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def set_created_at(client, incident_id, created_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_decision_incidents "
                "SET created_at = :at WHERE id = :id"
            ).bindparams(at=created_at, id=incident_id)
        )


def fetch_all_pages(client, query="", limit=1):
    """Walk the listing to its end, returning the collected items."""
    collected = []
    cursor = None
    separator = "&" if query else ""
    while True:
        page_query = f"?limit={limit}{separator}{query}"
        if cursor is not None:
            page_query += f"&cursor={cursor}"
        body = client.get(f"/incidents{page_query}").json()
        collected.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return collected


# --------------------------------------------------------------------------- #
# Envelope and item shape
# --------------------------------------------------------------------------- #


def test_empty_database_lists_empty_page(client):
    response = client.get("/incidents")
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


def test_item_shape_and_envelope_key_order(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    response = client.get("/incidents")
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["items", "next_cursor"]
    assert body["next_cursor"] is None
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert list(item.keys()) == [
        "id",
        "machine_id",
        "event_id",
        "incident_type",
        "summary",
        "status",
        "created_at",
        "responsibility_assignment_count",
    ]
    assert item["id"] == incident["id"]
    assert item["machine_id"] == machine_id
    assert item["event_id"] == event_id
    assert item["incident_type"] == "breach"
    assert item["summary"] == "something happened"
    assert item["status"] == "open"
    assert item["created_at"] == incident["created_at"]
    assert item["responsibility_assignment_count"] == 0


def test_body_is_compact_utf8_json_with_single_trailing_newline(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id, summary="héllo")

    response = client.get("/incidents")
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert "héllo".encode("utf-8") in raw
    assert b", " not in raw
    assert b'": ' not in raw


def test_repeat_calls_are_byte_identical(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    first = client.get("/incidents").content
    second = client.get("/incidents").content
    assert first == second


# --------------------------------------------------------------------------- #
# Cross-machine scope and responsibility counts
# --------------------------------------------------------------------------- #


def test_lists_incidents_across_machines(client):
    machine_a = create_machine(client, external_id="machine-a")
    machine_b = create_machine(client, external_id="machine-b")
    event_a = record_event(client, machine_a)
    event_b = record_event(client, machine_b)
    first = create_incident(client, machine_a, event_a, summary="on a")
    second = create_incident(client, machine_b, event_b, summary="on b")

    items = client.get("/incidents").json()["items"]
    assert [item["id"] for item in items] == [first["id"], second["id"]]
    assert {item["machine_id"] for item in items} == {machine_a, machine_b}


def test_responsibility_assignment_count_is_current_deduplicated_count(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    other_event = record_event(client, machine_id, resource="res/y")
    with_assignments = create_incident(client, machine_id, event_id)
    without_assignments = create_incident(
        client, machine_id, other_event, summary="other"
    )
    assign(client, machine_id, event_id, with_assignments["id"])
    assign(
        client,
        machine_id,
        event_id,
        with_assignments["id"],
        party="team-b",
        role="reviewer",
    )

    items = {item["id"]: item for item in client.get("/incidents").json()["items"]}
    assert items[with_assignments["id"]]["responsibility_assignment_count"] == 2
    assert items[without_assignments["id"]]["responsibility_assignment_count"] == 0


def test_count_does_not_leak_across_incidents_or_machines(client):
    machine_a = create_machine(client, external_id="machine-a")
    machine_b = create_machine(client, external_id="machine-b")
    event_a = record_event(client, machine_a)
    event_b = record_event(client, machine_b)
    incident_a = create_incident(client, machine_a, event_a)
    incident_b = create_incident(client, machine_b, event_b)
    assign(client, machine_a, event_a, incident_a["id"])

    items = {item["id"]: item for item in client.get("/incidents").json()["items"]}
    assert items[incident_a["id"]]["responsibility_assignment_count"] == 1
    assert items[incident_b["id"]]["responsibility_assignment_count"] == 0


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_status_filter_selects_only_matching_incidents(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{i}") for i in range(3)]
    open_incident = create_incident(client, machine_id, events[0], summary="open")
    acknowledged = create_incident(client, machine_id, events[1], summary="ack")
    resolved = create_incident(client, machine_id, events[2], summary="resolved")
    transition(client, machine_id, events[1], acknowledged["id"], "acknowledged")
    transition(client, machine_id, events[2], resolved["id"], "acknowledged")
    transition(client, machine_id, events[2], resolved["id"], "resolved")

    body = client.get("/incidents?status=open").json()
    assert [item["id"] for item in body["items"]] == [open_incident["id"]]
    body = client.get("/incidents?status=acknowledged").json()
    assert [item["id"] for item in body["items"]] == [acknowledged["id"]]
    body = client.get("/incidents?status=resolved").json()
    assert [item["id"] for item in body["items"]] == [resolved["id"]]


def test_machine_id_filter_is_exact_and_isolated(client):
    machine_a = create_machine(client, external_id="machine-a")
    machine_b = create_machine(client, external_id="machine-b")
    event_a = record_event(client, machine_a)
    event_b = record_event(client, machine_b)
    incident_a = create_incident(client, machine_a, event_a)
    create_incident(client, machine_b, event_b)

    body = client.get(f"/incidents?machine_id={machine_a}").json()
    assert [item["id"] for item in body["items"]] == [incident_a["id"]]
    assert body["next_cursor"] is None


def test_machine_id_filter_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)

    body = client.get("/incidents", params={"machine_id": f"  {machine_id}\t"}).json()
    assert [item["id"] for item in body["items"]] == [incident["id"]]


def test_unknown_machine_filter_returns_empty_page(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    body = client.get(f"/incidents?machine_id={MISSING_ID}").json()
    assert body == {"items": [], "next_cursor": None}


def test_status_and_machine_id_filters_combine_with_logical_and(client):
    machine_a = create_machine(client, external_id="machine-a")
    machine_b = create_machine(client, external_id="machine-b")
    event_a = record_event(client, machine_a)
    event_b = record_event(client, machine_b)
    incident_a = create_incident(client, machine_a, event_a)
    incident_b = create_incident(client, machine_b, event_b)
    transition(client, machine_b, event_b, incident_b["id"], "acknowledged")

    # The acknowledged incident lives on machine b: each single-filter match
    # is excluded by the other filter.
    body = client.get(
        f"/incidents?status=acknowledged&machine_id={machine_a}"
    ).json()
    assert body == {"items": [], "next_cursor": None}
    body = client.get(f"/incidents?status=open&machine_id={machine_b}").json()
    assert body == {"items": [], "next_cursor": None}
    body = client.get(
        f"/incidents?status=acknowledged&machine_id={machine_b}"
    ).json()
    assert [item["id"] for item in body["items"]] == [incident_b["id"]]
    body = client.get(f"/incidents?status=open&machine_id={machine_a}").json()
    assert [item["id"] for item in body["items"]] == [incident_a["id"]]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_ordering_uses_created_at_utc_instant_then_id(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{i}") for i in range(3)]
    incidents = [
        create_incident(client, machine_id, event, summary=f"s{i}")
        for i, event in enumerate(events)
    ]
    ids = sorted(incident["id"] for incident in incidents)

    # Same second: the incident with the larger id gets the exact-second
    # stamp and the one with the smaller id gets a later fractional stamp.
    # Naive text ordering would reverse both; instant ordering must win.
    set_created_at(client, ids[2], "2026-01-01T00:00:00Z")
    set_created_at(client, ids[1], "2026-01-01T00:00:00.5Z")
    set_created_at(client, ids[0], "2026-01-01T00:00:01Z")

    items = client.get("/incidents").json()["items"]
    assert [item["id"] for item in items] == [ids[2], ids[1], ids[0]]
    # Stamps are emitted verbatim.
    assert [item["created_at"] for item in items] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00.5Z",
        "2026-01-01T00:00:01Z",
    ]


def test_same_instant_ties_break_by_id_ascending(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{i}") for i in range(3)]
    incidents = [
        create_incident(client, machine_id, event, summary=f"s{i}")
        for i, event in enumerate(events)
    ]
    stamp = "2026-02-02T08:09:10.25Z"
    for incident in incidents:
        set_created_at(client, incident["id"], stamp)

    items = client.get("/incidents").json()["items"]
    assert [item["id"] for item in items] == sorted(
        incident["id"] for incident in incidents
    )
    assert all(item["created_at"] == stamp for item in items)


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


def test_default_limit_is_100(client):
    machine_id = create_machine(client)
    for index in range(101):
        event_id = record_event(client, machine_id, resource=f"res/{index}")
        create_incident(client, machine_id, event_id, summary=f"s{index}")

    first = client.get("/incidents").json()
    assert len(first["items"]) == 100
    assert first["next_cursor"] == first["items"][-1]["id"]

    second = client.get(f"/incidents?cursor={first['next_cursor']}").json()
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None


@pytest.mark.parametrize("page_limit", [1, 2, 3, 7])
def test_pagination_never_repeats_or_omits(client, page_limit):
    machine_id = create_machine(client)
    for index in range(7):
        event_id = record_event(client, machine_id, resource=f"res/{index}")
        create_incident(client, machine_id, event_id, summary=f"s{index}")
    expected = [item["id"] for item in client.get("/incidents").json()["items"]]

    collected = fetch_all_pages(client, limit=page_limit)
    assert [item["id"] for item in collected] == expected
    assert len({item["id"] for item in collected}) == len(expected)


def test_next_cursor_is_last_item_id_only_when_more_follow(client):
    machine_id = create_machine(client)
    for index in range(3):
        event_id = record_event(client, machine_id, resource=f"res/{index}")
        create_incident(client, machine_id, event_id, summary=f"s{index}")
    expected = [item["id"] for item in client.get("/incidents").json()["items"]]

    first = client.get("/incidents?limit=2").json()
    assert [item["id"] for item in first["items"]] == expected[:2]
    assert first["next_cursor"] == expected[1]

    second = client.get(f"/incidents?limit=2&cursor={first['next_cursor']}").json()
    assert [item["id"] for item in second["items"]] == expected[2:]
    assert second["next_cursor"] is None


def test_pagination_applies_within_the_filtered_set(client):
    machine_a = create_machine(client, external_id="machine-a")
    machine_b = create_machine(client, external_id="machine-b")
    for index in range(4):
        event_id = record_event(client, machine_a, resource=f"res/{index}")
        create_incident(client, machine_a, event_id, summary=f"a{index}")
    event_b = record_event(client, machine_b)
    create_incident(client, machine_b, event_b, summary="on b")

    expected = [
        item["id"]
        for item in client.get(f"/incidents?machine_id={machine_a}").json()["items"]
    ]
    assert len(expected) == 4
    collected = fetch_all_pages(client, query=f"machine_id={machine_a}", limit=2)
    assert [item["id"] for item in collected] == expected


def test_cursor_positions_by_incident_even_when_filters_exclude_it(client):
    machine_id = create_machine(client)
    events = [record_event(client, machine_id, resource=f"res/{i}") for i in range(3)]
    incidents = [
        create_incident(client, machine_id, event, summary=f"s{i}")
        for i, event in enumerate(events)
    ]
    # The cursor names the first incident, since moved out of the filter;
    # the position still resolves and the page continues strictly after it.
    transition(client, machine_id, events[0], incidents[0]["id"], "acknowledged")

    body = client.get(
        f"/incidents?status=open&cursor={incidents[0]['id']}"
    ).json()
    assert [item["id"] for item in body["items"]] == [
        incidents[1]["id"],
        incidents[2]["id"],
    ]
    assert body["next_cursor"] is None


# --------------------------------------------------------------------------- #
# 422 invalid_query
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=201",
        "limit=-1",
        "limit=1.5",
        "limit=true",
        "limit=abc",
        "limit=",
        "limit=+5",
        "limit=%205",
        "status=closed",
        "status=OPEN",
        "status=",
        "machine_id=",
        "machine_id=%20%20",
        "cursor=",
        "cursor=%20%20",
        "unknown=1",
        "limit=1&limit=2",
        "status=open&status=resolved",
        "machine_id=a&machine_id=b",
        "cursor=a&cursor=b",
    ],
)
def test_invalid_query_shape_is_422(client, query):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    response = client.get(f"/incidents?{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_carried_body_is_422_invalid_query(client):
    response = client.request("GET", "/incidents", content=b"{}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.request(
        "GET", "/incidents?limit=1", content=b"anything"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("limit", ["1", "100", "200"])
def test_limit_boundaries_are_accepted(client, limit):
    response = client.get(f"/incidents?limit={limit}")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# 404 cursor_not_found
# --------------------------------------------------------------------------- #


def test_well_shaped_cursor_naming_no_incident_is_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    response = client.get(f"/incidents?cursor={MISSING_ID}")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "cursor_not_found"}}


def test_cursor_not_found_also_under_filters(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    create_incident(client, machine_id, event_id)

    response = client.get(f"/incidents?status=open&cursor={MISSING_ID}")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "cursor_not_found"}}


# --------------------------------------------------------------------------- #
# Method routing
# --------------------------------------------------------------------------- #


def test_non_get_methods_are_405(client):
    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)("/incidents")
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Read-only guarantees
# --------------------------------------------------------------------------- #


def test_listing_writes_nothing_and_leaves_existing_entries_unchanged(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident = create_incident(client, machine_id, event_id)
    assign(client, machine_id, event_id, incident["id"])

    client.get("/incidents")
    client.get("/incidents?status=open&limit=1")

    # The incident and its responsibility records are exactly as before.
    listed = client.get(incidents_url(machine_id, event_id)).json()
    assert [record["id"] for record in listed] == [incident["id"]]
    assert listed[0]["status"] == "open"
    integrity = client.get(
        f"/machines/{machine_id}/responsibility-assignments/integrity"
    ).json()
    assert integrity == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
    incident_integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/incidents/integrity"
    ).json()
    assert incident_integrity["valid"] is True
    assert incident_integrity["checked_count"] == 1
