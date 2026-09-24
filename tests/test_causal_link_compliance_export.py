"""Tests for the read-only machine-level causal-link compliance export.

Covers `GET /machines/{machine_id}/authorization-decision-events/causal-links/
compliance-export`: closed-UTC-window filtering on the link's own
``created_at`` (independently of its endpoint events), ordering by the actual
UTC instant then record id (exact-second links before fractional-second links
of the same second), verbatim export when a cause/effect event is missing,
misowned, duplicated, or otherwise damaged, the ``bad_time`` /
``invalid_query`` / ``not_found`` outcomes, GET-only routing, strict
read-only byte stability, machine isolation, and persistence across a restart.
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


def export_url(machine_id, from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/authorization-decision-events/causal-links/"
        f"compliance-export?from_created_at={from_created_at}"
        f"&to_created_at={to_created_at}"
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


def allow_read(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )


def record_event(client, machine_id, resource="res/1"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": resource},
    )
    assert response.status_code == 201
    return response.json()


def create_link(client, machine_id, cause_event_id, effect_event_id):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{cause_event_id}/causal-links",
        json={"effect_event_id": effect_event_id},
    )
    assert response.status_code == 201
    return response.json()


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


LINK_KEYS = {"id", "machine_id", "cause_event_id", "effect_event_id", "created_at"}


def insert_link_row(client, link_id, machine_id, cause_event_id, effect_event_id,
                    created_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :machine_id, :cause_event_id, :effect_event_id, "
                ":created_at)"
            ),
            {
                "id": link_id,
                "machine_id": machine_id,
                "cause_event_id": cause_event_id,
                "effect_event_id": effect_event_id,
                "created_at": created_at,
            },
        )


def insert_event_row(client, machine_id, event_id, created_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_events "
                "(id, machine_id, action_type, resource, allowed, reason, "
                "created_at) "
                "VALUES (:id, :machine_id, 'read', 'res/x', 1, "
                "'allowed_by_policy', :created_at)"
            ),
            {"id": event_id, "machine_id": machine_id, "created_at": created_at},
        )


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/causal-links/"
        f"compliance-export{query}"
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
    response = client.get(export_url(machine_id, value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_seconds_are_accepted(client):
    machine_id = create_machine(client)
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T2)
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.250Z",
            "2026-03-01T00:00:03.750000Z",
        )
    )
    assert response.status_code == 200
    assert [link["id"] for link in response.json()["causal_links"]] == [rid(1)]


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [link["id"] for link in response.json()["causal_links"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/causal-links/"
        f"compliance-export?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    base = f"/machines/{missing}/authorization-decision-events/causal-links/compliance-export"

    bad_time = client.get(f"{base}?from_created_at=nope&to_created_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and windowing
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_links(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "causal_links",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == WIDE[0]
    assert body["to_created_at"] == WIDE[1]
    assert body["causal_links"] == []


def test_links_exported_with_list_endpoint_fields(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    cause = record_event(client, machine_id, "res/1")
    effect = record_event(client, machine_id, "res/2")
    link = create_link(client, machine_id, cause["id"], effect["id"])

    body = client.get(export_url(machine_id)).json()
    assert len(body["causal_links"]) == 1
    exported = body["causal_links"][0]
    assert set(exported.keys()) == LINK_KEYS
    assert exported == {
        "id": link["id"],
        "machine_id": machine_id,
        "cause_event_id": cause["id"],
        "effect_event_id": effect["id"],
        "created_at": link["created_at"],
    }


def test_window_is_closed_on_link_created_at(client):
    machine_id = create_machine(client)
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T0)
    insert_link_row(client, rid(2), machine_id, rid(102), rid(103), T2)
    insert_link_row(client, rid(3), machine_id, rid(104), rid(105), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(2)]

    # Equal bounds include the boundary link.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["causal_links"] == []


def test_window_ignores_endpoint_event_timestamps(client):
    """A link is keyed on its own created_at, even when both endpoint events
    were created far outside the window."""
    machine_id = create_machine(client)
    # Events in 2025 and 2027; the link itself lands at T2.
    insert_event_row(client, machine_id, rid(100), "2025-01-01T00:00:00Z")
    insert_event_row(client, machine_id, rid(101), "2027-01-01T00:00:00Z")
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T2)

    body = client.get(export_url(machine_id, T2, T2)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(1)]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional link first.
    insert_link_row(client, rid(2), machine_id, rid(200), rid(201), fractional)
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T0)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(1), rid(2)]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_link_row(client, rid(30), machine_id, rid(300), rid(301), T2)
    insert_link_row(client, rid(20), machine_id, rid(200), rid(201), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(20), rid(30)]


# --------------------------------------------------------------------------- #
# Verbatim export despite missing / misowned / duplicated / damaged endpoints
# --------------------------------------------------------------------------- #


def test_links_with_damaged_endpoints_are_exported_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    dangling_event_id = str(uuid.uuid4())

    # cause missing entirely
    insert_link_row(client, rid(1), machine_one, dangling_event_id, rid(900), T1)
    # effect owned by another machine
    insert_link_row(
        client, rid(2), machine_one, rid(901), foreign_event["id"], T2
    )
    # both endpoints missing, and a self-link (which the integrity audit breaks)
    insert_link_row(client, rid(3), machine_one, rid(902), rid(902), T3)
    # duplicated endpoint: two links share the same (missing) cause event
    insert_link_row(client, rid(4), machine_one, rid(903), rid(904), T4)
    insert_link_row(client, rid(5), machine_one, rid(903), rid(905), T4)

    body = client.get(export_url(machine_one, T0, T5)).json()
    links = body["causal_links"]
    assert [link["id"] for link in links] == [rid(1), rid(2), rid(3), rid(4), rid(5)]
    assert all(link["machine_id"] == machine_one for link in links)
    assert links[0]["cause_event_id"] == dangling_event_id
    assert links[1]["effect_event_id"] == foreign_event["id"]
    assert links[2]["cause_event_id"] == links[2]["effect_event_id"] == rid(902)


def test_other_machine_links_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_link_row(client, rid(1), machine_one, rid(100), rid(101), T1)
    insert_link_row(client, rid(2), machine_two, rid(200), rid(201), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(1)]

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    insert_link_row(client, rid(1), machine_id, rid(100), rid(101), T1)
    insert_link_row(client, rid(2), machine_id, rid(102), rid(103), T2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (
                    "authorization_decision_causal_links",
                    "authorization_decision_events",
                )
            }

    before = table_state()
    first = client.get(export_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(export_url(machine_id, T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_empty_database_needs_no_migration_and_exports_empty(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json()["causal_links"] == []


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_link_row(first, rid(1), machine_id, rid(100), rid(101), T1)
        insert_link_row(first, rid(2), machine_id, rid(102), rid(103), T3)
        expected = first.get(export_url(machine_id, T0, T5)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.content == expected
    assert [link["id"] for link in response.json()["causal_links"]] == [
        rid(1),
        rid(2),
    ]
