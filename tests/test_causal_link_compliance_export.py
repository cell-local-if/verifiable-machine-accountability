"""Tests for the read-only machine-level causal-link compliance export.

Covers `GET /machines/{machine_id}/causal-links/compliance-export`: the
closed-UTC-window applied to the links' own `created_at`, ordering by the
actual UTC instant then link id (exact-second links before fractional-second
links of the same second), verbatim export when an endpoint event is missing,
misowned, duplicated, or otherwise damaged, the `bad_time` / `invalid_query` /
`not_found` outcomes, GET-only access, machine isolation by stored
`machine_id`, strict read-only byte stability, and persistence across a
restart.
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
        f"/machines/{machine_id}/causal-links/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
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


def lid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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
            {
                "id": event_id,
                "machine_id": machine_id,
                "created_at": created_at,
            },
        )


def insert_link_row(client, machine_id, link_id, cause_event_id, effect_event_id,
                    created_at):
    """Insert a causal link row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO authorization_decision_causal_links "
                "(id, machine_id, cause_event_id, effect_event_id, created_at) "
                "VALUES (:id, :machine_id, :cause, :effect, :created_at)"
            ),
            {
                "id": link_id,
                "machine_id": machine_id,
                "cause": cause_event_id,
                "effect": effect_event_id,
                "created_at": created_at,
            },
        )


LINK_KEYS = {"id", "machine_id", "cause_event_id", "effect_event_id", "created_at"}


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
        f"/machines/{machine_id}/causal-links/compliance-export{query}"
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
    insert_event_row(client, machine_id, lid(100), T0)
    insert_link_row(client, machine_id, lid(1), lid(100), lid(100),
                    "2026-03-01T00:00:02.250000Z")
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:02.0Z",
            "2026-03-01T00:00:02.999999999Z",
        )
    )
    assert response.status_code == 200
    assert [link["id"] for link in response.json()["causal_links"]] == [lid(1)]


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, lid(100), T0)
    insert_link_row(client, machine_id, lid(1), lid(100), lid(100), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [link["id"] for link in response.json()["causal_links"]] == [lid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/causal-links/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    bad_time = client.get(
        f"/machines/{missing}/causal-links/compliance-export"
        f"?from_created_at=nope&to_created_at={T5}"
    )
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(
        f"/machines/{missing}/causal-links/compliance-export"
        f"?from_created_at={T0}&to_created_at={T5}&x=1"
    )
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_only_get_is_allowed(client):
    machine_id = create_machine(client)
    response = client.post(export_url(machine_id), json={})
    assert response.status_code == 405


def test_missing_machine_returns_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Envelope shape, windowing, and ordering
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
    cause = record_event(client, machine_id, "res/a")
    effect = record_event(client, machine_id, "res/b")
    link = create_link(client, machine_id, cause["id"], effect["id"])

    body = client.get(export_url(machine_id)).json()
    exported = body["causal_links"][0]
    assert set(exported.keys()) == LINK_KEYS
    assert exported == {
        "id": link["id"],
        "machine_id": machine_id,
        "cause_event_id": cause["id"],
        "effect_event_id": effect["id"],
        "created_at": link["created_at"],
    }


def test_window_is_closed_on_the_links_own_created_at(client):
    machine_id = create_machine(client)
    # All endpoint events sit at T0, far outside the [T2, T3] window; a link
    # is included or excluded solely by its own created_at, regardless of the
    # endpoint events' timestamps.
    for n in range(100, 105):
        insert_event_row(client, machine_id, lid(n), T0)
    pairs = [
        (lid(100), lid(101)),
        (lid(101), lid(102)),
        (lid(102), lid(103)),
        (lid(103), lid(104)),
    ]
    for link_n, ((cause, effect), at) in enumerate(zip(pairs, [T1, T2, T3, T4]),
                                                    start=1):
        insert_link_row(client, machine_id, lid(link_n), cause, effect, at)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(2), lid(3)]

    # Equal bounds include the boundary link.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(4)]


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, lid(100), T0)
    insert_event_row(client, machine_id, lid(101), T0)
    fractional = "2026-03-01T00:00:02.500000Z"
    # Insert so lexicographic order would put the fractional link first.
    insert_link_row(client, machine_id, lid(2), lid(101), lid(100), fractional)
    insert_link_row(client, machine_id, lid(1), lid(100), lid(101), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(1), lid(2)]


def test_same_instant_tie_breaks_by_link_id(client):
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, lid(100), T0)
    insert_event_row(client, machine_id, lid(101), T0)
    insert_link_row(client, machine_id, lid(30), lid(101), lid(100), T2)
    insert_link_row(client, machine_id, lid(20), lid(100), lid(101), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(20), lid(30)]


# --------------------------------------------------------------------------- #
# Verbatim export despite endpoint damage; machine isolation by stored value
# --------------------------------------------------------------------------- #


def test_dangling_and_misowned_endpoints_are_exported_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    allow_read(client, machine_two)
    foreign_event = record_event(client, machine_two)
    dangling_event_id = str(uuid.uuid4())

    insert_link_row(client, machine_one, lid(1), dangling_event_id,
                    dangling_event_id, T1)
    insert_link_row(client, machine_one, lid(2), dangling_event_id,
                    foreign_event["id"], T2)
    insert_link_row(client, machine_one, lid(3), foreign_event["id"],
                    dangling_event_id, T3)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(1), lid(2), lid(3)]
    assert body["causal_links"][0]["cause_event_id"] == dangling_event_id
    assert body["causal_links"][0]["effect_event_id"] == dangling_event_id
    assert body["causal_links"][1]["effect_event_id"] == foreign_event["id"]
    assert all(link["machine_id"] == machine_one
               for link in body["causal_links"])


def test_corrupt_rows_are_exported_verbatim(client):
    """Export never repairs, dedupes, or normalizes stored rows.

    Rows the create endpoint would reject outright — a self-loop, a dangling
    endpoint, and an empty cause — are written straight to the table and must
    come back byte-for-byte rather than being filtered or "fixed".
    """
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, lid(100), T0)
    dangling = str(uuid.uuid4())
    insert_link_row(client, machine_id, lid(1), lid(100), lid(100), T1)
    insert_link_row(client, machine_id, lid(2), dangling, dangling, T2)
    insert_link_row(client, machine_id, lid(3), "", lid(100), T3)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(1), lid(2), lid(3)]
    assert body["causal_links"][0]["cause_event_id"] == lid(100)
    assert body["causal_links"][0]["effect_event_id"] == lid(100)
    assert body["causal_links"][1]["cause_event_id"] == dangling
    assert body["causal_links"][2]["cause_event_id"] == ""


def test_other_machine_links_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_event_row(client, machine_one, lid(100), T0)
    insert_event_row(client, machine_two, lid(200), T0)
    # Even a link owned by machine_two that points at machine_one's event must
    # never appear in machine_one's slice: ownership is the stored machine_id.
    insert_link_row(client, machine_one, lid(1), lid(100), lid(100), T1)
    insert_link_row(client, machine_two, lid(2), lid(100), lid(200), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [link["id"] for link in body["causal_links"]] == [lid(1)]
    assert all(link["machine_id"] == machine_one
               for link in body["causal_links"])


def test_event_window_export_rule_is_unchanged(client):
    """The existing event-window export still gates links on both endpoints
    being events in that export's event set, independent of the new slice."""
    machine_id = create_machine(client)
    insert_event_row(client, machine_id, lid(100), T0)
    # Link created at T2, endpoint event at T0. The event window [T2, T3]
    # contains no events, so its causal_links must stay empty even though the
    # new link-only export over the same window returns the link.
    insert_link_row(client, machine_id, lid(1), lid(100), lid(100), T2)

    windowed = client.get(
        f"/machines/{machine_id}/authorization-decision-events/compliance-export"
        f"?from_created_at={T2}&to_created_at={T3}"
    ).json()
    assert windowed["events"] == []
    assert windowed["causal_links"] == []

    link_only = client.get(export_url(machine_id, T2, T3)).json()
    assert [link["id"] for link in link_only["causal_links"]] == [lid(1)]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    allow_read(client, machine_id)
    cause = record_event(client, machine_id, "res/a")
    effect = record_event(client, machine_id, "res/b")
    create_link(client, machine_id, cause["id"], effect["id"])

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in (
                    "authorization_decision_events",
                    "authorization_decision_causal_links",
                )
            }

    before = table_state()
    first = client.get(export_url(machine_id))
    middle = table_state()
    second = client.get(export_url(machine_id))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_empty_database_exports_empty_array(client):
    machine_id = create_machine(client)
    body = client.get(export_url(machine_id)).json()
    assert body["causal_links"] == []


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_event_row(first, machine_id, lid(100), T0)
        insert_link_row(first, machine_id, lid(1), lid(100), lid(100), T2)
        expected = first.get(export_url(machine_id, T0, T5)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T5))

    assert response.status_code == 200
    assert response.content == expected
    assert [link["id"] for link in response.json()["causal_links"]] == [lid(1)]
