"""Tests for the read-only authorization grant compliance export.

The export entry is::

    GET /machines/{machine_id}/authorization-grants/compliance-export

It is strictly read-only: it never creates, repairs, updates, or deletes a
grant, its consumption record, a lifecycle event, an authorization decision,
a decision basis, or any chain hash. These tests cover the fixed six-field
response shape and per-item view (issuance fields, derived
``active``/``expired``/``consumed``/``revoked`` status, ``use`` or ``null``,
and the lifecycle events with their chain fields), the left-closed
right-open ``start_at``/``end_at`` window with both defaults covering
everything, ordering by the actual UTC instant of ``issued_at`` then id,
keyset pagination that never repeats or omits an item, strict machine
isolation, every 422 validation case (``invalid_query`` / ``invalid_range``
/ ``invalid_cursor``) and its priority over the machine lookup, 404 for a
missing machine, method routing, verbatim emission of damaged stored values,
and non-interference with issue/consume/revoke and the lifecycle chain.
"""
from datetime import datetime, timedelta, timezone

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


def create_rule(client, action_type="read", resource_pattern="res/*",
                effect="allow", priority=0):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )
    assert response.status_code == 201
    return response.json()


def declare(client, machine_id, action_type="read", resource_pattern="res/*",
            enabled=True):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201
    return response.json()


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def export_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/compliance-export"


def issue(client, machine_id, event_id, ttl_seconds=300):
    return client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )


def consume(client, machine_id, grant_id):
    return client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke(client, machine_id, grant_id):
    return client.post(
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def allowed_machine(client, external_id="machine-1"):
    """A machine with an enabled declaration and an allow rule."""
    machine_id = create_machine(client, external_id=external_id)
    declare(client, machine_id)
    # Policy rules are global: a second machine's setup reuses the allow
    # rule the first one created.
    response = client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )
    assert response.status_code in (201, 409)
    return machine_id


def issue_grants(client, machine_id, count, ttl_seconds=300):
    grants = []
    for index in range(count):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        response = issue(client, machine_id, event["id"], ttl_seconds)
        assert response.status_code == 201
        grants.append(response.json())
    return grants


def set_issued_at(client, grant_id, issued_at):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET issued_at = :at WHERE id = :id"
            ).bindparams(at=issued_at, id=grant_id)
        )


def backdate_expires_at(client, grant_id, seconds=10):
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant_id)
        )
    return past


def fetch_all_pages(client, machine_id, limit=1, query=""):
    """Walk the export to its end, returning the collected items."""
    collected = []
    cursor = None
    seen_cursors = set()
    while True:
        separator = "&" if query else "?"
        url = export_url(machine_id) + query + f"{separator}limit={limit}"
        if cursor is not None:
            url += f"&cursor={cursor}"
        response = client.get(url)
        assert response.status_code == 200
        body = response.json()
        collected.extend(body["items"])
        assert body["has_more"] is (body["next_cursor"] is not None)
        cursor = body["next_cursor"]
        if cursor is None:
            break
        assert cursor not in seen_cursors
        seen_cursors.add(cursor)
    return collected


MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"

RESPONSE_FIELDS = [
    "machine_id",
    "start_at",
    "end_at",
    "items",
    "next_cursor",
    "has_more",
]

ITEM_FIELDS = [
    "id",
    "machine_id",
    "event_id",
    "issued_at",
    "expires_at",
    "status",
    "use",
    "lifecycle_events",
]

EVENT_FIELDS = [
    "id",
    "machine_id",
    "grant_id",
    "authorization_event_id",
    "type",
    "occurred_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
]


# --------------------------------------------------------------------------- #
# Empty export and basic shape
# --------------------------------------------------------------------------- #


def test_machine_without_grants_exports_empty_page(client):
    machine_id = allowed_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "start_at": None,
        "end_at": None,
        "items": [],
        "next_cursor": None,
        "has_more": False,
    }


def test_response_field_order_and_compact_body(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert list(response.json()) == RESPONSE_FIELDS
    assert response.text.endswith("\n")
    assert "\n" not in response.text[:-1]
    assert '", "' not in response.text


def test_item_shape_for_active_grant(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    (item,) = response.json()["items"]
    assert list(item) == ITEM_FIELDS
    assert item["id"] == grant["id"]
    assert item["machine_id"] == machine_id
    assert item["event_id"] == grant["event_id"]
    assert item["issued_at"] == grant["issued_at"]
    assert item["expires_at"] == grant["expires_at"]
    assert item["status"] == "active"
    assert item["use"] is None
    # The issue action appended exactly one lifecycle event, carrying its
    # content fields and the chain fields.
    assert len(item["lifecycle_events"]) == 1
    event = item["lifecycle_events"][0]
    assert list(event) == EVENT_FIELDS
    assert event["grant_id"] == grant["id"]
    assert event["authorization_event_id"] == grant["event_id"]
    assert event["type"] == "issued"
    assert event["occurred_at"] == grant["issued_at"]
    assert event["previous_event_id"] is None
    assert len(event["content_hash"]) == 64
    assert len(event["chain_hash"]) == 64


def test_bounds_echo_null_when_omitted_and_verbatim_when_given(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(
        export_url(machine_id)
        + "?start_at=2020-01-01T00:00:00Z&end_at=2999-01-01T00:00:00Z"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["start_at"] == "2020-01-01T00:00:00Z"
    assert body["end_at"] == "2999-01-01T00:00:00Z"
    assert len(body["items"]) == 1


# --------------------------------------------------------------------------- #
# Derived status and use record
# --------------------------------------------------------------------------- #


def test_consumed_grant_keeps_terminal_status_and_carries_use(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    outcome = consume(client, machine_id, grant["id"])
    assert outcome.status_code == 200
    use = outcome.json()

    (item,) = client.get(export_url(machine_id)).json()["items"]
    assert item["status"] == "consumed"
    assert item["use"] == {
        "use_id": use["use_id"],
        "consumed_at": use["consumed_at"],
    }
    assert [event["type"] for event in item["lifecycle_events"]] == [
        "issued",
        "consumed",
    ]


def test_revoked_grant_keeps_terminal_status_and_null_use(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    outcome = revoke(client, machine_id, grant["id"])
    assert outcome.status_code == 200

    (item,) = client.get(export_url(machine_id)).json()["items"]
    assert item["status"] == "revoked"
    assert item["use"] is None
    assert [event["type"] for event in item["lifecycle_events"]] == [
        "issued",
        "revoked",
    ]


def test_expired_grant_presents_expired_without_write_back(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    past = backdate_expires_at(client, grant["id"])

    (item,) = client.get(export_url(machine_id)).json()["items"]
    assert item["status"] == "expired"
    assert item["expires_at"] == past
    # The derived status is never persisted.
    with client.app.state.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT status FROM authorization_grants WHERE id = :id")
            .bindparams(id=grant["id"])
        ).scalar_one()
    assert stored == "active"


def test_terminal_status_survives_elapsed_ttl(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    assert consume(client, machine_id, grant["id"]).status_code == 200
    backdate_expires_at(client, grant["id"])

    (item,) = client.get(export_url(machine_id)).json()["items"]
    assert item["status"] == "consumed"


# --------------------------------------------------------------------------- #
# Window filtering
# --------------------------------------------------------------------------- #


def test_window_is_left_closed_right_open(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    for index, stamp in enumerate(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
    ):
        set_issued_at(client, grants[index]["id"], stamp)

    # start_at includes its exact instant; end_at excludes its own.
    response = client.get(
        export_url(machine_id)
        + "?start_at=2026-01-02T00:00:00Z&end_at=2026-01-03T00:00:00Z"
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["id"] for item in items] == [grants[1]["id"]]


def test_start_at_alone_and_end_at_alone(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    for index, stamp in enumerate(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
    ):
        set_issued_at(client, grants[index]["id"], stamp)

    body = client.get(
        export_url(machine_id) + "?start_at=2026-01-02T00:00:00Z"
    ).json()
    assert [item["id"] for item in body["items"]] == [
        grants[1]["id"],
        grants[2]["id"],
    ]

    body = client.get(
        export_url(machine_id) + "?end_at=2026-01-03T00:00:00Z"
    ).json()
    assert [item["id"] for item in body["items"]] == [
        grants[0]["id"],
        grants[1]["id"],
    ]


def test_fractional_second_bound_applies_to_the_actual_instant(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    set_issued_at(client, grant["id"], "2026-01-01T00:00:00.500000Z")

    body = client.get(
        export_url(machine_id) + "?end_at=2026-01-01T00:00:00Z"
    ).json()
    assert body["items"] == []
    body = client.get(
        export_url(machine_id) + "?end_at=2026-01-01T00:00:00.500001Z"
    ).json()
    assert len(body["items"]) == 1


# --------------------------------------------------------------------------- #
# Ordering and pagination
# --------------------------------------------------------------------------- #


def test_items_order_by_issued_instant_then_id(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    # An exact-second stamp sorts before a fractional-second stamp of the
    # same second even though the ISO text orders them the other way.
    set_issued_at(client, grants[0]["id"], "2026-01-01T00:00:00.500000Z")
    set_issued_at(client, grants[1]["id"], "2026-01-01T00:00:00Z")
    set_issued_at(client, grants[2]["id"], "2025-12-31T23:59:59Z")

    items = client.get(export_url(machine_id)).json()["items"]
    assert [item["id"] for item in items] == [
        grants[2]["id"],
        grants[1]["id"],
        grants[0]["id"],
    ]


def test_pagination_never_repeats_or_omits(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 5)
    expected = sorted(
        (grant["id"] for grant in grants),
        key=lambda grant_id: next(
            g["issued_at"] for g in grants if g["id"] == grant_id
        ),
    )
    collected = fetch_all_pages(client, machine_id, limit=2)
    assert [item["id"] for item in collected] == [
        grant["id"] for grant in grants
    ]
    assert len({item["id"] for item in collected}) == 5
    assert sorted(item["id"] for item in collected) == sorted(expected)


def test_last_page_has_null_next_cursor_and_no_has_more(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 2)
    body = client.get(export_url(machine_id) + "?limit=2").json()
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert len(body["items"]) == 2


def test_cursor_points_after_last_item_of_page(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 3)
    first = client.get(export_url(machine_id) + "?limit=1").json()
    assert first["has_more"] is True
    assert first["items"][0]["id"] == grants[0]["id"]
    second = client.get(
        export_url(machine_id) + f"?limit=1&cursor={first['next_cursor']}"
    ).json()
    assert second["items"][0]["id"] == grants[1]["id"]


def test_default_limit_is_fifty(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 55)
    body = client.get(export_url(machine_id)).json()
    assert len(body["items"]) == 50
    assert body["has_more"] is True


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?unknown=1",
        "?limit=1&limit=2",
        "?cursor=a&cursor=b",
        "?start_at=2026-01-01T00:00:00Z&start_at=2026-01-02T00:00:00Z",
        "?start_at=not-a-time",
        "?start_at=2026-01-01T00:00:00",  # missing Z
        "?start_at=2026-01-01T00:00:00+00:00",  # offset form
        "?start_at=2026-13-01T00:00:00Z",  # out-of-range month
        "?end_at=2026-01-01",  # date only
        "?limit=",
        "?limit=0",
        "?limit=101",
        "?limit=-1",
        "?limit=1.0",
        "?limit=true",
        "?limit=abc",
        "?limit=+1",
        "?cursor=",
        "?cursor=only-one-segment",
        "?cursor=|id",
        "?cursor=stamp|",
    ],
)
def test_malformed_query_is_invalid_query(client, query):
    machine_id = allowed_machine(client)
    response = client.get(export_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "query",
    [
        "?start_at=2026-01-02T00:00:00Z&end_at=2026-01-01T00:00:00Z",
        # Equal bounds: start not earlier than end.
        "?start_at=2026-01-01T00:00:00Z&end_at=2026-01-01T00:00:00Z",
    ],
)
def test_inverted_or_empty_range_is_invalid_range(client, query):
    machine_id = allowed_machine(client)
    response = client.get(export_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_range"}}


def test_unlocatable_cursor_is_invalid_cursor(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 1)
    response = client.get(
        export_url(machine_id) + "?cursor=2026-01-01T00:00:00Z|no-such-grant"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_of_other_machine_is_invalid_cursor(client):
    machine_one = allowed_machine(client, "machine-1")
    machine_two = allowed_machine(client, "machine-2")
    issue_grants(client, machine_one, 2)
    issue_grants(client, machine_two, 2)

    first = client.get(export_url(machine_two) + "?limit=1").json()
    response = client.get(
        export_url(machine_one) + f"?cursor={first['next_cursor']}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_query_validation_precedes_machine_lookup(client):
    response = client.get(export_url(MISSING_MACHINE) + "?limit=0")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.get(export_url(MISSING_MACHINE) + "?cursor=x|y")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_cursor_error_precedes_machine_lookup(client):
    response = client.get(
        export_url(MISSING_MACHINE) + "?cursor=2026-01-01T00:00:00Z|missing"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_cursor"}}


def test_missing_machine_is_not_found(client):
    response = client.get(export_url(MISSING_MACHINE))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_body_is_invalid_query(client):
    machine_id = allowed_machine(client)
    response = client.request("GET", export_url(machine_id), content=b"{}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_are_405(client, method):
    machine_id = allowed_machine(client)
    response = getattr(client, method)(export_url(machine_id))
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Isolation, damage tolerance, and read-only behavior
# --------------------------------------------------------------------------- #


def test_export_is_machine_isolated(client):
    machine_one = allowed_machine(client, "machine-1")
    machine_two = allowed_machine(client, "machine-2")
    (grant_one,) = issue_grants(client, machine_one, 1)
    issue_grants(client, machine_two, 2)

    body = client.get(export_url(machine_one)).json()
    assert [item["id"] for item in body["items"]] == [grant_one["id"]]
    # The other machine's lifecycle events never leak into the item.
    for event in body["items"][0]["lifecycle_events"]:
        assert event["machine_id"] == machine_one


def test_unparseable_issued_at_sorts_last_and_is_emitted_verbatim(client):
    machine_id = allowed_machine(client)
    grants = issue_grants(client, machine_id, 2)
    set_issued_at(client, grants[0]["id"], "not-a-timestamp")

    body = client.get(export_url(machine_id)).json()
    assert [item["id"] for item in body["items"]] == [
        grants[1]["id"],
        grants[0]["id"],
    ]
    assert body["items"][1]["issued_at"] == "not-a-timestamp"
    # A damaged stamp falls outside any finite window.
    body = client.get(
        export_url(machine_id) + "?end_at=2999-01-01T00:00:00Z"
    ).json()
    assert [item["id"] for item in body["items"]] == [grants[1]["id"]]


def test_contradictory_state_is_reported_not_corrected(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    # Damage: terminal status without its use record.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET status = 'consumed' "
                "WHERE id = :id"
            ).bindparams(id=grant["id"])
        )
    (item,) = client.get(export_url(machine_id)).json()["items"]
    assert item["status"] == "consumed"
    assert item["use"] is None


def test_export_does_not_modify_any_records(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    assert consume(client, machine_id, grant["id"]).status_code == 200

    def snapshot():
        with client.app.state.engine.connect() as conn:
            return {
                table: conn.execute(text(f"SELECT * FROM {table} ORDER BY id"))
                .mappings()
                .all()
                for table in (
                    "authorization_grants",
                    "authorization_grant_uses",
                    "authorization_grant_lifecycle_events",
                )
            }

    before = snapshot()
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert snapshot() == before


def test_repeat_calls_are_byte_identical(client):
    machine_id = allowed_machine(client)
    issue_grants(client, machine_id, 2)
    first = client.get(export_url(machine_id))
    second = client.get(export_url(machine_id))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_existing_grant_endpoints_keep_their_behavior(client):
    machine_id = allowed_machine(client)
    (grant,) = issue_grants(client, machine_id, 1)
    # The export does not consume, revoke, or otherwise disturb the grant.
    client.get(export_url(machine_id))
    outcome = consume(client, machine_id, grant["id"])
    assert outcome.status_code == 200
    # A second consumption still reports the terminal conflict.
    again = consume(client, machine_id, grant["id"])
    assert again.status_code == 409
    assert again.json() == {"error": {"code": "grant_consumed"}}
    # The lifecycle chain still verifies.
    integrity = client.get(
        f"/machines/{machine_id}/authorization-grant-lifecycle-events/integrity"
    )
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
