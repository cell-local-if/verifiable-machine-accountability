"""Tests for the batch consumption of one-time authorization grants.

    POST /machines/{machine_id}/authorization-grants/batch-consume

The batch entry confirms the use of several grants of one machine in a
single all-or-nothing action: the body is exactly ``{"grant_ids": [...]}``
with 1..100 distinct, blank-free strings in input order; every grant is
checked in input order against exactly the single-consume rules, the first
failing grant decides the whole batch's outcome, and a failure writes
nothing at all. A successful batch flips every grant to ``consumed``,
inserts exactly one use record per grant, and appends one ``consumed``
lifecycle event per grant, all stamped with one shared UTC consume moment.
These tests cover the success shape, every validation and lookup outcome,
the per-item failure precedence, atomicity (no partial consumption),
concurrency against the single consume and against another batch, the
suspended-machine gate, and non-interference with the single-consume
entry.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
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


def issue(client, machine_id, event_id, ttl_seconds=60):
    response = client.post(
        f"/machines/{machine_id}/authorization-grants",
        json={"event_id": event_id, "ttl_seconds": ttl_seconds},
    )
    assert response.status_code == 201
    return response.json()


def batch_consume_url(machine_id):
    return f"/machines/{machine_id}/authorization-grants/batch-consume"


def consume_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/consume"
    )


def revoke_url(machine_id, grant_id):
    return (
        f"/machines/{machine_id}/authorization-grants/{grant_id}/revoke"
    )


def batch_consume(client, machine_id, grant_ids):
    return client.post(
        batch_consume_url(machine_id), json={"grant_ids": grant_ids}
    )


def set_machine_status(client, machine_id, status):
    response = client.post(
        f"/machines/{machine_id}/status", json={"status": status}
    )
    assert response.status_code == 200
    return response


def _parse_z(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture
def granted_machine(client):
    """A machine with three active grants on three allowed events."""
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    grants = []
    for resource in ("res/1", "res/2", "res/3"):
        event = record_event(client, machine_id, resource=resource).json()
        assert event["allowed"] is True
        grants.append(issue(client, machine_id, event["id"]))
    return machine_id, grants


def _grant_row(client, grant_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT status, consumed_at, revoked_at FROM "
                "authorization_grants WHERE id = :id"
            ).bindparams(id=grant_id)
        ).one()


def _use_rows(client, grant_id):
    with client.app.state.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT id, grant_id, machine_id, event_id, consumed_at "
                "FROM authorization_grant_uses WHERE grant_id = :id"
            ).bindparams(id=grant_id)
        ).all()


def _lifecycle_types(client, grant_id):
    with client.app.state.engine.connect() as conn:
        return [
            row[0]
            for row in conn.execute(
                text(
                    "SELECT type FROM authorization_grant_lifecycle_events "
                    "WHERE grant_id = :id ORDER BY occurred_at, id"
                ).bindparams(id=grant_id)
            )
        ]


# --------------------------------------------------------------------------- #
# Success shape
# --------------------------------------------------------------------------- #


def test_batch_consume_success_shape(granted_machine, client):
    machine_id, grants = granted_machine
    grant_ids = [g["id"] for g in grants]

    before = datetime.now(timezone.utc)
    response = batch_consume(client, machine_id, grant_ids)
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    body = response.json()
    assert [item["grant_id"] for item in body] == grant_ids
    assert all(list(item.keys()) == ["grant_id", "use_id", "consumed_at"]
               for item in body)
    # One shared UTC commit moment for the whole batch.
    assert len({item["consumed_at"] for item in body}) == 1
    consumed_at = body[0]["consumed_at"]
    assert consumed_at.endswith("Z")
    assert before <= _parse_z(consumed_at) <= after
    # Every use id is fresh and distinct.
    assert len({item["use_id"] for item in body}) == len(grant_ids)


def test_batch_consume_of_single_grant_matches_single_shape(
    granted_machine, client
):
    machine_id, grants = granted_machine
    response = batch_consume(client, machine_id, [grants[0]["id"]])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert list(body[0].keys()) == ["grant_id", "use_id", "consumed_at"]
    assert body[0]["grant_id"] == grants[0]["id"]


def test_batch_consume_persists_state_uses_and_lifecycle(
    granted_machine, client
):
    machine_id, grants = granted_machine
    events = {g["id"]: g["event_id"] for g in grants}
    body = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    ).json()
    consumed_at = body[0]["consumed_at"]

    for item in body:
        row = _grant_row(client, item["grant_id"])
        assert row.status == "consumed"
        assert row.consumed_at == consumed_at
        assert row.revoked_at is None
        uses = _use_rows(client, item["grant_id"])
        assert len(uses) == 1
        assert uses[0].id == item["use_id"]
        assert uses[0].machine_id == machine_id
        assert uses[0].event_id == events[item["grant_id"]]
        assert uses[0].consumed_at == consumed_at
        assert _lifecycle_types(client, item["grant_id"]) == [
            "issued",
            "consumed",
        ]


def test_batch_consume_accepts_whitespace_padded_ids(granted_machine, client):
    machine_id, grants = granted_machine
    response = batch_consume(
        client, machine_id, [f"  {grants[0]['id']}  "]
    )
    assert response.status_code == 200
    assert response.json()[0]["grant_id"] == grants[0]["id"]


# --------------------------------------------------------------------------- #
# Body and query validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"grant_ids": []},
        {"grant_ids": "x"},
        {"grant_ids": 12},
        {"grant_ids": None},
        {"grant_ids": ["a"], "extra": 1},
        {"ids": ["a"]},
        {"grant_ids": [""]},
        {"grant_ids": ["   "]},
        {"grant_ids": [None]},
        {"grant_ids": [True]},
        {"grant_ids": [12]},
        {"grant_ids": [["a"]]},
        {"grant_ids": [{"id": "a"}]},
        {"grant_ids": ["a", "a"]},
        {"grant_ids": ["a", " b ", "b"]},
    ],
)
def test_invalid_body_is_invalid_grant_request(granted_machine, client, payload):
    machine_id, _ = granted_machine
    response = client.post(batch_consume_url(machine_id), json=payload)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_oversized_batch_is_invalid_grant_request(granted_machine, client):
    machine_id, _ = granted_machine
    response = batch_consume(
        client, machine_id, [f"grant-{i}" for i in range(101)]
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


def test_maximum_batch_size_is_accepted(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    grant_ids = []
    for index in range(100):
        event = record_event(
            client, machine_id, resource=f"res/{index}"
        ).json()
        grant_ids.append(issue(client, machine_id, event["id"])["id"])
    response = batch_consume(client, machine_id, grant_ids)
    assert response.status_code == 200
    assert len(response.json()) == 100


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'"x"', b"12", b"null"])
def test_unparseable_or_typed_body_is_invalid_grant_request(
    granted_machine, client, raw
):
    machine_id, _ = granted_machine
    response = client.post(
        batch_consume_url(machine_id),
        content=raw,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_grant_request"}}


@pytest.mark.parametrize("query", ["?x=1", "?=", "?foo", "?x=1&x=2"])
def test_any_query_parameter_is_invalid_query_before_lookup(
    granted_machine, client, query
):
    machine_id, grants = granted_machine
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.post(
        batch_consume_url(machine_id) + query,
        json={"grant_ids": [grants[0]["id"]]},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    # The query check wins even over a non-existent machine.
    response = client.post(
        batch_consume_url(missing_machine) + query,
        json={"grant_ids": [grants[0]["id"]]},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_validation_runs_before_lookup(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = client.post(
        batch_consume_url(missing_machine), json={"grant_ids": [True]}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_grant_request"


def test_validation_failure_writes_nothing(granted_machine, client):
    machine_id, grants = granted_machine
    response = batch_consume(
        client, machine_id, [grants[0]["id"], grants[0]["id"]]
    )
    assert response.status_code == 422
    assert _grant_row(client, grants[0]["id"]).status == "active"
    assert _use_rows(client, grants[0]["id"]) == []


# --------------------------------------------------------------------------- #
# Lookup outcomes
# --------------------------------------------------------------------------- #


def test_missing_machine_is_not_found(granted_machine, client):
    _, grants = granted_machine
    missing_machine = "00000000-0000-0000-0000-000000000000"
    response = batch_consume(client, missing_machine, [grants[0]["id"]])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_grant_is_not_found(granted_machine, client):
    machine_id, grants = granted_machine
    response = batch_consume(
        client, machine_id, [grants[0]["id"], "no-such-grant"]
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_grant_owned_by_another_machine_is_not_found(granted_machine, client):
    machine_id, grants = granted_machine
    other = create_machine(client, external_id="machine-2")
    response = batch_consume(client, other, [grants[0]["id"]])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The grant is untouched and still consumable by its real owner.
    assert client.post(
        consume_url(machine_id, grants[0]["id"])
    ).status_code == 200


def test_not_found_batch_writes_nothing(granted_machine, client):
    machine_id, grants = granted_machine
    response = batch_consume(
        client, machine_id, [grants[0]["id"], "no-such-grant"]
    )
    assert response.status_code == 404
    # The first, otherwise-consumable grant was not flipped.
    assert _grant_row(client, grants[0]["id"]).status == "active"
    assert _use_rows(client, grants[0]["id"]) == []
    assert _lifecycle_types(client, grants[0]["id"]) == ["issued"]


# --------------------------------------------------------------------------- #
# Per-item failure precedence and atomicity
# --------------------------------------------------------------------------- #


def test_first_consumed_item_decides_and_rolls_back(granted_machine, client):
    machine_id, grants = granted_machine
    assert client.post(
        consume_url(machine_id, grants[1]["id"])
    ).status_code == 200

    response = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}

    # The earlier items in input order were not consumed by the batch.
    assert _grant_row(client, grants[0]["id"]).status == "active"
    assert _use_rows(client, grants[0]["id"]) == []
    assert _grant_row(client, grants[2]["id"]).status == "active"
    # The pre-consumed grant still has exactly its one use record.
    assert len(_use_rows(client, grants[1]["id"])) == 1


def test_first_revoked_item_decides(granted_machine, client):
    machine_id, grants = granted_machine
    assert client.post(
        revoke_url(machine_id, grants[0]["id"])
    ).status_code == 200

    response = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}
    assert _grant_row(client, grants[1]["id"]).status == "active"
    assert _grant_row(client, grants[2]["id"]).status == "active"


def _backdate(client, grant_id):
    past = (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    ).isoformat().replace("+00:00", "Z")
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE authorization_grants SET expires_at = :at "
                "WHERE id = :id"
            ).bindparams(at=past, id=grant_id)
        )


def test_first_expired_item_decides(granted_machine, client):
    machine_id, grants = granted_machine
    _backdate(client, grants[2]["id"])

    response = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}
    assert _grant_row(client, grants[0]["id"]).status == "active"
    assert _grant_row(client, grants[1]["id"]).status == "active"


def test_input_order_selects_the_first_failure(granted_machine, client):
    machine_id, grants = granted_machine
    # Both a revoked and a consumed grant are in the batch; the one
    # earlier in input order decides.
    assert client.post(
        revoke_url(machine_id, grants[0]["id"])
    ).status_code == 200
    assert client.post(
        consume_url(machine_id, grants[1]["id"])
    ).status_code == 200

    response = batch_consume(
        client, machine_id, [grants[1]["id"], grants[0]["id"]]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}

    response = batch_consume(
        client, machine_id, [grants[0]["id"], grants[1]["id"]]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}


def test_terminal_states_keep_precedence_over_expiry(granted_machine, client):
    machine_id, grants = granted_machine
    assert client.post(
        revoke_url(machine_id, grants[0]["id"])
    ).status_code == 200
    _backdate(client, grants[0]["id"])

    response = batch_consume(client, machine_id, [grants[0]["id"]])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}


# --------------------------------------------------------------------------- #
# Suspended machines
# --------------------------------------------------------------------------- #


def test_suspended_machine_batch_is_machine_suspended(granted_machine, client):
    machine_id, grants = granted_machine
    set_machine_status(client, machine_id, "suspended")

    response = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "machine_suspended"}}
    for grant in grants:
        assert _grant_row(client, grant["id"]).status == "active"
        assert _use_rows(client, grant["id"]) == []
        assert _lifecycle_types(client, grant["id"]) == ["issued"]


def test_suspended_machine_still_reports_terminal_states(
    granted_machine, client
):
    machine_id, grants = granted_machine
    assert client.post(
        consume_url(machine_id, grants[0]["id"])
    ).status_code == 200
    assert client.post(
        revoke_url(machine_id, grants[1]["id"])
    ).status_code == 200
    _backdate(client, grants[2]["id"])
    set_machine_status(client, machine_id, "suspended")

    response = batch_consume(client, machine_id, [grants[0]["id"]])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_consumed"}}
    response = batch_consume(client, machine_id, [grants[1]["id"]])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_revoked"}}
    response = batch_consume(client, machine_id, [grants[2]["id"]])
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "grant_expired"}}


def test_batch_consume_resumes_after_reactivation(granted_machine, client):
    machine_id, grants = granted_machine
    set_machine_status(client, machine_id, "suspended")
    rejected = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert rejected.status_code == 409

    set_machine_status(client, machine_id, "active")
    response = batch_consume(
        client, machine_id, [g["id"] for g in grants]
    )
    assert response.status_code == 200
    assert len(response.json()) == 3


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_batch_and_single_consume_have_one_winner(
    granted_machine, client
):
    machine_id, grants = granted_machine
    grant_ids = [g["id"] for g in grants]
    gate = threading.Event()

    def batch_hit():
        gate.wait()
        return batch_consume(client, machine_id, grant_ids)

    def single_hit():
        gate.wait()
        return client.post(consume_url(machine_id, grants[1]["id"]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(batch_hit) for _ in range(4)]
        futures += [pool.submit(single_hit) for _ in range(4)]
        gate.set()
        responses = [f.result() for f in futures]

    successes = [r for r in responses if r.status_code == 200]
    # Either exactly one batch succeeded, or exactly one single consume
    # succeeded (and then every batch fails on the consumed grant).
    batch_wins = [r for r in successes if isinstance(r.json(), list)]
    single_wins = [r for r in successes if isinstance(r.json(), dict)]
    assert len(batch_wins) + len(single_wins) == 1

    for grant_id in grant_ids:
        assert len(_use_rows(client, grant_id)) == (
            1 if (batch_wins or grant_id == grants[1]["id"]) else 0
        )
    if single_wins:
        for response in responses:
            if response.status_code == 409:
                assert response.json() == {
                    "error": {"code": "grant_consumed"}
                }


def test_concurrent_batches_have_exactly_one_winner(granted_machine, client):
    machine_id, grants = granted_machine
    grant_ids = [g["id"] for g in grants]
    gate = threading.Event()

    def hit():
        gate.wait()
        return batch_consume(client, machine_id, grant_ids)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(hit) for _ in range(10)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(200) == 1
    assert statuses.count(409) == 9
    for response in responses:
        if response.status_code == 409:
            assert response.json() == {"error": {"code": "grant_consumed"}}

    for grant_id in grant_ids:
        uses = _use_rows(client, grant_id)
        assert len(uses) == 1
        assert _grant_row(client, grant_id).status == "consumed"


# --------------------------------------------------------------------------- #
# Non-interference with the single-consume entry
# --------------------------------------------------------------------------- #


def test_single_consume_still_works_after_batch(granted_machine, client):
    machine_id, grants = granted_machine
    assert batch_consume(
        client, machine_id, [grants[0]["id"]]
    ).status_code == 200
    # A grant the batch left alone still consumes through the single entry.
    response = client.post(consume_url(machine_id, grants[1]["id"]))
    assert response.status_code == 200
    assert list(response.json().keys()) == ["grant_id", "use_id", "consumed_at"]
    # And the batch-consumed grant reports consumed through the single entry.
    repeat = client.post(consume_url(machine_id, grants[0]["id"]))
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "grant_consumed"}}


def test_repeated_batch_after_success_is_grant_consumed(
    granted_machine, client
):
    machine_id, grants = granted_machine
    grant_ids = [g["id"] for g in grants]
    assert batch_consume(client, machine_id, grant_ids).status_code == 200
    repeat = batch_consume(client, machine_id, grant_ids)
    assert repeat.status_code == 409
    assert repeat.json() == {"error": {"code": "grant_consumed"}}
    # Still exactly one use record per grant.
    for grant in grants:
        assert len(_use_rows(client, grant["id"])) == 1
