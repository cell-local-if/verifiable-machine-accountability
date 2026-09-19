import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "machine_id",
    "action_type",
    "resource",
    "allowed",
    "reason",
    "created_at",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def canonical_content_hash(event: dict) -> str:
    document = json.dumps(
        {key: event[key] for key in CONTENT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


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


def allow_read(client):
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )


def declare_read(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )


def record(client, machine_id, resource="res/x", action_type="read"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def integrity(client, machine_id):
    return client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )


@pytest.fixture
def machine_with_policy(client):
    machine_id = create_machine(client)
    declare_read(client, machine_id)
    allow_read(client)
    return machine_id


def test_created_event_contains_chain_fields(machine_with_policy, client):
    event = record(client, machine_with_policy, resource="res/a").json()

    assert event["previous_event_id"] is None
    assert HEX64_RE.match(event["content_hash"])
    assert HEX64_RE.match(event["chain_hash"])


def test_content_hash_is_canonical_sha256(machine_with_policy, client):
    event = record(client, machine_with_policy, resource="res/a").json()
    assert event["content_hash"] == canonical_content_hash(event)


def test_unicode_and_key_ordering_do_not_affect_agreement(machine_with_policy, client):
    response = record(client, machine_with_policy, resource="res/数据/✓")
    assert response.status_code == 201
    event = response.json()
    assert event["content_hash"] == canonical_content_hash(event)


def test_first_event_chain_hash_uses_empty_previous(machine_with_policy, client):
    event = record(client, machine_with_policy, resource="res/a").json()
    assert event["chain_hash"] == chain_hash("", event["content_hash"])


def test_events_link_in_created_order(machine_with_policy, client):
    events = [
        record(client, machine_with_policy, resource=f"res/{i}").json()
        for i in range(4)
    ]

    for index, event in enumerate(events):
        if index == 0:
            assert event["previous_event_id"] is None
            assert event["chain_hash"] == chain_hash("", event["content_hash"])
        else:
            previous = events[index - 1]
            assert event["previous_event_id"] == previous["id"]
            assert event["chain_hash"] == chain_hash(
                previous["chain_hash"], event["content_hash"]
            )


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="m-1")
    machine_two = create_machine(client, external_id="m-2")
    for machine_id in (machine_one, machine_two):
        declare_read(client, machine_id)
    allow_read(client)

    first_one = record(client, machine_one, resource="res/one").json()
    first_two = record(client, machine_two, resource="res/two").json()
    second_two = record(client, machine_two, resource="res/three").json()

    assert first_one["previous_event_id"] is None
    assert first_two["previous_event_id"] is None
    assert second_two["previous_event_id"] == first_two["id"]
    # Each machine's first link is rooted in the empty string.
    assert first_one["chain_hash"] == chain_hash("", first_one["content_hash"])
    assert first_two["chain_hash"] == chain_hash("", first_two["content_hash"])


def test_listed_events_carry_chain_fields(machine_with_policy, client):
    record(client, machine_with_policy, resource="res/a")
    record(client, machine_with_policy, resource="res/b")

    events = client.get(
        f"/machines/{machine_with_policy}/authorization-decision-events"
    ).json()
    assert all(
        {"previous_event_id", "content_hash", "chain_hash"} <= set(event)
        for event in events
    )


def test_integrity_valid_reports_count(machine_with_policy, client):
    for resource in ("res/a", "res/b", "res/c"):
        record(client, machine_with_policy, resource=resource)

    response = integrity(client, machine_with_policy)
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_event_id": None,
    }


def test_integrity_empty_machine_is_valid(machine_with_policy, client):
    response = integrity(client, machine_with_policy)
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_integrity_missing_machine_returns_404(client):
    response = integrity(client, "00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_is_read_only(machine_with_policy, client, tmp_path):
    for resource in ("res/a", "res/b"):
        record(client, machine_with_policy, resource=resource)
    before = client.get(
        f"/machines/{machine_with_policy}/authorization-decision-events"
    ).json()

    assert integrity(client, machine_with_policy).json()["valid"] is True
    assert integrity(client, machine_with_policy).json()["valid"] is True

    after = client.get(
        f"/machines/{machine_with_policy}/authorization-decision-events"
    ).json()
    assert after == before


def test_integrity_detects_tampered_content(machine_with_policy, client, tmp_path):
    events = [
        record(client, machine_with_policy, resource=f"res/{i}").json()
        for i in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE authorization_decision_events SET allowed = 0 WHERE id = ?",
        (events[1]["id"],),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_with_policy).json()
    assert response == {
        "valid": False,
        "checked_count": 3,
        "broken_event_id": events[1]["id"],
    }


def test_integrity_detects_tampered_previous_link(machine_with_policy, client, tmp_path):
    events = [
        record(client, machine_with_policy, resource=f"res/{i}").json()
        for i in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE authorization_decision_events SET previous_event_id = ? WHERE id = ?",
        (None, events[2]["id"]),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_with_policy).json()
    assert response["valid"] is False
    assert response["checked_count"] == 3
    assert response["broken_event_id"] == events[2]["id"]


def test_integrity_detects_tampered_chain_hash(machine_with_policy, client, tmp_path):
    events = [
        record(client, machine_with_policy, resource=f"res/{i}").json()
        for i in range(2)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE authorization_decision_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, events[0]["id"]),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_with_policy).json()
    assert response["valid"] is False
    assert response["broken_event_id"] == events[0]["id"]


def test_integrity_reports_first_broken_event(machine_with_policy, client, tmp_path):
    events = [
        record(client, machine_with_policy, resource=f"res/{i}").json()
        for i in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE authorization_decision_events SET reason = 'forged' WHERE id = ?",
        (events[0]["id"],),
    )
    connection.execute(
        "UPDATE authorization_decision_events SET reason = 'forged' WHERE id = ?",
        (events[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_with_policy).json()
    assert response["valid"] is False
    assert response["broken_event_id"] == events[0]["id"]


def test_concurrent_appends_form_one_unbroken_chain(machine_with_policy, client):
    count = 30

    def append(index):
        return record(client, machine_with_policy, resource=f"res/{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(append, range(count)))

    assert all(response.status_code == 201 for response in responses)
    events = client.get(
        f"/machines/{machine_with_policy}/authorization-decision-events"
    ).json()
    assert len(events) == count
    assert len({event["id"] for event in events}) == count

    ids = [event["id"] for event in events]
    previous_ids = [event["previous_event_id"] for event in events]
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]

    assert integrity(client, machine_with_policy).json() == {
        "valid": True,
        "checked_count": count,
        "broken_event_id": None,
    }


def test_legacy_events_are_backfilled_on_startup(tmp_path, monkeypatch):
    machine_id = "11111111-1111-1111-1111-111111111111"
    db_path = tmp_path / "legacy.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE machines (
            id VARCHAR(36) PRIMARY KEY, external_id VARCHAR UNIQUE,
            display_name VARCHAR, public_key VARCHAR, status VARCHAR,
            version INTEGER, created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE authorization_decision_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            action_type VARCHAR, resource VARCHAR, allowed BOOLEAN,
            reason VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    legacy_rows = [
        (
            "aaaaaaaa-0000-0000-0000-000000000002",
            machine_id,
            "read",
            "res/b",
            0,
            "denied_by_policy",
            "2026-01-01T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000003",
            machine_id,
            "read",
            "res/c",
            1,
            "allowed_by_policy",
            "2026-01-02T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            machine_id,
            "read",
            "res/a",
            1,
            "allowed_by_policy",
            "2026-01-01T00:00:00.000000Z",
        ),
    ]
    connection.executemany(
        "INSERT INTO authorization_decision_events VALUES (?,?,?,?,?,?,?)",
        legacy_rows,
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        events = client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()

        # Ordered by (created_at, id), not insertion order.
        assert [event["resource"] for event in events] == [
            "res/a",
            "res/b",
            "res/c",
        ]
        assert events[0]["previous_event_id"] is None
        assert events[1]["previous_event_id"] == events[0]["id"]
        assert events[2]["previous_event_id"] == events[1]["id"]
        for event in events:
            assert HEX64_RE.match(event["content_hash"])
            assert HEX64_RE.match(event["chain_hash"])
            assert event["content_hash"] == canonical_content_hash(event)
        assert events[0]["chain_hash"] == chain_hash("", events[0]["content_hash"])
        assert events[1]["chain_hash"] == chain_hash(
            events[0]["chain_hash"], events[1]["content_hash"]
        )

        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_event_id": None,
        }


def test_chain_hashes_stay_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare_read(first, machine_id)
        allow_read(first)
        created = [
            record(first, machine_id, resource=f"res/{i}").json()
            for i in range(3)
        ]

    with TestClient(app) as second:
        events = second.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        assert events == created
        assert integrity(second, machine_id).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_event_id": None,
        }

    # A further restart must still be a no-op (deterministic recomputation).
    with TestClient(app) as third:
        events_again = third.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        assert events_again == created
