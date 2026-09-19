import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


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


def create_rule(client, action_type="read", resource_pattern="res/*"):
    response = client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": "allow",
            "priority": 0,
        },
    )
    assert response.status_code == 201


def declare(client, machine_id, action_type="read", resource_pattern="res/*"):
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": True,
        },
    )
    assert response.status_code == 201


def record_event(client, machine_id, action_type="read", resource="res/x"):
    return client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )


def expected_content_hash(event):
    payload = {
        "id": event["id"],
        "machine_id": event["machine_id"],
        "action_type": event["action_type"],
        "resource": event["resource"],
        "allowed": event["allowed"],
        "reason": event["reason"],
        "created_at": event["created_at"],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_chain_hash(previous_chain_hash, content_hash):
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


def test_first_event_has_null_previous_and_valid_hashes(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    body = record_event(client, machine_id).json()

    assert body["previous_event_id"] is None
    assert HEX64_RE.match(body["content_hash"])
    assert HEX64_RE.match(body["chain_hash"])
    assert body["content_hash"] == expected_content_hash(body)
    assert body["chain_hash"] == expected_chain_hash("", body["content_hash"])


def test_events_form_linked_chain_in_created_order(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    recorded = [
        record_event(client, machine_id, resource=f"res/{i}").json()
        for i in range(3)
    ]

    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert [e["id"] for e in events] == [e["id"] for e in recorded]

    previous_id = None
    previous_chain_hash = ""
    for event in events:
        assert event["previous_event_id"] == previous_id
        assert event["content_hash"] == expected_content_hash(event)
        assert event["chain_hash"] == expected_chain_hash(
            previous_chain_hash, event["content_hash"]
        )
        previous_id = event["id"]
        previous_chain_hash = event["chain_hash"]


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    declare(client, machine_one)
    declare(client, machine_two)
    create_rule(client)

    first_one = record_event(client, machine_one, resource="res/a").json()
    first_two = record_event(client, machine_two, resource="res/b").json()

    assert first_one["previous_event_id"] is None
    assert first_two["previous_event_id"] is None
    assert first_one["chain_hash"] != first_two["chain_hash"]


def test_integrity_missing_machine_returns_404(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000"
        "/authorization-decision-events/integrity"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_empty_chain_is_valid(client):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_event_id": None,
    }


def test_integrity_intact_chain_is_valid(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    for i in range(3):
        record_event(client, machine_id, resource=f"res/{i}")

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_event_id": None,
    }


def test_integrity_detects_tampered_event(client, tmp_path):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    for i in range(3):
        record_event(client, machine_id, resource=f"res/{i}")
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()

    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE authorization_decision_events SET reason = 'tampered' "
            "WHERE id = ?",
            (events[1]["id"],),
        )

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 3,
        "broken_event_id": events[1]["id"],
    }


def test_integrity_detects_relinked_chain(client, tmp_path):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)
    for i in range(3):
        record_event(client, machine_id, resource=f"res/{i}")
    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()

    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE authorization_decision_events SET previous_event_id = NULL "
            "WHERE id = ?",
            (events[2]["id"],),
        )

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["broken_event_id"] == events[2]["id"]


def test_startup_backfills_chain_for_legacy_events(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        for i in range(3):
            record_event(first, machine_id, resource=f"res/{i}")
        events = first.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()

    # Simulate a pre-chain database: strip all chain data.
    with sqlite3.connect(tmp_path / "legacy.db") as conn:
        conn.execute(
            "UPDATE authorization_decision_events "
            "SET previous_event_id = NULL, content_hash = NULL, chain_hash = NULL"
        )

    with TestClient(app) as second:
        response = second.get(
            f"/machines/{machine_id}/authorization-decision-events"
        )
        backfilled = response.json()
        assert [e["id"] for e in backfilled] == [e["id"] for e in events]

        previous_id = None
        previous_chain_hash = ""
        for event in backfilled:
            assert event["previous_event_id"] == previous_id
            assert event["content_hash"] == expected_content_hash(event)
            assert event["chain_hash"] == expected_chain_hash(
                previous_chain_hash, event["content_hash"]
            )
            previous_id = event["id"]
            previous_chain_hash = event["chain_hash"]

        integrity = second.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).json()
        assert integrity == {
            "valid": True,
            "checked_count": 3,
            "broken_event_id": None,
        }

    # A second restart must not change the backfilled chain data.
    with TestClient(app) as third:
        again = third.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
    assert again == backfilled


def test_restart_preserves_chain_data(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declare(first, machine_id)
        create_rule(first)
        for i in range(2):
            record_event(first, machine_id, resource=f"res/{i}")
        events = first.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()

    with TestClient(app) as second:
        assert (
            second.get(
                f"/machines/{machine_id}/authorization-decision-events"
            ).json()
            == events
        )


def test_concurrent_appends_form_single_intact_chain(client):
    machine_id = create_machine(client)
    declare(client, machine_id)
    create_rule(client)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda i: record_event(client, machine_id, resource=f"res/{i}"),
                range(16),
            )
        )
    assert all(r.status_code == 201 for r in responses)

    events = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    assert len(events) == 16

    heads = [e for e in events if e["previous_event_id"] is None]
    assert len(heads) == 1
    referenced = {e["previous_event_id"] for e in events} - {None}
    assert len(referenced) == 15

    integrity = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert integrity == {
        "valid": True,
        "checked_count": 16,
        "broken_event_id": None,
    }
