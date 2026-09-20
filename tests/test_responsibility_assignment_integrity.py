import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

MISSING_ID = "00000000-0000-0000-0000-000000000000"

CONTENT_KEYS = (
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party",
    "role",
    "created_at",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def canonical_content_hash(record: dict) -> str:
    document = json.dumps(
        {key: record[key] for key in CONTENT_KEYS},
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


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def create_incident(client, machine_id, event_id, summary="something happened"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": "breach", "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


def assignments_url(machine_id, event_id, incident_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/responsibility-assignments"
    )


def integrity_url(machine_id, event_id, incident_id):
    return assignments_url(machine_id, event_id, incident_id) + "/integrity"


def assign(client, machine_id, event_id, incident_id, party, role):
    return client.post(
        assignments_url(machine_id, event_id, incident_id),
        json={"party": party, "role": role},
    )


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = create_incident(client, machine_id, event_id)
    return machine_id, event_id, record["id"]


def test_created_assignment_contains_chain_fields(client, incident):
    machine_id, event_id, incident_id = incident

    record = assign(client, machine_id, event_id, incident_id, "alice", "owner").json()

    assert record["previous_assignment_id"] is None
    assert HEX64_RE.match(record["content_hash"])
    assert HEX64_RE.match(record["chain_hash"])


def test_content_hash_is_canonical_sha256(client, incident):
    machine_id, event_id, incident_id = incident

    record = assign(
        client, machine_id, event_id, incident_id, "alice/数据", "owner ✓"
    ).json()

    assert record["content_hash"] == canonical_content_hash(record)


def test_first_assignment_chain_hash_uses_empty_previous(client, incident):
    machine_id, event_id, incident_id = incident

    record = assign(client, machine_id, event_id, incident_id, "alice", "owner").json()

    assert record["chain_hash"] == chain_hash("", record["content_hash"])


def test_assignments_link_in_created_order(client, incident):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}").json()
        for n in range(4)
    ]

    for index, record in enumerate(records):
        if index == 0:
            assert record["previous_assignment_id"] is None
            assert record["chain_hash"] == chain_hash("", record["content_hash"])
        else:
            previous = records[index - 1]
            assert record["previous_assignment_id"] == previous["id"]
            assert record["chain_hash"] == chain_hash(
                previous["chain_hash"], record["content_hash"]
            )


def test_chains_are_independent_per_incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_one = create_incident(client, machine_id, event_id, summary="one")
    incident_two = create_incident(client, machine_id, event_id, summary="two")

    first_one = assign(
        client, machine_id, event_id, incident_one["id"], "alice", "owner"
    ).json()
    first_two = assign(
        client, machine_id, event_id, incident_two["id"], "bob", "owner"
    ).json()
    second_two = assign(
        client, machine_id, event_id, incident_two["id"], "carol", "reviewer"
    ).json()

    assert first_one["previous_assignment_id"] is None
    assert first_two["previous_assignment_id"] is None
    assert second_two["previous_assignment_id"] == first_two["id"]
    # Each incident's first link is rooted in the empty string.
    assert first_one["chain_hash"] == chain_hash("", first_one["content_hash"])
    assert first_two["chain_hash"] == chain_hash("", first_two["content_hash"])


def test_listed_assignments_carry_chain_fields(client, incident):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")

    records = client.get(assignments_url(machine_id, event_id, incident_id)).json()
    assert all(
        {"previous_assignment_id", "content_hash", "chain_hash"} <= set(record)
        for record in records
    )


def test_integrity_valid_reports_count(client, incident):
    machine_id, event_id, incident_id = incident
    for n in range(3):
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}")

    response = client.get(integrity_url(machine_id, event_id, incident_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_assignment_id": None,
    }


def test_integrity_empty_incident_is_valid(client, incident):
    machine_id, event_id, incident_id = incident

    response = client.get(integrity_url(machine_id, event_id, incident_id))
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_assignment_id": None,
    }


@pytest.mark.parametrize(
    "which",
    ["missing_machine", "missing_event", "missing_incident"],
)
def test_integrity_missing_owner_returns_404(client, incident, which):
    machine_id, event_id, incident_id = incident
    mid = MISSING_ID if which == "missing_machine" else machine_id
    eid = MISSING_ID if which == "missing_event" else event_id
    iid = MISSING_ID if which == "missing_incident" else incident_id

    response = client.get(integrity_url(mid, eid, iid))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_ownership_mismatch_returns_404(client, incident):
    machine_id, event_id, incident_id = incident
    other_machine = create_machine(client, external_id="machine-2")
    other_event = record_event(client, machine_id, resource="res/other")
    other_incident = create_incident(client, machine_id, event_id, summary="other")

    for mid, eid, iid in (
        (other_machine, event_id, incident_id),
        (machine_id, other_event, incident_id),
        (machine_id, event_id, other_incident["id"]),
    ):
        # The last combination is a real incident, so only the first two are
        # ownership mismatches; the third is a valid empty chain.
        response = client.get(integrity_url(mid, eid, iid))
        if (mid, eid, iid) == (machine_id, event_id, other_incident["id"]):
            assert response.status_code == 200
        else:
            assert response.status_code == 404
            assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_is_read_only(client, incident):
    machine_id, event_id, incident_id = incident
    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    assign(client, machine_id, event_id, incident_id, "bob", "reviewer")
    before = client.get(assignments_url(machine_id, event_id, incident_id)).json()

    url = integrity_url(machine_id, event_id, incident_id)
    assert client.get(url).json()["valid"] is True
    assert client.get(url).json()["valid"] is True

    after = client.get(assignments_url(machine_id, event_id, incident_id)).json()
    assert after == before


def test_integrity_detects_tampered_content(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}").json()
        for n in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'mallory' WHERE id = ?",
        (records[1]["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(integrity_url(machine_id, event_id, incident_id)).json()
    assert response == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": records[1]["id"],
    }


def test_integrity_detects_tampered_previous_link(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}").json()
        for n in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments "
        "SET previous_assignment_id = NULL WHERE id = ?",
        (records[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(integrity_url(machine_id, event_id, incident_id)).json()
    assert response["valid"] is False
    assert response["checked_count"] == 3
    assert response["broken_assignment_id"] == records[2]["id"]


def test_integrity_detects_tampered_chain_hash(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}").json()
        for n in range(2)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments SET chain_hash = ? WHERE id = ?",
        ("0" * 64, records[0]["id"]),
    )
    connection.commit()
    connection.close()

    response = client.get(integrity_url(machine_id, event_id, incident_id)).json()
    assert response["valid"] is False
    assert response["broken_assignment_id"] == records[0]["id"]


def test_integrity_reports_first_broken_assignment(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    records = [
        assign(client, machine_id, event_id, incident_id, f"party-{n}", f"role-{n}").json()
        for n in range(3)
    ]
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (records[0]["id"],),
    )
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (records[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(integrity_url(machine_id, event_id, incident_id)).json()
    assert response["valid"] is False
    assert response["broken_assignment_id"] == records[0]["id"]


def test_integrity_is_scoped_to_the_path_incident(client, tmp_path):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_one = create_incident(client, machine_id, event_id, summary="one")
    incident_two = create_incident(client, machine_id, event_id, summary="two")
    damaged = assign(
        client, machine_id, event_id, incident_one["id"], "alice", "owner"
    ).json()
    assign(client, machine_id, event_id, incident_two["id"], "bob", "owner")
    db_path = tmp_path / "test.db"

    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'mallory' WHERE id = ?",
        (damaged["id"],),
    )
    connection.commit()
    connection.close()

    # Damage under another incident of the same event never fails this audit.
    assert client.get(
        integrity_url(machine_id, event_id, incident_two["id"])
    ).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_assignment_id": None,
    }
    assert client.get(
        integrity_url(machine_id, event_id, incident_one["id"])
    ).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_assignment_id": damaged["id"],
    }


def test_concurrent_appends_form_one_unbroken_chain(client, incident):
    machine_id, event_id, incident_id = incident
    count = 30

    def append(index):
        return assign(
            client, machine_id, event_id, incident_id, f"party-{index}", f"role-{index}"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(append, range(count)))

    assert all(response.status_code == 201 for response in responses)
    records = client.get(assignments_url(machine_id, event_id, incident_id)).json()
    assert len(records) == count
    assert len({record["id"] for record in records}) == count

    ids = [record["id"] for record in records]
    previous_ids = [record["previous_assignment_id"] for record in records]
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]

    assert client.get(integrity_url(machine_id, event_id, incident_id)).json() == {
        "valid": True,
        "checked_count": count,
        "broken_assignment_id": None,
    }


def test_legacy_assignments_are_backfilled_on_startup(tmp_path, monkeypatch):
    machine_id = "11111111-1111-1111-1111-111111111111"
    event_id = "22222222-2222-2222-2222-222222222222"
    incident_id = "33333333-3333-3333-3333-333333333333"
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
        """
        CREATE TABLE authorization_decision_incidents (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            event_id VARCHAR(36), incident_type VARCHAR, summary VARCHAR,
            status VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE incident_responsibility_assignments (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            event_id VARCHAR(36), incident_id VARCHAR(36),
            party VARCHAR, role VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_events VALUES (?,?,?,?,?,?,?)",
        (event_id, machine_id, "read", "res/x", 1, "allowed_by_policy", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_incidents VALUES (?,?,?,?,?,?,?)",
        (incident_id, machine_id, event_id, "breach", "legacy", "open", "t0"),
    )
    legacy_rows = [
        (
            "aaaaaaaa-0000-0000-0000-000000000002",
            machine_id,
            event_id,
            incident_id,
            "bob",
            "reviewer",
            "2026-01-02T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            machine_id,
            event_id,
            incident_id,
            "alice",
            "owner",
            "2026-01-01T00:00:00.000000Z",
        ),
    ]
    connection.executemany(
        "INSERT INTO incident_responsibility_assignments VALUES (?,?,?,?,?,?,?)",
        legacy_rows,
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        records = client.get(
            assignments_url(machine_id, event_id, incident_id)
        ).json()

        # Ordered by (created_at, id), not insertion order.
        assert [record["party"] for record in records] == ["alice", "bob"]
        assert records[0]["previous_assignment_id"] is None
        assert records[1]["previous_assignment_id"] == records[0]["id"]
        for record in records:
            assert HEX64_RE.match(record["content_hash"])
            assert HEX64_RE.match(record["chain_hash"])
            assert record["content_hash"] == canonical_content_hash(record)
        assert records[0]["chain_hash"] == chain_hash("", records[0]["content_hash"])
        assert records[1]["chain_hash"] == chain_hash(
            records[0]["chain_hash"], records[1]["content_hash"]
        )

        assert client.get(
            integrity_url(machine_id, event_id, incident_id)
        ).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_assignment_id": None,
        }


def test_chain_hashes_stay_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        created = [
            assign(
                first, machine_id, event_id, incident["id"], f"party-{n}", f"role-{n}"
            ).json()
            for n in range(3)
        ]

    with TestClient(app) as second:
        records = second.get(
            assignments_url(machine_id, event_id, incident["id"])
        ).json()
        assert records == created
        assert second.get(
            integrity_url(machine_id, event_id, incident["id"])
        ).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_assignment_id": None,
        }

    # A further restart must still be a no-op (deterministic recomputation).
    with TestClient(app) as third:
        records_again = third.get(
            assignments_url(machine_id, event_id, incident["id"])
        ).json()
        assert records_again == created


def test_assignments_do_not_modify_incident_event_chain_or_history(client, incident):
    machine_id, event_id, incident_id = incident
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    events_before = client.get(events_url).json()
    incidents_before = client.get(f"{events_url}/{event_id}/incidents").json()
    integrity_before = client.get(f"{events_url}/integrity").json()

    assign(client, machine_id, event_id, incident_id, "alice", "owner")
    client.get(integrity_url(machine_id, event_id, incident_id))

    assert client.get(events_url).json() == events_before
    assert client.get(f"{events_url}/{event_id}/incidents").json() == incidents_before
    assert client.get(f"{events_url}/integrity").json() == integrity_before
    assert integrity_before["valid"] is True
