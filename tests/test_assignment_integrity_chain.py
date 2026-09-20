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
    "event_id",
    "incident_id",
    "party",
    "role",
    "created_at",
)

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def create_incident(
    client, machine_id, event_id, incident_type="breach", summary="something happened"
):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
        json={"incident_type": incident_type, "summary": summary},
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = create_incident(client, machine_id, event_id)
    return machine_id, event_id, record


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


def assign_n(client, machine_id, event_id, incident_id, count):
    created = []
    for index in range(count):
        response = assign(
            client, machine_id, event_id, incident_id, f"party-{index}", f"role-{index}"
        )
        assert response.status_code == 201
        created.append(response.json())
    return created


def test_created_assignment_carries_chain_fields(client, incident):
    machine_id, event_id, record = incident

    body = assign(client, machine_id, event_id, record["id"], "alice", "owner").json()

    assert {"previous_assignment_id", "content_hash", "chain_hash"} <= set(body)
    assert body["previous_assignment_id"] is None
    assert HEX64_RE.match(body["content_hash"])
    assert HEX64_RE.match(body["chain_hash"])
    assert body["content_hash"] == canonical_content_hash(body)
    assert body["chain_hash"] == chain_hash("", body["content_hash"])


def test_listed_assignments_carry_chain_fields(client, incident):
    machine_id, event_id, record = incident
    assign_n(client, machine_id, event_id, record["id"], 2)

    body = client.get(
        assignments_url(machine_id, event_id, record["id"])
    ).json()
    assert len(body) == 2
    for entry in body:
        assert HEX64_RE.match(entry["content_hash"])
        assert HEX64_RE.match(entry["chain_hash"])
        assert entry["content_hash"] == canonical_content_hash(entry)


def test_assignments_link_in_created_order(client, incident):
    machine_id, event_id, record = incident
    created = assign_n(client, machine_id, event_id, record["id"], 4)

    body = client.get(
        assignments_url(machine_id, event_id, record["id"])
    ).json()
    assert [r["id"] for r in body] == [r["id"] for r in created]
    assert body[0]["previous_assignment_id"] is None
    for index in range(1, len(body)):
        assert body[index]["previous_assignment_id"] == body[index - 1]["id"]
        assert body[index]["chain_hash"] == chain_hash(
            body[index - 1]["chain_hash"], body[index]["content_hash"]
        )


def test_chains_are_independent_per_incident(client, incident):
    machine_id, event_id, first = incident
    second = create_incident(client, machine_id, event_id, summary="other")

    one = assign_n(client, machine_id, event_id, first["id"], 1)
    two = assign_n(client, machine_id, event_id, second["id"], 2)

    assert one[0]["previous_assignment_id"] is None
    assert two[0]["previous_assignment_id"] is None
    assert two[1]["previous_assignment_id"] == two[0]["id"]
    assert one[0]["chain_hash"] == chain_hash("", one[0]["content_hash"])
    assert two[0]["chain_hash"] == chain_hash("", two[0]["content_hash"])


def test_integrity_valid_reports_count(client, incident):
    machine_id, event_id, record = incident
    assign_n(client, machine_id, event_id, record["id"], 3)

    response = client.get(integrity_url(machine_id, event_id, record["id"]))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_assignment_id": None,
    }


def test_integrity_empty_incident_is_valid(client, incident):
    machine_id, event_id, record = incident

    assert client.get(
        integrity_url(machine_id, event_id, record["id"])
    ).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_assignment_id": None,
    }


@pytest.mark.parametrize(
    "path",
    [
        ("missing_machine", "existing_event", "existing_incident"),
        ("existing_machine", "missing_event", "existing_incident"),
        ("existing_machine", "existing_event", "missing_incident"),
    ],
)
def test_integrity_missing_owner_returns_404(client, incident, path):
    machine_id, event_id, record = incident
    which_machine, which_event, which_incident = path
    mid = MISSING_ID if which_machine == "missing_machine" else machine_id
    eid = MISSING_ID if which_event == "missing_event" else event_id
    iid = MISSING_ID if which_incident == "missing_incident" else record["id"]

    response = client.get(integrity_url(mid, eid, iid))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_event_of_other_machine_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_machine = create_machine(client, external_id="machine-2")

    response = client.get(
        integrity_url(other_machine, event_id, record["id"])
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_incident_of_other_event_returns_404(client, incident):
    machine_id, event_id, record = incident
    other_event = record_event(client, machine_id, resource="res/other")

    response = client.get(
        integrity_url(machine_id, other_event, record["id"])
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_is_read_only(client, incident):
    machine_id, event_id, record = incident
    assign_n(client, machine_id, event_id, record["id"], 2)
    url = assignments_url(machine_id, event_id, record["id"])
    before = client.get(url).json()

    assert client.get(integrity_url(machine_id, event_id, record["id"])).json()[
        "valid"
    ] is True
    assert client.get(integrity_url(machine_id, event_id, record["id"])).json()[
        "valid"
    ] is True
    assert client.get(url).json() == before


def test_integrity_detects_tampered_content(client, tmp_path, incident):
    machine_id, event_id, record = incident
    created = assign_n(client, machine_id, event_id, record["id"], 3)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (created[1]["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(
        integrity_url(machine_id, event_id, record["id"])
    ).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": created[1]["id"],
    }


def test_integrity_detects_tampered_previous_link(client, tmp_path, incident):
    machine_id, event_id, record = incident
    created = assign_n(client, machine_id, event_id, record["id"], 3)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments "
        "SET previous_assignment_id = NULL WHERE id = ?",
        (created[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(
        integrity_url(machine_id, event_id, record["id"])
    ).json()
    assert response == {
        "valid": False,
        "checked_count": 3,
        "broken_assignment_id": created[2]["id"],
    }


def test_integrity_detects_tampered_chain_hash(client, tmp_path, incident):
    machine_id, event_id, record = incident
    created = assign_n(client, machine_id, event_id, record["id"], 2)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET chain_hash = ? WHERE id = ?",
        ("0" * 64, created[0]["id"]),
    )
    connection.commit()
    connection.close()

    response = client.get(
        integrity_url(machine_id, event_id, record["id"])
    ).json()
    assert response["valid"] is False
    assert response["broken_assignment_id"] == created[0]["id"]


def test_integrity_reports_first_broken_assignment(client, tmp_path, incident):
    machine_id, event_id, record = incident
    created = assign_n(client, machine_id, event_id, record["id"], 3)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (created[0]["id"],),
    )
    connection.execute(
        "UPDATE incident_responsibility_assignments SET role = 'forged' WHERE id = ?",
        (created[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(
        integrity_url(machine_id, event_id, record["id"])
    ).json()
    assert response["valid"] is False
    assert response["broken_assignment_id"] == created[0]["id"]


def test_integrity_is_isolated_per_incident(client, tmp_path, incident):
    machine_id, event_id, first = incident
    second = create_incident(client, machine_id, event_id, summary="other")
    assign_n(client, machine_id, event_id, first["id"], 2)
    other = assign_n(client, machine_id, event_id, second["id"], 2)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_responsibility_assignments SET party = 'forged' WHERE id = ?",
        (other[0]["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id, event_id, first["id"])).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_assignment_id": None,
    }
    assert client.get(integrity_url(machine_id, event_id, second["id"])).json()[
        "valid"
    ] is False


def test_concurrent_assignments_keep_chain_unbroken(client, incident):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])

    def do_assign(index):
        return client.post(
            url, json={"party": f"party-{index}", "role": f"role-{index}"}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_assign, range(24)))

    assert all(response.status_code == 201 for response in responses)
    body = client.get(url).json()
    assert len(body) == 24
    assert body[0]["previous_assignment_id"] is None
    for index in range(1, len(body)):
        assert body[index]["previous_assignment_id"] == body[index - 1]["id"]
        assert body[index]["chain_hash"] == chain_hash(
            body[index - 1]["chain_hash"], body[index]["content_hash"]
        )
    assert client.get(integrity_url(machine_id, event_id, record["id"])).json() == {
        "valid": True,
        "checked_count": 24,
        "broken_assignment_id": None,
    }


def test_concurrent_duplicate_assignments_write_at_most_one(client, incident):
    machine_id, event_id, record = incident
    url = assignments_url(machine_id, event_id, record["id"])

    def do_assign(_):
        return client.post(url, json={"party": "alice", "role": "owner"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_assign, range(16)))

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [201] + [409] * 15
    assert len(client.get(url).json()) == 1
    assert client.get(integrity_url(machine_id, event_id, record["id"])).json()[
        "valid"
    ] is True


def test_assignments_do_not_modify_incident_event_or_other_chains(client, incident):
    machine_id, event_id, record = incident
    events_url = f"/machines/{machine_id}/authorization-decision-events"
    incidents_url = f"{events_url}/{event_id}/incidents"
    incidents_before = client.get(incidents_url).json()
    events_before = client.get(events_url).json()
    event_integrity_before = client.get(f"{events_url}/integrity").json()
    rotation_before = client.get(
        f"/machines/{machine_id}/key-rotation-events"
    ).json()

    assign_n(client, machine_id, event_id, record["id"], 3)

    assert client.get(incidents_url).json() == incidents_before
    assert client.get(events_url).json() == events_before
    assert client.get(f"{events_url}/integrity").json() == event_integrity_before
    assert (
        client.get(f"/machines/{machine_id}/key-rotation-events").json()
        == rotation_before
    )


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
        (machine_id, "legacy", "Legacy", "key-1", "active", 1, "t0", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_events VALUES (?,?,?,?,?,?,?)",
        (event_id, machine_id, "read", "res/x", 1, "allowed_by_policy", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_incidents VALUES (?,?,?,?,?,?,?)",
        (incident_id, machine_id, event_id, "breach", "summary", "open", "t0"),
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
            "aaaaaaaa-0000-0000-0000-000000000003",
            machine_id,
            event_id,
            incident_id,
            "carol",
            "approver",
            "2026-01-03T00:00:00.000000Z",
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
        "INSERT INTO incident_responsibility_assignments "
        "VALUES (?,?,?,?,?,?,?)",
        legacy_rows,
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        body = client.get(
            assignments_url(machine_id, event_id, incident_id)
        ).json()

        # Ordered by (created_at, id), not insertion order.
        assert [r["party"] for r in body] == ["alice", "bob", "carol"]
        assert body[0]["previous_assignment_id"] is None
        assert body[1]["previous_assignment_id"] == body[0]["id"]
        assert body[2]["previous_assignment_id"] == body[1]["id"]
        for entry in body:
            assert HEX64_RE.match(entry["content_hash"])
            assert HEX64_RE.match(entry["chain_hash"])
            assert entry["content_hash"] == canonical_content_hash(entry)
        assert body[0]["chain_hash"] == chain_hash("", body[0]["content_hash"])
        assert body[1]["chain_hash"] == chain_hash(
            body[0]["chain_hash"], body[1]["content_hash"]
        )

        assert client.get(
            integrity_url(machine_id, event_id, incident_id)
        ).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_assignment_id": None,
        }


def test_chain_hashes_stay_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        assign_n(first, machine_id, event_id, incident["id"], 3)
        created = first.get(
            assignments_url(machine_id, event_id, incident["id"])
        ).json()

    with TestClient(app) as second:
        assert (
            second.get(
                assignments_url(machine_id, event_id, incident["id"])
            ).json()
            == created
        )
        assert second.get(
            integrity_url(machine_id, event_id, incident["id"])
        ).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_assignment_id": None,
        }

    # A further restart must still be a no-op (deterministic recomputation).
    with TestClient(app) as third:
        assert (
            third.get(
                assignments_url(machine_id, event_id, incident["id"])
            ).json()
            == created
        )


def test_restart_with_complete_database_performs_no_writes(tmp_path, monkeypatch):
    db_path = tmp_path / "nowrite.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        incident = create_incident(first, machine_id, event_id)
        assign_n(first, machine_id, event_id, incident["id"], 3)

    def table_digest() -> str:
        connection = sqlite3.connect(db_path)
        rows = connection.execute(
            "SELECT id, previous_assignment_id, content_hash, chain_hash "
            "FROM incident_responsibility_assignments "
            "ORDER BY created_at, id"
        ).fetchall()
        connection.close()
        return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()

    digest_before = table_digest()
    with TestClient(app):
        pass
    assert table_digest() == digest_before
