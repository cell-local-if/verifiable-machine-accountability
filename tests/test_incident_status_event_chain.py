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
    "from_status",
    "to_status",
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


def canonical_chain_hash(previous_chain_hash: str, content_hash: str) -> str:
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
    return response.json()["id"]


def transition(client, machine_id, event_id, incident_id, status):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status",
        json={"status": status},
    )
    assert response.status_code == 200
    return response.json()


def history(client, machine_id, event_id, incident_id):
    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/{event_id}"
        f"/incidents/{incident_id}/status-history"
    )
    assert response.status_code == 200
    return response.json()


def integrity_url(machine_id):
    return f"/machines/{machine_id}/incident-status-events/integrity"


@pytest.fixture
def incident(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_id = create_incident(client, machine_id, event_id)
    return machine_id, event_id, incident_id


def test_empty_chain_is_valid(client):
    machine_id = create_machine(client)

    response = client.get(integrity_url(machine_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
    assert response.content == (
        b'{"valid":true,"checked_count":0,"broken_status_event_id":null}\n'
    )


def test_history_entries_carry_chain_fields(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")

    entries = history(client, machine_id, event_id, incident_id)

    assert len(entries) == 2
    first, second = entries
    for entry in entries:
        assert set(entry) == {
            "id",
            "machine_id",
            "event_id",
            "incident_id",
            "from_status",
            "to_status",
            "created_at",
            "previous_status_event_id",
            "content_hash",
            "chain_hash",
        }
        assert HEX64_RE.match(entry["content_hash"])
        assert HEX64_RE.match(entry["chain_hash"])
        assert entry["content_hash"] == canonical_content_hash(entry)
    assert first["previous_status_event_id"] is None
    assert first["chain_hash"] == canonical_chain_hash("", first["content_hash"])
    assert second["previous_status_event_id"] == first["id"]
    assert second["chain_hash"] == canonical_chain_hash(
        first["chain_hash"], second["content_hash"]
    )


def test_complete_chain_is_valid(client, incident):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }


def test_chain_spans_incidents_of_one_machine(client, incident):
    machine_id, event_id, incident_id = incident
    other_incident = create_incident(client, machine_id, event_id, summary="two")
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, other_incident, "acknowledged")

    first = history(client, machine_id, event_id, incident_id)[0]
    second = history(client, machine_id, event_id, other_incident)[0]

    # One per-machine chain: the second transition links to the first, even
    # though it belongs to a different incident.
    assert second["previous_status_event_id"] == first["id"]
    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }


def test_chains_are_isolated_per_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)
    transition(client, machine_one, event_one, incident_one, "acknowledged")
    transition(client, machine_two, event_two, incident_two, "acknowledged")

    first = history(client, machine_one, event_one, incident_one)[0]
    second = history(client, machine_two, event_two, incident_two)[0]

    # Each machine roots its own chain at the empty prefix.
    assert first["previous_status_event_id"] is None
    assert second["previous_status_event_id"] is None
    assert first["chain_hash"] != second["chain_hash"]


def test_missing_machine_returns_404(client):
    response = client.get(integrity_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_extra_query_param_returns_422_before_machine_lookup(client):
    response = client.get(integrity_url(MISSING_ID), params={"from": "x"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_returns_422_before_machine_lookup(client):
    response = client.request("GET", integrity_url(MISSING_ID), content=b"{}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_non_get_methods_return_405(client, incident):
    machine_id, _, _ = incident
    url = integrity_url(machine_id)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405, method


def test_detects_tampered_content(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    first = history(client, machine_id, event_id, incident_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET to_status = 'forged' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_detects_tampered_previous_link(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    second = history(client, machine_id, event_id, incident_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET previous_status_event_id = NULL "
        "WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_detects_tampered_chain_hash(client, incident, tmp_path):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    first = history(client, machine_id, event_id, incident_id)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_reports_first_broken_record_only(client, tmp_path):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_ids = [
        create_incident(client, machine_id, event_id, summary=f"incident-{n}")
        for n in range(3)
    ]
    for incident_id in incident_ids:
        transition(client, machine_id, event_id, incident_id, "acknowledged")
    records = [
        history(client, machine_id, event_id, incident_id)[0]
        for incident_id in incident_ids
    ]

    connection = sqlite3.connect(tmp_path / "test.db")
    for record in records[1:]:
        connection.execute(
            "UPDATE incident_status_events SET to_status = 'forged' WHERE id = ?",
            (record["id"],),
        )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_id)).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_status_event_id": records[1]["id"],
    }


def test_damaged_created_at_counts_and_breaks_without_crash(
    client, incident, tmp_path
):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    transition(client, machine_id, event_id, incident_id, "resolved")
    second = history(client, machine_id, event_id, incident_id)[1]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET created_at = 'not-a-time' WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    response = client.get(integrity_url(machine_id))

    assert response.status_code == 200
    assert response.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_other_machines_damage_does_not_affect_result(client, tmp_path):
    machine_one = create_machine(client, external_id="machine-1")
    event_one = record_event(client, machine_one)
    incident_one = create_incident(client, machine_one, event_one)
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    incident_two = create_incident(client, machine_two, event_two)
    transition(client, machine_one, event_one, incident_one, "acknowledged")
    transition(client, machine_two, event_two, incident_two, "acknowledged")
    damaged = history(client, machine_two, event_two, incident_two)[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET to_status = 'forged' WHERE id = ?",
        (damaged["id"],),
    )
    connection.commit()
    connection.close()

    assert client.get(integrity_url(machine_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
    assert client.get(integrity_url(machine_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": damaged["id"],
    }


def test_chain_survives_restart(client, incident, tmp_path, monkeypatch):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")
    before = history(client, machine_id, event_id, incident_id)

    # A second boot over the same database performs no writes and keeps the
    # chain byte-identical.
    with TestClient(app) as restarted:
        assert history(restarted, machine_id, event_id, incident_id) == before
        assert restarted.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_status_event_id": None,
        }


def test_startup_backfills_legacy_rows(client, incident, tmp_path, monkeypatch):
    machine_id, event_id, incident_id = incident
    transition(client, machine_id, event_id, incident_id, "acknowledged")

    # Simulate a database written before the chain feature: no chain columns.
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE incident_status_events SET previous_status_event_id = NULL, "
        "content_hash = NULL, chain_hash = NULL"
    )
    connection.commit()
    connection.close()

    # The next startup backfills the missing chain data deterministically.
    with TestClient(app) as restarted:
        entries = history(restarted, machine_id, event_id, incident_id)
        assert entries[0]["previous_status_event_id"] is None
        assert HEX64_RE.match(entries[0]["content_hash"])
        assert HEX64_RE.match(entries[0]["chain_hash"])
        assert restarted.get(integrity_url(machine_id)).json() == {
            "valid": True,
            "checked_count": 1,
            "broken_status_event_id": None,
        }

    # A further restart over the complete chain performs no writes.
    with TestClient(app) as again:
        assert history(again, machine_id, event_id, incident_id) == entries


def test_concurrent_transitions_keep_one_unforked_chain(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    incident_ids = [
        create_incident(client, machine_id, event_id, summary=f"incident-{n}")
        for n in range(8)
    ]

    def ack(incident_id):
        return transition(
            client, machine_id, event_id, incident_id, "acknowledged"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(ack, incident_ids))

    # The committed result is equivalent to one definite serial order: every
    # record links to a distinct predecessor and the whole chain verifies.
    entries = [
        history(client, machine_id, event_id, incident_id)[0]
        for incident_id in incident_ids
    ]
    prevs = [entry["previous_status_event_id"] for entry in entries]
    assert prevs.count(None) == 1
    assert len({prev for prev in prevs if prev is not None}) == len(entries) - 1
    assert client.get(integrity_url(machine_id)).json() == {
        "valid": True,
        "checked_count": len(entries),
        "broken_status_event_id": None,
    }
