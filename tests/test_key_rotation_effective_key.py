import sqlite3

import pytest
from fastapi.testclient import TestClient

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def create_machine(client, external_id="machine-1", public_key="key-1"):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "Machine One",
            "public_key": public_key,
        },
    )
    assert response.status_code == 201
    return response.json()


def rotate(client, machine_id, public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": public_key, "expected_version": expected_version},
    )
    assert response.status_code == 200
    return response


def events(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events").json()


def effective_key(client, machine_id, at="2030-01-01T00:00:00Z", **kwargs):
    return client.get(
        f"/machines/{machine_id}/key-rotation-events/effective-key",
        params={"at": at},
        **kwargs,
    )


def test_no_rotations_returns_current_machine_key(client):
    machine = create_machine(client)

    response = effective_key(client, machine["id"])

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": "2030-01-01T00:00:00Z",
        "effective_key": "key-1",
        "key_version": 1,
        "effective_from": machine["created_at"],
        "rotation_id": None,
    }


def test_at_is_echoed_verbatim_with_fractional_seconds(client):
    machine = create_machine(client)

    response = effective_key(client, machine["id"], at="2030-06-30T12:34:56.789Z")

    assert response.status_code == 200
    assert response.json()["at"] == "2030-06-30T12:34:56.789Z"


def test_instant_before_machine_creation_yields_nulls(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    response = effective_key(client, machine["id"], at="2000-01-01T00:00:00Z")

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": "2000-01-01T00:00:00Z",
        "effective_key": None,
        "key_version": 0,
        "effective_from": None,
        "rotation_id": None,
    }


def test_before_first_rotation_uses_old_public_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    # After the machine exists but before the first rotation commits.
    created = client.get(f"/machines/{machine['id']}").json()
    response = effective_key(client, machine["id"], at=created["created_at"])

    assert response.status_code == 200
    body = response.json()
    assert body["effective_key"] == "key-1"
    assert body["key_version"] == 1
    assert body["effective_from"] == machine["created_at"]
    assert body["rotation_id"] is None


def test_rotation_instant_uses_new_public_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    (first,) = events(client, machine["id"])

    response = effective_key(client, machine["id"], at=first["created_at"])

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": first["created_at"],
        "effective_key": "key-2",
        "key_version": 2,
        "effective_from": first["created_at"],
        "rotation_id": first["id"],
    }


def test_segments_across_multiple_rotations(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    rotate(client, machine["id"], "key-4", 3)
    first, second, third = events(client, machine["id"])

    # At the second rotation's instant its new key applies.
    body = effective_key(client, machine["id"], at=second["created_at"]).json()
    assert body["effective_key"] == "key-3"
    assert body["key_version"] == 3
    assert body["effective_from"] == second["created_at"]
    assert body["rotation_id"] == second["id"]

    # Far in the future the last segment applies and matches the machine.
    body = effective_key(client, machine["id"], at="2099-01-01T00:00:00Z").json()
    current = client.get(f"/machines/{machine['id']}").json()
    assert body["effective_key"] == current["public_key"] == "key-4"
    assert body["key_version"] == current["version"] == 4
    assert body["effective_from"] == third["created_at"]
    assert body["rotation_id"] == third["id"]


def test_missing_machine_returns_404(client):
    response = effective_key(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_invalid_query_takes_priority_over_missing_machine(client):
    missing = "00000000-0000-0000-0000-000000000000"

    response = client.get(
        f"/machines/{missing}/key-rotation-events/effective-key",
        params={"at": "2030-01-01T00:00:00Z", "extra": "1"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(
        f"/machines/{missing}/key-rotation-events/effective-key",
        params={"at": "not-a-time"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_unknown_parameter_rejected(client):
    machine = create_machine(client)

    response = client.get(
        f"/machines/{machine['id']}/key-rotation-events/effective-key",
        params={"at": "2030-01-01T00:00:00Z", "machine_id": "x"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_at_rejected(client):
    machine = create_machine(client)

    response = client.get(
        f"/machines/{machine['id']}/key-rotation-events/effective-key"
        "?at=2030-01-01T00:00:00Z&at=2031-01-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_rejected(client):
    machine = create_machine(client)

    response = client.request(
        "GET",
        f"/machines/{machine['id']}/key-rotation-events/effective-key",
        params={"at": "2030-01-01T00:00:00Z"},
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_empty_and_malformed_at_rejected(client):
    machine = create_machine(client)
    base = f"/machines/{machine['id']}/key-rotation-events/effective-key"

    for url in (
        base,
        f"{base}?at=",
        f"{base}?at=2030-01-01",
        f"{base}?at=2030-01-01T00:00:00",
        f"{base}?at=2030-01-01T00:00:00%2B00:00",
        f"{base}?at=%202030-01-01T00:00:00Z",
        f"{base}?at=2030-13-01T00:00:00Z",
    ):
        response = client.get(url)
        assert response.status_code == 422, url
        assert response.json() == {"error": {"code": "bad_time"}}, url


def test_only_get_is_routed(client):
    machine = create_machine(client)
    url = f"/machines/{machine['id']}/key-rotation-events/effective-key"

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url, params={"at": "2030-01-01T00:00:00Z"})
        assert response.status_code == 405, method


def test_effective_key_isolated_per_machine(client):
    first = create_machine(client, external_id="machine-a", public_key="key-a1")
    second = create_machine(client, external_id="machine-b", public_key="key-b1")
    rotate(client, first["id"], "key-a2", 1)

    body = effective_key(client, second["id"]).json()
    assert body["effective_key"] == "key-b1"
    assert body["key_version"] == 1
    assert body["rotation_id"] is None


def test_query_is_read_only_and_repeatable(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    before_events = events(client, machine["id"])

    first = effective_key(client, machine["id"])
    second = effective_key(client, machine["id"])

    assert first.status_code == 200
    assert first.content == second.content
    assert events(client, machine["id"]) == before_events
    assert client.get(f"/machines/{machine['id']}").json()["version"] == 2


def test_results_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first_client:
        machine = create_machine(first_client)
        rotate(first_client, machine["id"], "key-2", 1)
        before = effective_key(first_client, machine["id"]).json()

    with TestClient(app) as second_client:
        after = effective_key(second_client, machine["id"]).json()

    assert after == before


def test_tampered_event_field_returns_500(client, tmp_path):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    (record,) = events(client, machine["id"])

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET old_public_key = 'forged' WHERE id = ?",
        (record["id"],),
    )
    connection.commit()
    connection.close()

    response = effective_key(client, machine["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_tampered_chain_hash_returns_500(client, tmp_path):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    (record,) = events(client, machine["id"])

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, record["id"]),
    )
    connection.commit()
    connection.close()

    response = effective_key(client, machine["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_machine_key_mismatch_with_last_segment_returns_500(client, tmp_path):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET public_key = 'forged' WHERE id = ?",
        (machine["id"],),
    )
    connection.commit()
    connection.close()

    response = effective_key(client, machine["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_discontinuous_version_chain_returns_500(client, tmp_path):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    records = events(client, machine["id"])

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "DELETE FROM key_rotation_events WHERE id = ?",
        (records[0]["id"],),
    )
    connection.commit()
    connection.close()

    response = effective_key(client, machine["id"])
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_other_endpoints_unaffected(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    effective_key(client, machine["id"])

    assert client.get(f"/machines/{machine['id']}").json()["public_key"] == "key-2"
    assert len(events(client, machine["id"])) == 1
    integrity = client.get(
        f"/machines/{machine['id']}/key-rotation-events/integrity"
    ).json()
    assert integrity == {"valid": True, "checked_count": 1, "broken_rotation_id": None}
