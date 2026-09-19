from concurrent.futures import ThreadPoolExecutor

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


def create_machine(client, external_id="machine-1", display_name="Machine One", public_key="key-1"):
    return client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": display_name,
            "public_key": public_key,
        },
    )


def rotate(client, machine_id, public_key, expected_version):
    return client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": public_key, "expected_version": expected_version},
    )


def events(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events")


def test_rotation_events_empty_for_new_machine(client):
    created = create_machine(client).json()

    response = events(client, created["id"])

    assert response.status_code == 200
    assert response.json() == []


def test_rotation_events_missing_machine_returns_404(client):
    response = events(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_successful_rotation_records_event(client):
    created = create_machine(client).json()

    rotated = rotate(client, created["id"], "key-2", 1)
    assert rotated.status_code == 200

    response = events(client, created["id"])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    record = body[0]
    import uuid

    uuid.UUID(record["id"])
    assert record["machine_id"] == created["id"]
    assert record["old_public_key"] == "key-1"
    assert record["new_public_key"] == "key-2"
    assert record["version"] == rotated.json()["version"] == 2
    assert record["created_at"].endswith("Z")


def test_multiple_rotations_recorded_in_order(client):
    created = create_machine(client).json()

    rotate(client, created["id"], "key-2", 1)
    rotate(client, created["id"], "key-3", 2)
    rotate(client, created["id"], "key-4", 3)

    body = events(client, created["id"]).json()
    assert len(body) == 3
    assert [r["old_public_key"] for r in body] == ["key-1", "key-2", "key-3"]
    assert [r["new_public_key"] for r in body] == ["key-2", "key-3", "key-4"]
    assert [r["version"] for r in body] == [2, 3, 4]
    assert body == sorted(body, key=lambda r: (r["created_at"], r["id"]))


def test_failed_rotations_leave_no_records(client):
    created = create_machine(client).json()

    assert rotate(client, created["id"], "key-1", 1).status_code == 422
    assert rotate(client, created["id"], "key-2", 99).status_code == 409
    assert rotate(client, "00000000-0000-0000-0000-000000000000", "key-9", 1).status_code == 404

    assert events(client, created["id"]).json() == []


def test_concurrent_rotation_leaves_exactly_one_record(client):
    created = create_machine(client).json()

    def do_rotate(i):
        return rotate(client, created["id"], f"key-{i}", 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(do_rotate, [2, 3]))

    assert sorted(r.status_code for r in responses) == [200, 409]

    body = events(client, created["id"]).json()
    assert len(body) == 1
    final = client.get(f"/machines/{created['id']}").json()
    assert body[0]["new_public_key"] == final["public_key"]
    assert body[0]["version"] == final["version"] == 2
    assert body[0]["old_public_key"] == "key-1"


def test_rotation_events_isolated_per_machine(client):
    first = create_machine(client, external_id="machine-a", public_key="key-a1").json()
    second = create_machine(client, external_id="machine-b", public_key="key-b1").json()

    rotate(client, first["id"], "key-a2", 1)

    assert len(events(client, first["id"]).json()) == 1
    assert events(client, second["id"]).json() == []


def test_rotation_events_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first_client:
        created = create_machine(first_client).json()
        rotate(first_client, created["id"], "key-2", 1)
        before = events(first_client, created["id"]).json()

    with TestClient(app) as second_client:
        after = events(second_client, created["id"]).json()

    assert len(before) == 1
    assert after == before


def test_listing_events_does_not_modify_machine(client):
    created = create_machine(client).json()
    rotate(client, created["id"], "key-2", 1)

    events(client, created["id"])
    events(client, created["id"])

    assert client.get(f"/machines/{created['id']}").json()["version"] == 2
    assert len(events(client, created["id"]).json()) == 1
