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


def test_create_machine_returns_201_with_full_record(client):
    response = create_machine(client)

    assert response.status_code == 201
    body = response.json()
    assert body["external_id"] == "machine-1"
    assert body["display_name"] == "Machine One"
    assert body["public_key"] == "key-1"
    assert body["status"] == "active"
    assert body["version"] == 1
    assert body["created_at"].endswith("Z")
    assert body["updated_at"].endswith("Z")
    assert body["created_at"] == body["updated_at"]
    import uuid

    uuid.UUID(body["id"])  # must be a valid UUID


def test_create_machine_strips_surrounding_whitespace(client):
    response = client.post(
        "/machines",
        json={
            "external_id": "  machine-2  ",
            "display_name": "  Machine Two ",
            "public_key": " key-2\t",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["external_id"] == "machine-2"
    assert body["display_name"] == "Machine Two"
    assert body["public_key"] == "key-2"


@pytest.mark.parametrize("field", ["external_id", "display_name", "public_key"])
def test_create_machine_rejects_blank_fields(client, field):
    payload = {
        "external_id": "machine-3",
        "display_name": "Machine Three",
        "public_key": "key-3",
    }
    payload[field] = "   "

    response = client.post("/machines", json=payload)

    assert response.status_code == 422


def test_create_machine_duplicate_external_id_returns_409(client):
    assert create_machine(client).status_code == 201

    response = create_machine(client, display_name="Other", public_key="key-2")

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_external_id"}}


def test_get_machine_returns_record(client):
    created = create_machine(client).json()

    response = client.get(f"/machines/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_missing_machine_returns_404(client):
    response = client.get("/machines/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_rotate_key_success_bumps_version(client):
    created = create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == created["id"]
    assert body["public_key"] == "key-2"
    assert body["version"] == 2
    assert body["created_at"] == created["created_at"]
    assert body["updated_at"] >= created["updated_at"]

    fetched = client.get(f"/machines/{created['id']}").json()
    assert fetched == body


def test_rotate_key_stale_version_returns_409(client):
    created = create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 99},
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "version_conflict"}}

    unchanged = client.get(f"/machines/{created['id']}").json()
    assert unchanged == created


def test_rotate_key_same_key_returns_422(client):
    created = create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-1", "expected_version": 1},
    )

    assert response.status_code == 422

    unchanged = client.get(f"/machines/{created['id']}").json()
    assert unchanged == created


def test_rotate_key_missing_machine_returns_404(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000/rotate-key",
        json={"public_key": "key-9", "expected_version": 1},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_concurrent_rotate_same_version_allows_at_most_one_success(client):
    created = create_machine(client).json()

    def rotate(i):
        return client.post(
            f"/machines/{created['id']}/rotate-key",
            json={"public_key": f"key-{i}", "expected_version": 1},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(rotate, [2, 3]))

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 409]

    final = client.get(f"/machines/{created['id']}").json()
    assert final["version"] == 2
    assert final["public_key"] in {"key-2", "key-3"}


def test_data_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        created = create_machine(first).json()

    with TestClient(app) as second:
        response = second.get(f"/machines/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created
