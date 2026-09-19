from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from accountability.app import app
from accountability.db import get_session, init_db, make_engine


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/machines.db"


@pytest.fixture
def client(db_url):
    engine = make_engine(db_url)
    init_db(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session():
        session = testing_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_get_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def _create_machine(client, external_id="machine-001", display_name="Machine 001", public_key="key-1"):
    return client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": display_name,
            "public_key": public_key,
        },
    )


def test_health_check_unchanged(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_machine_returns_201_with_full_record(client):
    response = _create_machine(client)

    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["external_id"] == "machine-001"
    assert body["display_name"] == "Machine 001"
    assert body["public_key"] == "key-1"
    assert body["status"] == "active"
    assert body["version"] == 1
    assert body["created_at"].endswith("+00:00")
    assert body["updated_at"].endswith("+00:00")


def test_create_machine_trims_whitespace(client):
    response = _create_machine(client, external_id="  machine-002  ")

    assert response.status_code == 201
    assert response.json()["external_id"] == "machine-002"


@pytest.mark.parametrize(
    "payload",
    [
        {"external_id": "  ", "display_name": "n", "public_key": "k"},
        {"external_id": "e", "display_name": "", "public_key": "k"},
        {"external_id": "e", "display_name": "n", "public_key": "   "},
        {"display_name": "n", "public_key": "k"},
    ],
)
def test_create_machine_rejects_blank_or_missing_fields(client, payload):
    response = client.post("/machines", json=payload)

    assert response.status_code == 422


def test_create_machine_duplicate_external_id_returns_409(client):
    assert _create_machine(client).status_code == 201

    response = _create_machine(client, display_name="Other", public_key="key-2")

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_external_id"}}


def test_get_machine_returns_record(client):
    created = _create_machine(client).json()

    response = client.get(f"/machines/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_get_machine_not_found_returns_404(client):
    response = client.get("/machines/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_rotate_key_success_bumps_version(client):
    created = _create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["public_key"] == "key-2"
    assert body["version"] == 2
    assert body["created_at"] == created["created_at"]
    assert body["updated_at"] >= created["updated_at"]


def test_rotate_key_stale_version_returns_409_and_no_partial_update(client):
    created = _create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-2", "expected_version": 99},
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "version_conflict"}}
    current = client.get(f"/machines/{created['id']}").json()
    assert current == created


def test_rotate_key_same_key_returns_422_and_no_partial_update(client):
    created = _create_machine(client).json()

    response = client.post(
        f"/machines/{created['id']}/rotate-key",
        json={"public_key": "key-1", "expected_version": 1},
    )

    assert response.status_code == 422
    current = client.get(f"/machines/{created['id']}").json()
    assert current == created


def test_rotate_key_not_found_returns_404(client):
    response = client.post(
        "/machines/does-not-exist/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_concurrent_rotations_with_same_version_at_most_one_succeeds(client):
    created = _create_machine(client).json()
    url = f"/machines/{created['id']}/rotate-key"

    def rotate(i):
        return client.post(url, json={"public_key": f"key-{i}", "expected_version": 1})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(rotate, range(2, 12)))

    successes = [r for r in responses if r.status_code == 200]
    conflicts = [r for r in responses if r.status_code == 409]
    assert len(successes) == 1
    assert len(conflicts) == len(responses) - 1
    for conflict in conflicts:
        assert conflict.json() == {"error": {"code": "version_conflict"}}

    current = client.get(f"/machines/{created['id']}").json()
    assert current["version"] == 2
    assert current["public_key"] == successes[0].json()["public_key"]


def test_data_survives_restart(client, db_url):
    created = _create_machine(client).json()

    # Simulate an application restart: drop the current engine and rebuild
    # the session factory against the same database file.
    app.dependency_overrides.clear()
    engine = make_engine(db_url)
    restarted_session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_session():
        session = restarted_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_get_session
    try:
        response = client.get(f"/machines/{created['id']}")
    finally:
        engine.dispose()

    assert response.status_code == 200
    assert response.json() == created
