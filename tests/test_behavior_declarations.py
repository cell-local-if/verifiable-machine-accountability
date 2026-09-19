import uuid

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


def create_declaration(client, machine_id, action_type="read", resource_pattern="res/*", enabled=True):
    return client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )


def test_create_declaration_returns_201_with_full_record(client):
    machine_id = create_machine(client)

    response = create_declaration(client, machine_id)

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert body["machine_id"] == machine_id
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "res/*"
    assert body["enabled"] is True
    assert body["created_at"].endswith("Z")
    assert body["updated_at"].endswith("Z")
    assert body["created_at"] == body["updated_at"]


def test_create_declaration_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": " write \t", "resource_pattern": "  s3://x ", "enabled": False},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["action_type"] == "write"
    assert body["resource_pattern"] == "s3://x"
    assert body["enabled"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "   ", "resource_pattern": "r", "enabled": True},
        {"action_type": "a", "resource_pattern": "\t", "enabled": True},
        {"action_type": "a", "resource_pattern": "r", "enabled": "yes"},
        {"action_type": "a", "resource_pattern": "r", "enabled": 1},
        {"action_type": "a", "resource_pattern": "r"},
        {"resource_pattern": "r", "enabled": True},
    ],
)
def test_create_declaration_rejects_invalid_payload(client, payload):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations", json=payload
    )

    assert response.status_code == 422


def test_create_declaration_missing_machine_returns_404(client):
    response = create_declaration(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_duplicate_declaration_returns_409_and_adds_nothing(client):
    machine_id = create_machine(client)
    assert create_declaration(client, machine_id).status_code == 201

    response = create_declaration(client, machine_id, enabled=False)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_behavior_declaration"}}

    declarations = client.get(
        f"/machines/{machine_id}/behavior-declarations"
    ).json()
    assert len(declarations) == 1
    assert declarations[0]["enabled"] is True


def test_same_action_and_resource_allowed_on_other_machines(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")

    assert create_declaration(client, first).status_code == 201
    assert create_declaration(client, second).status_code == 201


def test_list_declarations_sorted_by_created_at_ascending(client):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id, action_type="read").json()
    second = create_declaration(client, machine_id, action_type="write").json()

    response = client.get(f"/machines/{machine_id}/behavior-declarations")

    assert response.status_code == 200
    assert [d["id"] for d in response.json()] == [first["id"], second["id"]]


def test_list_declarations_empty_when_none(client):
    machine_id = create_machine(client)

    response = client.get(f"/machines/{machine_id}/behavior-declarations")

    assert response.status_code == 200
    assert response.json() == []


def test_list_declarations_isolated_between_machines(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    create_declaration(client, first, action_type="read")
    create_declaration(client, second, action_type="write")

    first_list = client.get(f"/machines/{first}/behavior-declarations").json()
    second_list = client.get(f"/machines/{second}/behavior-declarations").json()

    assert [d["action_type"] for d in first_list] == ["read"]
    assert [d["action_type"] for d in second_list] == ["write"]


def test_list_declarations_missing_machine_returns_404(client):
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/behavior-declarations"
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_declarations_persist_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        machine_id = create_machine(first)
        created = create_declaration(first, machine_id).json()

    with TestClient(app) as second:
        response = second.get(f"/machines/{machine_id}/behavior-declarations")

    assert response.status_code == 200
    assert response.json() == [created]
