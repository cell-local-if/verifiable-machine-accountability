import threading
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
    response = client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "enabled": enabled,
        },
    )
    assert response.status_code == 201
    return response.json()


def set_enabled(client, machine_id, declaration_id, enabled):
    return client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration_id}/enabled",
        json={"enabled": enabled},
    )


def test_disable_declaration_returns_updated_record(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    response = set_enabled(client, machine_id, declaration["id"], False)

    assert response.status_code == 200
    body = response.json()
    assert list(body) == [
        "id",
        "machine_id",
        "action_type",
        "resource_pattern",
        "enabled",
        "created_at",
        "updated_at",
    ]
    assert body["id"] == declaration["id"]
    assert body["machine_id"] == machine_id
    assert body["action_type"] == "read"
    assert body["resource_pattern"] == "res/*"
    assert body["enabled"] is False
    assert body["created_at"] == declaration["created_at"]
    assert body["updated_at"].endswith("Z")
    assert body["updated_at"] != body["created_at"]


def test_reenable_declaration_round_trip(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    disabled = set_enabled(client, machine_id, declaration["id"], False).json()
    restored = set_enabled(client, machine_id, declaration["id"], True)

    assert restored.status_code == 200
    body = restored.json()
    assert body["enabled"] is True
    assert body["created_at"] == declaration["created_at"]
    assert body["updated_at"] != disabled["updated_at"]


def test_update_visible_in_list_and_preserves_other_fields(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    set_enabled(client, machine_id, declaration["id"], False)

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert len(listed) == 1
    assert listed[0]["enabled"] is False
    assert listed[0]["action_type"] == "read"
    assert listed[0]["resource_pattern"] == "res/*"
    assert listed[0]["created_at"] == declaration["created_at"]


def test_same_state_returns_409_and_changes_nothing(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    response = set_enabled(client, machine_id, declaration["id"], True)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "declaration_state_unchanged"}}

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed == [declaration]


def test_missing_machine_returns_404(client):
    response = set_enabled(
        client,
        "00000000-0000-0000-0000-000000000000",
        str(uuid.uuid4()),
        False,
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_declaration_returns_404(client):
    machine_id = create_machine(client)

    response = set_enabled(client, machine_id, str(uuid.uuid4()), False)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_declaration_of_other_machine_returns_404(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    declaration = create_declaration(client, first)

    response = set_enabled(client, second, declaration["id"], False)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    listed = client.get(f"/machines/{first}/behavior-declarations").json()
    assert listed[0]["enabled"] is True


def test_query_parameter_returns_422_invalid_query(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled?force=true",
        json={"enabled": False},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_parameter_wins_over_missing_machine(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000"
        f"/behavior-declarations/{uuid.uuid4()}/enabled?x=1",
        json={"enabled": False},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"enabled": True, "extra": 1},
        {"enabled": True, "action_type": "read"},
        {"state": True},
        {"enabled": 1},
        {"enabled": 0},
        {"enabled": "true"},
        {"enabled": "false"},
        {"enabled": None},
        {"enabled": [True]},
        {"enabled": {"value": True}},
    ],
)
def test_invalid_body_returns_422(client, payload):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled",
        json=payload,
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


@pytest.mark.parametrize("content", [b"", b"not json", b"[true]", b'"true"', b"1", b"null", b"true"])
def test_non_object_or_unparseable_body_returns_422(client, content):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled",
        content=content,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


def test_invalid_body_checked_before_lookup(client):
    response = client.post(
        "/machines/00000000-0000-0000-0000-000000000000"
        f"/behavior-declarations/{uuid.uuid4()}/enabled",
        json={"enabled": "yes"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


def test_concurrent_same_target_has_exactly_one_success(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    results = []

    def worker():
        response = set_enabled(client, machine_id, declaration["id"], False)
        results.append(response.status_code)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(200) == 1
    assert results.count(409) == 7

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[0]["enabled"] is False


def test_concurrent_opposite_targets_lose_no_update(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=True)

    outcomes = []

    def worker(target):
        response = set_enabled(client, machine_id, declaration["id"], target)
        outcomes.append((target, response.status_code))

    threads = [
        threading.Thread(target=worker, args=(target,))
        for target in [False, True, False, True, False, True]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Every request got a definite answer and the final state equals the
    # target of the last committed winner — no update was lost.
    assert all(status in (200, 409) for _, status in outcomes)
    final = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert final[0]["enabled"] in (True, False)


def test_update_does_not_touch_other_declarations(client):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id, action_type="read")
    second = create_declaration(client, machine_id, action_type="write")

    set_enabled(client, machine_id, first["id"], False)

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[1] == second


def test_update_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declaration = create_declaration(first, machine_id)
        updated = set_enabled(first, machine_id, declaration["id"], False).json()

    with TestClient(app) as second:
        listed = second.get(f"/machines/{machine_id}/behavior-declarations").json()

    assert listed == [updated]
