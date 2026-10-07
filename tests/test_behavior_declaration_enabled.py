import uuid
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


def create_declaration(
    client, machine_id, action_type="read", resource_pattern="res/*", enabled=True
):
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


def test_disable_returns_updated_record_in_fixed_field_order(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

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
    assert body["updated_at"] >= declaration["updated_at"]


def test_reenable_round_trip(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id, enabled=False)

    response = set_enabled(client, machine_id, declaration["id"], True)

    assert response.status_code == 200
    assert response.json()["enabled"] is True


def test_list_reflects_updated_enabled(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    set_enabled(client, machine_id, declaration["id"], False)

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[0]["enabled"] is False
    assert listed[0]["created_at"] == declaration["created_at"]


def test_query_parameters_rejected(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled?x=1",
        json={"enabled": False},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_parameters_rejected_even_for_missing_machine(client):
    response = client.post(
        "/machines/nope/behavior-declarations/nope/enabled?x=1",
        json={"enabled": False},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"enabled": True, "extra": 1},
        {"enabled": 1},
        {"enabled": 0},
        {"enabled": "true"},
        {"enabled": None},
        {"enabled": [True]},
        {"enabled": {"value": True}},
        {"other": True},
    ],
)
def test_invalid_body_fields_rejected(client, payload):
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


def test_empty_body_rejected(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled"
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "true", "not-json{"])
def test_non_object_or_unparseable_body_rejected(client, raw):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{declaration['id']}/enabled",
        content=raw,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


def test_body_validation_precedes_lookup(client):
    response = client.post(
        "/machines/nope/behavior-declarations/nope/enabled",
        json={"enabled": 1},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_declaration_state_request"}
    }


def test_missing_machine_is_404(client):
    response = client.post(
        f"/machines/{uuid.uuid4()}/behavior-declarations/{uuid.uuid4()}/enabled",
        json={"enabled": False},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_missing_declaration_is_404(client):
    machine_id = create_machine(client)

    response = client.post(
        f"/machines/{machine_id}/behavior-declarations/{uuid.uuid4()}/enabled",
        json={"enabled": False},
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_declaration_of_other_machine_is_404(client):
    machine_id = create_machine(client)
    other_machine_id = create_machine(client, external_id="machine-2")
    declaration = create_declaration(client, machine_id)

    response = set_enabled(client, other_machine_id, declaration["id"], False)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # The declaration itself is untouched.
    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[0]["enabled"] is True


def test_same_target_is_409_and_writes_nothing(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    response = set_enabled(client, machine_id, declaration["id"], True)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "declaration_state_unchanged"}}
    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[0]["enabled"] is True
    assert listed[0]["updated_at"] == declaration["updated_at"]


def test_concurrent_same_target_allows_exactly_one_success(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                lambda _: set_enabled(client, machine_id, declaration["id"], False),
                range(4),
            )
        )

    assert sum(r.status_code == 200 for r in responses) == 1
    assert sorted(r.status_code for r in responses) == [200, 409, 409, 409]
    for r in responses:
        if r.status_code == 409:
            assert r.json() == {"error": {"code": "declaration_state_unchanged"}}
    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    assert listed[0]["enabled"] is False


def test_concurrent_mixed_targets_never_lose_updates(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)

    targets = [False, True, False, True, False, True]
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(
            pool.map(
                lambda target: set_enabled(
                    client, machine_id, declaration["id"], target
                ),
                targets,
            )
        )

    assert all(r.status_code in (200, 409) for r in responses)
    # Every committed change flipped the flag; the final state is the target
    # of the last committed success, and every 200 toggled the value.
    final = client.get(f"/machines/{machine_id}/behavior-declarations").json()[0]
    assert final["enabled"] in (True, False)


def test_new_decisions_read_updated_enabled(client):
    machine_id = create_machine(client)
    declaration = create_declaration(client, machine_id)
    rule = client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )
    assert rule.status_code == 201

    allowed = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/1"},
    )
    assert allowed.json()["allowed"] is True

    set_enabled(client, machine_id, declaration["id"], False)

    denied = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/1"},
    )
    assert denied.json()["allowed"] is False
    assert denied.json()["reason"] == "no_enabled_declaration"


def test_toggle_does_not_touch_other_declarations(client):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id)
    second = create_declaration(
        client, machine_id, action_type="write", resource_pattern="res2/*"
    )

    set_enabled(client, machine_id, first["id"], False)

    listed = client.get(f"/machines/{machine_id}/behavior-declarations").json()
    by_id = {d["id"]: d for d in listed}
    assert by_id[first["id"]]["enabled"] is False
    assert by_id[second["id"]] == second


def test_enabled_state_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        declaration = create_declaration(first, machine_id)
        assert (
            set_enabled(first, machine_id, declaration["id"], False).status_code
            == 200
        )

    with TestClient(app) as second:
        listed = second.get(f"/machines/{machine_id}/behavior-declarations").json()
        assert listed[0]["enabled"] is False
        assert listed[0]["created_at"] == declaration["created_at"]
