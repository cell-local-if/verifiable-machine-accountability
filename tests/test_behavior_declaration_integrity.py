import sqlite3
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


def integrity(client, machine_id, **kwargs):
    return client.get(
        f"/machines/{machine_id}/behavior-declarations/integrity", **kwargs
    )


def raw_update(db_path, sql, params=()):
    connection = sqlite3.connect(db_path)
    connection.execute(sql, params)
    connection.commit()
    connection.close()


def raw_insert_declaration(db_path, row):
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO behavior_declarations VALUES (?,?,?,?,?,?,?)", row
    )
    connection.commit()
    connection.close()


def test_integrity_empty_machine_is_valid(client):
    machine_id = create_machine(client)

    response = integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_declaration_id": None,
    }


def test_integrity_valid_machine_reports_count(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read")
    create_declaration(client, machine_id, action_type="write", enabled=False)

    response = integrity(client, machine_id)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_declaration_id": None,
    }


def test_integrity_missing_machine_returns_404(client):
    response = integrity(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_rejects_any_query_parameter(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id)

    response = integrity(client, machine_id, params={"action_type": "read"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_integrity_query_parameter_checked_before_machine_lookup(client):
    response = integrity(
        client,
        "00000000-0000-0000-0000-000000000000",
        params={"unknown": "x"},
    )

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_integrity_rejects_non_get_methods(client):
    machine_id = create_machine(client)

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(
            f"/machines/{machine_id}/behavior-declarations/integrity"
        )
        assert response.status_code == 405


def test_integrity_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read")
    create_declaration(client, machine_id, action_type="write")
    before = client.get(f"/machines/{machine_id}/behavior-declarations").json()

    first = integrity(client, machine_id)
    second = integrity(client, machine_id)

    assert first.content == second.content
    assert client.get(f"/machines/{machine_id}/behavior-declarations").json() == before


def test_integrity_detects_blank_action_type(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET action_type = '   ' WHERE id = ?",
        (record["id"],),
    )

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": record["id"],
    }


def test_integrity_detects_non_boolean_enabled(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET enabled = 2 WHERE id = ?",
        (record["id"],),
    )

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["broken_declaration_id"] == record["id"]


def test_integrity_detects_non_utc_timestamp(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET created_at = '2026-01-01 00:00:00'"
        " WHERE id = ?",
        (record["id"],),
    )

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["broken_declaration_id"] == record["id"]


def test_integrity_detects_offset_timestamp(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00+00:00", record["id"]),
    )

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["broken_declaration_id"] == record["id"]


def test_integrity_detects_non_uuid_id(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET id = 'not-a-uuid' WHERE id = ?",
        (record["id"],),
    )

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": "not-a-uuid",
    }


def test_integrity_detects_duplicate_trimmed_combination(client, tmp_path):
    machine_id = create_machine(client)
    first = create_declaration(
        client, machine_id, action_type="read", resource_pattern="res/*"
    )
    # The stored unique constraint covers raw values, so a duplicate under
    # trimming can only arrive through direct storage manipulation.
    raw_insert_declaration(
        tmp_path / "test.db",
        (
            str(uuid.uuid4()),
            machine_id,
            "read ",
            "res/*",
            1,
            "2027-01-01T00:00:00Z",
            "2027-01-01T00:00:00Z",
        ),
    )

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["checked_count"] == 2
    # Inside the duplicate group the record sorting first is judged broken.
    assert response["broken_declaration_id"] == first["id"]


def test_integrity_reports_first_broken_in_created_order(client, tmp_path):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id, action_type="read")
    second = create_declaration(client, machine_id, action_type="write")
    db_path = tmp_path / "test.db"
    raw_update(
        db_path,
        "UPDATE behavior_declarations SET action_type = '' WHERE id = ?",
        (second["id"],),
    )
    raw_update(
        db_path,
        "UPDATE behavior_declarations SET enabled = 'yes' WHERE id = ?",
        (first["id"],),
    )

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["checked_count"] == 2
    assert response["broken_declaration_id"] == first["id"]


def test_integrity_is_isolated_per_machine(client, tmp_path):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    create_declaration(client, machine_one, action_type="read")
    other = create_declaration(client, machine_two, action_type="write")
    raw_update(
        tmp_path / "test.db",
        "UPDATE behavior_declarations SET action_type = '' WHERE id = ?",
        (other["id"],),
    )

    assert integrity(client, machine_one).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_declaration_id": None,
    }
    assert integrity(client, machine_two).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": other["id"],
    }


def test_integrity_stable_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'persist.db'}"
    )

    with TestClient(app) as first:
        machine_id = create_machine(first)
        create_declaration(first, machine_id, action_type="read")
        create_declaration(first, machine_id, action_type="write")
        created = integrity(first, machine_id).content

    with TestClient(app) as second:
        assert integrity(second, machine_id).content == created
        assert second.get(
            f"/machines/{machine_id}/behavior-declarations"
        ).status_code == 200
