"""Tests for the read-only desensitized machine identity privacy export.

Covers `GET /machines/{machine_id}/privacy-export`: the strict query
validation (any query parameter is ``invalid_query`` before any machine
read), ``404 not_found`` with no identity data, GET-only ``405`` routing,
the ``ext_ref`` / ``name_ref`` / ``key_ref`` SHA-256 desensitizing digests
(including ``null`` for non-string or blank stored values), the absence of
raw external-id/display-name/public-key text, verbatim ``version``/``status``
/``created_at``/``updated_at``, machine isolation, the fixed-field-order
compact newline-terminated body, strict read-only byte stability, and
persistence across a restart.
"""
import hashlib
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def export_url(machine_id):
    return f"/machines/{machine_id}/privacy-export"


def create_machine(
    client,
    external_id="machine-1",
    display_name="Machine One",
    public_key="key-1",
):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": display_name,
            "public_key": public_key,
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def expected_ref(kind, machine_id, value):
    return hashlib.sha256(
        f"privacy:v1|{kind}{machine_id}{value}".encode("utf-8")
    ).hexdigest()


EXPORT_KEYS = [
    "id",
    "ext_ref",
    "name_ref",
    "key_ref",
    "version",
    "status",
    "created_at",
    "updated_at",
]


# --------------------------------------------------------------------------- #
# Parameter validation and routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?unexpected=1",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
        "?from_created_at=2026-03-01T00:00:00Z&to_created_at=2026-03-01T00:00:05Z",
        "?limit=10",
        "?x=",
    ],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(f"{export_url(machine_id)}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"{export_url(missing)}?unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_identity_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_database_returns_404(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'empty.db'}"
    )
    with TestClient(app) as empty_client:
        response = empty_client.get(export_url("any-machine-id"))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(export_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and desensitized fields
# --------------------------------------------------------------------------- #


def test_identity_exported_with_desensitized_fields(client):
    machine_id = create_machine(
        client,
        external_id="machine-1",
        display_name="Machine One",
        public_key="key-1",
    )
    listed = client.get(f"/machines/{machine_id}").json()

    body = client.get(export_url(machine_id)).json()
    assert list(body.keys()) == EXPORT_KEYS
    assert body["id"] == machine_id
    assert body["ext_ref"] == expected_ref("external", machine_id, "machine-1")
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")
    assert body["version"] == listed["version"]
    assert isinstance(body["version"], int)
    assert body["status"] == listed["status"]
    assert body["created_at"] == listed["created_at"]
    assert body["updated_at"] == listed["updated_at"]


def test_digest_strips_surrounding_whitespace(client, tmp_path):
    machine_id = create_machine(client)
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ?, display_name = ?, public_key = ? "
        "WHERE id = ?",
        ("  machine-1\t", "\n Machine One  ", "  key-1 ", machine_id),
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["ext_ref"] == expected_ref("external", machine_id, "machine-1")
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_identity_field_digest_is_null_but_record_kept(client, blank, tmp_path):
    machine_id = create_machine(client)
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ? WHERE id = ?", (blank, machine_id)
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["id"] == machine_id
    assert body["ext_ref"] is None
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")
    assert body["version"] == 1


def test_non_string_identity_field_digest_is_null_but_record_kept(client, tmp_path):
    machine_id = create_machine(client)
    # SQLite TEXT affinity would coerce a numeric literal to text, so store
    # BLOBs (which TEXT affinity leaves untouched) to get genuinely non-string
    # values. The privacy view must still return the record with null refs
    # rather than coercing or leaking either value.
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET display_name = ?, public_key = ? WHERE id = ?",
        (b"\xff\xfe binary-name", b"\x00\x01 bin", machine_id),
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["id"] == machine_id
    assert body["ext_ref"] == expected_ref("external", machine_id, "machine-1")
    assert body["name_ref"] is None
    assert body["key_ref"] is None


def test_raw_identity_values_never_appear_in_response(client, tmp_path):
    machine_id = create_machine(client)
    secret_external = "the-secret-external-id"
    secret_name = "the-secret-display-name"
    secret_key = "the-secret-public-key"
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ?, display_name = ?, public_key = ? "
        "WHERE id = ?",
        (f"  {secret_external}  ", secret_name, secret_key, machine_id),
    )
    connection.commit()
    connection.close()

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert secret_external not in response.text
    assert secret_name not in response.text
    assert secret_key not in response.text


def test_digest_is_scoped_to_machine(client):
    machine_one = create_machine(client, "machine-1", "Shared Name", "shared-key")
    machine_two = create_machine(client, "machine-2", "Shared Name", "shared-key")

    body_one = client.get(export_url(machine_one)).json()
    body_two = client.get(export_url(machine_two)).json()
    assert body_one["name_ref"] == expected_ref("display", machine_one, "Shared Name")
    assert body_two["name_ref"] == expected_ref("display", machine_two, "Shared Name")
    assert body_one["name_ref"] != body_two["name_ref"]
    assert body_one["key_ref"] != body_two["key_ref"]


def test_export_reflects_current_stored_identity(client):
    machine_id = create_machine(client, public_key="key-1")
    client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )
    client.post(f"/machines/{machine_id}/status", json={"status": "suspended"})

    listed = client.get(f"/machines/{machine_id}").json()
    body = client.get(export_url(machine_id)).json()
    assert body["key_ref"] == expected_ref("public", machine_id, "key-2")
    assert body["version"] == listed["version"] == 2
    assert body["status"] == "suspended"
    assert body["created_at"] == listed["created_at"]
    assert body["updated_at"] == listed["updated_at"]


# --------------------------------------------------------------------------- #
# Body format, read-only byte stability, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_newline_terminated_json_with_fixed_field_order(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert b"\n" not in raw[:-1]
    # Compact separators: no spaces after ':' or ','.
    assert b'": ' not in raw
    assert b", " not in raw
    listed = client.get(f"/machines/{machine_id}").json()
    expected = {
        "id": machine_id,
        "ext_ref": expected_ref("external", machine_id, "machine-1"),
        "name_ref": expected_ref("display", machine_id, "Machine One"),
        "key_ref": expected_ref("public", machine_id, "key-1"),
        "version": 1,
        "status": "active",
        "created_at": listed["created_at"],
        "updated_at": listed["updated_at"],
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")

    machine_url = f"/machines/{machine_id}"
    other_url = f"/machines/{other_machine}"
    before_machine = client.get(machine_url).json()
    before_other = client.get(other_url).json()

    first = client.get(export_url(machine_id)).content
    second = client.get(export_url(machine_id)).content
    assert first == second

    assert client.get(machine_url).json() == before_machine
    assert client.get(other_url).json() == before_other


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
