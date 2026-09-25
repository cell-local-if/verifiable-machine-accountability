"""Tests for the read-only desensitized machine identity privacy export.

Covers `GET /machines/{machine_id}/privacy-export`: the strict query
validation (``invalid_query`` before any machine read), ``404 not_found``
with no partial identity, GET-only ``405`` routing, the
``ext_ref`` / ``name_ref`` / ``key_ref`` SHA-256 desensitizing digests
(including ``null`` for non-string or blank values while the other fields
stay present), the absence of raw external-id/display-name/public-key text,
verbatim return of the stored version/status/timestamps, machine isolation,
the fixed-field-order compact newline-terminated body with an integer
version and no floating-point values, strict read-only byte stability, and
persistence across a restart.
"""
import hashlib
import json
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


def export_url(machine_id):
    return f"/machines/{machine_id}/privacy-export"


def create_machine(
    client,
    external_id="ext-1",
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
    return response.json()


IDENTITY_KEYS = [
    "machine_id",
    "ext_ref",
    "name_ref",
    "key_ref",
    "version",
    "status",
    "created_at",
    "updated_at",
]


def expected_ref(kind, machine_id, value):
    return hashlib.sha256(
        f"privacy:v1|{kind}{machine_id}{value}".encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Parameter validation, missing machine, method routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "?unexpected=1",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?x",
        "?=",
        "?x=1&y=2",
        "?machine_id=something",
    ],
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)["id"]
    response = client.get(f"/machines/{machine_id}/privacy-export{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_unknown_parameter_is_invalid_query(client):
    machine_id = create_machine(client)["id"]
    response = client.get(
        f"/machines/{machine_id}/privacy-export?x=1&x=2"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/machines/{missing}/privacy-export?unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_identity_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_database_returns_404(client):
    # No machines created at all: the current no-identity result is 404.
    response = client.get(
        "/machines/00000000-0000-0000-0000-000000000000/privacy-export"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)["id"]
    url = export_url(machine_id)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        response = client.request(method, url)
        assert response.status_code == 405


def test_non_get_with_query_still_405_without_validation_or_reads(client):
    # A non-GET request is routed out before query validation or any identity
    # read: a disallowed method is 405, not 422.
    machine_id = create_machine(client)["id"]
    response = client.post(
        f"/machines/{machine_id}/privacy-export?unexpected=1", json={}
    )
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and desensitized fields
# --------------------------------------------------------------------------- #


def test_envelope_shape_and_stored_fields(client):
    machine = create_machine(
        client, external_id="ext-1", display_name="Machine One",
        public_key="key-1",
    )
    machine_id = machine["id"]

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == IDENTITY_KEYS
    assert body["machine_id"] == machine_id
    assert body["version"] == 1
    assert isinstance(body["version"], int)
    assert body["status"] == "active"
    assert body["created_at"] == machine["created_at"]
    assert body["updated_at"] == machine["updated_at"]
    assert body["ext_ref"] == expected_ref("external", machine_id, "ext-1")
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")


def test_digests_strip_surrounding_whitespace(client, tmp_path):
    machine = create_machine(client)
    machine_id = machine["id"]
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ?, display_name = ?, public_key = ? "
        "WHERE id = ?",
        ("  ext-1\t", "\n Machine One  ", "  key-1 \t", machine_id),
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["ext_ref"] == expected_ref("external", machine_id, "ext-1")
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_value_digest_is_null_but_other_fields_returned(
    client, tmp_path, blank
):
    machine = create_machine(client)
    machine_id = machine["id"]
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ? WHERE id = ?", (blank, machine_id)
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert list(body.keys()) == IDENTITY_KEYS
    assert body["ext_ref"] is None
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")
    assert body["version"] == 1
    assert body["status"] == machine["status"]
    assert body["created_at"] == machine["created_at"]
    assert body["updated_at"] == machine["updated_at"]


def test_each_blank_ref_is_independent(client, tmp_path):
    machine = create_machine(client)
    machine_id = machine["id"]
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ' ', display_name = '  ', "
        "public_key = '' WHERE id = ?",
        (machine_id,),
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["ext_ref"] is None
    assert body["name_ref"] is None
    assert body["key_ref"] is None
    # The non-identity fields are still returned in full.
    assert body["machine_id"] == machine_id
    assert body["version"] == 1
    assert body["status"] == "active"
    assert body["created_at"]
    assert body["updated_at"]


def test_non_string_value_digest_is_null_but_identity_kept(client, tmp_path):
    machine = create_machine(client)
    machine_id = machine["id"]
    connection = sqlite3.connect(tmp_path / "test.db")
    # TEXT affinity leaves BLOB values untouched (numeric literals would be
    # coerced to text), so a BLOB yields a genuinely non-string stored value.
    connection.execute(
        "UPDATE machines SET external_id = ? WHERE id = ?",
        (b"\xff\xfe binary-ext", machine_id),
    )
    connection.commit()
    connection.close()

    body = client.get(export_url(machine_id)).json()
    assert body["ext_ref"] is None
    assert body["name_ref"] == expected_ref("display", machine_id, "Machine One")
    assert body["key_ref"] == expected_ref("public", machine_id, "key-1")


def test_raw_identity_values_never_appear_in_response(client, tmp_path):
    secret_ext = "the-secret-external-id"
    secret_name = "the-secret-display-name"
    secret_key = "the-secret-key"
    machine = create_machine(
        client, external_id=secret_ext, display_name=secret_name,
        public_key=secret_key,
    )
    machine_id = machine["id"]
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET external_id = ?, display_name = ? WHERE id = ?",
        (f"  {secret_ext}  ", f"\t{secret_name}\n", machine_id),
    )
    connection.commit()
    connection.close()

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert secret_ext not in response.text
    assert secret_name not in response.text
    assert secret_key not in response.text


def test_digests_are_scoped_to_machine(client):
    # external_id is globally unique, so the two machines differ there; the
    # display name and public key are deliberately identical. Every ref still
    # embeds the machine id, so identical raw values on different machines
    # hash differently.
    one = create_machine(client, "same-ext-1", "Same Name", "same-key")
    two = create_machine(client, "same-ext-2", "Same Name", "same-key")
    assert one["id"] != two["id"]

    body_one = client.get(export_url(one["id"])).json()
    body_two = client.get(export_url(two["id"])).json()
    assert body_one["ext_ref"] == expected_ref("external", one["id"], "same-ext-1")
    assert body_two["ext_ref"] == expected_ref("external", two["id"], "same-ext-2")
    assert body_one["name_ref"] == expected_ref("display", one["id"], "Same Name")
    assert body_two["name_ref"] == expected_ref("display", two["id"], "Same Name")
    assert body_one["key_ref"] == expected_ref("public", one["id"], "same-key")
    assert body_two["key_ref"] == expected_ref("public", two["id"], "same-key")
    assert body_one["ext_ref"] != body_two["ext_ref"]
    assert body_one["name_ref"] != body_two["name_ref"]
    assert body_one["key_ref"] != body_two["key_ref"]
    # The same raw value under different machine ids must never collide.
    assert expected_ref("display", one["id"], "Same Name") != expected_ref(
        "display", two["id"], "Same Name"
    )


def test_digest_kind_prefixes_are_distinct(client):
    machine = create_machine(
        client, external_id="same-value", display_name="same-value",
        public_key="same-value",
    )
    body = client.get(export_url(machine["id"])).json()
    assert body["ext_ref"] == expected_ref("external", machine["id"], "same-value")
    assert body["name_ref"] == expected_ref("display", machine["id"], "same-value")
    assert body["key_ref"] == expected_ref("public", machine["id"], "same-value")
    assert len({body["ext_ref"], body["name_ref"], body["key_ref"]}) == 3


def test_export_reflects_stored_status_version_and_key_after_mutations(client):
    machine = create_machine(client, public_key="key-1")
    machine_id = machine["id"]

    suspend = client.post(
        f"/machines/{machine_id}/status", json={"status": "suspended"}
    )
    assert suspend.status_code == 200
    rotated = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )
    assert rotated.status_code == 200

    body = client.get(export_url(machine_id)).json()
    assert body["status"] == "suspended"
    assert body["version"] == 2
    assert isinstance(body["version"], int)
    assert body["key_ref"] == expected_ref("public", machine_id, "key-2")
    assert body["updated_at"] == rotated.json()["updated_at"]
    assert body["created_at"] == machine["created_at"]


# --------------------------------------------------------------------------- #
# Body format, read-only byte stability, persistence
# --------------------------------------------------------------------------- #


def test_body_is_compact_newline_terminated_json_with_fixed_field_order(client):
    machine = create_machine(
        client, external_id="ext-1", display_name="Machine One",
        public_key="key-1",
    )
    machine_id = machine["id"]

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    raw = response.content
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    # Compact separators: no spaces after ':' or ','.
    assert b'": ' not in raw
    assert b", " not in raw

    expected = {
        "machine_id": machine_id,
        "ext_ref": expected_ref("external", machine_id, "ext-1"),
        "name_ref": expected_ref("display", machine_id, "Machine One"),
        "key_ref": expected_ref("public", machine_id, "key-1"),
        "version": 1,
        "status": "active",
        "created_at": machine["created_at"],
        "updated_at": machine["updated_at"],
    }
    assert raw == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    # The integer version is serialized without a decimal point: no floats,
    # -0.0, or non-finite tokens anywhere.
    assert b"1.0" not in raw
    assert b"-0.0" not in raw
    assert b"NaN" not in raw
    assert b"Infinity" not in raw


def test_export_is_read_only_and_byte_stable(client):
    machine = create_machine(
        client, external_id="ext-1", display_name="Machine One",
        public_key="key-1",
    )
    machine_id = machine["id"]
    other = create_machine(client, external_id="ext-2")
    client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": "key-2", "expected_version": 1},
    )
    machine_url = f"/machines/{machine_id}"
    other_url = f"/machines/{other['id']}"
    before = client.get(machine_url).json()
    before_other = client.get(other_url).json()
    rotations_url = f"/machines/{machine_id}/key-rotation-events"
    before_rotations = client.get(rotations_url).json()

    first = client.get(export_url(machine_id)).content
    second = client.get(export_url(machine_id)).content
    assert first == second

    # Nothing about either machine or the rotation history changed.
    assert client.get(machine_url).json() == before
    assert client.get(other_url).json() == before_other
    assert client.get(rotations_url).json() == before_rotations


def test_export_never_contains_another_machine_identity(client):
    one = create_machine(client, "ext-one", "Machine One", "key-one")
    create_machine(client, "ext-two", "Machine Two", "key-two")

    body = client.get(export_url(one["id"])).json()
    assert body["machine_id"] == one["id"]
    assert body["ext_ref"] == expected_ref("external", one["id"], "ext-one")
    assert body["name_ref"] == expected_ref("display", one["id"], "Machine One")
    assert body["key_ref"] == expected_ref("public", one["id"], "key-one")


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        expected = first.get(export_url(machine["id"])).content

    with TestClient(app) as second:
        response = second.get(export_url(machine["id"]))

    assert response.status_code == 200
    assert response.content == expected
    assert response.json()["machine_id"] == machine["id"]


def test_missing_machine_after_restart_is_404(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app):
        pass
    with TestClient(app) as second:
        response = second.get(
            "/machines/00000000-0000-0000-0000-000000000000/privacy-export"
        )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
