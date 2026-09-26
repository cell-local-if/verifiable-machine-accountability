"""Tests for the read-only behavior-declaration completeness audit.

Covers `GET /machines/{machine_id}/behavior-declarations/integrity`: GET-only
405, ``invalid_query`` 422 for any query parameter before the machine lookup,
``not_found`` 404 with no conclusion, the empty-machine ``true/0/null``
result, audit ordering by the actual UTC instant of ``created_at`` then id
(exact second before fractional second), every stored-field anomaly
(machine_id, UUID id, non-blank action/resource after trimming, boolean
``enabled``, Z-terminated UTC timestamps), the stripped-pair uniqueness rule
that flags the earliest-sorted record of a duplicate group, per-machine
isolation, strict read-only byte stability, and persistence across restarts.
"""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T0_HALF = "2026-03-01T00:00:00.500000Z"
T1 = "2026-03-01T00:00:01Z"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def integrity_url(machine_id):
    return f"/machines/{machine_id}/behavior-declarations/integrity"


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


def raw_connect(tmp_path):
    return sqlite3.connect(tmp_path / "test.db")


def tamper(tmp_path, statement, parameters=()):
    connection = raw_connect(tmp_path)
    connection.execute(statement, parameters)
    connection.commit()
    connection.close()


def insert_raw(tmp_path, row):
    connection = raw_connect(tmp_path)
    connection.execute(
        """
        INSERT INTO behavior_declarations
            (id, machine_id, action_type, resource_pattern, enabled,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row["machine_id"],
            row["action_type"],
            row["resource_pattern"],
            row["enabled"],
            row["created_at"],
            row["updated_at"],
        ),
    )
    connection.commit()
    connection.close()


def fetch_raw(tmp_path, machine_id):
    connection = raw_connect(tmp_path)
    rows = connection.execute(
        "SELECT id, machine_id, action_type, resource_pattern, enabled, "
        "created_at, updated_at FROM behavior_declarations "
        "WHERE machine_id = ? ORDER BY id",
        (machine_id,),
    ).fetchall()
    connection.close()
    return rows


# --------------------------------------------------------------------------- #
# Method and query-string handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(integrity_url(machine_id))
    assert response.status_code == 405
    # A rejected method neither reads for an audit nor writes anything.
    assert client.get(
        f"/machines/{machine_id}/behavior-declarations"
    ).json() == []


@pytest.mark.parametrize(
    "query", ["?unexpected=1", "?action_type=read", "?limit=1", "?foo="]
)
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_query_param_is_422_even_for_missing_machine(client):
    response = client.get(integrity_url(MISSING_ID) + "?foo=bar")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_conclusion(client):
    response = client.get(integrity_url(MISSING_ID))
    assert response.status_code == 404
    body = response.json()
    assert body == {"error": {"code": "not_found"}}
    assert "valid" not in body
    assert "checked_count" not in body
    assert "broken_declaration_id" not in body


# --------------------------------------------------------------------------- #
# Basic results
# --------------------------------------------------------------------------- #


def test_empty_machine_is_valid_true_zero_null(client):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_declaration_id": None,
    }
    # checked_count is a JSON integer, even for the empty-machine zero.
    assert json.loads(response.content)["checked_count"] == 0
    assert isinstance(json.loads(response.content)["checked_count"], int)


def test_sound_declarations_are_valid(client):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id, action_type="read", enabled=False)
    create_declaration(client, machine_id, action_type="write")

    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 2,
        "broken_declaration_id": None,
    }
    # enabled=false is a legitimate stored boolean and stays sound.
    assert first["enabled"] is False


def test_response_has_exactly_three_fields_in_order(client):
    machine_id = create_machine(client)
    create_declaration(client, machine_id)
    response = client.get(integrity_url(machine_id))
    assert list(response.json().keys()) == [
        "valid",
        "checked_count",
        "broken_declaration_id",
    ]


# --------------------------------------------------------------------------- #
# Stored-field anomalies
# --------------------------------------------------------------------------- #


def test_non_uuid_id_is_broken(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET id = 'not-a-uuid' WHERE id = ?",
        (record["id"],),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 1
    assert body["broken_declaration_id"] == "not-a-uuid"


def test_uppercase_uuid_is_broken(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET id = ? WHERE id = ?",
        (record["id"].upper(), record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["broken_declaration_id"] == record["id"].upper()


@pytest.mark.parametrize("value", ["   ", "\t\n", b"\x00bad"])
def test_bad_action_type_is_broken(client, tmp_path, value):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET action_type = ? WHERE id = ?",
        (value, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": record["id"],
    }


@pytest.mark.parametrize("value", ["  ", "\n\t ", b"\x00bad"])
def test_bad_resource_pattern_is_broken(client, tmp_path, value):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET resource_pattern = ? WHERE id = ?",
        (value, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": record["id"],
    }


@pytest.mark.parametrize("stored_enabled", [2, "yes", "true", b"x"])
def test_non_boolean_enabled_is_broken(client, tmp_path, stored_enabled):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET enabled = ? WHERE id = ?",
        (stored_enabled, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": record["id"],
    }


@pytest.mark.parametrize("stored_enabled", [0, 1])
def test_zero_and_one_enabled_are_sound(client, tmp_path, stored_enabled):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET enabled = ? WHERE id = ?",
        (stored_enabled, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": True,
        "checked_count": 1,
        "broken_declaration_id": None,
    }


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-13-01T00:00:00Z",  # shape passes, calendar value invalid
        "2026-01-01T00:00:00+00:00",  # offset form instead of Z
        "2026-01-01T00:00:00",  # missing Z
        "garbage",
        b"\x00bad",
    ],
)
def test_bad_created_at_is_broken(client, tmp_path, stamp):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET created_at = ? WHERE id = ?",
        (stamp, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 1
    assert body["broken_declaration_id"] == record["id"]


@pytest.mark.parametrize(
    "stamp",
    [
        "2026-02-30T00:00:00Z",
        "2026-01-01T24:00:00Z",
        "2026-01-01 00:00:00Z",
        b"\x00bad",
    ],
)
def test_bad_updated_at_is_broken(client, tmp_path, stamp):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET updated_at = ? WHERE id = ?",
        (stamp, record["id"]),
    )

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["broken_declaration_id"] == record["id"]


# --------------------------------------------------------------------------- #
# Ordering: actual UTC instant of created_at, then id
# --------------------------------------------------------------------------- #


def _declaration_row(machine_id, index, created_at, *, enabled=1, action="read",
                     resource="res/*"):
    return {
        "id": f"00000000-0000-0000-0000-{index:012d}",
        "machine_id": machine_id,
        "action_type": action,
        "resource_pattern": resource,
        "enabled": enabled,
        "created_at": created_at,
        "updated_at": created_at,
    }


def test_first_broken_record_uses_actual_utc_instant(client, tmp_path):
    machine_id = create_machine(client)
    # The fractional-second record is lexicographically earlier ('.' < 'Z')
    # but chronologically later; both are broken via enabled, so the
    # exact-second record must be reported first.
    later = _declaration_row(machine_id, 2, T0_HALF, enabled=2, action="a",
                             resource="r2")
    earlier = _declaration_row(machine_id, 1, T0, enabled=2, action="a",
                               resource="r1")
    insert_raw(tmp_path, later)
    insert_raw(tmp_path, earlier)

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["broken_declaration_id"] == earlier["id"]


def test_first_broken_record_tie_breaks_by_id(client, tmp_path):
    machine_id = create_machine(client)
    larger = _declaration_row(machine_id, 9, T0, enabled=2, action="a",
                              resource="r9")
    smaller = _declaration_row(machine_id, 1, T0, enabled=2, action="a",
                               resource="r1")
    insert_raw(tmp_path, larger)
    insert_raw(tmp_path, smaller)

    body = client.get(integrity_url(machine_id)).json()
    assert body["broken_declaration_id"] == smaller["id"]


def test_checked_count_counts_all_rows_even_when_later_broken(client, tmp_path):
    machine_id = create_machine(client)
    sound = _declaration_row(machine_id, 1, T0, action="a", resource="r1")
    broken = _declaration_row(machine_id, 2, T1, enabled=2, action="a",
                              resource="r2")
    insert_raw(tmp_path, sound)
    insert_raw(tmp_path, broken)

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 2,
        "broken_declaration_id": broken["id"],
    }


# --------------------------------------------------------------------------- #
# Stripped-pair uniqueness
# --------------------------------------------------------------------------- #


def test_duplicate_after_stripping_flags_earliest_record(client, tmp_path):
    machine_id = create_machine(client)
    first = create_declaration(client, machine_id, action_type="read",
                               resource_pattern="res/*")
    # A second stored row whose raw text differs only in surrounding
    # whitespace, created after the first: the creation API strips it, so
    # insert raw to simulate a legacy/external writer. The earliest-sorted
    # member — the API row — is the duplicate-group anomaly.
    duplicate = _declaration_row(machine_id, 2, "2030-01-01T00:00:00Z",
                                 action=" read ", resource="res/*")
    insert_raw(tmp_path, duplicate)

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 2,
        "broken_declaration_id": first["id"],
    }


def test_duplicate_group_uses_audit_order_not_insertion_order(client, tmp_path):
    machine_id = create_machine(client)
    # Inserted first but sorts later by created_at; the audit must flag the
    # earliest-sorted member of the duplicate group.
    inserted_first = _declaration_row(machine_id, 2, T1, action="read",
                                      resource="res/*")
    inserted_second = _declaration_row(machine_id, 1, T0, action="\tread",
                                       resource="  res/*  ")
    insert_raw(tmp_path, inserted_first)
    insert_raw(tmp_path, inserted_second)

    body = client.get(integrity_url(machine_id)).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["broken_declaration_id"] == inserted_second["id"]


def test_distinct_pairs_are_sound(client, tmp_path):
    machine_id = create_machine(client)
    insert_raw(tmp_path, _declaration_row(machine_id, 1, T0, action="read",
                                          resource="r1"))
    insert_raw(tmp_path, _declaration_row(machine_id, 2, T1, action="write",
                                          resource="r1"))
    insert_raw(tmp_path, _declaration_row(machine_id, 3, T1, action=" read",
                                          resource="r2"))

    body = client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": True,
        "checked_count": 3,
        "broken_declaration_id": None,
    }


def test_identical_raw_duplicates_flag_earliest_in_legacy_db(
    tmp_path, monkeypatch
):
    # A legacy/external table written without the unique constraint can hold
    # two byte-identical pairs.
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE machines (
            id VARCHAR(36) PRIMARY KEY, external_id VARCHAR UNIQUE,
            display_name VARCHAR, public_key VARCHAR, status VARCHAR,
            version INTEGER, created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE behavior_declarations (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            action_type VARCHAR, resource_pattern VARCHAR, enabled BOOLEAN,
            created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    machine_id = "11111111-1111-1111-1111-111111111111"
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, T0, T0),
    )
    connection.execute(
        "INSERT INTO behavior_declarations VALUES (?,?,?,?,?,?,?)",
        ("22222222-0000-0000-0000-000000000002", machine_id, "read", "r",
         1, T1, T1),
    )
    connection.execute(
        "INSERT INTO behavior_declarations VALUES (?,?,?,?,?,?,?)",
        ("22222222-0000-0000-0000-000000000001", machine_id, "read", "r",
         1, T0, T0),
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as test_client:
        body = test_client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 2,
        "broken_declaration_id": "22222222-0000-0000-0000-000000000001",
    }


# --------------------------------------------------------------------------- #
# Machine isolation, read-only, persistence
# --------------------------------------------------------------------------- #


def test_other_machines_broken_declarations_never_enter_audit(
    client, tmp_path
):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    sound = create_declaration(client, machine_one)
    damaged = create_declaration(client, machine_two)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET enabled = 2 WHERE id = ?",
        (damaged["id"],),
    )

    assert client.get(integrity_url(machine_one)).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_declaration_id": None,
    }
    assert client.get(integrity_url(machine_two)).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": damaged["id"],
    }
    assert sound["id"] != damaged["id"]


def test_audit_is_read_only_and_byte_stable(client, tmp_path):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read")
    create_declaration(client, machine_id, action_type="write", enabled=False)
    before_rows = fetch_raw(tmp_path, machine_id)
    before_machine = client.get(f"/machines/{machine_id}").json()

    first = client.get(integrity_url(machine_id)).content
    second = client.get(integrity_url(machine_id)).content
    assert first == second
    # Repeat calls are byte-stable and the count is a JSON integer.
    parsed = json.loads(first)
    assert isinstance(parsed["checked_count"], int)

    assert fetch_raw(tmp_path, machine_id) == before_rows
    assert client.get(f"/machines/{machine_id}").json() == before_machine
    # The listing is also untouched.
    listing = client.get(f"/machines/{machine_id}/behavior-declarations")
    assert [d["action_type"] for d in listing.json()] == ["read", "write"]


def test_audit_does_not_change_authorization_evaluation(client, tmp_path):
    machine_id = create_machine(client)
    create_declaration(client, machine_id, action_type="read",
                       resource_pattern="res/*", enabled=True)
    # Authorization also requires a matching global allow policy rule.
    policy = client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "*",
            "effect": "allow",
            "priority": 0,
        },
    )
    assert policy.status_code == 201
    before = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/thing"},
    ).json()

    client.get(integrity_url(machine_id))
    client.get(integrity_url(machine_id))

    after = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/thing"},
    ).json()
    assert after == before
    assert after["allowed"] is True


def test_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        create_declaration(first, machine_id, action_type="read")
        create_declaration(first, machine_id, action_type="write")
        sound_result = first.get(integrity_url(machine_id)).content

    with TestClient(app) as second:
        assert second.get(integrity_url(machine_id)).content == sound_result

    # And stays byte-identical after a further restart.
    with TestClient(app) as third:
        assert third.get(integrity_url(machine_id)).content == sound_result


def test_broken_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist-broken.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        record = create_declaration(first, machine_id)

    connection = sqlite3.connect(tmp_path / "persist-broken.db")
    connection.execute(
        "UPDATE behavior_declarations SET resource_pattern = '   ' WHERE id = ?",
        (record["id"],),
    )
    connection.commit()
    connection.close()

    with TestClient(app) as second:
        body = second.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": record["id"],
    }


def test_empty_database_serves_integrity(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}"
    )
    with TestClient(app) as test_client:
        # No machines at all: still a clean 404, never a startup error.
        response = test_client.get(integrity_url(MISSING_ID))
        assert response.status_code == 404


def test_null_fields_in_a_legacy_table_are_broken(tmp_path, monkeypatch):
    # A table written before the NOT NULL contract can hold NULL values; the
    # read-only audit must flag them rather than coerce or crash.
    db_path = tmp_path / "nulls.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE machines (
            id VARCHAR(36) PRIMARY KEY, external_id VARCHAR UNIQUE,
            display_name VARCHAR, public_key VARCHAR, status VARCHAR,
            version INTEGER, created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE behavior_declarations (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            action_type VARCHAR, resource_pattern VARCHAR, enabled,
            created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    machine_id = "11111111-1111-1111-1111-111111111111"
    broken_id = "33333333-3333-3333-3333-333333333333"
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, T0, T0),
    )
    connection.execute(
        "INSERT INTO behavior_declarations VALUES (?,?,?,?,?,?,?)",
        (broken_id, machine_id, "read", "r", None, T0, None),
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as test_client:
        body = test_client.get(integrity_url(machine_id)).json()
    assert body == {
        "valid": False,
        "checked_count": 1,
        "broken_declaration_id": broken_id,
    }


def test_non_text_id_is_surfaced_without_500(client, tmp_path):
    machine_id = create_machine(client)
    record = create_declaration(client, machine_id)
    tamper(
        tmp_path,
        "UPDATE behavior_declarations SET id = x'deadbeef' WHERE id = ?",
        (record["id"],),
    )

    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["checked_count"] == 1
    # The damaged identifier is surfaced as replacement text, never a 500.
    assert isinstance(body["broken_declaration_id"], str)
