import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "machine_id",
    "old_public_key",
    "new_public_key",
    "version",
    "created_at",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def canonical_content_hash(record: dict) -> str:
    document = json.dumps(
        {key: record[key] for key in CONTENT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


def create_machine(client, external_id="machine-1", public_key="key-1"):
    response = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "Machine One",
            "public_key": public_key,
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


def rotate(client, machine_id, public_key, expected_version):
    return client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": public_key, "expected_version": expected_version},
    )


def events(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events")


def integrity(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events/integrity")


def rotate_n(client, machine_id, count, start_version=1):
    for index in range(count):
        response = rotate(
            client, machine_id, f"key-{start_version + index + 1}", start_version + index
        )
        assert response.status_code == 200


def test_listed_rotations_carry_chain_fields(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)

    body = events(client, machine_id).json()
    assert len(body) == 2
    for record in body:
        assert {"previous_rotation_id", "content_hash", "chain_hash"} <= set(record)
        assert HEX64_RE.match(record["content_hash"])
        assert HEX64_RE.match(record["chain_hash"])
        assert record["content_hash"] == canonical_content_hash(record)


def test_first_rotation_chain_hash_uses_empty_previous(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 1)

    (record,) = events(client, machine_id).json()
    assert record["previous_rotation_id"] is None
    assert record["chain_hash"] == chain_hash("", record["content_hash"])


def test_rotations_link_in_created_order(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 4)

    body = events(client, machine_id).json()
    assert [r["version"] for r in body] == [2, 3, 4, 5]
    assert body[0]["previous_rotation_id"] is None
    for index in range(1, len(body)):
        assert body[index]["previous_rotation_id"] == body[index - 1]["id"]
        assert body[index]["chain_hash"] == chain_hash(
            body[index - 1]["chain_hash"], body[index]["content_hash"]
        )


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="m-1", public_key="key-a1")
    machine_two = create_machine(client, external_id="m-2", public_key="key-b1")

    rotate_n(client, machine_one, 1)
    rotate_n(client, machine_two, 2)

    first_one = events(client, machine_one).json()[0]
    first_two, second_two = events(client, machine_two).json()
    assert first_one["previous_rotation_id"] is None
    assert first_two["previous_rotation_id"] is None
    assert second_two["previous_rotation_id"] == first_two["id"]
    assert first_one["chain_hash"] == chain_hash("", first_one["content_hash"])
    assert first_two["chain_hash"] == chain_hash("", first_two["content_hash"])


def test_integrity_valid_reports_count(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)

    response = integrity(client, machine_id)
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 3,
        "broken_rotation_id": None,
    }


def test_integrity_empty_machine_is_valid(client):
    machine_id = create_machine(client)
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_rotation_id": None,
    }


def test_integrity_missing_machine_returns_404(client):
    response = integrity(client, "00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_integrity_is_read_only(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    before = events(client, machine_id).json()

    assert integrity(client, machine_id).json()["valid"] is True
    assert integrity(client, machine_id).json()["valid"] is True

    assert events(client, machine_id).json() == before


def test_integrity_detects_tampered_content(client, tmp_path):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    records = events(client, machine_id).json()

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET old_public_key = 'forged' WHERE id = ?",
        (records[1]["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_id).json() == {
        "valid": False,
        "checked_count": 3,
        "broken_rotation_id": records[1]["id"],
    }


def test_integrity_detects_tampered_previous_link(client, tmp_path):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    records = events(client, machine_id).json()

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET previous_rotation_id = NULL WHERE id = ?",
        (records[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["checked_count"] == 3
    assert response["broken_rotation_id"] == records[2]["id"]


def test_integrity_detects_tampered_chain_hash(client, tmp_path):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    records = events(client, machine_id).json()

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, records[0]["id"]),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["broken_rotation_id"] == records[0]["id"]


def test_integrity_reports_first_broken_record(client, tmp_path):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    records = events(client, machine_id).json()

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET new_public_key = 'forged' WHERE id = ?",
        (records[0]["id"],),
    )
    connection.execute(
        "UPDATE key_rotation_events SET new_public_key = 'forged' WHERE id = ?",
        (records[2]["id"],),
    )
    connection.commit()
    connection.close()

    response = integrity(client, machine_id).json()
    assert response["valid"] is False
    assert response["broken_rotation_id"] == records[0]["id"]


def test_integrity_is_isolated_per_machine(client, tmp_path):
    machine_one = create_machine(client, external_id="m-1", public_key="key-a1")
    machine_two = create_machine(client, external_id="m-2", public_key="key-b1")
    rotate_n(client, machine_one, 2)
    rotate_n(client, machine_two, 2)
    other_records = events(client, machine_two).json()

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events SET old_public_key = 'forged' WHERE id = ?",
        (other_records[0]["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine_one).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_rotation_id": None,
    }
    assert integrity(client, machine_two).json()["valid"] is False


def test_concurrent_rotations_keep_chain_unbroken(client):
    machine_id = create_machine(client)

    def do_rotate(index):
        return rotate(client, machine_id, f"key-r{index}", 1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_rotate, range(20)))

    # Exactly one rotation wins the version race; the rest conflict.
    assert sorted(r.status_code for r in responses) == [200] + [409] * 19

    body = events(client, machine_id).json()
    assert len(body) == 1
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_rotation_id": None,
    }

    # Further sequential rotations extend the same sound chain.
    rotate_n(client, machine_id, 5, start_version=2)
    assert integrity(client, machine_id).json() == {
        "valid": True,
        "checked_count": 6,
        "broken_rotation_id": None,
    }


def test_legacy_records_are_backfilled_on_startup(tmp_path, monkeypatch):
    machine_id = "11111111-1111-1111-1111-111111111111"
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
        CREATE TABLE key_rotation_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            old_public_key VARCHAR, new_public_key VARCHAR,
            version INTEGER, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key-3", "active", 3, "t0", "t3"),
    )
    legacy_rows = [
        (
            "aaaaaaaa-0000-0000-0000-000000000002",
            machine_id,
            "key-1",
            "key-2",
            2,
            "2026-01-01T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000003",
            machine_id,
            "key-2",
            "key-3",
            3,
            "2026-01-02T00:00:00.000000Z",
        ),
        (
            "aaaaaaaa-0000-0000-0000-000000000001",
            machine_id,
            "key-0",
            "key-1",
            1,
            "2026-01-01T00:00:00.000000Z",
        ),
    ]
    connection.executemany(
        "INSERT INTO key_rotation_events VALUES (?,?,?,?,?,?)",
        legacy_rows,
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        body = events(client, machine_id).json()

        # Ordered by (created_at, id), not insertion order.
        assert [r["old_public_key"] for r in body] == ["key-0", "key-1", "key-2"]
        assert body[0]["previous_rotation_id"] is None
        assert body[1]["previous_rotation_id"] == body[0]["id"]
        assert body[2]["previous_rotation_id"] == body[1]["id"]
        for record in body:
            assert HEX64_RE.match(record["content_hash"])
            assert HEX64_RE.match(record["chain_hash"])
            assert record["content_hash"] == canonical_content_hash(record)
        assert body[0]["chain_hash"] == chain_hash("", body[0]["content_hash"])
        assert body[1]["chain_hash"] == chain_hash(
            body[0]["chain_hash"], body[1]["content_hash"]
        )

        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_rotation_id": None,
        }


def test_chain_hashes_stay_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        rotate_n(first, machine_id, 3)
        created = events(first, machine_id).json()

    with TestClient(app) as second:
        assert events(second, machine_id).json() == created
        assert integrity(second, machine_id).json() == {
            "valid": True,
            "checked_count": 3,
            "broken_rotation_id": None,
        }

    # A further restart must still be a no-op (deterministic recomputation).
    with TestClient(app) as third:
        assert events(third, machine_id).json() == created
