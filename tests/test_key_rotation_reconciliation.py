"""Tests for the read-only key-rotation/current-state reconciliation endpoint.

    GET /machines/{machine_id}/key-rotation-events/reconciliation

The endpoint cross-checks, strictly read-only and only for the path
machine, that the machine's stored rotation records fully explain its
current key state: every record's ``created_at`` parses, the chain links
and hashes verify in check order, the first record carries version 2 and
every later record increments it by one, every record after the first
continues the key hand-off, and the chain tail matches the machine's
current ``public_key`` and ``version``. An empty history reconciles only
with a machine still at version 1.

These tests cover the sound flows (rotated, empty), every anomaly
category (timestamp, chain, version transition, key transition, current
state), the check order (instant then id, exact second before fractional,
broken moments last), first-anomaly reporting, machine isolation,
request-shape validation (422 before 404), 404, 405, the exact response
shape, and the read-only guarantee across repeated calls and restarts.
"""
import hashlib
import json
import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

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


def rotate_n(client, machine_id, count, start_version=1):
    for index in range(count):
        response = rotate(
            client, machine_id, f"key-{start_version + index + 1}", start_version + index
        )
        assert response.status_code == 200


def events(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events").json()


def reconciliation_url(machine_id):
    return f"/machines/{machine_id}/key-rotation-events/reconciliation"


def reconcile(client, machine_id):
    return client.get(reconciliation_url(machine_id))


def db_execute(client, statement, **params):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement).bindparams(**params))


def recompute_chain(records):
    """Recompute link, content hash, and chain hash over the given order."""
    previous_chain_hash = ""
    previous_id = None
    for record in records:
        record["previous_rotation_id"] = previous_id
        record["content_hash"] = canonical_content_hash(record)
        record["chain_hash"] = chain_hash(previous_chain_hash, record["content_hash"])
        previous_chain_hash = record["chain_hash"]
        previous_id = record["id"]
    return records


def update_record(client, record):
    db_execute(
        client,
        "UPDATE key_rotation_events SET previous_rotation_id = :previous_rotation_id,"
        " content_hash = :content_hash, chain_hash = :chain_hash WHERE id = :id",
        **{key: record[key] for key in
           ("previous_rotation_id", "content_hash", "chain_hash", "id")},
    )


# --------------------------------------------------------------------------- #
# Sound conclusions
# --------------------------------------------------------------------------- #


def test_valid_history_reports_tail(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    tail = events(client, machine_id)[-1]

    response = reconcile(client, machine_id)
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 3,
        "latest_rotation_id": tail["id"],
        "broken_rotation_id": None,
        "anomaly": None,
    }


def test_empty_history_at_version_one_is_valid(client):
    machine_id = create_machine(client)
    assert reconcile(client, machine_id).json() == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 0,
        "latest_rotation_id": None,
        "broken_rotation_id": None,
        "anomaly": None,
    }


def test_single_rotation_is_valid(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 1)
    (record,) = events(client, machine_id)

    body = reconcile(client, machine_id).json()
    assert body["valid"] is True
    assert body["checked_count"] == 1
    assert body["latest_rotation_id"] == record["id"]


def test_exact_second_sorts_before_fractional_of_same_second(client, tmp_path):
    # Two records in one second: the exact-second stamp is the earlier
    # instant even though it sorts *after* the fractional stamp as text.
    machine_id = create_machine(client)
    first = {
        "id": "bbbbbbbb-0000-0000-0000-000000000001",
        "machine_id": machine_id,
        "old_public_key": "key-1",
        "new_public_key": "key-2",
        "version": 2,
        "created_at": "2026-01-01T00:00:00Z",
    }
    second = {
        "id": "aaaaaaaa-0000-0000-0000-000000000002",
        "machine_id": machine_id,
        "old_public_key": "key-2",
        "new_public_key": "key-3",
        "version": 3,
        "created_at": "2026-01-01T00:00:00.500000Z",
    }
    first, second = recompute_chain([first, second])
    connection = sqlite3.connect(tmp_path / "test.db")
    for record in (first, second):
        connection.execute(
            "INSERT INTO key_rotation_events VALUES (?,?,?,?,?,?,?,?,?)",
            (
                record["id"],
                machine_id,
                record["old_public_key"],
                record["new_public_key"],
                record["version"],
                record["created_at"],
                record["previous_rotation_id"],
                record["content_hash"],
                record["chain_hash"],
            ),
        )
    connection.execute(
        "UPDATE machines SET public_key = 'key-3', version = 3 WHERE id = ?",
        (machine_id,),
    )
    connection.commit()
    connection.close()

    body = reconcile(client, machine_id).json()
    assert body["valid"] is True
    assert body["checked_count"] == 2
    assert body["latest_rotation_id"] == second["id"]


# --------------------------------------------------------------------------- #
# Request shape, machine lookup, and method validation
# --------------------------------------------------------------------------- #


def test_query_string_is_rejected_before_lookup(client):
    response = client.get(reconciliation_url("missing"), params={"x": "1"})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_query_param_is_rejected(client):
    machine_id = create_machine(client)
    response = client.get(reconciliation_url(machine_id) + "?a=1&a=2")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_body_is_rejected_before_lookup(client):
    response = client.request("GET", reconciliation_url("missing"), content=b"{}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_404(client):
    response = reconcile(client, "00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_non_get_methods_are_405(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(reconciliation_url(machine_id))
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Response shape
# --------------------------------------------------------------------------- #


def test_response_is_compact_json_with_fixed_field_order(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)

    response = reconcile(client, machine_id)
    raw = response.content.decode("utf-8")
    assert raw.endswith("\n") and raw.count("\n") == 1
    assert '":' in raw and '", "' not in raw
    assert list(json.loads(raw).keys()) == [
        "machine_id",
        "valid",
        "checked_count",
        "latest_rotation_id",
        "broken_rotation_id",
        "anomaly",
    ]


# --------------------------------------------------------------------------- #
# Anomaly: timestamps
# --------------------------------------------------------------------------- #


def test_unparseable_created_at_is_timestamp_anomaly(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 1)
    (record,) = events(client, machine_id)

    db_execute(
        client,
        "UPDATE key_rotation_events SET created_at = 'not-a-time' WHERE id = :id",
        id=record["id"],
    )

    assert reconcile(client, machine_id).json() == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 1,
        "latest_rotation_id": record["id"],
        "broken_rotation_id": record["id"],
        "anomaly": "timestamp_unparseable",
    }


def test_unparseable_moment_sorts_last_in_check_order(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    first, second = events(client, machine_id)

    # Damaging the tail's moment leaves it last in the check order; the
    # sound first record is checked before the damage is reported.
    db_execute(
        client,
        "UPDATE key_rotation_events SET created_at = 'garbage' WHERE id = :id",
        id=second["id"],
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["latest_rotation_id"] == second["id"]
    assert body["broken_rotation_id"] == second["id"]
    assert body["anomaly"] == "timestamp_unparseable"


# --------------------------------------------------------------------------- #
# Anomaly: chain
# --------------------------------------------------------------------------- #


def test_tampered_content_is_chain_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    records = events(client, machine_id)

    db_execute(
        client,
        "UPDATE key_rotation_events SET old_public_key = 'forged' WHERE id = :id",
        id=records[1]["id"],
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["checked_count"] == 3
    assert body["latest_rotation_id"] == records[2]["id"]
    assert body["broken_rotation_id"] == records[1]["id"]
    assert body["anomaly"] == "chain_mismatch"


def test_tampered_previous_link_is_chain_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    records = events(client, machine_id)

    db_execute(
        client,
        "UPDATE key_rotation_events SET previous_rotation_id = NULL WHERE id = :id",
        id=records[1]["id"],
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == records[1]["id"]
    assert body["anomaly"] == "chain_mismatch"


def test_tampered_chain_hash_is_chain_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    records = events(client, machine_id)

    db_execute(
        client,
        "UPDATE key_rotation_events SET chain_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=records[0]["id"],
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == records[0]["id"]
    assert body["anomaly"] == "chain_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: version and key transitions
# --------------------------------------------------------------------------- #


def test_wrong_version_is_version_transition_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    first, second = events(client, machine_id)

    # Forge a self-consistent chain whose second record skips a version.
    second["version"] = 9
    db_execute(
        client,
        "UPDATE key_rotation_events SET version = 9 WHERE id = :id",
        id=second["id"],
    )
    update_record(client, recompute_chain([first, second])[1])

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == second["id"]
    assert body["anomaly"] == "version_transition_mismatch"


def test_first_record_must_be_version_two(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 1)
    (record,) = events(client, machine_id)

    record["version"] = 1
    db_execute(
        client,
        "UPDATE key_rotation_events SET version = 1 WHERE id = :id",
        id=record["id"],
    )
    update_record(client, recompute_chain([record])[0])

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == record["id"]
    assert body["anomaly"] == "version_transition_mismatch"


def test_broken_key_handoff_is_key_transition_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    first, second = events(client, machine_id)

    # Forge a self-consistent chain whose second record names a different
    # old key than the first record's new key.
    second["old_public_key"] = "key-unrelated"
    db_execute(
        client,
        "UPDATE key_rotation_events SET old_public_key = 'key-unrelated'"
        " WHERE id = :id",
        id=second["id"],
    )
    update_record(client, recompute_chain([first, second])[1])

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == second["id"]
    assert body["anomaly"] == "key_transition_mismatch"


# --------------------------------------------------------------------------- #
# Anomaly: current state
# --------------------------------------------------------------------------- #


def test_machine_key_drift_is_current_state_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    tail = events(client, machine_id)[-1]

    db_execute(
        client,
        "UPDATE machines SET public_key = 'key-elsewhere' WHERE id = :id",
        id=machine_id,
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["checked_count"] == 2
    assert body["latest_rotation_id"] == tail["id"]
    assert body["broken_rotation_id"] == tail["id"]
    assert body["anomaly"] == "current_state_mismatch"


def test_machine_version_drift_is_current_state_mismatch(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    tail = events(client, machine_id)[-1]

    db_execute(
        client,
        "UPDATE machines SET version = 7 WHERE id = :id",
        id=machine_id,
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == tail["id"]
    assert body["anomaly"] == "current_state_mismatch"


def test_empty_history_with_version_drift_blames_no_record(client):
    machine_id = create_machine(client)

    db_execute(
        client,
        "UPDATE machines SET version = 3, public_key = 'key-3' WHERE id = :id",
        id=machine_id,
    )

    assert reconcile(client, machine_id).json() == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 0,
        "latest_rotation_id": None,
        "broken_rotation_id": None,
        "anomaly": "current_state_mismatch",
    }


# --------------------------------------------------------------------------- #
# Ordering, isolation, and the read-only guarantee
# --------------------------------------------------------------------------- #


def test_first_anomaly_in_check_order_is_reported(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 3)
    records = events(client, machine_id)

    db_execute(
        client,
        "UPDATE key_rotation_events SET new_public_key = 'forged' WHERE id = :id",
        id=records[0]["id"],
    )
    db_execute(
        client,
        "UPDATE key_rotation_events SET new_public_key = 'forged' WHERE id = :id",
        id=records[2]["id"],
    )

    body = reconcile(client, machine_id).json()
    assert body["valid"] is False
    assert body["broken_rotation_id"] == records[0]["id"]
    assert body["anomaly"] == "chain_mismatch"


def test_reconciliation_is_isolated_per_machine(client):
    machine_one = create_machine(client, external_id="m-1", public_key="key-a1")
    machine_two = create_machine(client, external_id="m-2", public_key="key-b1")
    rotate_n(client, machine_one, 2)
    rotate_n(client, machine_two, 2)
    other_records = events(client, machine_two)

    db_execute(
        client,
        "UPDATE key_rotation_events SET old_public_key = 'forged' WHERE id = :id",
        id=other_records[0]["id"],
    )

    assert reconcile(client, machine_one).json()["valid"] is True
    assert reconcile(client, machine_two).json()["valid"] is False


def test_reconciliation_is_read_only(client):
    machine_id = create_machine(client)
    rotate_n(client, machine_id, 2)
    before = events(client, machine_id)
    machine_before = client.get(f"/machines/{machine_id}").json()

    first = reconcile(client, machine_id)
    second = reconcile(client, machine_id)
    assert first.content == second.content
    assert first.json()["valid"] is True

    assert events(client, machine_id) == before
    assert client.get(f"/machines/{machine_id}").json() == machine_before


def test_conclusion_is_stable_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        rotate_n(first, machine_id, 3)
        created = reconcile(first, machine_id).content

    with TestClient(app) as second:
        assert reconcile(second, machine_id).content == created
