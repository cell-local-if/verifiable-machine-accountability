"""Tests for the per-machine evidence tamper-evident hash chain.

Covers the chain fields stored by
`POST /machines/{machine_id}/authorization-decision-events/{event_id}/evidence`
(registration and append are one locked transaction), the read-only
`GET /machines/{machine_id}/authorization-decision-events/evidence-chain/integrity`
sub-entry, and chain fields carried by the evidence list and compliance
export: GET-only 405, ``invalid_query`` 422 for any query parameter before the
machine lookup, ``not_found`` 404 with no partial conclusion, empty-chain
``true/0/null``, chain ordering by the actual UTC instant of ``created_at``
then id, tamper detection (content digest, previous link, chain digest,
malformed timestamp, cross-machine/duplicate pointers), per-machine
isolation, read-only stability and byte-identical repeats, concurrent
registration safety, startup migration/backfill and no-write restarts.
"""
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
    "event_id",
    "evidence_type",
    "content_hash",
    "created_at",
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASHES = ["".join(chr(ord("0") + (i % 10))) * 64 for i in range(10)]

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()["id"]


def evidence_url(machine_id, event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/{event_id}/evidence"
    )


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "evidence-chain/integrity"
    )


def create_evidence(client, machine_id, event_id, evidence_type="log",
                    content_hash=HASH_A):
    return client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )


def register(client, machine_id, event_id, content_hash=HASH_A, evidence_type="log"):
    response = create_evidence(client, machine_id, event_id, evidence_type,
                               content_hash)
    assert response.status_code == 201
    return response.json()


def get_integrity(client, machine_id):
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_path_of(client):
    return client.app.state.engine.url.database


def tamper(db_path, statement, parameters=()):
    connection = sqlite3.connect(db_path)
    connection.execute(statement, parameters)
    connection.commit()
    connection.close()


# --------------------------------------------------------------------------- #
# Chain fields on registration
# --------------------------------------------------------------------------- #


def test_first_evidence_is_chain_head(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    record = register(client, machine_id, event_id)

    assert set(record.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "evidence_type",
        "content_hash",
        "created_at",
        "previous_evidence_id",
        "chain_hash",
    }
    assert record["previous_evidence_id"] is None
    assert record["content_hash"] == HASH_A
    assert record["chain_hash"] == chain_hash("", canonical_content_hash(record))


def test_content_digest_covers_all_six_content_fields(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    record = register(client, machine_id, event_id, evidence_type=" sig ",
                      content_hash=HASH_C)

    # The fingerprint field stays the submitted fingerprint; the content
    # digest over all six fields is exposed through the chain digest.
    assert record["content_hash"] == HASH_C
    assert record["chain_hash"] == chain_hash("", canonical_content_hash(record))
    assert HEX64_RE.match(record["chain_hash"])

    # The digest covers the evidence type: a differently typed record with the
    # same fingerprint gets a different head chain digest.
    other = register(client, machine_id, event_id, evidence_type="trace",
                     content_hash=HASH_B)
    assert other["previous_evidence_id"] == record["id"]
    assert other["chain_hash"] == chain_hash(
        record["chain_hash"], canonical_content_hash(other)
    )


def test_records_link_in_registration_order_across_events(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    records = [
        register(client, machine_id, event_one, content_hash=HASH_A),
        register(client, machine_id, event_one, content_hash=HASH_B),
        register(client, machine_id, event_two, content_hash=HASH_C),
    ]

    for index, record in enumerate(records):
        if index == 0:
            assert record["previous_evidence_id"] is None
            assert record["chain_hash"] == chain_hash(
                "", canonical_content_hash(record)
            )
        else:
            previous = records[index - 1]
            assert record["previous_evidence_id"] == previous["id"]
            assert record["chain_hash"] == chain_hash(
                previous["chain_hash"], canonical_content_hash(record)
            )


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)

    first_one = register(client, machine_one, event_one, content_hash=HASH_A)
    first_two = register(client, machine_two, event_two, content_hash=HASH_A)

    assert first_one["previous_evidence_id"] is None
    assert first_two["previous_evidence_id"] is None
    assert first_one["chain_hash"] != first_two["chain_hash"]
    for machine_id in (machine_one, machine_two):
        assert get_integrity(client, machine_id) == {
            "valid": True,
            "checked_count": 1,
            "broken_evidence_id": None,
        }


def test_duplicate_fingerprint_is_409_and_appends_nothing(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    register(client, machine_id, event_id, content_hash=HASH_A)

    response = create_evidence(client, machine_id, event_id, content_hash=HASH_A)

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_evidence"}}
    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }


def test_same_fingerprint_on_different_events_forms_two_links(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")

    first = register(client, machine_id, event_one, content_hash=HASH_A)
    second = register(client, machine_id, event_two, content_hash=HASH_A)

    assert first["previous_evidence_id"] is None
    assert second["previous_evidence_id"] == first["id"]


# --------------------------------------------------------------------------- #
# List and export carry the same chain fields
# --------------------------------------------------------------------------- #


def test_list_carries_chain_fields_consistent_with_registration(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    created = [
        register(client, machine_id, event_id, content_hash=h)
        for h in (HASH_A, HASH_B, HASH_C)
    ]

    response = client.get(evidence_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.text.endswith("\n")
    listed = response.json()
    assert [r["id"] for r in listed] == [r["id"] for r in created]
    for record, expected in zip(listed, created, strict=True):
        assert record == expected
        assert list(record.keys()) == [
            "id",
            "machine_id",
            "event_id",
            "evidence_type",
            "content_hash",
            "created_at",
            "previous_evidence_id",
            "chain_hash",
        ]


def test_list_empty_chain_is_empty_array_ending_newline(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.get(evidence_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.text == "[]\n"


def test_compliance_export_carries_identical_chain_fields(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    register(client, machine_id, event_id, content_hash=HASH_A)
    register(client, machine_id, event_id, content_hash=HASH_B)
    listed = client.get(evidence_url(machine_id, event_id)).json()

    response = client.get(
        f"/machines/{machine_id}/authorization-decision-events/evidence/"
        "compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z&to_created_at=2100-01-01T00:00:00Z"
    )

    assert response.status_code == 200
    exported = response.json()["evidence"]
    assert exported == listed
    for record in exported:
        assert list(record.keys()) == [
            "id",
            "machine_id",
            "event_id",
            "evidence_type",
            "content_hash",
            "created_at",
            "previous_evidence_id",
            "chain_hash",
        ]


# --------------------------------------------------------------------------- #
# Integrity entry method/query/machine handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ("post", "put", "patch", "delete"))
def test_only_get_is_accepted(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(integrity_url(machine_id))
    assert response.status_code == 405


@pytest.mark.parametrize("query", ["?unexpected=1", "?x=", "?from_created_at=2026-01-01T00:00:00Z"])
def test_any_query_parameter_is_invalid_query(client, query):
    machine_id = create_machine(client)
    response = client.get(integrity_url(machine_id) + query)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_invalid_query_precedes_missing_machine(client):
    response = client.get(integrity_url(MISSING_ID) + "?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_conclusion(client):
    response = client.get(integrity_url(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_chain_is_valid_zero_null(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


def test_sound_chain_is_valid(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    register(client, machine_id, event_one, content_hash=HASH_A)
    register(client, machine_id, event_two, content_hash=HASH_B)
    register(client, machine_id, event_two, content_hash=HASH_C)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 3,
        "broken_evidence_id": None,
    }


# --------------------------------------------------------------------------- #
# Tamper detection
# --------------------------------------------------------------------------- #


def test_tampered_fingerprint_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = register(client, machine_id, event_id)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET content_hash = ? WHERE id = ?",
        (HASH_B, record["id"]),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }


def test_tampered_evidence_type_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = register(client, machine_id, event_id)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET evidence_type = '   ' "
        "WHERE id = ?",
        (record["id"],),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }


def test_tampered_previous_link_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    register(client, machine_id, event_id, content_hash=HASH_A)
    second = register(client, machine_id, event_id, content_hash=HASH_B)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET previous_evidence_id = NULL "
        "WHERE id = ?",
        (second["id"],),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": second["id"],
    }


def test_tampered_chain_hash_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = register(client, machine_id, event_id, content_hash=HASH_A)
    register(client, machine_id, event_id, content_hash=HASH_B)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": first["id"],
    }


def test_cross_machine_previous_pointer_is_broken(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    foreign = register(client, machine_one, event_one, content_hash=HASH_A)
    register(client, machine_two, event_two, content_hash=HASH_A)
    second_two = register(client, machine_two, event_two, content_hash=HASH_B)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET previous_evidence_id = ? "
        "WHERE id = ?",
        (foreign["id"], second_two["id"]),
    )

    assert get_integrity(client, machine_two) == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": second_two["id"],
    }
    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }


def test_duplicate_previous_pointer_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = register(client, machine_id, event_id, content_hash=HASH_A)
    register(client, machine_id, event_id, content_hash=HASH_B)
    third = register(client, machine_id, event_id, content_hash=HASH_C)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET previous_evidence_id = ? "
        "WHERE id = ?",
        # The third record must point at the second; pointing at the first
        # repeats first as a predecessor and forks the link order.
        (first["id"], third["id"]),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": third["id"],
    }


def test_malformed_created_at_is_an_anomaly_in_the_total(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    register(client, machine_id, event_id, content_hash=HASH_A)
    damaged = register(client, machine_id, event_id, content_hash=HASH_B)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET created_at = 'not-a-time' "
        "WHERE id = ?",
        (damaged["id"],),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": damaged["id"],
    }


def test_missing_associated_event_fails_but_record_stays_counted(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = register(client, machine_id, event_id)

    tamper(
        db_path_of(client),
        "DELETE FROM authorization_decision_events WHERE id = ?",
        (event_id,),
    )

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }
    # The record itself is retained, never removed by the check.
    with sqlite3.connect(db_path_of(client)) as conn:
        remaining = conn.execute(
            "SELECT id FROM authorization_decision_evidence"
        ).fetchall()
    assert [row[0] for row in remaining] == [record["id"]]


def test_first_broken_record_is_reported(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    records = [
        register(client, machine_id, event_id, content_hash=HASHES[0]),
        register(client, machine_id, event_id, content_hash=HASHES[1]),
        register(client, machine_id, event_id, content_hash=HASHES[2]),
    ]
    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET evidence_type = '  ' "
        "WHERE id = ?",
        (records[1]["id"],),
    )
    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
        ("0" * 64, records[2]["id"]),
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": records[1]["id"],
    }


def test_other_machine_damage_does_not_fail_this_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    register(client, machine_one, event_one, content_hash=HASH_A)
    damaged = register(client, machine_two, event_two, content_hash=HASH_A)

    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
        ("0" * 64, damaged["id"],),
    )

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }
    assert get_integrity(client, machine_two)["valid"] is False


# --------------------------------------------------------------------------- #
# Read-only, repeatability, concurrency
# --------------------------------------------------------------------------- #


def test_check_is_read_only_and_byte_repeatable(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    good = register(client, machine_id, event_id, content_hash=HASH_A)
    broken = register(client, machine_id, event_id, content_hash=HASH_B)
    tamper(
        db_path_of(client),
        "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
        ("0" * 64, broken["id"],),
    )

    with sqlite3.connect(db_path_of(client)) as conn:
        before = conn.execute(
            "SELECT id, previous_evidence_id, chain_hash FROM "
            "authorization_decision_evidence ORDER BY created_at, id"
        ).fetchall()

    first = client.get(integrity_url(machine_id))
    second = client.get(integrity_url(machine_id))

    assert first.status_code == 200
    assert first.content == second.content
    assert first.json() == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": broken["id"],
    }
    with sqlite3.connect(db_path_of(client)) as conn:
        after = conn.execute(
            "SELECT id, previous_evidence_id, chain_hash FROM "
            "authorization_decision_evidence ORDER BY created_at, id"
        ).fetchall()
    assert after == before
    assert good["id"] != broken["id"]


def test_concurrent_registrations_keep_chain_unbroken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    def do_register(index):
        return create_evidence(client, machine_id, event_id, content_hash=HASHES[index % 10] if index < 10 else f"{index:064x}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(do_register, range(30)))

    assert all(response.status_code == 201 for response in responses)
    records = client.get(evidence_url(machine_id, event_id)).json()
    assert len(records) == 30
    assert records[0]["previous_evidence_id"] is None
    seen = set()
    for previous, current in zip(records, records[1:]):
        assert current["previous_evidence_id"] == previous["id"]
        seen.add(current["previous_evidence_id"])
    # No predecessor is ever referenced twice.
    assert len(seen) == 29
    result = get_integrity(client, machine_id)
    assert result == {
        "valid": True,
        "checked_count": 30,
        "broken_evidence_id": None,
    }


def test_concurrent_duplicate_fingerprint_has_one_winner(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: create_evidence(client, machine_id, event_id,
                                          content_hash=HASH_A),
                range(8),
            )
        )

    statuses = sorted(response.status_code for response in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 7
    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }


# --------------------------------------------------------------------------- #
# Restart: migration, backfill, no-write stability, empty database
# --------------------------------------------------------------------------- #


def test_restart_backfills_null_chain_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "persist.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        register(first, machine_id, event_id, content_hash=HASH_A)
        register(first, machine_id, event_id, content_hash=HASH_B)

    # Simulate rows written before the chain feature existed.
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE authorization_decision_evidence SET "
        "previous_evidence_id = NULL, chain_hash = NULL"
    )
    connection.commit()
    connection.close()

    with TestClient(app) as second:
        listed = second.get(evidence_url(machine_id, event_id)).json()
        assert listed[0]["previous_evidence_id"] is None
        assert listed[1]["previous_evidence_id"] == listed[0]["id"]
        for index, record in enumerate(listed):
            assert HEX64_RE.match(record["chain_hash"])
            expected_previous = "" if index == 0 else listed[index - 1]["chain_hash"]
            assert record["chain_hash"] == chain_hash(
                expected_previous, canonical_content_hash(record)
            )
        assert get_integrity(second, machine_id) == {
            "valid": True,
            "checked_count": 2,
            "broken_evidence_id": None,
        }


def test_restart_over_complete_database_writes_nothing(tmp_path, monkeypatch):
    db_path = tmp_path / "persist.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        register(first, machine_id, event_id, content_hash=HASH_A)
        register(first, machine_id, event_id, content_hash=HASH_B)
        before = first.get(evidence_url(machine_id, event_id)).text
        integrity = get_integrity(first, machine_id)

    with TestClient(app) as second:
        after = second.get(evidence_url(machine_id, event_id)).text
        assert after == before
        assert get_integrity(second, machine_id) == integrity


def test_old_schema_database_is_migrated_and_backfilled(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    machine_id = "11111111-1111-1111-1111-111111111111"
    event_id = "22222222-2222-2222-2222-222222222222"
    evidence_one = "33333333-3333-3333-3333-333333333333"
    evidence_two = "44444444-4444-4444-4444-444444444444"
    # A database written before evidence chain columns existed.
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE machines (id VARCHAR(36) PRIMARY KEY, external_id VARCHAR, "
        "display_name VARCHAR, public_key VARCHAR, status VARCHAR, version INTEGER, "
        "created_at VARCHAR, updated_at VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE authorization_decision_events (id VARCHAR(36) PRIMARY KEY, "
        "machine_id VARCHAR, action_type VARCHAR, resource VARCHAR, allowed BOOLEAN, "
        "reason VARCHAR, created_at VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE authorization_decision_evidence (id VARCHAR(36) PRIMARY KEY, "
        "machine_id VARCHAR, event_id VARCHAR, evidence_type VARCHAR, "
        "content_hash VARCHAR(64), created_at VARCHAR)"
    )
    connection.execute(
        "INSERT INTO machines VALUES (?, 'ext', 'Name', 'key', 'active', 1, "
        "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (machine_id,),
    )
    connection.execute(
        "INSERT INTO authorization_decision_events VALUES "
        "(?, ?, 'read', 'res/x', 1, 'allowed_by_policy', '2026-01-01T00:00:00Z')",
        (event_id, machine_id),
    )
    # Insert in reverse chain order; backfill must follow (created_at, id).
    connection.execute(
        "INSERT INTO authorization_decision_evidence VALUES (?, ?, ?, 'log', ?, "
        "'2026-01-01T00:00:02Z')",
        (evidence_two, machine_id, event_id, HASH_B),
    )
    connection.execute(
        "INSERT INTO authorization_decision_evidence VALUES (?, ?, ?, 'log', ?, "
        "'2026-01-01T00:00:01Z')",
        (evidence_one, machine_id, event_id, HASH_A),
    )
    connection.commit()
    connection.close()

    with TestClient(app) as client:
        listed = client.get(evidence_url(machine_id, event_id)).json()
        assert [r["id"] for r in listed] == [evidence_one, evidence_two]
        assert listed[0]["previous_evidence_id"] is None
        assert listed[1]["previous_evidence_id"] == evidence_one
        assert get_integrity(client, machine_id) == {
            "valid": True,
            "checked_count": 2,
            "broken_evidence_id": None,
        }


def test_empty_database_can_register_and_query_after_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as first:
        machine_id = create_machine(first)
        assert get_integrity(first, machine_id) == {
            "valid": True,
            "checked_count": 0,
            "broken_evidence_id": None,
        }
        event_id = record_event(first, machine_id)
        record = register(first, machine_id, event_id)

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == {
            "valid": True,
            "checked_count": 1,
            "broken_evidence_id": None,
        }
        assert second.get(evidence_url(machine_id, event_id)).json() == [record]
