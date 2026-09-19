import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "0123456789abcdef" * 4

MISSING_ID = "00000000-0000-0000-0000-000000000000"


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


def create_evidence(client, machine_id, event_id, evidence_type="log",
                    content_hash=HASH_A):
    response = client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )
    assert response.status_code == 201
    return response.json()


def integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "evidence/integrity"
    )


def get_integrity(client, machine_id):
    response = client.get(integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def db_path_of(client):
    return client.app.state.engine.url.database


def insert_evidence_row(db_path, *, evidence_id, machine_id, event_id,
                        evidence_type, content_hash, created_at):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO authorization_decision_evidence "
            "(id, machine_id, event_id, evidence_type, content_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                evidence_id,
                machine_id,
                event_id,
                evidence_type,
                content_hash,
                created_at,
            ),
        )


def delete_event_row(db_path, event_id):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM authorization_decision_events WHERE id = ?",
            (event_id,),
        )


def test_missing_machine_returns_404(client):
    response = client.get(integrity_url(MISSING_ID))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_no_evidence_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


def test_consistent_evidence_is_valid(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    first = create_evidence(client, machine_id, event_one, content_hash=HASH_A)
    second = create_evidence(client, machine_id, event_two, content_hash=HASH_B)
    # An evidence type that required trimming at creation time is stored
    # trimmed and must still verify.
    third = create_evidence(
        client, machine_id, event_two, evidence_type=" sig ", content_hash=HASH_C
    )

    assert get_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 3,
        "broken_evidence_id": None,
    }


def test_only_counts_evidence_of_path_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_two = record_event(client, machine_two)
    create_evidence(client, machine_two, event_two)

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }
    assert get_integrity(client, machine_two)["checked_count"] == 1


def test_dangling_event_reference_is_broken(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = create_evidence(client, machine_id, event_id)

    delete_event_row(db_path_of(client), event_id)

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }


def test_reference_to_event_of_other_machine_is_broken(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign_event = record_event(client, machine_two)

    broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=broken_id,
        machine_id=machine_one,
        event_id=foreign_event,
        evidence_type="log",
        content_hash=HASH_A,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_one) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": broken_id,
    }
    # The foreign machine's own audit is untouched.
    assert get_integrity(client, machine_two) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


@pytest.mark.parametrize("bad_type", ["   ", "\t\n", " "])
def test_whitespace_only_evidence_type_is_broken(client, bad_type):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=broken_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type=bad_type,
        content_hash=HASH_A,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": broken_id,
    }


@pytest.mark.parametrize(
    "bad_hash",
    [
        "A" * 64,                  # uppercase rejected, never case-folded
        "a" * 63,                  # too short
        "a" * 65,                  # too long
        "g" * 64,                  # non-hex characters
        "a" * 63 + "Z",            # one uppercase at the tail
        " " + "a" * 63,            # leading whitespace
        "0123456789ABCDEF" * 4,    # uppercase hex digits
    ],
)
def test_non_lowercase_hex_content_hash_is_broken(client, bad_hash):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=broken_id,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type="log",
        content_hash=bad_hash,
        created_at="2026-01-01T00:00:00Z",
    )

    assert get_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": broken_id,
    }


def test_first_broken_record_in_scan_order_is_reported(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    good = create_evidence(client, machine_id, event_one, content_hash=HASH_A)

    first_broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=first_broken_id,
        machine_id=machine_id,
        event_id=event_two,
        evidence_type="log",
        content_hash=HASH_B,
        created_at="2020-01-01T00:00:00Z",
    )
    # Corrupt the first record's hash after insert so it is broken while a
    # later record is also broken: the earliest in (created_at, id) wins.
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET content_hash = ? "
            "WHERE id = ?",
            ("A" * 64, first_broken_id),
        )
    second_broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=second_broken_id,
        machine_id=machine_id,
        event_id=event_two,
        evidence_type="   ",
        content_hash=HASH_C,
        created_at="2020-01-02T00:00:00Z",
    )

    result = get_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": first_broken_id,
    }
    assert result["broken_evidence_id"] != good["id"]


def test_other_machine_damage_does_not_fail_this_machine(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)
    create_evidence(client, machine_one, event_one)
    create_evidence(client, machine_two, event_two)

    # Corrupt only machine two's evidence.
    rows = client.get(evidence_url(machine_two, event_two)).json()
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET content_hash = ? "
            "WHERE id = ?",
            ("A" * 64, rows[0]["id"]),
        )

    assert get_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }
    assert get_integrity(client, machine_two)["valid"] is False


def test_integrity_check_writes_nothing_and_is_repeatable(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    good = create_evidence(client, machine_id, event_one, content_hash=HASH_A)
    broken_id = str(uuid.uuid4())
    insert_evidence_row(
        db_path_of(client),
        evidence_id=broken_id,
        machine_id=machine_id,
        event_id=event_two,
        evidence_type="log",
        content_hash=HASH_B,
        created_at="2020-01-01T00:00:00Z",
    )
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET evidence_type = '  ' "
            "WHERE id = ?",
            (broken_id,),
        )

    evidence_before = client.get(
        evidence_url(machine_id, event_one)
    ).json()
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    chain_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    first = get_integrity(client, machine_id)
    assert first == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": broken_id,
    }
    assert get_integrity(client, machine_id) == first

    # Nothing was repaired or deleted: the good record lists identically and
    # the damaged record keeps its damaged stored value.
    assert (
        client.get(evidence_url(machine_id, event_one)).json()
        == evidence_before
        == [good]
    )
    with sqlite3.connect(db_path_of(client)) as conn:
        stored_type = conn.execute(
            "SELECT evidence_type FROM authorization_decision_evidence "
            "WHERE id = ?",
            (broken_id,),
        ).fetchone()[0]
    assert stored_type == "  "
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events"
        ).json()
        == events_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).json()
        == chain_before
    )


def test_integrity_result_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        record = create_evidence(first, machine_id, event_id)
        expected_good = get_integrity(first, machine_id)
        assert expected_good == {
            "valid": True,
            "checked_count": 1,
            "broken_evidence_id": None,
        }

        # Damage the record after the first process checked it; a restarted
        # process reads the same stored bytes and reports the damage.
        with sqlite3.connect(db_path_of(first)) as conn:
            conn.execute(
                "UPDATE authorization_decision_evidence SET content_hash = ? "
                "WHERE id = ?",
                ("A" * 64, record["id"]),
            )
        expected_bad = get_integrity(first, machine_id)
        assert expected_bad == {
            "valid": False,
            "checked_count": 1,
            "broken_evidence_id": record["id"],
        }

    with TestClient(app) as second:
        assert get_integrity(second, machine_id) == expected_bad
