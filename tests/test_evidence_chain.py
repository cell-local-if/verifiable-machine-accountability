import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app
from accountability.evidence_chain import compute_content_digest, compute_chain_hash

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "0123456789abcdef" * 4

MISSING_ID = "00000000-0000-0000-0000-000000000000"

CONTENT_KEYS = (
    "id",
    "machine_id",
    "event_id",
    "evidence_type",
    "content_hash",
    "created_at",
)


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


def record_event(client, machine_id, resource="res/x", action_type="read"):
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


def add_evidence(client, machine_id, event_id, *, evidence_type="log", content_hash):
    response = client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )
    assert response.status_code == 201
    return response.json()


def chain_integrity_url(machine_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        "evidence-chain/integrity"
    )


def get_chain_integrity(client, machine_id):
    response = client.get(chain_integrity_url(machine_id))
    assert response.status_code == 200
    return response.json()


def expected_digest(record: dict) -> str:
    return compute_content_digest(**{key: record[key] for key in CONTENT_KEYS})


# --- registration carries the chain fields ---------------------------------


def test_first_evidence_roots_the_machine_chain(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    record = add_evidence(
        client, machine_id, event_id, content_hash=HASH_A
    )

    assert record["previous_evidence_id"] is None
    assert HEX64_RE.match(record["chain_hash"])
    assert record["chain_hash"] == compute_chain_hash(
        "", expected_digest(record)
    )


def test_records_link_across_events_in_one_machine_chain(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")

    first = add_evidence(client, machine_id, event_one, content_hash=HASH_A)
    second = add_evidence(client, machine_id, event_two, content_hash=HASH_B)
    third = add_evidence(client, machine_id, event_one, content_hash=HASH_C)

    assert first["previous_evidence_id"] is None
    assert second["previous_evidence_id"] == first["id"]
    assert third["previous_evidence_id"] == second["id"]
    assert second["chain_hash"] == compute_chain_hash(
        first["chain_hash"], expected_digest(second)
    )
    assert third["chain_hash"] == compute_chain_hash(
        second["chain_hash"], expected_digest(third)
    )


def test_chains_are_independent_per_machine(client):
    machine_one = create_machine(client, external_id="m-1")
    machine_two = create_machine(client, external_id="m-2")
    event_one = record_event(client, machine_one)
    event_two = record_event(client, machine_two)

    first_one = add_evidence(client, machine_one, event_one, content_hash=HASH_A)
    first_two = add_evidence(client, machine_two, event_two, content_hash=HASH_B)
    second_two = add_evidence(client, machine_two, event_two, content_hash=HASH_C)

    assert first_one["previous_evidence_id"] is None
    assert first_two["previous_evidence_id"] is None
    assert second_two["previous_evidence_id"] == first_two["id"]
    assert first_one["chain_hash"] == compute_chain_hash(
        "", expected_digest(first_one)
    )
    assert first_two["chain_hash"] == compute_chain_hash(
        "", expected_digest(first_two)
    )


def test_duplicate_fingerprint_still_returns_409_and_adds_no_link(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    add_evidence(client, machine_id, event_id, content_hash=HASH_A)

    response = client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": "other", "content_hash": HASH_A},
    )
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_evidence"}}

    # Only the one record exists; the rejected duplicate added no chain link.
    records = client.get(evidence_url(machine_id, event_id)).json()
    assert [r["content_hash"] for r in records] == [HASH_A]
    assert get_chain_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }


def test_invalid_body_is_422_and_missing_or_misowned_event_is_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    bad = client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": "  ", "content_hash": "nope"},
    )
    assert bad.status_code == 422

    assert (
        client.post(
            evidence_url(MISSING_ID, event_id),
            json={"evidence_type": "log", "content_hash": HASH_A},
        ).status_code
        == 404
    )
    assert (
        client.post(
            evidence_url(machine_id, MISSING_ID),
            json={"evidence_type": "log", "content_hash": HASH_A},
        ).status_code
        == 404
    )


# --- list and export carry the same chain values ---------------------------


def test_list_and_export_carry_consistent_chain_fields(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    created = [
        add_evidence(client, machine_id, event_id, content_hash=h)
        for h in (HASH_A, HASH_B, HASH_D)
    ]

    listed = client.get(evidence_url(machine_id, event_id)).json()
    assert listed == created
    for record in listed:
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

    exported = client.get(
        f"/machines/{machine_id}/authorization-decision-events/evidence/"
        "compliance-export",
        params={
            "from_created_at": "2000-01-01T00:00:00Z",
            "to_created_at": "2100-01-01T00:00:00Z",
        },
    ).json()["evidence"]
    assert exported == listed


def test_list_body_is_compact_json_ending_in_one_newline(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    add_evidence(client, machine_id, event_id, content_hash=HASH_A)

    response = client.get(evidence_url(machine_id, event_id))
    raw = response.content
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b": " not in raw and b", " not in raw
    assert json.loads(raw)  # parses


def test_empty_event_list_is_empty_array(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    response = client.get(evidence_url(machine_id, event_id))
    assert response.content == b"[]\n"


# --- read-only chain integrity endpoint ------------------------------------


def test_chain_integrity_empty_chain_is_valid(client):
    machine_id = create_machine(client)
    record_event(client, machine_id)

    assert get_chain_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }


def test_chain_integrity_consistent_records_are_valid(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    add_evidence(client, machine_id, event_one, content_hash=HASH_A)
    add_evidence(client, machine_id, event_two, content_hash=HASH_B)
    add_evidence(
        client, machine_id, event_two, evidence_type=" sig ", content_hash=HASH_C
    )

    assert get_chain_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 3,
        "broken_evidence_id": None,
    }


def test_chain_integrity_missing_machine_is_404_with_no_conclusion(client):
    response = client.get(chain_integrity_url(MISSING_ID))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert "valid" not in response.json()


def test_chain_integrity_unknown_query_is_422_before_machine_lookup(client):
    response = client.get(f"{chain_integrity_url(MISSING_ID)}?bogus=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_chain_integrity_non_get_is_405(client):
    machine_id = create_machine(client)
    for method in ("post", "put", "delete", "patch"):
        response = getattr(client, method)(chain_integrity_url(machine_id))
        assert response.status_code == 405


def db_path_of(client):
    return client.app.state.engine.url.database


def test_chain_integrity_detects_tampered_link(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    records = [
        add_evidence(client, machine_id, event_id, content_hash=h)
        for h in (HASH_A, HASH_B, HASH_C)
    ]

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET previous_evidence_id = ? "
            "WHERE id = ?",
            (None, records[2]["id"]),
        )

    assert get_chain_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": records[2]["id"],
    }


def test_chain_integrity_detects_tampered_chain_hash(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = add_evidence(client, machine_id, event_id, content_hash=HASH_A)
    add_evidence(client, machine_id, event_id, content_hash=HASH_B)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
            ("0" * 64, first["id"]),
        )

    # The first record's broken chain hash is reported even though the later
    # record still chains off the stored value.
    assert get_chain_integrity(client, machine_id) == {
        "valid": False,
        "checked_count": 2,
        "broken_evidence_id": first["id"],
    }


def test_chain_integrity_detects_tampered_fingerprint(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = add_evidence(client, machine_id, event_id, content_hash=HASH_A)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET content_hash = ? WHERE id = ?",
            ("d" * 64, record["id"]),
        )

    result = get_chain_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }


def test_corrupted_timestamp_is_counted_and_flagged(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = add_evidence(client, machine_id, event_id, content_hash=HASH_A)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET created_at = 'not-a-time' "
            "WHERE id = ?",
            (record["id"],),
        )

    result = get_chain_integrity(client, machine_id)
    assert result["valid"] is False
    assert result["checked_count"] == 1
    assert result["broken_evidence_id"] == record["id"]


def test_dangling_event_reference_is_flagged_but_kept(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    record = add_evidence(client, machine_id, event_id, content_hash=HASH_A)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "DELETE FROM authorization_decision_events WHERE id = ?", (event_id,)
        )

    result = get_chain_integrity(client, machine_id)
    assert result == {
        "valid": False,
        "checked_count": 1,
        "broken_evidence_id": record["id"],
    }
    # The record is still present; the audit never deletes or repairs it.
    with sqlite3.connect(db_path_of(client)) as conn:
        count = conn.execute(
            "SELECT count(*) FROM authorization_decision_evidence"
        ).fetchone()[0]
    assert count == 1


def test_other_machine_damage_does_not_fail_this_machine(client):
    machine_one = create_machine(client, external_id="m-1")
    machine_two = create_machine(client, external_id="m-2")
    event_two = record_event(client, machine_two)
    other = add_evidence(client, machine_two, event_two, content_hash=HASH_A)

    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
            ("f" * 64, other["id"]),
        )

    assert get_chain_integrity(client, machine_one) == {
        "valid": True,
        "checked_count": 0,
        "broken_evidence_id": None,
    }
    assert get_chain_integrity(client, machine_two)["valid"] is False


def test_chain_integrity_reports_only_first_error_and_is_byte_stable(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    records = [
        add_evidence(client, machine_id, event_id, content_hash=h)
        for h in (HASH_A, HASH_B, HASH_C)
    ]
    with sqlite3.connect(db_path_of(client)) as conn:
        conn.execute(
            "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
            ("1" * 64, records[0]["id"]),
        )
        conn.execute(
            "UPDATE authorization_decision_evidence SET chain_hash = ? WHERE id = ?",
            ("2" * 64, records[2]["id"]),
        )

    url = chain_integrity_url(machine_id)
    first = client.get(url).content
    second = client.get(url).content
    assert first == second
    assert json.loads(first) == {
        "valid": False,
        "checked_count": 3,
        "broken_evidence_id": records[0]["id"],
    }


# --- concurrency ------------------------------------------------------------


def test_concurrent_registrations_form_one_unbroken_chain(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    count = 30

    def register(index):
        # Distinct fingerprints so none collide on the same event.
        return client.post(
            evidence_url(machine_id, event_id),
            json={
                "evidence_type": "log",
                "content_hash": f"{index:064x}",
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(register, range(count)))

    assert all(response.status_code == 201 for response in responses)
    records = [response.json() for response in responses]
    assert len({record["id"] for record in records}) == count

    exported = client.get(
        f"/machines/{machine_id}/authorization-decision-events/evidence/"
        "compliance-export",
        params={
            "from_created_at": "2000-01-01T00:00:00Z",
            "to_created_at": "2100-01-01T00:00:00Z",
        },
    ).json()["evidence"]
    assert len(exported) == count

    ids = [record["id"] for record in exported]
    previous_ids = [record["previous_evidence_id"] for record in exported]
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]
    assert len(set(previous_ids[1:])) == count - 1  # no repeated predecessor

    assert get_chain_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": count,
        "broken_evidence_id": None,
    }


def test_concurrent_identical_fingerprints_register_once(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    def register(_):
        return client.post(
            evidence_url(machine_id, event_id),
            json={"evidence_type": "log", "content_hash": HASH_A},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(register, range(12)))

    statuses = sorted(response.status_code for response in responses)
    assert statuses.count(201) == 1
    assert statuses.count(409) == 11
    records = client.get(evidence_url(machine_id, event_id)).json()
    assert len(records) == 1
    assert records[0]["previous_evidence_id"] is None
    assert get_chain_integrity(client, machine_id) == {
        "valid": True,
        "checked_count": 1,
        "broken_evidence_id": None,
    }


# --- legacy migration and restart ------------------------------------------


def test_legacy_evidence_is_backfilled_in_chain_order(tmp_path, monkeypatch):
    machine_id = "22222222-2222-2222-2222-222222222222"
    event_id = "33333333-3333-3333-3333-333333333333"
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
        CREATE TABLE authorization_decision_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            action_type VARCHAR, resource VARCHAR, allowed BOOLEAN,
            reason VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE authorization_decision_evidence (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            event_id VARCHAR(36), evidence_type VARCHAR,
            content_hash VARCHAR(64), created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    connection.execute(
        "INSERT INTO authorization_decision_events VALUES (?,?,?,?,?,?,?)",
        (event_id, machine_id, "read", "res/x", 1, "allowed_by_policy", "t0"),
    )
    # Inserted out of chain order; (created_at, id) must define the links.
    connection.executemany(
        "INSERT INTO authorization_decision_evidence "
        "(id, machine_id, event_id, evidence_type, content_hash, created_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("b" * 36, machine_id, event_id, "log", HASH_B,
             "2026-01-02T00:00:00.000000Z"),
            ("a" * 36, machine_id, event_id, "log", HASH_A,
             "2026-01-01T00:00:00.000000Z"),
            ("c" * 36, machine_id, event_id, "log", HASH_C,
             "2026-01-03T00:00:00.000000Z"),
        ],
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        listed = client.get(evidence_url(machine_id, event_id)).json()
        assert [record["id"] for record in listed] == [
            "a" * 36,
            "b" * 36,
            "c" * 36,
        ]
        assert listed[0]["previous_evidence_id"] is None
        assert listed[1]["previous_evidence_id"] == "a" * 36
        assert listed[2]["previous_evidence_id"] == "b" * 36
        for record in listed:
            assert HEX64_RE.match(record["chain_hash"])
            assert record["chain_hash"] is not None

        assert get_chain_integrity(client, machine_id) == {
            "valid": True,
            "checked_count": 3,
            "broken_evidence_id": None,
        }
        first_bytes = client.get(chain_integrity_url(machine_id)).content

    # A second restart over the now-complete database must not rewrite rows:
    # the audit bytes are identical and the chain still verifies.
    with TestClient(app) as client:
        assert client.get(chain_integrity_url(machine_id)).content == first_bytes
        listed_again = client.get(evidence_url(machine_id, event_id)).json()
        assert [r["chain_hash"] for r in listed_again] == [
            r["chain_hash"] for r in listed
        ]


def test_empty_database_can_register_and_query_after_startup(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        machine_id = create_machine(client)
        event_id = record_event(client, machine_id)
        record = add_evidence(client, machine_id, event_id, content_hash=HASH_A)
        assert record["previous_evidence_id"] is None
        assert get_chain_integrity(client, machine_id)["checked_count"] == 1
