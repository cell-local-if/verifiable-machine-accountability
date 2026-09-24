"""Tests for the privacy-preserving responsibility compliance export.

Covers
`GET /machines/{machine_id}/authorization-decision-events/privacy-responsibility/compliance-export`:
the `bad_time` / `invalid_query` / `not_found` outcomes (validation before any
data access), closed-UTC-window membership on each record's own `created_at`,
ordering by the actual UTC instant then record id (exact-second records before
fractional-second records of the same second), the `party_ref` / `role_ref`
SHA-256 pseudonyms (with `null` for non-string or blank stored values and the
raw `party` / `role` never appearing), verbatim export of damaged records,
machine isolation, strict read-only byte stability, and persistence across a
restart.
"""
import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app, privacy_responsibility_ref


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"


def export_url(machine_id, from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


HASH_A = "a" * 64
HASH_B = "b" * 64


def insert_assignment_row(client, machine_id, assignment_id, created_at, *,
                          event_id=None, incident_id=None, party="ops",
                          role="lead", previous_assignment_id=None,
                          content_hash=HASH_A, chain_hash=HASH_B):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO incident_responsibility_assignments "
                "(id, machine_id, event_id, incident_id, party, role, "
                "created_at, previous_assignment_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :event_id, :incident_id, :party, "
                ":role, :created_at, :previous_assignment_id, :content_hash, "
                ":chain_hash)"
            ),
            {
                "id": assignment_id,
                "machine_id": machine_id,
                "event_id": event_id if event_id is not None else str(uuid.uuid4()),
                "incident_id": (
                    incident_id if incident_id is not None else str(uuid.uuid4())
                ),
                "party": party,
                "role": role,
                "created_at": created_at,
                "previous_assignment_id": previous_assignment_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        )


def expected_ref(kind, machine_id, raw):
    stripped = raw.strip()
    payload = f"privacy:v1|{kind}{machine_id}{stripped}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_non_get_returns_405(client):
    machine_id = create_machine(client)
    response = client.post(export_url(machine_id))
    assert response.status_code == 405


def test_unknown_query_param_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id) + "&extra=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_query_param_checked_before_machine_lookup(client):
    response = client.get(export_url(rid(12345)) + "&extra=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "query",
    [
        "",  # both missing
        "?from_created_at=2026-03-01T00:00:00Z",  # to missing
        "?to_created_at=2026-03-01T00:00:00Z",  # from missing
        "?from_created_at=&to_created_at=2026-03-01T00:00:00Z",  # blank
        "?from_created_at=%20&to_created_at=2026-03-01T00:00:00Z",  # whitespace
        "?from_created_at=2026-03-01T00:00:00&to_created_at=2026-03-01T00:00:00Z",  # no Z
        "?from_created_at=2026-03-01T00:00:00%2B00:00&to_created_at=2026-03-01T00:00:00Z",  # offset
        "?from_created_at=2026-03-01&to_created_at=2026-03-01T00:00:00Z",  # date only
        "?from_created_at=2026-13-01T00:00:00Z&to_created_at=2026-03-01T00:00:00Z",  # month 13
        "?from_created_at=not-a-time&to_created_at=2026-03-01T00:00:00Z",
        "?from_created_at=2026-03-02T00:00:00Z&to_created_at=2026-03-01T00:00:00Z",  # inverted
    ],
)
def test_bad_time_rejected(client, query):
    machine_id = create_machine(client)
    url = (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"privacy-responsibility/compliance-export{query}"
    )
    response = client.get(url)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_bad_time_checked_before_machine_lookup(client):
    response = client.get(
        f"/machines/{rid(12345)}/authorization-decision-events/"
        "privacy-responsibility/compliance-export"
        "?from_created_at=nope&to_created_at=2026-03-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_legal(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T1, T1))
    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine_id,
        "from_created_at": T1,
        "to_created_at": T1,
        "assignments": [],
    }


def test_missing_machine_is_404_without_responsibility_data(client):
    response = client.get(export_url(rid(12345)))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_empty_window_keeps_assignments_array(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), T2)
    response = client.get(export_url(machine_id, T3, T3))
    assert response.status_code == 200
    assert response.json()["assignments"] == []


def test_closed_window_membership_and_echoed_bounds(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), T0)  # before
    insert_assignment_row(client, machine_id, rid(2), T1)  # lower bound
    insert_assignment_row(client, machine_id, rid(3), T2)  # inside
    insert_assignment_row(client, machine_id, rid(4), T3)  # upper bound
    insert_assignment_row(client, machine_id, rid(5), "2026-03-01T00:00:04Z")

    response = client.get(export_url(machine_id, T1, T3))
    assert response.status_code == 200
    body = response.json()
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == T1
    assert body["to_created_at"] == T3
    assert [a["id"] for a in body["assignments"]] == [rid(2), rid(3), rid(4)]


def test_fractional_bounds_are_inclusive(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), "2026-03-01T00:00:01.500Z")
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.500Z",
            "2026-03-01T00:00:01.500Z",
        )
    )
    assert response.status_code == 200
    assert [a["id"] for a in response.json()["assignments"]] == [rid(1)]


def test_ordering_exact_second_before_fractional_then_id(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(3), "2026-03-01T00:00:01.5Z")
    insert_assignment_row(client, machine_id, rid(2), "2026-03-01T00:00:01Z")
    insert_assignment_row(client, machine_id, rid(1), "2026-03-01T00:00:01.5Z")
    insert_assignment_row(client, machine_id, rid(4), "2026-03-01T00:00:00Z")

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert [a["id"] for a in response.json()["assignments"]] == [
        rid(4),  # earlier second
        rid(2),  # exact second before fractional of the same second
        rid(1),  # same instant as rid(3), id tie-break
        rid(3),
    ]


def test_record_fields_and_pseudonyms(client):
    machine_id = create_machine(client)
    insert_assignment_row(
        client,
        machine_id,
        rid(1),
        T1,
        event_id=rid(10),
        incident_id=rid(20),
        party="  Alice Example  ",
        role="incident commander",
        previous_assignment_id=rid(0),
        content_hash=HASH_A,
        chain_hash=HASH_B,
    )

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    (record,) = response.json()["assignments"]
    assert record == {
        "id": rid(1),
        "machine_id": machine_id,
        "event_id": rid(10),
        "incident_id": rid(20),
        "party_ref": expected_ref("party", machine_id, "  Alice Example  "),
        "role_ref": expected_ref("role", machine_id, "incident commander"),
        "created_at": T1,
        "previous_assignment_id": rid(0),
        "content_hash": HASH_A,
        "chain_hash": HASH_B,
    }
    # The raw party/role text never appears anywhere in the response.
    raw = response.content.decode()
    assert "Alice" not in raw
    assert "commander" not in raw
    assert '"party"' not in raw
    assert '"role"' not in raw


def test_blank_values_yield_null_refs(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), T1, party="   ", role="")

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    (record,) = response.json()["assignments"]
    assert record["party_ref"] is None
    assert record["role_ref"] is None


def test_non_string_values_yield_null_refs():
    # TEXT-affinity columns coerce on storage, so the non-string guard is
    # exercised against the helper directly.
    assert privacy_responsibility_ref("party", "machine-1", 42) is None
    assert privacy_responsibility_ref("role", "machine-1", None) is None
    assert privacy_responsibility_ref("party", "machine-1", "  ") is None
    assert privacy_responsibility_ref(
        "party", "machine-1", " alice "
    ) == hashlib.sha256(b"privacy:v1|partymachine-1alice").hexdigest()
    assert privacy_responsibility_ref(
        "role", "machine-1", "lead"
    ) == hashlib.sha256(b"privacy:v1|rolemachine-1lead").hexdigest()


def test_damaged_record_is_exported_verbatim(client):
    machine_id = create_machine(client)
    # Dangling event/incident references and corrupt chain fields: the record
    # is still exported exactly as stored.
    insert_assignment_row(
        client,
        machine_id,
        rid(1),
        T1,
        event_id="missing-event",
        incident_id="missing-incident",
        content_hash="not-a-hash",
        chain_hash="also-not-a-hash",
    )
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    (record,) = response.json()["assignments"]
    assert record["event_id"] == "missing-event"
    assert record["incident_id"] == "missing-incident"
    assert record["content_hash"] == "not-a-hash"
    assert record["chain_hash"] == "also-not-a-hash"


def test_other_machine_records_never_exported(client):
    machine_id = create_machine(client, "machine-1")
    other_id = create_machine(client, "machine-2")
    insert_assignment_row(client, machine_id, rid(1), T1, party="alice")
    insert_assignment_row(client, other_id, rid(2), T1, party="bob")

    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    records = response.json()["assignments"]
    assert [r["id"] for r in records] == [rid(1)]
    assert records[0]["party_ref"] == expected_ref("party", machine_id, "alice")
    assert "bob" not in response.content.decode()


def test_repeat_calls_are_byte_identical_and_read_only(client):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), T1)

    first = client.get(export_url(machine_id))
    second = client.get(export_url(machine_id))
    assert first.status_code == 200
    assert first.content == second.content

    with client.app.state.engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM incident_responsibility_assignments")
        ).scalar()
    assert count == 1


def test_records_persist_across_restart(client, tmp_path, monkeypatch):
    machine_id = create_machine(client)
    insert_assignment_row(client, machine_id, rid(1), T1, party="alice")
    first = client.get(export_url(machine_id))
    assert first.status_code == 200

    # Simulate a restart against the same database file.
    with TestClient(app) as restarted:
        second = restarted.get(export_url(machine_id))
    assert second.status_code == 200
    assert second.content == first.content
