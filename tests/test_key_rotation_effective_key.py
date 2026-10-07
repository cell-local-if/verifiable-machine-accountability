"""Tests for the read-only effective-key query over the rotation timeline.

    GET /machines/{machine_id}/key-rotation-events/effective-key?at=<moment>

The endpoint answers, strictly read-only and only for the path machine,
which public key was in effect at the requested UTC moment: moments before
the machine's creation report nulls with ``key_version`` ``0``; a machine
that never rotated reports its current key since its ``created_at``; with
rotations, the records segment the timeline in (created-at instant, id)
order — before the first record the answer is its ``old_public_key``,
afterwards the most recent record's ``new_public_key`` — and the chain
tail must line up with the machine's current ``public_key``.

These tests cover the pre-creation, no-rotation, pre-first-rotation,
on-the-moment, between-rotations, and after-the-tail segments; the
``at``-echo and fixed response shape; request-shape validation
(``invalid_query``/``bad_time`` before 404); 404; 405; the 500
``internal_error`` on a damaged or non-reconciling history; machine
isolation; and the read-only, byte-identical, restart-stable guarantee.
"""
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability.rotation_chain import (
    compute_chain_hash,
    compute_content_hash,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


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
    return response.json()


def rotate(client, machine_id, new_public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": new_public_key, "expected_version": expected_version},
    )
    assert response.status_code == 200
    return response.json()


def events(client, machine_id):
    response = client.get(f"/machines/{machine_id}/key-rotation-events")
    assert response.status_code == 200
    return response.json()


def effective_key(client, machine_id, at):
    return client.get(
        f"/machines/{machine_id}/key-rotation-events/effective-key",
        params={"at": at},
    )


def moment_before(stamp):
    """The stamp's instant shifted one microsecond earlier, as a Z stamp."""
    instant = datetime.fromisoformat(stamp[:-1] + "+00:00")
    return (instant - timedelta(microseconds=1)).isoformat().replace(
        "+00:00", "Z"
    )


def db_execute(client, statement, **params):
    with client.app.state.engine.begin() as conn:
        conn.execute(text(statement).bindparams(**params))


def rotation_rows(client, machine_id):
    with client.app.state.engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT id, machine_id, old_public_key, new_public_key, "
                    "version, created_at, previous_rotation_id, content_hash, "
                    "chain_hash FROM key_rotation_events "
                    "WHERE machine_id = :machine_id ORDER BY created_at, id"
                ).bindparams(machine_id=machine_id)
            ).mappings()
        )


def recompute_chain(client, machine_id):
    """Rewrite a machine's chain columns to the recomputed values, leaving
    every other stored field — including any damaged one — exactly as it
    is, so a single chosen anomaly remains."""
    rows = rotation_rows(client, machine_id)
    previous_rotation_id = None
    previous_chain_hash = ""
    with client.app.state.engine.begin() as conn:
        for row in rows:
            content_hash = compute_content_hash(
                **{
                    key: row[key]
                    for key in (
                        "id",
                        "machine_id",
                        "old_public_key",
                        "new_public_key",
                        "version",
                        "created_at",
                    )
                }
            )
            chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
            conn.execute(
                text(
                    "UPDATE key_rotation_events SET previous_rotation_id = :p, "
                    "content_hash = :c, chain_hash = :h WHERE id = :id"
                ).bindparams(
                    p=previous_rotation_id,
                    c=content_hash,
                    h=chain_hash,
                    id=row["id"],
                )
            )
            previous_rotation_id = row["id"]
            previous_chain_hash = chain_hash


# --------------------------------------------------------------------------- #
# Timeline segments
# --------------------------------------------------------------------------- #


def test_before_machine_creation_reports_nulls(client):
    machine = create_machine(client)

    response = effective_key(client, machine["id"], "2000-01-01T00:00:00Z")

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": "2000-01-01T00:00:00Z",
        "effective_key": None,
        "key_version": 0,
        "effective_from": None,
        "rotation_id": None,
    }


def test_never_rotated_reports_current_key_since_creation(client):
    machine = create_machine(client)

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": "2999-01-01T00:00:00Z",
        "effective_key": "key-1",
        "key_version": 1,
        "effective_from": machine["created_at"],
        "rotation_id": None,
    }


def test_at_exactly_machine_created_at_counts_as_existing(client):
    machine = create_machine(client)

    response = effective_key(client, machine["id"], machine["created_at"])

    assert response.status_code == 200
    assert response.json()["effective_key"] == "key-1"
    assert response.json()["effective_from"] == machine["created_at"]
    assert response.json()["rotation_id"] is None


def test_before_first_rotation_reports_initial_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    response = effective_key(client, machine["id"], machine["created_at"])

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": machine["created_at"],
        "effective_key": "key-1",
        "key_version": 1,
        "effective_from": machine["created_at"],
        "rotation_id": None,
    }


def test_at_rotation_moment_uses_the_new_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    event = events(client, machine["id"])[0]

    response = effective_key(client, machine["id"], event["created_at"])

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": event["created_at"],
        "effective_key": "key-2",
        "key_version": 2,
        "effective_from": event["created_at"],
        "rotation_id": event["id"],
    }


def test_one_microsecond_before_rotation_still_uses_the_old_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    event = events(client, machine["id"])[0]
    at = moment_before(event["created_at"])

    response = effective_key(client, machine["id"], at)

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": at,
        "effective_key": "key-1",
        "key_version": 1,
        "effective_from": machine["created_at"],
        "rotation_id": None,
    }


def test_between_rotations_reports_the_most_recent_event(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    rotate(client, machine["id"], "key-4", 3)
    first, second, third = events(client, machine["id"])

    at = moment_before(third["created_at"])
    response = effective_key(client, machine["id"], at)

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": at,
        "effective_key": "key-3",
        "key_version": 3,
        "effective_from": second["created_at"],
        "rotation_id": second["id"],
    }
    assert first["id"] != second["id"]


def test_after_last_rotation_matches_the_machine_current_key(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    tail = events(client, machine["id"])[-1]
    current = client.get(f"/machines/{machine['id']}").json()

    response = effective_key(client, machine["id"], "2999-12-31T23:59:59Z")

    assert response.status_code == 200
    assert response.json() == {
        "machine_id": machine["id"],
        "at": "2999-12-31T23:59:59Z",
        "effective_key": "key-3",
        "key_version": 3,
        "effective_from": tail["created_at"],
        "rotation_id": tail["id"],
    }
    # The last segment is exactly the machine's current key and version.
    assert current["public_key"] == "key-3"
    assert current["version"] == 3


def test_at_is_echoed_verbatim_with_fractional_seconds(client):
    machine = create_machine(client)
    at = "2001-02-03T04:05:06.123456Z"

    response = effective_key(client, machine["id"], at)

    assert response.status_code == 200
    assert response.json()["at"] == at


def test_response_body_is_compact_ordered_and_newline_terminated(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    event = events(client, machine["id"])[0]

    response = effective_key(client, machine["id"], event["created_at"])

    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert body.endswith("\n") and not body.endswith("\n\n")
    assert body == (
        '{"machine_id":"%s","at":"%s","effective_key":"key-2",'
        '"key_version":2,"effective_from":"%s","rotation_id":"%s"}\n'
    ) % (machine["id"], event["created_at"], event["created_at"], event["id"])


# --------------------------------------------------------------------------- #
# Request validation (before any machine lookup)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "suffix",
    ["?at=2000-01-01T00:00:00Z&limit=1", "?machine_id=x&at=2000-01-01T00:00:00Z"],
)
def test_unknown_parameter_is_invalid_query(client, suffix):
    machine = create_machine(client)
    response = client.get(
        f"/machines/{machine['id']}/key-rotation-events/effective-key{suffix}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_repeated_at_is_invalid_query(client):
    machine = create_machine(client)
    response = client.get(
        f"/machines/{machine['id']}/key-rotation-events/effective-key"
        "?at=2000-01-01T00:00:00Z&at=2001-01-01T00:00:00Z"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_carried_body_is_invalid_query(client):
    machine = create_machine(client)
    response = client.request(
        "GET",
        f"/machines/{machine['id']}/key-rotation-events/effective-key"
        "?at=2000-01-01T00:00:00Z",
        content=b'{"x": 1}',
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "suffix",
    [
        "",  # missing
        "?at=",  # blank
        "?at=not-a-time",  # malformed
        "?at=2000-01-01T00:00:00",  # missing Z
        "?at=2000-01-01T00:00:00+00:00",  # offset form
        "?at=%202000-01-01T00:00:00Z%20",  # surrounding whitespace
        "?at=2000-13-01T00:00:00Z",  # month out of range
        "?at=2000-01-01T24:00:00Z",  # hour out of range
    ],
)
def test_missing_or_invalid_at_is_bad_time(client, suffix):
    machine = create_machine(client)
    response = client.get(
        f"/machines/{machine['id']}/key-rotation-events/effective-key{suffix}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_query_validation_precedes_machine_lookup(client):
    missing = str(uuid.uuid4())
    base = f"/machines/{missing}/key-rotation-events/effective-key"

    response = client.get(base + "?at=2000-01-01T00:00:00Z&extra=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}

    response = client.get(base)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}

    response = client.request(
        "GET", base + "?at=2000-01-01T00:00:00Z", content=b"{}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_not_found(client):
    response = effective_key(
        client, str(uuid.uuid4()), "2000-01-01T00:00:00Z"
    )
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_is_routed(client, method):
    machine = create_machine(client)
    response = getattr(client, method)(
        f"/machines/{machine['id']}/key-rotation-events/effective-key"
    )
    assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Damaged or non-reconciling history is a 500, never a partial timeline
# --------------------------------------------------------------------------- #


def test_current_key_drift_is_internal_error(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    db_execute(
        client,
        "UPDATE machines SET public_key = 'key-9' WHERE id = :id",
        id=machine["id"],
    )

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_version_chain_gap_is_internal_error(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    second = rotation_rows(client, machine["id"])[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET version = 5 WHERE id = :id",
        id=second["id"],
    )
    recompute_chain(client, machine["id"])

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_key_chain_gap_is_internal_error(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    second = rotation_rows(client, machine["id"])[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET old_public_key = 'key-9' WHERE id = :id",
        id=second["id"],
    )
    recompute_chain(client, machine["id"])

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_damaged_event_moment_is_internal_error(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    row = rotation_rows(client, machine["id"])[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET created_at = 'garbage' WHERE id = :id",
        id=row["id"],
    )
    recompute_chain(client, machine["id"])

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_tampered_chain_field_is_internal_error(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    row = rotation_rows(client, machine["id"])[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=row["id"],
    )

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_damaged_machine_created_at_is_internal_error(client):
    machine = create_machine(client)
    db_execute(
        client,
        "UPDATE machines SET created_at = 'not-a-moment' WHERE id = :id",
        id=machine["id"],
    )

    response = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


def test_damaged_history_reports_500_even_before_creation(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    db_execute(
        client,
        "UPDATE machines SET public_key = 'key-9' WHERE id = :id",
        id=machine["id"],
    )

    response = effective_key(client, machine["id"], "2000-01-01T00:00:00Z")

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}


# --------------------------------------------------------------------------- #
# Isolation, stability, and the read-only guarantee
# --------------------------------------------------------------------------- #


def test_other_machines_never_change_the_outcome(client):
    broken = create_machine(client, external_id="machine-a")
    sound = create_machine(client, external_id="machine-b", public_key="k1")
    rotate(client, sound["id"], "k2", 1)
    rotate(client, sound["id"], "k3", 2)
    before = effective_key(client, sound["id"], "2999-01-01T00:00:00Z").content

    # Damage the other machine's history and current row beyond recognition.
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash, "
        "created_at = 'garbage' WHERE machine_id = :id",
        hash="0" * 64,
        id=broken["id"],
    )
    db_execute(
        client,
        "UPDATE machines SET version = 9 WHERE id = :id",
        id=broken["id"],
    )

    response = effective_key(client, sound["id"], "2999-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.content == before


def test_repeated_reads_are_byte_identical_and_never_write(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    row = rotation_rows(client, machine["id"])[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=row["id"],
    )

    one = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")
    two = effective_key(client, machine["id"], "2999-01-01T00:00:00Z")
    assert one.status_code == two.status_code == 500
    assert one.content == two.content
    # The damaged value was reported, never repaired.
    assert rotation_rows(client, machine["id"])[0]["content_hash"] == "0" * 64


def test_query_does_not_modify_machine_or_events(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    before_machine = client.get(f"/machines/{machine['id']}").json()
    before_events = events(client, machine["id"])

    effective_key(client, machine["id"], "2000-01-01T00:00:00Z")
    effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    assert client.get(f"/machines/{machine['id']}").json() == before_machine
    assert events(client, machine["id"]) == before_events


def test_result_survives_restart(client, tmp_path, monkeypatch):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)
    rotate(client, machine["id"], "key-3", 2)
    before = effective_key(client, machine["id"], "2999-01-01T00:00:00Z").content

    # A fresh app instance over the same database file answers the same.
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as second_client:
        assert (
            effective_key(second_client, machine["id"], "2999-01-01T00:00:00Z").content
            == before
        )


def test_existing_rotation_endpoints_keep_their_behavior(client):
    machine = create_machine(client)
    rotate(client, machine["id"], "key-2", 1)

    effective_key(client, machine["id"], "2999-01-01T00:00:00Z")

    listing = client.get(f"/machines/{machine['id']}/key-rotation-events")
    assert listing.status_code == 200
    assert len(listing.json()) == 1
    integrity = client.get(
        f"/machines/{machine['id']}/key-rotation-events/integrity"
    )
    assert integrity.status_code == 200
    assert integrity.json()["valid"] is True
    # rotate-key itself is untouched: a follow-up rotation still works.
    rotate(client, machine["id"], "key-3", 2)
    assert client.get(f"/machines/{machine['id']}").json()["version"] == 3
