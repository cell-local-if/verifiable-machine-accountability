"""Tests for the read-only key-rotation history/current-state reconciliation.

    GET /machines/{machine_id}/key-rotation-events/reconciliation

The endpoint cross-checks, strictly read-only and only for the path
machine, whether the machine's current ``public_key`` and ``version`` can
be explained by its complete rotation history: every record must carry a
parseable moment, a sound previous-rotation link and content/chain hashes,
the next version of the 2-then-plus-one sequence, and — after the first
record — an ``old_public_key`` continuing its predecessor's
``new_public_key``; the chain tail must line up with the machine's current
row, and an empty history explains only ``version`` ``1``.

These tests cover the sound flows (empty history, one and several
rotations), every anomaly category (timestamp, chain, version transition,
key transition, current state), the check order and damaged-timestamp
ordering, machine isolation, request-shape validation (422 before 404),
404, 405, and the read-only guarantee across repeated calls and restarts.
"""
import uuid
from datetime import datetime, timezone

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
    return response.json()["id"]


def rotate(client, machine_id, new_public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": new_public_key, "expected_version": expected_version},
    )
    assert response.status_code == 200
    return response.json()


def reconciliation_url(machine_id):
    return f"/machines/{machine_id}/key-rotation-events/reconciliation"


def reconcile(client, machine_id):
    response = client.get(reconciliation_url(machine_id))
    assert response.status_code == 200
    return response.json()


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
                    "WHERE machine_id = :machine_id"
                ).bindparams(machine_id=machine_id)
            ).mappings()
        )


def _instant(value):
    """The reconciliation's ordering instant; damaged stamps sort last."""
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except (ValueError, TypeError):
        return datetime.max.replace(tzinfo=timezone.utc)


def ordered_rows(client, machine_id):
    """The machine's rotation rows in the reconciliation's check order."""
    return sorted(
        rotation_rows(client, machine_id),
        key=lambda row: (_instant(row["created_at"]), row["id"]),
    )


def recompute_chain(client, machine_id):
    """Rewrite a machine's chain columns to the recomputed values, in the
    reconciliation's (instant, id) check order, leaving every other stored
    field — including any damaged one — exactly as it is."""
    rows = ordered_rows(client, machine_id)
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
# Sound flows reconcile clean
# --------------------------------------------------------------------------- #


def test_empty_history_at_initial_version_is_valid(client):
    machine_id = create_machine(client)
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 0,
        "latest_rotation_id": None,
        "broken_rotation_id": None,
        "anomaly": None,
    }


def test_single_rotation_is_valid(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rows = ordered_rows(client, machine_id)
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 1,
        "latest_rotation_id": rows[0]["id"],
        "broken_rotation_id": None,
        "anomaly": None,
    }


def test_several_rotations_are_valid(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    rotate(client, machine_id, "key-4", 3)
    tail = ordered_rows(client, machine_id)[-1]
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": True,
        "checked_count": 3,
        "latest_rotation_id": tail["id"],
        "broken_rotation_id": None,
        "anomaly": None,
    }


def test_response_body_is_compact_ordered_and_newline_terminated(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    response = client.get(reconciliation_url(machine_id))
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert body.endswith("\n") and not body.endswith("\n\n")
    assert body == (
        '{"machine_id":"%s","valid":true,"checked_count":1,'
        '"latest_rotation_id":"%s","broken_rotation_id":null,'
        '"anomaly":null}\n'
    ) % (machine_id, rotation_rows(client, machine_id)[0]["id"])


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #


def test_damaged_created_at_reports_timestamp_unparseable(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    tail = ordered_rows(client, machine_id)[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET created_at = 'not-a-moment' "
        "WHERE id = :id",
        id=tail["id"],
    )
    # Keep the chain itself sound so the moment is the first anomaly.
    recompute_chain(client, machine_id)
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 2,
        "latest_rotation_id": tail["id"],
        "broken_rotation_id": tail["id"],
        "anomaly": "timestamp_unparseable",
    }


def test_damaged_stamp_sorts_last_so_earlier_chain_break_wins(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    first, second = ordered_rows(client, machine_id)
    # Damage the first record's moment without repairing anything: the
    # damaged stamp sorts last, so the second record is checked first and
    # its previous-rotation link no longer matches the check order.
    db_execute(
        client,
        "UPDATE key_rotation_events SET created_at = 'garbage' WHERE id = :id",
        id=first["id"],
    )
    conclusion = reconcile(client, machine_id)
    assert conclusion["valid"] is False
    assert conclusion["broken_rotation_id"] == second["id"]
    assert conclusion["anomaly"] == "chain_mismatch"
    assert conclusion["latest_rotation_id"] == first["id"]


def test_tampered_content_hash_reports_chain_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    first = ordered_rows(client, machine_id)[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=first["id"],
    )
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 2,
        "latest_rotation_id": ordered_rows(client, machine_id)[1]["id"],
        "broken_rotation_id": first["id"],
        "anomaly": "chain_mismatch",
    }


def test_broken_previous_link_reports_chain_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    second = ordered_rows(client, machine_id)[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET previous_rotation_id = :p "
        "WHERE id = :id",
        p=str(uuid.uuid4()),
        id=second["id"],
    )
    conclusion = reconcile(client, machine_id)
    assert conclusion["valid"] is False
    assert conclusion["broken_rotation_id"] == second["id"]
    assert conclusion["anomaly"] == "chain_mismatch"


def test_version_skip_reports_version_transition_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    second = ordered_rows(client, machine_id)[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET version = 5 WHERE id = :id",
        id=second["id"],
    )
    # Keep the chain sound so the version transition is the first anomaly.
    recompute_chain(client, machine_id)
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 2,
        "latest_rotation_id": second["id"],
        "broken_rotation_id": second["id"],
        "anomaly": "version_transition_mismatch",
    }


def test_first_record_version_not_two_reports_version_transition(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    first = ordered_rows(client, machine_id)[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET version = 3 WHERE id = :id",
        id=first["id"],
    )
    recompute_chain(client, machine_id)
    conclusion = reconcile(client, machine_id)
    assert conclusion["valid"] is False
    assert conclusion["broken_rotation_id"] == first["id"]
    assert conclusion["anomaly"] == "version_transition_mismatch"


def test_key_gap_reports_key_transition_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    second = ordered_rows(client, machine_id)[1]
    db_execute(
        client,
        "UPDATE key_rotation_events SET old_public_key = 'key-9' "
        "WHERE id = :id",
        id=second["id"],
    )
    # Keep the chain sound so the key transition is the first anomaly.
    recompute_chain(client, machine_id)
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 2,
        "latest_rotation_id": second["id"],
        "broken_rotation_id": second["id"],
        "anomaly": "key_transition_mismatch",
    }


def test_current_key_drift_reports_current_state_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    tail = ordered_rows(client, machine_id)[0]
    db_execute(
        client,
        "UPDATE machines SET public_key = 'key-9' WHERE id = :id",
        id=machine_id,
    )
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 1,
        "latest_rotation_id": tail["id"],
        "broken_rotation_id": None,
        "anomaly": "current_state_mismatch",
    }


def test_current_version_drift_reports_current_state_mismatch(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    tail = ordered_rows(client, machine_id)[0]
    db_execute(
        client,
        "UPDATE machines SET version = 7 WHERE id = :id",
        id=machine_id,
    )
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 1,
        "latest_rotation_id": tail["id"],
        "broken_rotation_id": None,
        "anomaly": "current_state_mismatch",
    }


def test_empty_history_with_drifted_version_reports_current_state(client):
    machine_id = create_machine(client)
    db_execute(
        client,
        "UPDATE machines SET version = 2 WHERE id = :id",
        id=machine_id,
    )
    assert reconcile(client, machine_id) == {
        "machine_id": machine_id,
        "valid": False,
        "checked_count": 0,
        "latest_rotation_id": None,
        "broken_rotation_id": None,
        "anomaly": "current_state_mismatch",
    }


# --------------------------------------------------------------------------- #
# Isolation, stability, and the read-only guarantee
# --------------------------------------------------------------------------- #


def test_other_machines_never_change_the_outcome(client):
    broken_id = create_machine(client, external_id="machine-a")
    sound_id = create_machine(client, external_id="machine-b", public_key="k1")
    rotate(client, sound_id, "k2", 1)
    rotate(client, sound_id, "k3", 2)
    before = reconcile(client, sound_id)
    assert before["valid"] is True

    # Damage the other machine's only rotation beyond recognition.
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash, "
        "created_at = 'garbage' WHERE machine_id = :id",
        hash="0" * 64,
        id=broken_id,
    )
    db_execute(
        client,
        "UPDATE machines SET version = 9 WHERE id = :id",
        id=broken_id,
    )
    assert reconcile(client, sound_id) == before


def test_repeated_reads_are_byte_identical_and_never_write(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    first = ordered_rows(client, machine_id)[0]
    db_execute(
        client,
        "UPDATE key_rotation_events SET content_hash = :hash WHERE id = :id",
        hash="0" * 64,
        id=first["id"],
    )
    one = client.get(reconciliation_url(machine_id))
    two = client.get(reconciliation_url(machine_id))
    assert one.status_code == two.status_code == 200
    assert one.content == two.content
    # The damaged value was reported, never repaired.
    assert rotation_rows(client, machine_id)[0]["content_hash"] == "0" * 64


def test_conclusion_survives_restart(client, tmp_path, monkeypatch):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    before = client.get(reconciliation_url(machine_id)).content

    # A fresh app instance over the same database file reaches the same
    # conclusion; the read-only query never rewrote anything.
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as second_client:
        assert (
            second_client.get(reconciliation_url(machine_id)).content == before
        )


# --------------------------------------------------------------------------- #
# Request shape, 404, and 405
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "suffix",
    ["?limit=1", "?machine_id=x", "?a=1&a=2", "?a=", "?a"],
)
def test_any_query_string_is_invalid_query(client, suffix):
    machine_id = create_machine(client)
    response = client.get(reconciliation_url(machine_id) + suffix)
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_carried_body_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.request(
        "GET", reconciliation_url(machine_id), content=b'{"x": 1}'
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_query_validation_precedes_machine_lookup(client):
    missing = str(uuid.uuid4())
    response = client.get(reconciliation_url(missing) + "?limit=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}
    response = client.request(
        "GET", reconciliation_url(missing), content=b"{}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_is_not_found(client):
    response = client.get(reconciliation_url(str(uuid.uuid4())))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_only_get_is_routed(client, method):
    machine_id = create_machine(client)
    response = getattr(client, method)(reconciliation_url(machine_id))
    assert response.status_code == 405
