"""Per-machine tamper-evident chain over machine status history.

Covers the chain fields on ``GET /machines/{id}/status-history`` and the
independent read-only verification sub-entry
``GET /machines/{id}/status-history/integrity``.
"""

import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app
from accountability.machine_status_chain import compute_chain_hash

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "machine_id",
    "from_status",
    "to_status",
    "created_at",
)

MISSING_MACHINE = "00000000-0000-0000-0000-000000000000"


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
    return response.json()


def set_status(client, machine_id, status):
    return client.post(f"/machines/{machine_id}/status", json={"status": status})


def history(client, machine_id):
    return client.get(f"/machines/{machine_id}/status-history").json()


def integrity_url(machine_id):
    return f"/machines/{machine_id}/status-history/integrity"


def integrity(client, machine_id):
    return client.get(integrity_url(machine_id))


def canonical_content_hash(record: dict) -> str:
    document = json.dumps(
        {key: record[key] for key in CONTENT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hashes(records):
    previous = ""
    expected = []
    for record in records:
        digest = canonical_content_hash(record)
        previous = compute_chain_hash(previous, digest)
        expected.append((digest, previous))
    return expected


# --- chain fields on the list ----------------------------------------------


def test_first_record_is_rooted_at_empty_prefix(client):
    machine = create_machine(client)
    assert set_status(client, machine["id"], "suspended").status_code == 200

    record = history(client, machine["id"])[0]

    assert list(record.keys()) == [
        "id",
        "machine_id",
        "from_status",
        "to_status",
        "created_at",
        "previous_status_event_id",
        "content_hash",
        "chain_hash",
    ]
    assert record["previous_status_event_id"] is None
    assert record["content_hash"] == canonical_content_hash(record)
    assert record["chain_hash"] == compute_chain_hash(
        "", record["content_hash"]
    )


def test_records_link_in_created_order(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")

    records = history(client, machine["id"])

    assert records[0]["previous_status_event_id"] is None
    assert records[1]["previous_status_event_id"] == records[0]["id"]
    for record, (content_hash, chain_hash) in zip(
        records, chain_hashes(records)
    ):
        assert record["content_hash"] == content_hash
        assert record["chain_hash"] == chain_hash


def test_status_change_stamp_matches_updated_at_and_history(client):
    machine = create_machine(client)
    updated = set_status(client, machine["id"], "suspended").json()

    record = history(client, machine["id"])[0]
    assert record["created_at"] == updated["updated_at"]


def test_chains_are_independent_per_machine(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    set_status(client, first["id"], "suspended")
    set_status(client, second["id"], "suspended")

    first_records = history(client, first["id"])
    second_records = history(client, second["id"])

    assert first_records[0]["previous_status_event_id"] is None
    assert second_records[0]["previous_status_event_id"] is None
    assert integrity(client, first["id"]).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
    assert integrity(client, second["id"]).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }


# --- integrity endpoint basics ---------------------------------------------


def test_empty_machine_reports_valid_zero_null(client):
    machine = create_machine(client)

    response = integrity(client, machine["id"])

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
    assert response.content == (
        b'{"valid":true,"checked_count":0,"broken_status_event_id":null}\n'
    )


def test_complete_chain_reports_valid(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")

    assert integrity(client, machine["id"]).json() == {
        "valid": True,
        "checked_count": 2,
        "broken_status_event_id": None,
    }


def test_missing_machine_returns_404_with_no_conclusion(client):
    response = integrity(client, MISSING_MACHINE)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    assert b"checked_count" not in response.content


def test_extra_query_param_is_422_before_machine_lookup(client):
    response = client.get(integrity_url(MISSING_MACHINE), params={"foo": "bar"})

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_request_body_is_422_before_machine_lookup(client):
    response = client.request("GET", integrity_url(MISSING_MACHINE), content=b"{}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_non_get_methods_return_405_without_reading(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    # Drop the table so any history read or chain computation would 500;
    # method routing must win and neither read nor write must happen.
    with client.app.state.engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("DROP TABLE machine_status_events"))

    for method in ("head", "post", "put", "patch", "delete"):
        response = getattr(client, method)(integrity_url(machine["id"]))
        assert response.status_code == 405, method


def test_read_failure_returns_500_with_no_partial_conclusion(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    with client.app.state.engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("DROP TABLE machine_status_events"))

    response = integrity(client, machine["id"])

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error"}}
    assert b"checked_count" not in response.content


# --- tampering --------------------------------------------------------------


def _connect(tmp_path):
    return sqlite3.connect(tmp_path / "test.db")


def test_detects_tampered_to_status(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")
    first = history(client, machine["id"])[0]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET to_status = 'active' WHERE id = ?",
        (first["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": first["id"],
    }


def test_detects_tampered_ownership_and_does_not_cross_machines(
    client, tmp_path
):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    set_status(client, first["id"], "suspended")
    record = history(client, first["id"])[0]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET machine_id = ? WHERE id = ?",
        (second["id"], record["id"]),
    )
    connection.commit()
    connection.close()

    # The original machine loses the record, leaving an empty chain; the
    # foreign machine never adopts a record it did not write.
    first_conclusion = integrity(client, first["id"]).json()
    assert first_conclusion == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
    # Under the other machine the misowned row cannot verify (its digest
    # covers a different owner and it is a foreign record).
    assert integrity(client, second["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": record["id"],
    }


def test_detects_tampered_previous_link(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")
    second = history(client, machine["id"])[1]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET previous_status_event_id = NULL "
        "WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_detects_tampered_chain_hash(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    first = history(client, machine["id"])[0]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
        ("0" * 64, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_self_loop_is_illegal_even_with_matching_digests(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    first = history(client, machine["id"])[0]

    # Rewrite the first edge as a self loop and recompute its digests to
    # match: the status-edge legality check must reject it independently of
    # the hash chain.
    tampered = {**first, "from_status": "suspended", "to_status": "suspended"}
    content_hash = canonical_content_hash(tampered)
    chain_hash = compute_chain_hash("", content_hash)
    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET from_status = 'suspended', "
        "to_status = 'suspended', content_hash = ?, chain_hash = ? WHERE id = ?",
        (content_hash, chain_hash, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_non_contiguous_edge_is_illegal(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    first = history(client, machine["id"])[0]

    # The carried state starts at active and the first edge moves it to
    # suspended. Rewrite the edge to suspended->active with consistent
    # digests: it no longer continues the carried state, so the edge check
    # rejects it independently of the hash chain.
    tampered = {**first, "from_status": "suspended", "to_status": "active"}
    content_hash = canonical_content_hash(tampered)
    chain_hash = compute_chain_hash("", content_hash)
    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET from_status = 'suspended', "
        "to_status = 'active', content_hash = ?, chain_hash = ? WHERE id = ?",
        (content_hash, chain_hash, first["id"]),
    )
    connection.commit()
    connection.close()

    # Carried state starts at active; suspended->active does not continue it.
    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_illegal_status_value_is_reported(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    first = history(client, machine["id"])[0]

    tampered = {**first, "to_status": "retired"}
    content_hash = canonical_content_hash(tampered)
    chain_hash = compute_chain_hash("", content_hash)
    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET to_status = 'retired', "
        "content_hash = ?, chain_hash = ? WHERE id = ?",
        (content_hash, chain_hash, first["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": first["id"],
    }


def test_damaged_created_at_counts_and_breaks_without_crash(client, tmp_path):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")
    second = history(client, machine["id"])[1]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET created_at = 'not-a-time' WHERE id = ?",
        (second["id"],),
    )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 2,
        "broken_status_event_id": second["id"],
    }


def test_reports_first_broken_record_only(client, tmp_path):
    machine = create_machine(client)
    for _ in range(3):
        set_status(client, machine["id"], "suspended")
        set_status(client, machine["id"], "active")
    records = history(client, machine["id"])

    connection = _connect(tmp_path)
    for record in records[1:]:
        connection.execute(
            "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
            ("f" * 64, record["id"]),
        )
    connection.commit()
    connection.close()

    assert integrity(client, machine["id"]).json() == {
        "valid": False,
        "checked_count": 6,
        "broken_status_event_id": records[1]["id"],
    }


def test_other_machines_damage_is_isolated(client, tmp_path):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    set_status(client, first["id"], "suspended")
    set_status(client, second["id"], "suspended")
    damaged = history(client, second["id"])[0]

    connection = _connect(tmp_path)
    connection.execute(
        "UPDATE machine_status_events SET chain_hash = ? WHERE id = ?",
        ("1" * 64, damaged["id"]),
    )
    connection.commit()
    connection.close()

    assert integrity(client, first["id"]).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
    assert integrity(client, second["id"]).json() == {
        "valid": False,
        "checked_count": 1,
        "broken_status_event_id": damaged["id"],
    }


# --- read-only, restart, backfill ------------------------------------------


def test_query_is_read_only_and_byte_identical(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")
    listed_before = history(client, machine["id"])

    first = integrity(client, machine["id"])
    second = integrity(client, machine["id"])

    assert first.content == second.content
    assert first.content.endswith(b"\n") and not first.content.endswith(b"\n\n")
    assert history(client, machine["id"]) == listed_before


def test_conclusion_survives_restart_and_complete_chain_restart_writes_nothing(
    tmp_path, monkeypatch
):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        set_status(first, machine["id"], "suspended")
        set_status(first, machine["id"], "active")
        conclusion = integrity(first, machine["id"]).json()
        listed = history(first, machine["id"])

        raw = sqlite3.connect(tmp_path / "persist.db")
        before = raw.execute(
            "SELECT id, previous_status_event_id, content_hash, chain_hash "
            "FROM machine_status_events ORDER BY created_at, id"
        ).fetchall()
        raw.close()

    with TestClient(app) as second:
        assert integrity(second, machine["id"]).json() == conclusion
        assert history(second, machine["id"]) == listed

    raw = sqlite3.connect(tmp_path / "persist.db")
    after = raw.execute(
        "SELECT id, previous_status_event_id, content_hash, chain_hash "
        "FROM machine_status_events ORDER BY created_at, id"
    ).fetchall()
    raw.close()
    # A restart over an already complete chain performs no writes.
    assert after == before


def test_legacy_rows_are_backfilled_on_startup(tmp_path, monkeypatch):
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
        CREATE TABLE machine_status_events (
            id VARCHAR(36) PRIMARY KEY, machine_id VARCHAR(36),
            from_status VARCHAR, to_status VARCHAR, created_at VARCHAR
        )
        """
    )
    connection.execute(
        "INSERT INTO machines VALUES (?,?,?,?,?,?,?,?)",
        (machine_id, "legacy", "Legacy", "key", "active", 1, "t0", "t0"),
    )
    # Inserted out of (created_at, id) order; ids deliberately invert the
    # id tiebreak relative to insertion.
    connection.executemany(
        "INSERT INTO machine_status_events VALUES (?,?,?,?,?)",
        [
            (
                "bbbbbbbb-0000-0000-0000-000000000000",
                machine_id,
                "suspended",
                "active",
                "2026-01-02T00:00:00Z",
            ),
            (
                "aaaaaaaa-0000-0000-0000-000000000000",
                machine_id,
                "active",
                "suspended",
                "2026-01-01T00:00:00Z",
            ),
        ],
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        records = history(client, machine_id)

        assert records[0]["previous_status_event_id"] is None
        assert records[1]["previous_status_event_id"] == records[0]["id"]
        for record, (content_hash, chain_hash) in zip(
            records, chain_hashes(records)
        ):
            assert HEX64_RE.match(record["content_hash"])
            assert record["content_hash"] == content_hash
            assert record["chain_hash"] == chain_hash

        assert integrity(client, machine_id).json() == {
            "valid": True,
            "checked_count": 2,
            "broken_status_event_id": None,
        }

    # A second restart over the backfilled, complete chain writes nothing.
    with TestClient(app) as client:
        assert integrity(client, machine_id).json()["valid"] is True


def test_empty_database_is_directly_usable(client):
    machine = create_machine(client)
    assert integrity(client, machine["id"]).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }
    set_status(client, machine["id"], "suspended")
    assert integrity(client, machine["id"]).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }


# --- concurrency and unchanged status semantics -----------------------------


def test_concurrent_toggles_form_one_unbroken_chain(client):
    machine = create_machine(client)

    def toggle(_):
        set_status(client, machine["id"], "suspended")
        return set_status(client, machine["id"], "active")

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(toggle, range(8)))

    # Every worker ends on an accepted reactivation (409s may interleave but
    # the final active transition of each serialized winner is recorded).
    assert all(response.status_code in (200, 409) for response in responses)

    connection = sqlite3.connect(client.app.state.engine.url.database)
    rows = connection.execute(
        "SELECT id, previous_status_event_id FROM machine_status_events "
        "WHERE machine_id = ?",
        (machine["id"],),
    ).fetchall()
    connection.close()

    assert len(rows) >= 2
    previous_ids = [row[1] for row in rows]
    assert previous_ids[0] is None
    ids = {row[0] for row in rows}
    assert set(pid for pid in previous_ids if pid is not None) <= ids
    assert len(previous_ids) == len(set(previous_ids))

    conclusion = integrity(client, machine["id"]).json()
    assert conclusion["valid"] is True
    assert conclusion["checked_count"] == len(rows)
    assert conclusion["broken_status_event_id"] is None


def test_status_change_error_semantics_are_unchanged(client):
    machine = create_machine(client)

    # Bad body: 422 before lookup.
    assert (
        client.post(
            f"/machines/{MISSING_MACHINE}/status", json={"status": "gone"}
        ).status_code
        == 422
    )
    # Missing machine: 404.
    assert (
        set_status(client, MISSING_MACHINE, "suspended").status_code == 404
    )
    # Same-status transition: 409, no history appended.
    assert set_status(client, machine["id"], "active").status_code == 409
    assert integrity(client, machine["id"]).json() == {
        "valid": True,
        "checked_count": 0,
        "broken_status_event_id": None,
    }


def test_concurrent_same_target_has_at_most_one_success(client):
    machine = create_machine(client)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                lambda _: set_status(client, machine["id"], "suspended"),
                range(4),
            )
        )

    assert sum(response.status_code == 200 for response in responses) == 1
    assert integrity(client, machine["id"]).json() == {
        "valid": True,
        "checked_count": 1,
        "broken_status_event_id": None,
    }
