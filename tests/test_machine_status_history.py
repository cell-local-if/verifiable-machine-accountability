import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
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


def get_history(client, machine_id):
    return client.get(f"/machines/{machine_id}/status-history")


# --- basics ----------------------------------------------------------------


def test_empty_history_returns_empty_array(client):
    machine = create_machine(client)

    response = get_history(client, machine["id"])

    assert response.status_code == 200
    assert response.json() == []
    assert response.content == b"[]\n"


def test_suspend_appends_one_history_record(client):
    machine = create_machine(client)

    assert set_status(client, machine["id"], "suspended").status_code == 200

    response = get_history(client, machine["id"])
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1
    record = records[0]
    assert list(record.keys()) == [
        "id",
        "machine_id",
        "from_status",
        "to_status",
        "created_at",
    ]
    assert record["machine_id"] == machine["id"]
    assert record["from_status"] == "active"
    assert record["to_status"] == "suspended"
    assert RFC3339_Z_RE.match(record["created_at"])


def test_history_created_at_matches_commit_moment(client):
    machine = create_machine(client)
    updated = set_status(client, machine["id"], "suspended").json()

    record = get_history(client, machine["id"]).json()[0]
    # The history stamp is the same commit-moment clock read as updated_at.
    assert record["created_at"] == updated["updated_at"]


def test_suspend_reactivate_records_both_transitions_in_order(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")

    records = get_history(client, machine["id"]).json()

    assert [(r["from_status"], r["to_status"]) for r in records] == [
        ("active", "suspended"),
        ("suspended", "active"),
    ]
    assert records[0]["created_at"] <= records[1]["created_at"]


def test_response_is_compact_utf8_json_ending_in_newline(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    response = get_history(client, machine["id"])

    assert response.headers["content-type"].startswith("application/json")
    body = response.content
    assert body.endswith(b"\n")
    assert not body.endswith(b"\n\n")
    # Compact separators: no ", " or ": " anywhere.
    assert b", " not in body
    assert b'": ' not in body
    # Round-trips to the same structure.
    assert json.loads(body.decode("utf-8")) == response.json()


def test_rejected_status_change_appends_no_history(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    # Same-status (409), invalid payload (422), and missing machine (404).
    assert set_status(client, machine["id"], "suspended").status_code == 409
    assert (
        client.post(
            f"/machines/{machine['id']}/status", json={"status": "deleted"}
        ).status_code
        == 422
    )
    assert set_status(client, MISSING_MACHINE, "suspended").status_code == 404

    records = get_history(client, machine["id"]).json()
    assert len(records) == 1
    assert records[0]["to_status"] == "suspended"


def test_history_is_isolated_per_machine(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")
    set_status(client, first["id"], "suspended")

    assert get_history(client, second["id"]).json() == []
    records = get_history(client, first["id"]).json()
    assert len(records) == 1
    assert records[0]["machine_id"] == first["id"]


def test_missing_machine_returns_404(client):
    response = get_history(client, MISSING_MACHINE)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --- query-string and method rules -----------------------------------------


@pytest.mark.parametrize("query", ["?from_status=active", "?foo=bar", "?limit=1"])
def test_unknown_query_param_returns_422_invalid_query(client, query):
    machine = create_machine(client)

    response = client.get(f"/machines/{machine['id']}/status-history{query}")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_unknown_query_param_is_422_even_for_missing_machine(client):
    response = client.get(f"/machines/{MISSING_MACHINE}/status-history?foo=bar")

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_non_get_methods_return_405(client, method):
    machine = create_machine(client)

    response = getattr(client, method)(f"/machines/{machine['id']}/status-history")

    assert response.status_code == 405
    # The rejected method neither reads nor writes anything.
    assert get_history(client, machine["id"]).json() == []


# --- concurrency and persistence -------------------------------------------


def test_concurrent_same_target_leaves_exactly_one_history_record(client):
    machine = create_machine(client)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                lambda _: set_status(client, machine["id"], "suspended"), range(4)
            )
        )

    assert sum(r.status_code == 200 for r in responses) == 1
    records = get_history(client, machine["id"]).json()
    assert len(records) == 1
    assert records[0]["from_status"] == "active"
    assert records[0]["to_status"] == "suspended"


def test_history_survives_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        set_status(first, machine["id"], "suspended")
        set_status(first, machine["id"], "active")

    with TestClient(app) as second:
        records = get_history(second, machine["id"]).json()
        assert [(r["from_status"], r["to_status"]) for r in records] == [
            ("active", "suspended"),
            ("suspended", "active"),
        ]
        # History keeps growing after the restart.
        set_status(second, machine["id"], "suspended")

    with TestClient(app) as third:
        records = get_history(third, machine["id"]).json()
        assert len(records) == 3
        assert records[-1]["to_status"] == "suspended"


def test_old_database_without_history_table_starts_and_serves(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "old.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine = create_machine(first)
        # Simulate a database from before the history feature.
        engine = first.app.state.engine
        from accountability.db import MachineStatusEvent

        with engine.begin() as conn:
            conn.execute(MachineStatusEvent.__table__.delete())
        MachineStatusEvent.__table__.drop(engine)

    with TestClient(app) as second:
        # Startup recreated the table; the machine itself is untouched.
        assert second.get(f"/machines/{machine['id']}").json()["status"] == "active"
        assert get_history(second, machine["id"]).json() == []
        set_status(second, machine["id"], "suspended")
        records = get_history(second, machine["id"]).json()
        assert len(records) == 1
        assert records[0]["to_status"] == "suspended"


# --- the rest of the contract is untouched ----------------------------------


def test_status_change_response_and_machine_record_unchanged(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "id",
        "external_id",
        "display_name",
        "public_key",
        "status",
        "version",
        "created_at",
        "updated_at",
    }
    assert body["status"] == "suspended"
    assert body["version"] == 1
    assert client.get(f"/machines/{machine['id']}").json() == body


def test_history_query_does_not_modify_anything(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")
    before_machine = client.get(f"/machines/{machine['id']}").json()
    before_history = get_history(client, machine["id"]).content

    for _ in range(3):
        get_history(client, machine["id"])

    assert client.get(f"/machines/{machine['id']}").json() == before_machine
    assert get_history(client, machine["id"]).content == before_history
