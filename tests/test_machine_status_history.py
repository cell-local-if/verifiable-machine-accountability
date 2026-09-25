import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

RFC3339_Z_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
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
    return response.json()


def set_status(client, machine_id, status):
    return client.post(f"/machines/{machine_id}/status", json={"status": status})


def get_history(client, machine_id):
    return client.get(f"/machines/{machine_id}/status-history")


# --- history query basics ---------------------------------------------------


def test_new_machine_has_empty_history(client):
    machine = create_machine(client)

    response = get_history(client, machine["id"])

    assert response.status_code == 200
    assert response.json() == []
    assert response.content == b"[]\n"


def test_history_records_each_accepted_transition_in_order(client):
    machine = create_machine(client)

    assert get_history(client, machine["id"]).json() == []

    set_status(client, machine["id"], "suspended")
    set_status(client, machine["id"], "active")
    set_status(client, machine["id"], "suspended")

    response = get_history(client, machine["id"])
    assert response.status_code == 200
    records = response.json()
    assert [(r["from_status"], r["to_status"]) for r in records] == [
        ("active", "suspended"),
        ("suspended", "active"),
        ("active", "suspended"),
    ]
    for record in records:
        assert list(record.keys()) == [
            "id",
            "machine_id",
            "from_status",
            "to_status",
            "created_at",
        ]
        assert record["machine_id"] == machine["id"]
        assert RFC3339_Z_RE.match(record["created_at"])


def test_history_response_is_compact_json_ending_in_newline(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    response = get_history(client, machine["id"])

    assert response.headers["content-type"].startswith("application/json")
    body = response.content
    assert body.endswith(b"\n") and not body.endswith(b"\n\n")
    # Compact: no whitespace after separators.
    assert b", " not in body and b": " not in body
    parsed = json.loads(body.decode("utf-8"))
    assert len(parsed) == 1
    assert parsed[0]["from_status"] == "active"
    assert parsed[0]["to_status"] == "suspended"


def test_rejected_transitions_leave_no_history(client):
    machine = create_machine(client)

    # Same-status transition (409), invalid body (422), missing machine (404).
    assert set_status(client, machine["id"], "active").status_code == 409
    assert (
        client.post(
            f"/machines/{machine['id']}/status", json={"status": "deleted"}
        ).status_code
        == 422
    )
    assert (
        set_status(client, "00000000-0000-0000-0000-000000000000", "suspended")
        .status_code
        == 404
    )

    assert get_history(client, machine["id"]).json() == []


def test_history_missing_machine_returns_404(client):
    response = get_history(client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_history_rejects_any_query_parameter_before_machine_lookup(client):
    machine = create_machine(client)

    for path in (
        f"/machines/{machine['id']}/status-history?from_status=active",
        f"/machines/{machine['id']}/status-history?limit=1",
        "/machines/00000000-0000-0000-0000-000000000000/status-history?x=1",
    ):
        response = client.get(path)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}


def test_history_only_accepts_get(client):
    machine = create_machine(client)
    set_status(client, machine["id"], "suspended")

    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(f"/machines/{machine['id']}/status-history")
        assert response.status_code == 405

    # The failed attempts neither removed nor added records.
    assert len(get_history(client, machine["id"]).json()) == 1


def test_history_is_isolated_per_machine(client):
    first = create_machine(client, "machine-1")
    second = create_machine(client, "machine-2")

    set_status(client, first["id"], "suspended")

    assert get_history(client, second["id"]).json() == []
    records = get_history(client, first["id"]).json()
    assert len(records) == 1
    assert records[0]["machine_id"] == first["id"]


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


def test_concurrent_same_target_leaves_exactly_one_history_record(client):
    machine = create_machine(client)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: set_status(client, machine["id"], "suspended"), range(4))
        )

    assert sum(r.status_code == 200 for r in responses) == 1

    records = get_history(client, machine["id"]).json()
    assert len(records) == 1
    assert records[0]["from_status"] == "active"
    assert records[0]["to_status"] == "suspended"


def test_status_change_response_is_unchanged(client):
    machine = create_machine(client)

    response = set_status(client, machine["id"], "suspended")

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "id",
        "external_id",
        "display_name",
        "public_key",
        "status",
        "version",
        "created_at",
        "updated_at",
    }
