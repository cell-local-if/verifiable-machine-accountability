"""Smoke test for the machine accountability changes endpoint."""
import json

import pytest
from fastapi.testclient import TestClient

from accountability.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def changes_url(machine_id, query="limit=100"):
    return (
        f"/machines/{machine_id}/accountability/compliance-export/changes"
        f"?{query}"
    )


def make_machine(client, external_id="m-1"):
    r = client.post(
        "/machines",
        json={
            "external_id": external_id,
            "display_name": "M",
            "public_key": "k",
        },
    )
    return r.json()["id"]


def allow_read(client, machine_id):
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={"action_type": "read", "resource_pattern": "res/*", "enabled": True},
    )
    client.post(
        "/policy-rules",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "effect": "allow",
            "priority": 0,
        },
    )


def test_full_flow(client):
    machine_id = make_machine(client)
    allow_read(client, machine_id)
    ev = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/1"},
    ).json()
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{ev['id']}/evidence",
        json={"evidence_type": "log", "content_hash": "ab" * 32},
    )
    inc = client.post(
        f"/machines/{machine_id}/authorization-decision-events/{ev['id']}/incidents",
        json={"incident_type": "fault", "summary": "boom"},
    ).json()
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{ev['id']}"
        f"/incidents/{inc['id']}/status",
        json={"status": "acknowledged"},
    )
    client.post(
        f"/machines/{machine_id}/authorization-decision-events/{ev['id']}"
        f"/incidents/{inc['id']}/responsibility-assignments",
        json={"party": "team-a", "role": "owner"},
    )

    r = client.get(changes_url(machine_id))
    assert r.status_code == 200
    assert r.content.endswith(b"\n") and not r.content.endswith(b"\n\n")
    body = json.loads(r.content)
    assert body["machine_id"] == machine_id
    assert body["limit"] == 100
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    groups = [item["group"] for item in body["records"]]
    assert sorted(groups) == [
        "events",
        "evidence",
        "incidents",
        "responsibility_assignments",
        "status_history",
    ]
    for item in body["records"]:
        assert set(item) == {"group", "record"}

    # Paginate one at a time and confirm stability + exclusive cursor.
    seen = []
    cursor = None
    while True:
        q = "limit=1" + (f"&cursor={cursor}" if cursor else "")
        page = json.loads(client.get(changes_url(machine_id, q)).content)
        seen.extend((i["group"], i["record"]["id"]) for i in page["records"])
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor.count("|") >= 2
    assert len(seen) == 5
    assert len(set(seen)) == 5

    # Repeat same cursor -> byte-identical page.
    p1 = client.get(changes_url(machine_id, "limit=2")).content
    p2 = client.get(changes_url(machine_id, "limit=2")).content
    assert p1 == p2
    cur = json.loads(p1)["next_cursor"]
    n1 = client.get(changes_url(machine_id, f"limit=2&cursor={cur}")).content
    n2 = client.get(changes_url(machine_id, f"limit=2&cursor={cur}")).content
    assert n1 == n2


def test_validation(client):
    machine_id = make_machine(client)
    # bad limit
    for q in ["", "limit=0", "limit=101", "limit=1.5", "limit=true",
              "limit=1&limit=2", "limit=abc"]:
        r = client.get(changes_url(machine_id, q))
        assert r.status_code == 422, q
        assert r.json()["error"]["code"] == "bad_limit", q
    # invalid cursor
    for q in ["limit=1&cursor=", "limit=1&cursor=x", "limit=1&cursor=a|b",
              "limit=1&cursor=a|b|c|d|", "limit=1&cursor=a||c",
              "limit=1&cursor=a|b|c&cursor=a|b|c"]:
        r = client.get(changes_url(machine_id, q))
        assert r.status_code == 422, q
        assert r.json()["error"]["code"] == "invalid_cursor", q
    # invalid query
    r = client.get(changes_url(machine_id, "limit=1&foo=bar"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_query"
    r = client.request(
        "GET",
        changes_url(machine_id, "limit=1"),
        content=b"{}",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_query"
    # unlocatable cursor
    r = client.get(changes_url(machine_id, "limit=1&cursor=zzz|events|nope"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_cursor"
    # missing machine
    r = client.get(changes_url("no-such-machine"))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    # non-GET
    r = client.post(changes_url(machine_id))
    assert r.status_code == 405


def test_machine_isolation(client):
    m1 = make_machine(client, "m-1")
    m2 = make_machine(client, "m-2")
    allow_read(client, m1)
    allow_read(client, m2)
    client.post(
        f"/machines/{m1}/authorization-decision-events",
        json={"action_type": "read", "resource": "res/1"},
    )
    body = json.loads(client.get(changes_url(m2)).content)
    assert body["records"] == []
    assert body["has_more"] is False
