import re

import pytest
from fastapi.testclient import TestClient

from accountability.app import app

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
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
    return response.json()["id"]


def record_event(client, machine_id, action_type="read", resource="res/x"):
    response = client.post(
        f"/machines/{machine_id}/authorization-decision-events",
        json={"action_type": action_type, "resource": resource},
    )
    assert response.status_code == 201
    return response.json()


def links_url(machine_id, cause_event_id):
    return (
        f"/machines/{machine_id}/authorization-decision-events/"
        f"{cause_event_id}/causal-links"
    )


def create_link(client, machine_id, cause_event_id, effect_event_id):
    return client.post(
        links_url(machine_id, cause_event_id),
        json={"effect_event_id": effect_event_id},
    )


def test_create_link_returns_201_with_full_record(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")

    response = create_link(client, machine_id, cause["id"], effect["id"])

    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["cause_event_id"] == cause["id"]
    assert body["effect_event_id"] == effect["id"]
    assert RFC3339_Z_RE.match(body["created_at"])


def test_create_link_strips_whitespace_in_effect_id(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")

    response = create_link(client, machine_id, cause["id"], f"  {effect['id']}\t")

    assert response.status_code == 201
    assert response.json()["effect_event_id"] == effect["id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"effect_event_id": "   "},
        {"effect_event_id": ""},
        {},
        {"effect_event_id": None},
        {"effect_event_id": 1},
    ],
)
def test_create_link_invalid_body_returns_422(client, payload):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id)

    response = client.post(links_url(machine_id, cause["id"]), json=payload)

    assert response.status_code == 422


def test_create_link_invalid_body_checked_before_missing_events(client):
    response = client.post(
        links_url(
            "00000000-0000-0000-0000-000000000000",
            "00000000-0000-0000-0000-000000000000",
        ),
        json={"effect_event_id": "  "},
    )

    assert response.status_code == 422


def test_create_link_missing_machine_returns_404(client):
    response = create_link(
        client,
        "00000000-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_link_missing_cause_event_returns_404(client):
    machine_id = create_machine(client)
    effect = record_event(client, machine_id)

    response = create_link(
        client,
        machine_id,
        "00000000-0000-0000-0000-000000000000",
        effect["id"],
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_link_missing_effect_event_returns_404(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id)

    response = create_link(
        client,
        machine_id,
        cause["id"],
        "00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_link_events_from_other_machine_return_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    cause = record_event(client, machine_one)
    foreign = record_event(client, machine_two)

    response = create_link(client, machine_one, cause["id"], foreign["id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}

    response = create_link(client, machine_one, foreign["id"], cause["id"])
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_link_self_returns_422(client):
    machine_id = create_machine(client)
    event = record_event(client, machine_id)

    response = create_link(client, machine_id, event["id"], event["id"])

    assert response.status_code == 422
    assert response.json() == {"error": {"code": "self_causal_link"}}


def test_create_link_duplicate_returns_409(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")

    assert create_link(client, machine_id, cause["id"], effect["id"]).status_code == 201
    response = create_link(client, machine_id, cause["id"], effect["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_causal_link"}}


def test_create_link_reverse_direction_is_allowed(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")

    assert create_link(client, machine_id, event_a["id"], event_b["id"]).status_code == 201
    response = create_link(client, machine_id, event_b["id"], event_a["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "causal_cycle"}}


def test_create_link_indirect_cycle_returns_409(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")

    assert create_link(client, machine_id, event_a["id"], event_b["id"]).status_code == 201
    assert create_link(client, machine_id, event_b["id"], event_c["id"]).status_code == 201
    response = create_link(client, machine_id, event_c["id"], event_a["id"])

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "causal_cycle"}}


def test_failed_link_creation_writes_nothing(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")

    create_link(client, machine_id, event_a["id"], event_b["id"])
    create_link(client, machine_id, event_a["id"], event_b["id"])  # duplicate
    create_link(client, machine_id, event_b["id"], event_a["id"])  # cycle
    create_link(client, machine_id, event_a["id"], event_a["id"])  # self

    response = client.get(links_url(machine_id, event_a["id"]))
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_links_empty_returns_empty_list(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id)

    response = client.get(links_url(machine_id, cause["id"]))

    assert response.status_code == 200
    assert response.json() == []


def test_list_links_returns_links_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effects = [
        record_event(client, machine_id, resource=f"res/{suffix}")
        for suffix in ["b", "c", "d"]
    ]
    created = [
        create_link(client, machine_id, cause["id"], e["id"]).json() for e in effects
    ]

    response = client.get(links_url(machine_id, cause["id"]))

    assert response.status_code == 200
    links = response.json()
    expected = sorted(created, key=lambda link: (link["created_at"], link["id"]))
    assert [link["id"] for link in links] == [link["id"] for link in expected]
    for link in links:
        assert link["machine_id"] == machine_id
        assert link["cause_event_id"] == cause["id"]
        assert set(link.keys()) == {
            "id",
            "machine_id",
            "cause_event_id",
            "effect_event_id",
            "created_at",
        }


def test_list_links_only_returns_links_where_event_is_cause(client):
    machine_id = create_machine(client)
    event_a = record_event(client, machine_id, resource="res/a")
    event_b = record_event(client, machine_id, resource="res/b")
    event_c = record_event(client, machine_id, resource="res/c")

    create_link(client, machine_id, event_a["id"], event_b["id"])
    create_link(client, machine_id, event_b["id"], event_c["id"])

    response = client.get(links_url(machine_id, event_a["id"]))
    assert [link["effect_event_id"] for link in response.json()] == [event_b["id"]]

    response = client.get(links_url(machine_id, event_c["id"]))
    assert response.json() == []


def test_list_links_missing_cause_event_returns_404(client):
    machine_id = create_machine(client)

    response = client.get(
        links_url(machine_id, "00000000-0000-0000-0000-000000000000")
    )

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_list_links_cause_from_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    foreign = record_event(client, machine_two)

    response = client.get(links_url(machine_one, foreign["id"]))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_links_persist_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        cause = record_event(first, machine_id, resource="res/a")
        effect = record_event(first, machine_id, resource="res/b")
        link = create_link(first, machine_id, cause["id"], effect["id"]).json()

    with TestClient(app) as second:
        response = second.get(links_url(machine_id, cause["id"]))

    assert response.status_code == 200
    assert response.json() == [link]


def test_link_creation_does_not_modify_events_or_chain(client):
    machine_id = create_machine(client)
    cause = record_event(client, machine_id, resource="res/a")
    effect = record_event(client, machine_id, resource="res/b")
    before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    integrity_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    create_link(client, machine_id, cause["id"], effect["id"])

    after = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    integrity_after = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()
    assert after == before
    assert integrity_after == integrity_before
    assert integrity_after["valid"] is True
