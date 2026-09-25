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


def create_evidence(client, machine_id, event_id, evidence_type="log", content_hash=HASH_A):
    return client.post(
        evidence_url(machine_id, event_id),
        json={"evidence_type": evidence_type, "content_hash": content_hash},
    )


def test_create_evidence_returns_201_with_full_record(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_evidence(client, machine_id, event_id)

    assert response.status_code == 201
    body = response.json()
    assert UUID_RE.match(body["id"])
    assert body["machine_id"] == machine_id
    assert body["event_id"] == event_id
    assert body["evidence_type"] == "log"
    assert body["content_hash"] == HASH_A
    assert RFC3339_Z_RE.match(body["created_at"])
    # The first record of a machine roots the chain: null predecessor and a
    # chain digest alongside the fingerprint.
    assert body["previous_evidence_id"] is None
    assert re.match(r"^[0-9a-f]{64}$", body["chain_hash"])
    assert set(body.keys()) == {
        "id",
        "machine_id",
        "event_id",
        "evidence_type",
        "content_hash",
        "created_at",
        "previous_evidence_id",
        "chain_hash",
    }


def test_create_evidence_strips_evidence_type_whitespace(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_evidence(client, machine_id, event_id, evidence_type="  log\t")

    assert response.status_code == 201
    assert response.json()["evidence_type"] == "log"


def test_create_evidence_does_not_case_fold_content_hash(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_evidence(client, machine_id, event_id, content_hash=HASH_C)

    assert response.status_code == 201
    assert response.json()["content_hash"] == HASH_C


@pytest.mark.parametrize(
    "payload",
    [
        {"content_hash": HASH_A},
        {"evidence_type": "log"},
        {},
        {"evidence_type": "   ", "content_hash": HASH_A},
        {"evidence_type": "", "content_hash": HASH_A},
        {"evidence_type": None, "content_hash": HASH_A},
        {"evidence_type": 1, "content_hash": HASH_A},
        {"evidence_type": "log", "content_hash": None},
        {"evidence_type": "log", "content_hash": 1},
        {"evidence_type": "log", "content_hash": "a" * 63},
        {"evidence_type": "log", "content_hash": "a" * 65},
        {"evidence_type": "log", "content_hash": "A" * 64},
        {"evidence_type": "log", "content_hash": "g" * 64},
        {"evidence_type": "log", "content_hash": "a" * 63 + "Z"},
        {"evidence_type": "log", "content_hash": " " + "a" * 63},
    ],
)
def test_create_evidence_rejects_invalid_payload_with_422(client, payload):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.post(evidence_url(machine_id, event_id), json=payload)

    assert response.status_code == 422


def test_create_evidence_invalid_payload_is_422_before_path_lookup(client):
    # Body validation happens before the handler; malformed input is 422
    # regardless of whether the machine or event exists.
    response = client.post(
        evidence_url(MISSING_ID, MISSING_ID),
        json={"evidence_type": "  ", "content_hash": "nope"},
    )

    assert response.status_code == 422


def test_create_evidence_missing_machine_returns_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = create_evidence(client, MISSING_ID, event_id)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_evidence_missing_event_returns_404(client):
    machine_id = create_machine(client)

    response = create_evidence(client, machine_id, MISSING_ID)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_create_evidence_event_of_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_id = record_event(client, machine_one)

    response = create_evidence(client, machine_two, event_id)

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}
    # And nothing was written under the event's own machine either.
    assert client.get(evidence_url(machine_one, event_id)).json() == []


def test_create_evidence_duplicate_content_hash_returns_409_and_writes_nothing(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    first = create_evidence(client, machine_id, event_id).json()

    response = create_evidence(
        client, machine_id, event_id, evidence_type="other", content_hash=HASH_A
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_evidence"}}
    records = client.get(evidence_url(machine_id, event_id)).json()
    assert records == [first]


def test_same_content_hash_allowed_on_different_events(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")

    assert create_evidence(client, machine_id, event_one).status_code == 201
    assert create_evidence(client, machine_id, event_two).status_code == 201


def test_list_evidence_empty_returns_empty_list(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    response = client.get(evidence_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == []


def test_list_evidence_returns_records_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    created = [
        create_evidence(client, machine_id, event_id, content_hash=h).json()
        for h in (HASH_A, HASH_B, HASH_C)
    ]

    response = client.get(evidence_url(machine_id, event_id))

    assert response.status_code == 200
    records = response.json()
    expected = sorted(created, key=lambda r: (r["created_at"], r["id"]))
    assert [r["id"] for r in records] == [r["id"] for r in expected]
    ordering_key = [(r["created_at"], r["id"]) for r in records]
    assert ordering_key == sorted(ordering_key)
    for record in records:
        assert record["machine_id"] == machine_id
        assert record["event_id"] == event_id


def test_list_evidence_is_scoped_to_the_event(client):
    machine_id = create_machine(client)
    event_one = record_event(client, machine_id, resource="res/1")
    event_two = record_event(client, machine_id, resource="res/2")
    create_evidence(client, machine_id, event_one, content_hash=HASH_A)
    create_evidence(client, machine_id, event_two, content_hash=HASH_B)

    records = client.get(evidence_url(machine_id, event_one)).json()

    assert [r["content_hash"] for r in records] == [HASH_A]


def test_list_evidence_missing_machine_or_event_returns_404(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)

    for path in (
        evidence_url(MISSING_ID, event_id),
        evidence_url(machine_id, MISSING_ID),
    ):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}


def test_list_evidence_event_of_other_machine_returns_404(client):
    machine_one = create_machine(client, external_id="machine-1")
    machine_two = create_machine(client, external_id="machine-2")
    event_id = record_event(client, machine_one)
    create_evidence(client, machine_one, event_id)

    response = client.get(evidence_url(machine_two, event_id))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_evidence_does_not_modify_event_chain_or_links(client):
    machine_id = create_machine(client)
    event_id = record_event(client, machine_id)
    events_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events"
    ).json()
    integrity_before = client.get(
        f"/machines/{machine_id}/authorization-decision-events/integrity"
    ).json()

    create_evidence(client, machine_id, event_id, content_hash=HASH_A)
    create_evidence(client, machine_id, event_id, content_hash=HASH_B)
    client.get(evidence_url(machine_id, event_id))

    assert (
        client.get(f"/machines/{machine_id}/authorization-decision-events").json()
        == events_before
    )
    assert (
        client.get(
            f"/machines/{machine_id}/authorization-decision-events/integrity"
        ).json()
        == integrity_before
    )
    assert integrity_before["valid"] is True


def test_evidence_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        event_id = record_event(first, machine_id)
        record = create_evidence(first, machine_id, event_id).json()

    with TestClient(app) as second:
        response = second.get(evidence_url(machine_id, event_id))

    assert response.status_code == 200
    assert response.json() == [record]
