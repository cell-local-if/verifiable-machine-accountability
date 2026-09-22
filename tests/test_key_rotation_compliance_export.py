import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app


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


def rotate(client, machine_id, public_key, expected_version):
    response = client.post(
        f"/machines/{machine_id}/rotate-key",
        json={"public_key": public_key, "expected_version": expected_version},
    )
    assert response.status_code == 200
    return response.json()


def rotations(client, machine_id):
    return client.get(f"/machines/{machine_id}/key-rotation-events")


def export_url(machine_id, from_created_at, to_created_at):
    return (
        f"/machines/{machine_id}/key-rotation-events/compliance-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"

# Wide window for rotations minted by the API at wall-clock "now".
FROM_WIDE = "2000-01-01T00:00:00Z"
TO_WIDE = "2100-01-01T00:00:00Z"


def insert_rotation_row(
    client,
    machine_id,
    rotation_id,
    created_at,
    *,
    old_public_key="key-1",
    new_public_key="key-2",
    version=2,
    previous_rotation_id=None,
    content_hash="a" * 64,
    chain_hash="b" * 64,
):
    """Insert a rotation row directly with a fixed id and timestamp."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO key_rotation_events "
                "(id, machine_id, old_public_key, new_public_key, version, "
                "created_at, previous_rotation_id, content_hash, chain_hash) "
                "VALUES (:id, :machine_id, :old_public_key, :new_public_key, "
                ":version, :created_at, :previous_rotation_id, :content_hash, "
                ":chain_hash)"
            ),
            {
                "id": rotation_id,
                "machine_id": machine_id,
                "old_public_key": old_public_key,
                "new_public_key": new_public_key,
                "version": version,
                "created_at": created_at,
                "previous_rotation_id": previous_rotation_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        )


# --------------------------------------------------------------------------- #
# Parameter validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?from_created_at=2026-03-01T00:00:00Z",
        "?to_created_at=2026-03-01T00:00:05Z",
    ],
)
def test_missing_params_return_422(client, query):
    machine_id = create_machine(client)

    response = client.get(
        f"/machines/{machine_id}/key-rotation-events/compliance-export{query}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form instead of Z
        "2026-03-01T00:00:00z",           # lowercase suffix
        "2026-03-01 00:00:00Z",           # space separator
        "2026-03-01T00:00:00.Z",          # dot without fraction digits
        "not-a-time",
        "2026-13-01T00:00:00Z",           # invalid month
        "2026-02-30T00:00:00Z",           # invalid calendar day
        "2026-03-01T24:00:00Z",           # invalid hour
        "2026-03-01T00:60:00Z",           # invalid minute
        "2026-03-01T00:00:60Z",           # invalid second
        " 2026-03-01T00:00:00Z",          # surrounding whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
    ],
)
def test_invalid_from_created_at_returns_422(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, value, T4))

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",
        "2026-03-01T00:00:00+00:00",
        "garbage",
        "2026-03-01T00:00:00.123",
        " 2026-03-01T00:00:00Z",
    ],
)
def test_invalid_to_created_at_returns_422(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, value))

    assert response.status_code == 422


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00Z",
        "2026-03-01T00:00:00.5Z",
        "2026-03-01T00:00:00.123456789Z",
    ],
)
def test_fractional_second_bounds_are_accepted(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, value, TO_WIDE))

    assert response.status_code == 200


def test_inverted_range_returns_422(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T4, T0))

    assert response.status_code == 422


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    record = rotate(client, machine_id, "key-2", 1)
    # The rotation record's created_at equals the machine's updated_at.
    stamp = record["updated_at"]

    response = client.get(export_url(machine_id, stamp, stamp))

    listed = rotations(client, machine_id).json()
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["rotations"]] == [listed[0]["id"]]


def test_invalid_params_take_precedence_over_missing_machine(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(export_url(missing_machine, "2026-13-01T00:00:00Z", T4))
    assert response.status_code == 422

    response = client.get(
        f"/machines/{missing_machine}/key-rotation-events/compliance-export"
    )
    assert response.status_code == 422

    response = client.get(export_url(missing_machine, T4, T0))
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Machine existence
# --------------------------------------------------------------------------- #


def test_missing_machine_returns_404(client):
    missing_machine = "00000000-0000-0000-0000-000000000000"

    response = client.get(export_url(missing_machine, T0, T4))

    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


# --------------------------------------------------------------------------- #
# Response shape and windowing
# --------------------------------------------------------------------------- #


def test_export_response_shape_and_echoes_params(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "machine_id",
        "from_created_at",
        "to_created_at",
        "rotations",
    }
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == FROM_WIDE
    assert body["to_created_at"] == TO_WIDE
    assert len(body["rotations"]) == 1


def test_empty_window_returns_empty_array(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T0)
    insert_rotation_row(client, machine_id, rid(2), T4)

    response = client.get(
        export_url(machine_id, "2026-03-01T00:00:05Z", "2026-03-01T00:00:09Z")
    )

    assert response.status_code == 200
    assert response.json()["rotations"] == []


def test_machine_with_no_rotations_returns_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["rotations"] == []


def test_window_is_closed_on_both_ends(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T1)
    insert_rotation_row(client, machine_id, rid(2), T2)
    insert_rotation_row(client, machine_id, rid(3), T3)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["rotations"]] == [
        rid(1),
        rid(2),
        rid(3),
    ]


def test_window_excludes_rotations_outside_bounds(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(10), T0)
    insert_rotation_row(client, machine_id, rid(11), T1)
    insert_rotation_row(client, machine_id, rid(12), T2)
    insert_rotation_row(client, machine_id, rid(13), T3)
    insert_rotation_row(client, machine_id, rid(14), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["rotations"]] == [
        rid(11),
        rid(12),
        rid(13),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp sorts *after* "...:00Z" lexicographically only
    # by accident; the implementation must compare parsed instants so the
    # record falls inside [T0, T1].
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(11), "2026-03-01T00:00:00.500000Z")
    insert_rotation_row(client, machine_id, rid(12), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["rotations"]] == [rid(11), rid(12)]


def test_rotations_ordered_by_created_at_then_id(client):
    machine_id = create_machine(client)
    # Insert out of order; two rows share T2 and must sort by id.
    insert_rotation_row(client, machine_id, rid(30), T3)
    insert_rotation_row(client, machine_id, rid(21), T2)
    insert_rotation_row(client, machine_id, rid(20), T2)
    insert_rotation_row(client, machine_id, rid(10), T1)

    response = client.get(export_url(machine_id, T0, T4))

    assert [r["id"] for r in response.json()["rotations"]] == [
        rid(10),
        rid(20),
        rid(21),
        rid(30),
    ]


def test_same_second_exact_second_sorts_before_fractional_seconds(client):
    # As text, "...:00.5Z" sorts *before* "...:00Z" ('.' < 'Z'), so a naive
    # lexicographic order inverts the true order within one second. The export
    # must order by the actual UTC instant: the exact-second record first, then
    # fractional records earliest fraction first.
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(20), "2026-03-01T00:00:00.900000Z")
    insert_rotation_row(client, machine_id, rid(10), "2026-03-01T00:00:00.500000Z")
    insert_rotation_row(client, machine_id, rid(1), "2026-03-01T00:00:00Z")

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["rotations"]] == [
        rid(1),
        rid(10),
        rid(20),
    ]


def test_equal_instant_tie_breaks_by_id_with_fractional_stamps(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(21), "2026-03-01T00:00:00.250000Z")
    insert_rotation_row(client, machine_id, rid(20), "2026-03-01T00:00:00.250000Z")

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["rotations"]] == [rid(20), rid(21)]


def test_export_items_have_exactly_the_list_endpoint_fields(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    listed = rotations(client, machine_id).json()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["rotations"]
    assert exported == listed
    expected_keys = {
        "id",
        "machine_id",
        "old_public_key",
        "new_public_key",
        "version",
        "created_at",
        "previous_rotation_id",
        "content_hash",
        "chain_hash",
    }
    assert set(exported[0].keys()) == expected_keys


# --------------------------------------------------------------------------- #
# Damaged machine or chain data is exported exactly as stored
# --------------------------------------------------------------------------- #


def test_damaged_chain_fields_are_exported_unmodified(client, tmp_path):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    record = rotations(client, machine_id).json()[0]

    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE key_rotation_events "
        "SET previous_rotation_id = NULL, content_hash = ?, chain_hash = ? "
        "WHERE id = ?",
        ("0" * 64, "f" * 64, record["id"]),
    )
    connection.commit()
    connection.close()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    (exported,) = response.json()["rotations"]
    assert exported["previous_rotation_id"] is None
    assert exported["content_hash"] == "0" * 64
    assert exported["chain_hash"] == "f" * 64
    assert exported["id"] == record["id"]


def test_damaged_machine_public_key_does_not_change_export(client, tmp_path):
    machine_id = create_machine(client, public_key="key-1")
    rotate(client, machine_id, "key-2", 1)
    record = rotations(client, machine_id).json()[0]

    # Corrupt the machine's *current* public key; the stored rotation record
    # must be exported verbatim regardless.
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.execute(
        "UPDATE machines SET public_key = 'forged' WHERE id = ?",
        (machine_id,),
    )
    connection.commit()
    connection.close()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    (exported,) = response.json()["rotations"]
    assert exported["old_public_key"] == "key-1"
    assert exported["new_public_key"] == "key-2"
    assert exported["content_hash"] == record["content_hash"]
    assert exported["chain_hash"] == record["chain_hash"]


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_rotations(client):
    machine_one = create_machine(client, external_id="machine-1", public_key="key-a1")
    machine_two = create_machine(client, external_id="machine-2", public_key="key-b1")
    rotate(client, machine_one, "key-a2", 1)
    rotate(client, machine_two, "key-b2", 1)
    own = rotations(client, machine_one).json()

    response = client.get(export_url(machine_one, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    exported = response.json()["rotations"]
    assert [r["id"] for r in exported] == [r["id"] for r in own]
    assert all(r["machine_id"] == machine_one for r in exported)


# --------------------------------------------------------------------------- #
# Read-only, determinism, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_deterministic(client):
    machine_id = create_machine(client)
    other_machine = create_machine(client, external_id="machine-2")
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    rotate(client, other_machine, "key-x2", 1)

    rotations_url = f"/machines/{machine_id}/key-rotation-events"
    integrity_url = f"/machines/{machine_id}/key-rotation-events/integrity"
    machine_url = f"/machines/{machine_id}"
    before_rotations = client.get(rotations_url).json()
    before_integrity = client.get(integrity_url).json()
    before_machine = client.get(machine_url).json()

    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    assert first == second
    assert len(first["rotations"]) == 2

    assert client.get(rotations_url).json() == before_rotations
    assert client.get(integrity_url).json() == before_integrity
    assert client.get(machine_url).json() == before_machine
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        rotate(first, machine_id, "key-2", 1)
        rotate(first, machine_id, "key-3", 2)
        expected = first.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json() == expected
    assert len(response.json()["rotations"]) == 2
