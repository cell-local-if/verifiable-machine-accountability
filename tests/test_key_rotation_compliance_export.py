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

# Wide window for records minted by the API at wall-clock "now".
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
    """Insert a rotation row directly with a fixed id and timestamp.

    Chain columns are supplied non-NULL so the startup backfill (which only
    completes rows missing hashes) leaves the stored values untouched across
    restarts; callers may pass deliberately corrupt values.
    """
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
        " 2026-03-01T00:00:00Z",          # leading whitespace
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
        " 2026-03-01T00:00:00Z ",
    ],
)
def test_invalid_to_created_at_returns_422(client, value):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T0, value))

    assert response.status_code == 422


def test_inverted_range_returns_422(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, T4, T0))

    assert response.status_code == 422


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)

    # Equal valid bounds are accepted (empty window), not rejected as inverted.
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert response.json()["rotations"] == []

    insert_rotation_row(client, machine_id, rid(1), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["rotations"]] == [rid(1)]


def test_fractional_second_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), "2026-03-01T00:00:00.500000Z")

    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:00.250Z",
            "2026-03-01T00:00:00.750000000Z",
        )
    )

    assert response.status_code == 200
    assert [r["id"] for r in response.json()["rotations"]] == [rid(1)]


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
    insert_rotation_row(client, machine_id, rid(0), T0)
    insert_rotation_row(client, machine_id, rid(1), T1)
    insert_rotation_row(client, machine_id, rid(2), T2)
    insert_rotation_row(client, machine_id, rid(3), T3)
    insert_rotation_row(client, machine_id, rid(4), T4)

    response = client.get(export_url(machine_id, T1, T3))

    assert [r["id"] for r in response.json()["rotations"]] == [
        rid(1),
        rid(2),
        rid(3),
    ]


def test_fractional_second_timestamp_is_filtered_as_an_instant(client):
    # A stored fractional stamp sorts *after* "...:00Z" lexicographically only
    # by accident; the implementation must compare parsed instants so the
    # record falls inside [T0, T1].
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), "2026-03-01T00:00:00.500000Z")
    insert_rotation_row(client, machine_id, rid(2), T1)

    response = client.get(export_url(machine_id, T0, T1))

    assert [r["id"] for r in response.json()["rotations"]] == [rid(1), rid(2)]


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


def test_export_items_match_list_endpoint_items(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)
    listed = client.get(f"/machines/{machine_id}/key-rotation-events").json()

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    rotations = response.json()["rotations"]
    assert rotations == listed
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
    assert set(rotations[0].keys()) == expected_keys


def test_new_machine_with_no_rotations_exports_empty_array(client):
    machine_id = create_machine(client)

    response = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE))

    assert response.status_code == 200
    assert response.json()["rotations"] == []


# --------------------------------------------------------------------------- #
# Machine boundary
# --------------------------------------------------------------------------- #


def test_export_never_contains_other_machine_rotations(client):
    machine_one = create_machine(client, external_id="machine-1", public_key="key-a1")
    machine_two = create_machine(client, external_id="machine-2", public_key="key-b1")
    insert_rotation_row(
        client,
        machine_one,
        rid(100),
        T1,
        old_public_key="key-a1",
        new_public_key="key-a2",
    )
    insert_rotation_row(
        client,
        machine_two,
        rid(200),
        T1,
        old_public_key="key-b1",
        new_public_key="key-b2",
    )
    # A rotation row owned by another machine is impossible to forge through
    # the API, but the filter is nevertheless on the path machine id: verify
    # both exports stay isolated.
    response_one = client.get(export_url(machine_one, T0, T4))
    response_two = client.get(export_url(machine_two, T0, T4))

    assert [r["id"] for r in response_one.json()["rotations"]] == [rid(100)]
    assert all(r["machine_id"] == machine_one for r in response_one.json()["rotations"])
    assert [r["id"] for r in response_two.json()["rotations"]] == [rid(200)]
    assert all(r["machine_id"] == machine_two for r in response_two.json()["rotations"])


# --------------------------------------------------------------------------- #
# Damaged current key / damaged chain fields are exported as stored
# --------------------------------------------------------------------------- #


def test_damaged_chain_fields_exported_exactly_as_stored(client):
    machine_id = create_machine(client)
    # Deliberately corrupt: dangling previous link and bogus hashes. The
    # integrity endpoint must flag it, but the export must return the stored
    # bytes verbatim.
    insert_rotation_row(
        client,
        machine_id,
        rid(1),
        T1,
        previous_rotation_id="99999999-9999-9999-9999-999999999999",
        content_hash="0" * 64,
        chain_hash="1" * 64,
    )
    insert_rotation_row(
        client,
        machine_id,
        rid(2),
        T2,
        previous_rotation_id=rid(1),
        content_hash="c" * 64,
        chain_hash="d" * 64,
    )

    integrity = client.get(
        f"/machines/{machine_id}/key-rotation-events/integrity"
    ).json()
    assert integrity["valid"] is False

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    rotations = response.json()["rotations"]
    assert rotations[0]["previous_rotation_id"] == "99999999-9999-9999-9999-999999999999"
    assert rotations[0]["content_hash"] == "0" * 64
    assert rotations[0]["chain_hash"] == "1" * 64
    # Corrupt-but-present chain fields on the second row are returned as
    # stored rather than recomputed on read.
    assert rotations[1]["content_hash"] == "c" * 64
    assert rotations[1]["chain_hash"] == "d" * 64
    # The export items must match the list endpoint items byte-for-byte.
    assert rotations == client.get(
        f"/machines/{machine_id}/key-rotation-events"
    ).json()


def test_damaged_current_machine_public_key_does_not_affect_export(client):
    machine_id = create_machine(client)
    insert_rotation_row(
        client,
        machine_id,
        rid(1),
        T1,
        old_public_key="key-1",
        new_public_key="key-2",
    )
    # Corrupt the machine's *current* public key after the rotation was stored.
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE machines SET public_key = :key WHERE id = :id"),
            {"key": "", "id": machine_id},
        )

    response = client.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    rotations = response.json()["rotations"]
    assert len(rotations) == 1
    assert rotations[0]["old_public_key"] == "key-1"
    assert rotations[0]["new_public_key"] == "key-2"
    assert rotations[0]["machine_id"] == machine_id


# --------------------------------------------------------------------------- #
# Read-only, determinism, persistence
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_deterministic(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    rotate(client, machine_id, "key-3", 2)

    events_url = f"/machines/{machine_id}/key-rotation-events"
    integrity_url = f"{events_url}/integrity"
    machine_url = f"/machines/{machine_id}"
    before_events = client.get(events_url).json()
    before_integrity = client.get(integrity_url).json()
    before_machine = client.get(machine_url).json()

    first = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    second = client.get(export_url(machine_id, FROM_WIDE, TO_WIDE)).json()
    assert first == second

    assert client.get(events_url).json() == before_events
    assert client.get(integrity_url).json() == before_integrity
    assert client.get(machine_url).json() == before_machine
    assert before_integrity["valid"] is True


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        insert_rotation_row(
            first,
            machine_id,
            rid(1),
            T1,
            previous_rotation_id="99999999-9999-9999-9999-999999999999",
            content_hash="0" * 64,
            chain_hash="1" * 64,
        )
        expected = first.get(export_url(machine_id, T0, T4)).json()

    with TestClient(app) as second:
        response = second.get(export_url(machine_id, T0, T4))

    assert response.status_code == 200
    assert response.json() == expected
    rotation = response.json()["rotations"][0]
    # Non-NULL corrupt chain values must survive the restart backfill verbatim.
    assert rotation["previous_rotation_id"] == "99999999-9999-9999-9999-999999999999"
    assert rotation["content_hash"] == "0" * 64
    assert rotation["chain_hash"] == "1" * 64
