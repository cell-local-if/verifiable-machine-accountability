"""Tests for the read-only desensitized privacy export of key rotations.

Covers `GET /machines/{machine_id}/key-rotation-events/privacy-export`: the
strict query validation (``bad_time`` / ``invalid_query`` before any machine
or rotation read), ``404 not_found``, GET-only ``405`` routing, closed-UTC-
window filtering on each rotation's own ``created_at``, ordering by the
actual UTC instant then record id (exact-second before fractional-second),
the ``old_public_key_ref`` / ``new_public_key_ref`` SHA-256 desensitizing
digests (including ``null`` for non-string or blank values), the absence of
raw public keys, verbatim export of missing/misowned/duplicated/chain-damaged
records, machine isolation, the fixed field order with compact newline-
terminated JSON, strict read-only byte stability, and persistence across a
restart.
"""
import hashlib

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


WIDE = ("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")

T0 = "2026-03-01T00:00:00Z"
T1 = "2026-03-01T00:00:01Z"
T2 = "2026-03-01T00:00:02Z"
T3 = "2026-03-01T00:00:03Z"
T4 = "2026-03-01T00:00:04Z"
T5 = "2026-03-01T00:00:05Z"


def export_url(machine_id, from_created_at=WIDE[0], to_created_at=WIDE[1]):
    return (
        f"/machines/{machine_id}/key-rotation-events/privacy-export"
        f"?from_created_at={from_created_at}&to_created_at={to_created_at}"
    )


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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


HASH_A = "a" * 64
HASH_B = "b" * 64


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
    content_hash=HASH_A,
    chain_hash=HASH_B,
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


ROTATION_KEYS = [
    "id",
    "machine_id",
    "version",
    "created_at",
    "previous_rotation_id",
    "chain_hash",
    "old_public_key_ref",
    "new_public_key_ref",
]


def expected_ref(kind, machine_id, value):
    return hashlib.sha256(
        f"privacy:v1|{kind}{machine_id}{value}".encode("utf-8")
    ).hexdigest()


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
def test_missing_bounds_are_bad_time(client, query):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/key-rotation-events/privacy-export{query}"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-01T00:00:00",            # missing Z
        "2026-03-01T00:00:00+00:00",      # offset form
        "2026-03-01T00:00:00z",           # lowercase suffix
        " 2026-03-01T00:00:00Z",          # leading whitespace
        "2026-03-01T00:00:00Z ",          # trailing whitespace
        "2026-03-01 00:00:00Z",           # space separator
        "garbage",
        "",                               # blank
        "2026-13-01T00:00:00Z",           # bad month
        "2026-02-30T00:00:00Z",           # bad day
        "2026-03-01T24:00:00Z",           # bad hour
    ],
)
def test_malformed_bounds_are_bad_time(client, value):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, value, T5))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_fractional_seconds_are_accepted(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T2)
    response = client.get(
        export_url(
            machine_id,
            "2026-03-01T00:00:01.250Z",
            "2026-03-01T00:00:03.750000Z",
        )
    )
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["rotations"]] == [rid(1)]


def test_inverted_bounds_are_bad_time(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T5, T0))
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "bad_time"}}


def test_equal_bounds_are_accepted(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T2)
    response = client.get(export_url(machine_id, T2, T2))
    assert response.status_code == 200
    assert [r["id"] for r in response.json()["rotations"]] == [rid(1)]


def test_unknown_query_parameter_is_invalid_query(client):
    machine_id = create_machine(client)
    response = client.get(
        f"/machines/{machine_id}/key-rotation-events/privacy-export"
        f"?from_created_at={T0}&to_created_at={T5}&unexpected=1"
    )
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


def test_validation_runs_before_machine_lookup(client):
    missing = "00000000-0000-0000-0000-000000000000"
    base = f"/machines/{missing}/key-rotation-events/privacy-export"

    bad_time = client.get(f"{base}?from_created_at=nope&to_created_at={T5}")
    assert bad_time.status_code == 422
    assert bad_time.json() == {"error": {"code": "bad_time"}}

    unknown = client.get(f"{base}?from_created_at={T0}&to_created_at={T5}&x=1")
    assert unknown.status_code == 422
    assert unknown.json() == {"error": {"code": "invalid_query"}}


def test_missing_machine_returns_404_without_rotation_data(client):
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.get(export_url(missing, T0, T5))
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found"}}


def test_only_get_is_accepted(client):
    machine_id = create_machine(client)
    url = export_url(machine_id, T0, T5)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(url)
        assert response.status_code == 405


# --------------------------------------------------------------------------- #
# Envelope shape and desensitized fields
# --------------------------------------------------------------------------- #


def test_envelope_shape_with_empty_rotations(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == [
        "machine_id",
        "from_created_at",
        "to_created_at",
        "rotations",
    ]
    assert body["machine_id"] == machine_id
    assert body["from_created_at"] == WIDE[0]
    assert body["to_created_at"] == WIDE[1]
    assert body["rotations"] == []


def test_response_is_compact_json_ending_with_newline(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T1)
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    text_body = response.content.decode("utf-8")
    assert text_body.endswith("}\n")
    assert '": "' not in text_body  # compact separators
    assert response.text[:-1] == __import__("json").dumps(
        response.json(), ensure_ascii=False, separators=(",", ":")
    )


def test_rotation_exported_with_desensitized_fields(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    listed = client.get(f"/machines/{machine_id}/key-rotation-events").json()[0]

    body = client.get(export_url(machine_id)).json()
    exported = body["rotations"][0]
    assert list(exported.keys()) == ROTATION_KEYS
    assert exported["id"] == listed["id"]
    assert exported["machine_id"] == machine_id
    assert exported["version"] == 2
    assert exported["created_at"] == listed["created_at"]
    assert exported["previous_rotation_id"] is None
    assert exported["chain_hash"] == listed["chain_hash"]
    assert exported["old_public_key_ref"] == expected_ref(
        "old_public_key", machine_id, "key-1"
    )
    assert exported["new_public_key_ref"] == expected_ref(
        "new_public_key", machine_id, "key-2"
    )
    # The content hash is not part of the privacy view.
    assert "content_hash" not in exported


def test_digest_strips_surrounding_whitespace(client):
    machine_id = create_machine(client)
    insert_rotation_row(
        client, machine_id, rid(1), T1,
        old_public_key="  key-old\t", new_public_key="\n key-new  ",
    )
    exported = client.get(export_url(machine_id)).json()["rotations"][0]
    assert exported["old_public_key_ref"] == expected_ref(
        "old_public_key", machine_id, "key-old"
    )
    assert exported["new_public_key_ref"] == expected_ref(
        "new_public_key", machine_id, "key-new"
    )


@pytest.mark.parametrize("blank", ["", "   ", "\t\n "])
def test_blank_key_digest_is_null_but_record_kept(client, blank):
    machine_id = create_machine(client)
    insert_rotation_row(
        client, machine_id, rid(1), T1,
        old_public_key=blank, new_public_key="key-2",
    )
    insert_rotation_row(
        client, machine_id, rid(2), T2,
        old_public_key="key-2", new_public_key=blank,
    )
    rows = client.get(export_url(machine_id)).json()["rotations"]
    assert [r["id"] for r in rows] == [rid(1), rid(2)]
    assert rows[0]["old_public_key_ref"] is None
    assert rows[0]["new_public_key_ref"] == expected_ref(
        "new_public_key", machine_id, "key-2"
    )
    assert rows[1]["old_public_key_ref"] == expected_ref(
        "old_public_key", machine_id, "key-2"
    )
    assert rows[1]["new_public_key_ref"] is None


def test_non_string_key_digest_is_null_but_record_kept(client):
    machine_id = create_machine(client)
    # SQLite TEXT affinity would coerce a numeric literal to text, so store
    # BLOBs (which TEXT affinity leaves untouched) to get genuinely non-string
    # values. The privacy view must still return the record with null refs
    # rather than coercing or leaking either value.
    insert_rotation_row(
        client, machine_id, rid(1), T1,
        old_public_key=b"\xff\xfe binary-old", new_public_key=b"\x00\x01 bin",
    )
    rows = client.get(export_url(machine_id)).json()["rotations"]
    assert len(rows) == 1
    assert rows[0]["old_public_key_ref"] is None
    assert rows[0]["new_public_key_ref"] is None
    assert rows[0]["id"] == rid(1)


def test_raw_public_keys_never_appear_in_response(client):
    machine_id = create_machine(client)
    secret_old = "the-secret-old-key"
    secret_new = "the-secret-new-key"
    insert_rotation_row(
        client, machine_id, rid(1), T1,
        old_public_key=f"  {secret_old}  ", new_public_key=secret_new,
    )
    response = client.get(export_url(machine_id))
    assert response.status_code == 200
    assert secret_old not in response.text
    assert secret_new not in response.text


def test_digest_is_scoped_to_machine(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_rotation_row(
        client, machine_one, rid(1), T1,
        old_public_key="key-1", new_public_key="key-2",
    )
    insert_rotation_row(
        client, machine_two, rid(2), T1,
        old_public_key="key-1", new_public_key="key-2",
    )
    ref_one = client.get(export_url(machine_one)).json()["rotations"][0][
        "old_public_key_ref"
    ]
    ref_two = client.get(export_url(machine_two)).json()["rotations"][0][
        "old_public_key_ref"
    ]
    assert ref_one == expected_ref("old_public_key", machine_one, "key-1")
    assert ref_two == expected_ref("old_public_key", machine_two, "key-1")
    assert ref_one != ref_two


# --------------------------------------------------------------------------- #
# Windowing, ordering, machine isolation
# --------------------------------------------------------------------------- #


def test_window_is_closed_on_rotation_created_at(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(1), T0)
    insert_rotation_row(client, machine_id, rid(2), T2)
    insert_rotation_row(client, machine_id, rid(3), T4)

    body = client.get(export_url(machine_id, T2, T3)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(2)]

    # Equal bounds include the boundary rotation.
    body = client.get(export_url(machine_id, T4, T4)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(3)]

    # An empty window keeps the array rather than omitting it.
    body = client.get(export_url(machine_id, T3, T3)).json()
    assert body["rotations"] == []


def test_exact_second_sorts_before_fractional_same_second(client):
    machine_id = create_machine(client)
    fractional = "2026-03-01T00:00:00.500000Z"
    # Insert so lexicographic order would put the fractional record first.
    insert_rotation_row(client, machine_id, rid(2), fractional)
    insert_rotation_row(client, machine_id, rid(1), T0)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(1), rid(2)]


def test_same_instant_tie_breaks_by_record_id(client):
    machine_id = create_machine(client)
    insert_rotation_row(client, machine_id, rid(30), T2)
    insert_rotation_row(client, machine_id, rid(20), T2)

    body = client.get(export_url(machine_id, T0, T5)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(20), rid(30)]


def test_damaged_and_chain_broken_records_export_verbatim(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    rotate(client, machine_two, "key-2", 1)
    foreign = client.get(
        f"/machines/{machine_two}/key-rotation-events"
    ).json()[0]

    # A record whose previous pointer references another machine's rotation,
    # a dangling previous pointer, and deliberately corrupt/arbitrary chain
    # fields all export exactly as stored.
    insert_rotation_row(
        client, machine_one, rid(1), T1,
        old_public_key="key-x", new_public_key="key-y",
        version=9, previous_rotation_id=foreign["id"],
        content_hash="z" * 64, chain_hash="0" * 64,
    )
    insert_rotation_row(
        client, machine_one, rid(2), T2,
        previous_rotation_id=rid(99), chain_hash="1" * 64,
    )
    insert_rotation_row(client, machine_one, rid(3), T3)

    rows = client.get(export_url(machine_one, T0, T5)).json()["rotations"]
    assert [r["id"] for r in rows] == [rid(1), rid(2), rid(3)]
    assert all(r["machine_id"] == machine_one for r in rows)
    assert rows[0]["version"] == 9
    assert rows[0]["previous_rotation_id"] == foreign["id"]
    assert rows[0]["chain_hash"] == "0" * 64
    assert rows[1]["previous_rotation_id"] == rid(99)
    assert rows[1]["chain_hash"] == "1" * 64


def test_other_machine_rotations_are_never_exported(client):
    machine_one = create_machine(client, "machine-1")
    machine_two = create_machine(client, "machine-2")
    insert_rotation_row(client, machine_one, rid(1), T1)
    insert_rotation_row(client, machine_two, rid(2), T1)

    body = client.get(export_url(machine_one, T0, T5)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(1)]

    body = client.get(export_url(machine_two, T0, T5)).json()
    assert [r["id"] for r in body["rotations"]] == [rid(2)]


# --------------------------------------------------------------------------- #
# Read-only, byte-stable, persistent
# --------------------------------------------------------------------------- #


def test_export_is_read_only_and_byte_stable(client):
    machine_id = create_machine(client)
    rotate(client, machine_id, "key-2", 1)
    insert_rotation_row(
        client, machine_id, rid(99), T3, old_public_key="  key-9  "
    )

    def table_state():
        with client.app.state.engine.connect() as conn:
            return {
                name: list(conn.execute(text(f"SELECT * FROM {name}")))
                for name in ("key_rotation_events", "machines")
            }

    before = table_state()
    first = client.get(export_url(machine_id, T0, T5))
    middle = table_state()
    second = client.get(export_url(machine_id, T0, T5))
    after = table_state()

    assert first.status_code == 200
    assert first.content == second.content
    assert before == middle == after


def test_empty_database_needs_no_migration_and_exports_empty(client):
    machine_id = create_machine(client)
    response = client.get(export_url(machine_id, T0, T5))
    assert response.status_code == 200
    assert response.json()["rotations"] == []


def test_export_persists_across_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        machine_id = create_machine(first)
        rotate(first, machine_id, "key-2", 1)
        expected = first.get(export_url(machine_id)).content

    with TestClient(app) as second:
        response = second.get(export_url(machine_id))

    assert response.status_code == 200
    assert response.content == expected
    rows = response.json()["rotations"]
    assert len(rows) == 1
    assert rows[0]["old_public_key_ref"] == expected_ref(
        "old_public_key", machine_id, "key-1"
    )
    assert rows[0]["new_public_key_ref"] == expected_ref(
        "new_public_key", machine_id, "key-2"
    )
