"""Tests for the machine-level incremental privacy access changes query.

Covers `GET /machines/{machine_id}/privacy-accesses/changes`:

- strict query validation (``invalid_query`` / ``bad_limit`` /
  ``invalid_cursor``) completed before any machine or access read,
  ``404 not_found`` with no access data for a machine missing after
  validation, GET-only ``405``;
- stable exclusive-cursor pagination in (accessed-at instant, id) order with
  ``next_cursor``/``has_more`` semantics, empty pages, machine isolation, and
  exactly the seven visible record fields;
- read-only byte stability on repeat calls, no re-read of already-returned
  records after new inserts, and persistence across a restart.
"""
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


MISSING_ID = "00000000-0000-0000-0000-000000000000"

T0 = "2026-03-01T00:00:00Z"
T5 = "2026-03-01T00:00:05Z"


def changes_url(machine_id):
    return f"/machines/{machine_id}/privacy-accesses/changes"


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


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def insert_access_row(
    client,
    machine_id,
    access_id,
    accessed_at,
    *,
    window_start=T0,
    window_end=T5,
    result="success",
    matches_count=1,
):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO privacy_accesses "
                "(id, machine_id, accessed_at, window_start, window_end, "
                "result, matches_count) VALUES "
                "(:id, :machine_id, :accessed_at, :window_start, :window_end, "
                ":result, :matches_count)"
            ),
            {
                "id": access_id,
                "machine_id": machine_id,
                "accessed_at": accessed_at,
                "window_start": window_start,
                "window_end": window_end,
                "result": result,
                "matches_count": matches_count,
            },
        )


def register_access(client, machine_id, accessed_at, **overrides):
    body = {
        "accessed_at": accessed_at,
        "window_start": T0,
        "window_end": T5,
        "result": "success",
        "matches_count": 1,
        **overrides,
    }
    response = client.post(f"/machines/{machine_id}/privacy-accesses", json=body)
    assert response.status_code == 201
    return response.json()


def get_changes(client, machine_id, **params):
    return client.get(changes_url(machine_id), params=params)


class TestQueryValidation:
    def test_missing_limit_is_bad_limit(self, client):
        machine_id = create_machine(client)
        response = client.get(changes_url(machine_id))
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "bad_limit"}}

    @pytest.mark.parametrize(
        "raw_limit",
        ["0", "101", "-1", "3.0", "true", "false", "abc", "", "1e2", "+5", " 5"],
    )
    def test_invalid_limit_is_bad_limit(self, client, raw_limit):
        machine_id = create_machine(client)
        response = get_changes(client, machine_id, limit=raw_limit)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "bad_limit"}}

    @pytest.mark.parametrize("raw_limit", ["1", "50", "100"])
    def test_boundary_limits_are_accepted(self, client, raw_limit):
        machine_id = create_machine(client)
        response = get_changes(client, machine_id, limit=raw_limit)
        assert response.status_code == 200
        assert response.json()["limit"] == int(raw_limit)

    def test_unknown_parameter_is_invalid_query(self, client):
        machine_id = create_machine(client)
        response = get_changes(client, machine_id, limit="2", extra="1")
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}

    @pytest.mark.parametrize(
        "raw_cursor",
        [
            "",
            "garbage",
            "2026-03-01T00:00:00Z",
            f"|{rid(1)}",
            "2026-03-01T00:00:00Z|",
            "2026-03-01T00:00:00Z|not-a-uuid",
            "2026-03-01 00:00:00Z|" + rid(1),
            "2026-03-01T00:00:00+00:00|" + rid(1),
            "2026-13-01T00:00:00Z|" + rid(1),
            "2026-03-01T00:00:00Z|" + rid(1) + "|extra",
        ],
    )
    def test_malformed_cursor_is_invalid_cursor(self, client, raw_cursor):
        machine_id = create_machine(client)
        response = get_changes(client, machine_id, limit="2", cursor=raw_cursor)
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_cursor"}}

    def test_validation_precedes_machine_lookup(self, client):
        response = client.get(changes_url(MISSING_ID))
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "bad_limit"}}

        response = get_changes(client, MISSING_ID, limit="2", unknown="x")
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_query"}}

        response = get_changes(client, MISSING_ID, limit="2", cursor="garbage")
        assert response.status_code == 422
        assert response.json() == {"error": {"code": "invalid_cursor"}}

    def test_missing_machine_is_not_found_without_records(self, client):
        response = get_changes(client, MISSING_ID, limit="2")
        assert response.status_code == 404
        assert response.json() == {"error": {"code": "not_found"}}
        assert "records" not in response.json()

    @pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
    def test_non_get_methods_are_405(self, client, method):
        machine_id = create_machine(client)
        response = getattr(client, method)(changes_url(machine_id))
        assert response.status_code == 405


class TestPagination:
    def test_empty_machine_returns_complete_empty_result(self, client):
        machine_id = create_machine(client)
        response = get_changes(client, machine_id, limit="10")
        assert response.status_code == 200
        assert response.json() == {
            "machine_id": machine_id,
            "limit": 10,
            "records": [],
            "next_cursor": None,
            "has_more": False,
        }

    def test_pages_walk_the_full_order(self, client):
        machine_id = create_machine(client)
        created = [
            register_access(client, machine_id, f"2026-03-01T00:00:0{i}Z")
            for i in range(5)
        ]

        page1 = get_changes(client, machine_id, limit="2").json()
        assert [r["id"] for r in page1["records"]] == [created[0]["id"], created[1]["id"]]
        assert page1["limit"] == 2
        assert page1["has_more"] is True
        assert page1["next_cursor"] == (
            f"{created[1]['accessed_at']}|{created[1]['id']}"
        )

        page2 = get_changes(
            client, machine_id, limit="2", cursor=page1["next_cursor"]
        ).json()
        assert [r["id"] for r in page2["records"]] == [created[2]["id"], created[3]["id"]]
        assert page2["has_more"] is True

        page3 = get_changes(
            client, machine_id, limit="2", cursor=page2["next_cursor"]
        ).json()
        assert [r["id"] for r in page3["records"]] == [created[4]["id"]]
        assert page3["has_more"] is False
        assert page3["next_cursor"] is None

    def test_page_beyond_the_last_record_is_empty(self, client):
        machine_id = create_machine(client)
        record = register_access(client, machine_id, T0)
        response = get_changes(
            client,
            machine_id,
            limit="2",
            cursor=f"{record['accessed_at']}|{record['id']}",
        )
        assert response.json() == {
            "machine_id": machine_id,
            "limit": 2,
            "records": [],
            "next_cursor": None,
            "has_more": False,
        }

    def test_exact_page_boundary_ends_with_null_cursor(self, client):
        machine_id = create_machine(client)
        for i in range(2):
            register_access(client, machine_id, f"2026-03-01T00:00:0{i}Z")
        page = get_changes(client, machine_id, limit="2").json()
        assert len(page["records"]) == 2
        assert page["has_more"] is False
        assert page["next_cursor"] is None

    def test_records_carry_exactly_the_seven_visible_fields(self, client):
        machine_id = create_machine(client)
        register_access(client, machine_id, T0)
        page = get_changes(client, machine_id, limit="1").json()
        assert set(page["records"][0]) == {
            "id",
            "machine_id",
            "accessed_at",
            "window_start",
            "window_end",
            "result",
            "matches_count",
        }

    def test_ordering_uses_the_actual_utc_instant_then_id(self, client):
        machine_id = create_machine(client)
        # An exact-second stamp sorts before any fractional-second stamp of
        # the same second only after parsing (lexicographically '.' < 'Z');
        # equal instants under distinct raw texts break ties by id ascending.
        insert_access_row(client, machine_id, rid(3), "2026-03-01T00:00:00.5Z")
        insert_access_row(client, machine_id, rid(2), "2026-03-01T00:00:00Z")
        insert_access_row(client, machine_id, rid(1), "2026-03-01T00:00:00.50Z")
        page = get_changes(client, machine_id, limit="10").json()
        assert [r["id"] for r in page["records"]] == [rid(2), rid(1), rid(3)]

    def test_records_are_isolated_per_machine(self, client):
        machine_one = create_machine(client, "machine-1")
        machine_two = create_machine(client, "machine-2")
        register_access(client, machine_one, T0)
        register_access(client, machine_two, "2026-03-01T00:00:01Z")

        page_one = get_changes(client, machine_one, limit="100").json()
        assert [r["machine_id"] for r in page_one["records"]] == [machine_one]
        page_two = get_changes(client, machine_two, limit="100").json()
        assert [r["machine_id"] for r in page_two["records"]] == [machine_two]

    def test_empty_page_keeps_machine_isolation(self, client):
        machine_one = create_machine(client, "machine-1")
        machine_two = create_machine(client, "machine-2")
        register_access(client, machine_two, T0)
        page = get_changes(client, machine_one, limit="5").json()
        assert page["records"] == []
        assert page["has_more"] is False


class TestStability:
    def test_repeated_requests_are_byte_identical(self, client):
        machine_id = create_machine(client)
        for i in range(3):
            register_access(client, machine_id, f"2026-03-01T00:00:0{i}Z")
        first = get_changes(client, machine_id, limit="2")
        second = get_changes(client, machine_id, limit="2")
        assert first.content == second.content

        cursor = first.json()["next_cursor"]
        third = get_changes(client, machine_id, limit="2", cursor=cursor)
        fourth = get_changes(client, machine_id, limit="2", cursor=cursor)
        assert third.content == fourth.content

    def test_insert_behind_the_cursor_is_not_re_read(self, client):
        machine_id = create_machine(client)
        for i in range(4):
            register_access(client, machine_id, f"2026-03-01T00:00:0{i}Z")
        page1 = get_changes(client, machine_id, limit="2").json()
        cursor = page1["next_cursor"]
        before = get_changes(client, machine_id, limit="2", cursor=cursor).json()

        # A new record sorting behind the cursor must not enter the resumed
        # page; one sorting after it appears in order.
        register_access(client, machine_id, "2026-03-01T00:00:00.500Z")
        register_access(client, machine_id, "2026-03-01T00:00:04Z")
        after = get_changes(client, machine_id, limit="2", cursor=cursor).json()
        assert after["records"] == before["records"]

        rest = get_changes(client, machine_id, limit="10", cursor=cursor).json()
        assert [r["accessed_at"] for r in rest["records"]] == [
            "2026-03-01T00:00:02Z",
            "2026-03-01T00:00:03Z",
            "2026-03-01T00:00:04Z",
        ]

    def test_query_is_read_only(self, client):
        machine_id = create_machine(client)
        register_access(client, machine_id, T0)
        get_changes(client, machine_id, limit="1")
        get_changes(client, machine_id, limit="1")
        with client.app.state.engine.begin() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM privacy_accesses")
            ).scalar()
        assert count == 1

    def test_records_survive_a_restart(self, client, tmp_path, monkeypatch):
        machine_id = create_machine(client)
        for i in range(3):
            register_access(client, machine_id, f"2026-03-01T00:00:0{i}Z")
        page1 = get_changes(client, machine_id, limit="2").json()

        # Simulate a restart: a fresh client over the same database file.
        with TestClient(app) as restarted:
            replayed = get_changes(restarted, machine_id, limit="2").json()
            assert replayed == page1
            page2 = get_changes(
                restarted, machine_id, limit="2", cursor=page1["next_cursor"]
            ).json()
            assert len(page2["records"]) == 1
            assert page2["has_more"] is False
            assert page2["next_cursor"] is None
