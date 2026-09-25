"""Tests for the global policy-rule tamper-evident hash chain.

Covers `POST /policy-rules` (rule creation and chain-tail append in one
transaction), `GET /policy-rules/chain` (the full chain view) and
`GET /policy-rules/integrity` (read-only chain verification):

* creation links each new rule to the tail with content/chain hashes;
* the chain view and the creation response agree, field for field;
* GET-only routing (405), no-query-parameter validation (422
  ``invalid_query``), invalid bodies (422) and duplicate business triples
  (409 ``duplicate_policy_rule``) never write a link;
* an empty table returns the three integrity fields and an empty chain;
* the first tampered content / predecessor / chain hash is reported;
* a rule with an unparseable ``created_at`` still counts, sorts last, and is
  judged by its chain values;
* concurrent creations form one unbroken chain with unique predecessors;
* a legacy database gets its columns added and rows backfilled in stable
  order, a second startup writes nothing, and an empty database can create
  and query;
* the plain listing, the compliance export, and authorization evaluation
  keep their existing fields and semantics.
"""
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app
from accountability.policy_rule_chain import compute_chain_hash, compute_content_hash

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CHAIN_PATH = "/policy-rules/chain"
INTEGRITY_PATH = "/policy-rules/integrity"

VISIBLE_KEYS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)
CHAIN_KEYS = VISIBLE_KEYS + ("previous_rule_id", "content_hash", "chain_hash")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def create_rule(
    client,
    action_type="read",
    resource_pattern="res/*",
    effect="allow",
    priority=0,
):
    return client.post(
        "/policy-rules",
        json={
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
        },
    )


def get_chain(client):
    response = client.get(CHAIN_PATH)
    assert response.status_code == 200
    return response.json()


def get_integrity(client):
    response = client.get(INTEGRITY_PATH)
    assert response.status_code == 200
    return response.json()


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


def expected_content_hash(record: dict) -> str:
    return compute_content_hash(**{key: record[key] for key in VISIBLE_KEYS})


# --- creation links to the chain tail --------------------------------------


def test_first_rule_is_the_chain_root(client):
    response = create_rule(client, priority=1)

    assert response.status_code == 201
    body = response.json()
    assert body["previous_rule_id"] is None
    assert HEX64_RE.fullmatch(body["content_hash"])
    assert HEX64_RE.fullmatch(body["chain_hash"])
    assert body["content_hash"] == expected_content_hash(body)
    assert body["chain_hash"] == compute_chain_hash("", body["content_hash"])
    assert set(body.keys()) == set(CHAIN_KEYS)


def test_each_new_rule_links_to_the_previous_tail(client):
    first = create_rule(client, priority=1).json()
    second = create_rule(client, action_type="write", priority=2).json()
    third = create_rule(client, resource_pattern="res/**", priority=3).json()

    assert second["previous_rule_id"] == first["id"]
    assert third["previous_rule_id"] == second["id"]
    assert second["chain_hash"] == compute_chain_hash(
        first["chain_hash"], second["content_hash"]
    )
    assert third["chain_hash"] == compute_chain_hash(
        second["chain_hash"], third["content_hash"]
    )

    chain = get_chain(client)
    assert [rule["id"] for rule in chain] == [first["id"], second["id"], third["id"]]
    previous = [rule["previous_rule_id"] for rule in chain]
    assert previous == [None, first["id"], second["id"]]


def test_chain_view_matches_creation_responses_field_for_field(client):
    first = create_rule(client, priority=1).json()
    second = create_rule(client, action_type="write", effect="deny", priority=2).json()

    chain = get_chain(client)
    assert chain == [first, second]
    for rule in chain:
        assert set(rule.keys()) == set(CHAIN_KEYS)
        assert rule["content_hash"] == expected_content_hash(rule)


def test_chain_body_is_compact_json_ending_in_one_newline(client):
    create_rule(client)
    response = client.get(CHAIN_PATH)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.content.endswith(b"\n")
    assert not response.content.endswith(b"\n\n")
    payload = response.json()
    expected = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False,
                   separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    assert response.content == expected


def test_empty_chain_is_empty_array(client):
    assert get_chain(client) == []


def test_empty_table_integrity_reports_valid_zero_null(client):
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 0,
        "broken_policy_rule_id": None,
    }


def test_sound_chain_is_valid(client):
    create_rule(client, priority=1)
    create_rule(client, priority=2)
    create_rule(client, priority=3)

    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 3,
        "broken_policy_rule_id": None,
    }


# --- routing and query validation ------------------------------------------


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_chain_endpoints_accept_only_get(client, path):
    create_rule(client)
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(path)
        assert response.status_code == 405


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
@pytest.mark.parametrize(
    "query",
    ["?unexpected=1", "?limit=10", "?id=x", "?from_created_at=2026-01-01T00:00:00Z"],
)
def test_query_parameters_are_invalid_query(client, path, query):
    response = client.get(f"{path}{query}")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_extra_parameter_is_invalid_query_on_empty_database_too(client, path):
    # Empty database must not change the error type or become a 200/404.
    response = client.get(f"{path}?anything=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource_pattern": "r", "effect": "allow", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "maybe", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": -1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": True},
        {"action_type": "a", "resource_pattern": "r", "priority": 0},
        {},
    ],
)
def test_invalid_body_is_422_and_writes_no_link(client, payload):
    response = client.post("/policy-rules", json=payload)
    assert response.status_code == 422
    assert get_chain(client) == []
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 0,
        "broken_policy_rule_id": None,
    }


def test_duplicate_business_triple_is_409_and_appends_nothing(client):
    first = create_rule(client, priority=1)
    assert first.status_code == 201

    response = create_rule(client, effect="deny", priority=1)
    assert response.status_code == 409
    assert response.json() == {"error": {"code": "duplicate_policy_rule"}}

    # The loser shares no predecessor: the chain still holds exactly one rule.
    chain = get_chain(client)
    assert [rule["id"] for rule in chain] == [first.json()["id"]]
    assert get_integrity(client)["checked_count"] == 1
    assert get_integrity(client)["valid"] is True


# --- tamper detection -------------------------------------------------------


def insert_chain_rule(client, values):
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at, previous_rule_id, content_hash, "
                "chain_hash) VALUES "
                "(:id, :action_type, :resource_pattern, :effect, :priority, "
                ":created_at, :updated_at, :previous_rule_id, :content_hash, "
                ":chain_hash)"
            ),
            values,
        )


def _build_linked_rows(client, stamps):
    """Insert sound linked rules directly, returning their stored dicts."""
    rows = []
    previous_id = None
    previous_chain_hash = ""
    for index, stamp in enumerate(stamps):
        rule_id = rid(index + 1)
        rule = {
            "id": rule_id,
            "action_type": "read",
            "resource_pattern": f"res/{index}",
            "effect": "allow",
            "priority": index,
            "created_at": stamp,
            "updated_at": stamp,
        }
        content_hash = compute_content_hash(**rule)
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        stored = {
            **rule,
            "previous_rule_id": previous_id,
            "content_hash": content_hash,
            "chain_hash": chain_hash,
        }
        insert_chain_rule(client, stored)
        rows.append(stored)
        previous_id = rule_id
        previous_chain_hash = chain_hash
    return rows


def test_tampered_content_is_reported(client):
    rows = _build_linked_rows(
        client,
        ["2026-03-01T00:00:00Z", "2026-03-01T00:00:01Z", "2026-03-01T00:00:02Z"],
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET effect = 'deny' WHERE id = :id"),
            {"id": rid(2)},
        )

    integrity = get_integrity(client)
    assert integrity == {
        "valid": False,
        "checked_count": 3,
        "broken_policy_rule_id": rid(2),
    }
    # The tampered rule is still shown in the chain view with stored values.
    chain = get_chain(client)
    assert len(chain) == 3
    assert next(r for r in chain if r["id"] == rid(2))["effect"] == "deny"
    assert rows[0]["id"] == rid(1)


def test_tampered_previous_link_is_reported(client):
    _build_linked_rows(
        client,
        ["2026-03-01T00:00:00Z", "2026-03-01T00:00:01Z"],
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET previous_rule_id = NULL WHERE id = :id"),
            {"id": rid(2)},
        )

    assert get_integrity(client) == {
        "valid": False,
        "checked_count": 2,
        "broken_policy_rule_id": rid(2),
    }


def test_tampered_chain_hash_is_reported(client):
    _build_linked_rows(
        client,
        ["2026-03-01T00:00:00Z", "2026-03-01T00:00:01Z"],
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET chain_hash = :h WHERE id = :id"),
            {"h": "f" * 64, "id": rid(1)},
        )

    assert get_integrity(client) == {
        "valid": False,
        "checked_count": 2,
        "broken_policy_rule_id": rid(1),
    }


def test_only_the_first_broken_rule_is_reported(client):
    _build_linked_rows(
        client,
        ["2026-03-01T00:00:00Z", "2026-03-01T00:00:01Z", "2026-03-01T00:00:02Z"],
    )
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text("UPDATE policy_rules SET priority = 99 WHERE id = :id"),
            {"id": rid(1)},
        )
        conn.execute(
            text("UPDATE policy_rules SET effect = 'deny' WHERE id = :id"),
            {"id": rid(3)},
        )

    assert get_integrity(client)["broken_policy_rule_id"] == rid(1)


def test_unparseable_created_at_is_counted_last_and_judged_by_chain_values(client):
    # One sound early rule, one rule whose stored stamp cannot parse but whose
    # stored chain values are consistent with being the second link.
    _build_linked_rows(client, ["2026-03-01T00:00:00Z"])
    first_chain_hash = get_chain(client)[0]["chain_hash"]

    rule_id = rid(2)
    rule = {
        "id": rule_id,
        "action_type": "read",
        "resource_pattern": "res/x",
        "effect": "allow",
        "priority": 5,
        "created_at": "not-a-time",
        "updated_at": "not-a-time",
    }
    content_hash = compute_content_hash(**rule)
    insert_chain_rule(
        client,
        {
            **rule,
            "previous_rule_id": rid(1),
            "content_hash": content_hash,
            "chain_hash": compute_chain_hash(first_chain_hash, content_hash),
        },
    )

    # No crash; counted in the total, sorts last, and the chain still verifies.
    integrity = get_integrity(client)
    assert integrity == {
        "valid": True,
        "checked_count": 2,
        "broken_policy_rule_id": None,
    }
    assert [rule_["id"] for rule_ in get_chain(client)] == [rid(1), rid(2)]


def test_unparseable_created_at_with_broken_chain_values_is_flagged(client):
    _build_linked_rows(client, ["2026-03-01T00:00:00Z"])
    rule_id = rid(2)
    rule = {
        "id": rule_id,
        "action_type": "read",
        "resource_pattern": "res/x",
        "effect": "allow",
        "priority": 5,
        "created_at": "not-a-time",
        "updated_at": "not-a-time",
    }
    insert_chain_rule(
        client,
        {
            **rule,
            "previous_rule_id": rid(1),
            "content_hash": compute_content_hash(**rule),
            "chain_hash": "0" * 64,
        },
    )

    assert get_integrity(client) == {
        "valid": False,
        "checked_count": 2,
        "broken_policy_rule_id": rule_id,
    }


def test_exact_second_precedes_fractional_same_second(client):
    # Two rules at the same wall-clock second: the exact-second stamp must be
    # the predecessor of the fractional one for the chain to verify.
    first_id = rid(1)
    second_id = rid(2)
    first_rule = {
        "id": first_id,
        "action_type": "read",
        "resource_pattern": "res/1",
        "effect": "allow",
        "priority": 1,
        "created_at": "2026-03-01T00:00:00Z",
        "updated_at": "2026-03-01T00:00:00Z",
    }
    first_content = compute_content_hash(**first_rule)
    first_chain = compute_chain_hash("", first_content)
    insert_chain_rule(
        client,
        {
            **first_rule,
            "previous_rule_id": None,
            "content_hash": first_content,
            "chain_hash": first_chain,
        },
    )
    second_rule = {
        "id": second_id,
        "action_type": "read",
        "resource_pattern": "res/2",
        "effect": "allow",
        "priority": 2,
        "created_at": "2026-03-01T00:00:00.500000Z",
        "updated_at": "2026-03-01T00:00:00.500000Z",
    }
    second_content = compute_content_hash(**second_rule)
    insert_chain_rule(
        client,
        {
            **second_rule,
            "previous_rule_id": first_id,
            "content_hash": second_content,
            "chain_hash": compute_chain_hash(first_chain, second_content),
        },
    )

    assert [rule["id"] for rule in get_chain(client)] == [first_id, second_id]
    assert get_integrity(client)["valid"] is True


# --- read-only / byte stability ---------------------------------------------


def test_chain_and_integrity_are_read_only_and_byte_stable(client):
    create_rule(client, priority=1)
    create_rule(client, priority=2)

    def table_state():
        with client.app.state.engine.connect() as conn:
            return list(conn.execute(text("SELECT * FROM policy_rules")))

    before = table_state()
    chain_1 = client.get(CHAIN_PATH).content
    integrity_1 = client.get(INTEGRITY_PATH).content
    middle = table_state()
    chain_2 = client.get(CHAIN_PATH).content
    integrity_2 = client.get(INTEGRITY_PATH).content
    after = table_state()

    assert chain_1 == chain_2
    assert integrity_1 == integrity_2
    assert before == middle == after


# --- concurrency ------------------------------------------------------------


def test_concurrent_creations_form_one_unbroken_chain(client):
    count = 30
    gate = threading.Event()

    def create(index):
        gate.wait()
        return create_rule(client, resource_pattern=f"res/{index}", priority=index)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(create, i) for i in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    assert all(response.status_code == 201 for response in responses)

    chain = get_chain(client)
    assert len(chain) == count
    assert len({rule["id"] for rule in chain}) == count

    # One unbroken chain: unique predecessors, no skips, no shared predecessor.
    previous_ids = [rule["previous_rule_id"] for rule in chain]
    ids = [rule["id"] for rule in chain]
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]
    assert len({pid for pid in previous_ids if pid is not None}) == count - 1

    # Every link hashes consistently.
    previous_chain_hash = ""
    for rule in chain:
        assert rule["content_hash"] == expected_content_hash(rule)
        assert rule["chain_hash"] == compute_chain_hash(
            previous_chain_hash, rule["content_hash"]
        )
        previous_chain_hash = rule["chain_hash"]

    assert get_integrity(client) == {
        "valid": True,
        "checked_count": count,
        "broken_policy_rule_id": None,
    }


def test_concurrent_duplicates_lose_and_leave_one_rule(client):
    count = 12
    gate = threading.Event()

    def create(_):
        gate.wait()
        # Identical business triple: at most one may succeed.
        return create_rule(client, priority=1)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(create, i) for i in range(count)]
        gate.set()
        responses = [f.result() for f in futures]

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [201] + [409] * (count - 1)
    assert all(
        response.json() == {"error": {"code": "duplicate_policy_rule"}}
        for response in responses
        if response.status_code == 409
    )
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 1,
        "broken_policy_rule_id": None,
    }


# --- legacy migration, backfill, restart ------------------------------------


def test_legacy_rules_are_backfilled_in_chain_order(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY, action_type VARCHAR,
            resource_pattern VARCHAR, effect VARCHAR, priority INTEGER,
            created_at VARCHAR, updated_at VARCHAR
        )
        """
    )
    # Inserted out of order, including an exact/fractional same-second pair.
    connection.executemany(
        "INSERT INTO policy_rules "
        "(id, action_type, resource_pattern, effect, priority, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        [
            (rid(30), "read", "res/30", "allow", 30,
             "2026-03-01T00:00:03Z", "2026-03-01T00:00:03Z"),
            (rid(20), "read", "res/20", "allow", 20,
             "2026-03-01T00:00:02Z", "2026-03-01T00:00:02Z"),
            (rid(10), "read", "res/10", "allow", 10,
             "2026-03-01T00:00:01Z", "2026-03-01T00:00:01Z"),
            (rid(1), "read", "res/1", "allow", 1,
             "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
            (rid(2), "read", "res/2", "allow", 2,
             "2026-03-01T00:00:00.500000Z", "2026-03-01T00:00:00.500000Z"),
        ],
    )
    connection.commit()
    connection.close()

    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as first:
        chain = get_chain(first)
        assert [rule["id"] for rule in chain] == [
            rid(1), rid(2), rid(10), rid(20), rid(30),
        ]
        previous = [rule["previous_rule_id"] for rule in chain]
        assert previous == [None, rid(1), rid(2), rid(10), rid(20)]
        for rule in chain:
            assert HEX64_RE.fullmatch(rule["content_hash"])
            assert HEX64_RE.fullmatch(rule["chain_hash"])
        assert get_integrity(first) == {
            "valid": True,
            "checked_count": 5,
            "broken_policy_rule_id": None,
        }
        # A newly created rule attaches to the backfilled tail.
        created = create_rule(first, resource_pattern="res/new", priority=99).json()
        assert created["previous_rule_id"] == rid(30)
        assert get_integrity(first) == {
            "valid": True,
            "checked_count": 6,
            "broken_policy_rule_id": None,
        }
        first_bytes = first.get(CHAIN_PATH).content
        integrity_bytes = first.get(INTEGRITY_PATH).content

    # A second startup over the now-complete table must not rewrite anything.
    with TestClient(app) as second:
        assert second.get(CHAIN_PATH).content == first_bytes
        assert second.get(INTEGRITY_PATH).content == integrity_bytes
        chain_again = get_chain(second)
        assert [rule["id"] for rule in chain_again] == [
            rid(1), rid(2), rid(10), rid(20), rid(30), created["id"],
        ]
        assert get_integrity(second)["valid"] is True
        assert get_integrity(second)["checked_count"] == 6


def test_backfill_is_idempotent_across_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "persist.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as first:
        create_rule(first, priority=1)
        create_rule(first, priority=2)
        expected = first.get(CHAIN_PATH).content

    with TestClient(app) as second:
        assert second.get(CHAIN_PATH).content == expected
        assert get_integrity(second) == {
            "valid": True,
            "checked_count": 2,
            "broken_policy_rule_id": None,
        }


def test_empty_database_can_create_and_query_after_startup(tmp_path, monkeypatch):
    db_path = tmp_path / "empty.db"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")
    with TestClient(app) as client:
        assert get_chain(client) == []
        assert get_integrity(client) == {
            "valid": True,
            "checked_count": 0,
            "broken_policy_rule_id": None,
        }
        created = create_rule(client)
        assert created.status_code == 201
        assert created.json()["previous_rule_id"] is None
        assert len(get_chain(client)) == 1
        assert get_integrity(client)["checked_count"] == 1


# --- listing, export and evaluation keep their existing behavior ------------


def test_listing_keeps_exactly_the_visible_fields(client):
    created = create_rule(client, priority=4).json()

    [rule] = client.get("/policy-rules").json()

    assert set(rule.keys()) == set(VISIBLE_KEYS)
    assert rule == {key: created[key] for key in VISIBLE_KEYS}
    assert "content_hash" not in rule
    assert "chain_hash" not in rule
    assert "previous_rule_id" not in rule


def test_compliance_export_keeps_existing_fields(client):
    create_rule(client, priority=1)
    response = client.get(
        "/policy-rules/compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z&to_created_at=2100-01-01T00:00:00Z"
    )
    assert response.status_code == 200
    [rule] = response.json()["policy_rules"]
    assert set(rule.keys()) == set(VISIBLE_KEYS)


def test_chain_fields_and_checks_do_not_change_authorization(client):
    machine_id = client.post(
        "/machines",
        json={
            "external_id": "machine-1",
            "display_name": "Machine One",
            "public_key": "key-1",
        },
    ).json()["id"]
    client.post(
        f"/machines/{machine_id}/behavior-declarations",
        json={
            "action_type": "read",
            "resource_pattern": "res/*",
            "enabled": True,
        },
    )
    create_rule(client, effect="allow", priority=0)

    client.get(CHAIN_PATH)
    client.get(INTEGRITY_PATH)
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}
