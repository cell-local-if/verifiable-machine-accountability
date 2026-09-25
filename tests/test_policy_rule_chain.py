"""Tests for the global tamper-evident policy-rule hash chain.

Covers ``POST /policy-rules`` (atomic chain-tail append), ``GET
/policy-rules/chain`` (the complete chain with predecessor id, content digest
and chain digest), and ``GET /policy-rules/integrity`` (read-only chain
verification): canonical SHA-256 digests over the rule's existing visible
fields, empty-prefix chain hashing, creation/query consistency, exact-second
vs fractional-second ordering, unparseable stored timestamps counted and
flagged without crashing, 405 on non-GET, 422 ``invalid_query`` on any
parameter (including against an empty database), 422 body rejection and 409
duplicate rejection writing no chain link, concurrent creations forming one
unbroken chain, legacy-schema migration/backfill on startup, no-op restarts
over a complete chain, unchanged list/compliance-export fields, and unaffected
authorization evaluation.
"""
import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from accountability.app import app

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

CONTENT_KEYS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)

CHAIN_PATH = "/policy-rules/chain"
INTEGRITY_PATH = "/policy-rules/integrity"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}"
    )
    with TestClient(app) as test_client:
        yield test_client


def canonical_content_hash(rule: dict) -> str:
    document = json.dumps(
        {key: rule[key] for key in CONTENT_KEYS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    return hashlib.sha256(
        f"{previous_chain_hash}:{content_hash}".encode("utf-8")
    ).hexdigest()


def rid(n):
    return f"00000000-0000-0000-0000-{n:012d}"


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


def insert_rule_row(client, rule_id, created_at, *, updated_at=None, priority=1,
                    action_type="read", resource_pattern="res/*", effect="allow"):
    """Insert a policy rule directly, bypassing the chain write path."""
    with client.app.state.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO policy_rules "
                "(id, action_type, resource_pattern, effect, priority, "
                "created_at, updated_at) "
                "VALUES (:id, :action_type, :resource_pattern, :effect, "
                ":priority, :created_at, :updated_at)"
            ),
            {
                "id": rule_id,
                "action_type": action_type,
                "resource_pattern": resource_pattern,
                "effect": effect,
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at or created_at,
            },
        )


# --------------------------------------------------------------------------- #
# Empty database
# --------------------------------------------------------------------------- #


def test_empty_chain_is_empty_array(client):
    response = client.get(CHAIN_PATH)
    assert response.status_code == 200
    assert response.content == b"[]\n"


def test_empty_chain_integrity_reports_three_fields(client):
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 0,
        "broken_policy_rule_id": None,
    }


# --------------------------------------------------------------------------- #
# Creation appends one correctly linked record
# --------------------------------------------------------------------------- #


def test_created_rule_is_the_chain_root(client):
    created = create_rule(client, priority=3).json()
    [link] = get_chain(client)

    assert link["previous_rule_id"] is None
    assert HEX64_RE.match(link["content_hash"])
    assert HEX64_RE.match(link["chain_hash"])
    assert link["content_hash"] == canonical_content_hash(link)
    assert link["chain_hash"] == chain_hash("", link["content_hash"])


def test_chain_item_visible_fields_match_creation_response(client):
    created = create_rule(
        client, action_type="read", resource_pattern="res/*",
        effect="allow", priority=4,
    ).json()
    [link] = get_chain(client)

    for key in CONTENT_KEYS:
        assert link[key] == created[key]


def test_new_rule_links_to_chain_tail(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(4)
    ]
    links = get_chain(client)

    assert [link["id"] for link in links] == [rule["id"] for rule in rules]
    for index, link in enumerate(links):
        assert link["content_hash"] == canonical_content_hash(link)
        if index == 0:
            assert link["previous_rule_id"] is None
            assert link["chain_hash"] == chain_hash("", link["content_hash"])
        else:
            previous = links[index - 1]
            assert link["previous_rule_id"] == previous["id"]
            assert link["chain_hash"] == chain_hash(
                previous["chain_hash"], link["content_hash"]
            )


def test_chain_items_have_exactly_ten_fields(client):
    create_rule(client)
    [link] = get_chain(client)
    assert set(link.keys()) == {
        "id",
        "action_type",
        "resource_pattern",
        "effect",
        "priority",
        "created_at",
        "updated_at",
        "previous_rule_id",
        "content_hash",
        "chain_hash",
    }


def test_chain_body_is_compact_json_with_single_newline(client):
    create_rule(client)
    response = client.get(CHAIN_PATH)
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


def test_sound_chain_integrity_is_valid(client):
    for i in range(3):
        create_rule(client, action_type=f"action-{i}", priority=i)
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 3,
        "broken_policy_rule_id": None,
    }


# --------------------------------------------------------------------------- #
# Failed creations write no chain link
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "  ", "resource_pattern": "r", "effect": "allow", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "maybe", "priority": 0},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": -1},
        {"action_type": "a", "resource_pattern": "r", "effect": "allow", "priority": True},
        {"action_type": "a", "resource_pattern": "r", "priority": 0},
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


def test_duplicate_is_409_and_appends_nothing(client):
    first = create_rule(client, priority=1)
    assert first.status_code == 201
    duplicate = create_rule(client, effect="deny", priority=1)
    assert duplicate.status_code == 409
    assert duplicate.json() == {"error": {"code": "duplicate_policy_rule"}}

    links = get_chain(client)
    assert len(links) == 1
    assert links[0]["id"] == first.json()["id"]
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 1,
        "broken_policy_rule_id": None,
    }


def test_duplicate_after_several_rules_keeps_chain_intact(client):
    create_rule(client, action_type="read", priority=1)
    create_rule(client, action_type="write", priority=1)
    create_rule(client, action_type="read", priority=2)
    before = get_chain(client)

    assert create_rule(client, action_type="write", priority=1).status_code == 409

    assert get_chain(client) == before
    assert get_integrity(client)["valid"] is True


# --------------------------------------------------------------------------- #
# Method and query-string restrictions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_non_get_methods_return_405(client, path):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)(path)
        assert response.status_code == 405


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_any_query_parameter_is_invalid_query(client, path):
    response = client.get(f"{path}?unexpected=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_invalid_query_same_on_empty_database_and_with_rules(client, path):
    # Empty table does not change the error type.
    empty = client.get(f"{path}?x=1")
    assert empty.status_code == 422
    assert empty.json() == {"error": {"code": "invalid_query"}}

    create_rule(client)
    populated = client.get(f"{path}?x=1")
    assert populated.status_code == 422
    assert populated.json() == {"error": {"code": "invalid_query"}}


@pytest.mark.parametrize("path", (CHAIN_PATH, INTEGRITY_PATH))
def test_invalid_query_does_not_read_rule_content(client, path):
    # Create a rule, then corrupt it: an invalid query must still be the 422
    # validation error, never a 200 reflecting (or failing on) the row.
    created = create_rule(client).json()
    with sqlite3.connect(client.app.state.engine.url.database) as conn:
        conn.execute(
            "UPDATE policy_rules SET effect = 'forged' WHERE id = ?",
            (created["id"],),
        )
        conn.commit()
    response = client.get(f"{path}?x=1")
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_query"}}


# --------------------------------------------------------------------------- #
# Chain ordering
# --------------------------------------------------------------------------- #


def test_chain_orders_by_created_at_instant_then_id(tmp_path, monkeypatch):
    db_path = tmp_path / "ordered.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_pattern VARCHAR NOT NULL,
            effect VARCHAR NOT NULL,
            priority INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            UNIQUE (action_type, resource_pattern, priority)
        )
        """
    )
    # Rows deliberately written out of order, with one exact/fractional pair.
    connection.executemany(
        "INSERT INTO policy_rules "
        "(id, action_type, resource_pattern, effect, priority, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        [
            (rid(30), "read", "res/30", "allow", 30,
             "2026-03-01T00:00:03Z", "2026-03-01T00:00:03Z"),
            (rid(21), "read", "res/21", "allow", 21,
             "2026-03-01T00:00:02Z", "2026-03-01T00:00:02Z"),
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

    with TestClient(app) as client:
        links = get_chain(client)
        assert [link["id"] for link in links] == [
            rid(1),   # exact second sorts before the fractional stamp
            rid(2),
            rid(10),
            rid(20),  # tie at :02 breaks by id
            rid(21),
            rid(30),
        ]
        # The backfilled links follow that same order.
        assert links[0]["previous_rule_id"] is None
        for previous, link in zip(links, links[1:]):
            assert link["previous_rule_id"] == previous["id"]
            assert link["chain_hash"] == chain_hash(
                previous["chain_hash"], link["content_hash"]
            )
        assert get_integrity(client) == {
            "valid": True,
            "checked_count": 6,
            "broken_policy_rule_id": None,
        }


def test_chain_stored_values_are_emitted_without_normalization(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "raw.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_pattern VARCHAR NOT NULL,
            effect VARCHAR NOT NULL,
            priority INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            UNIQUE (action_type, resource_pattern, priority)
        )
        """
    )
    # The write path trims inputs; the chain view must echo stored bytes as-is,
    # including values the write path would never produce.
    connection.execute(
        "INSERT INTO policy_rules "
        "(id, action_type, resource_pattern, effect, priority, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (
            rid(1), "  Read ", " res/* ", "ALLOW", 9,
            "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z",
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as client:
        [link] = get_chain(client)
        assert link["action_type"] == "  Read "
        assert link["resource_pattern"] == " res/* "
        assert link["effect"] == "ALLOW"
        assert link["priority"] == 9
        assert link["created_at"] == "2026-03-01T00:00:00Z"
        assert link["updated_at"] == "2026-04-01T00:00:00Z"
        # The digest is computed over exactly the stored visible values.
        assert link["content_hash"] == canonical_content_hash(link)
        assert get_integrity(client)["valid"] is True


# --------------------------------------------------------------------------- #
# Damage detection
# --------------------------------------------------------------------------- #


def _db_path(client):
    return client.app.state.engine.url.database


def test_integrity_detects_tampered_content(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(3)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET effect = 'forged' WHERE id = ?",
            (rules[1]["id"],),
        )
        conn.commit()

    assert get_integrity(client) == {
        "valid": False,
        "checked_count": 3,
        "broken_policy_rule_id": rules[1]["id"],
    }


def test_integrity_detects_tampered_previous_link(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(3)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET previous_rule_id = NULL WHERE id = ?",
            (rules[2]["id"],),
        )
        conn.commit()

    result = get_integrity(client)
    assert result == {
        "valid": False,
        "checked_count": 3,
        "broken_policy_rule_id": rules[2]["id"],
    }


def test_integrity_detects_tampered_chain_hash(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(2)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET chain_hash = ? WHERE id = ?",
            ("0" * 64, rules[0]["id"]),
        )
        conn.commit()

    result = get_integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rules[0]["id"]


def test_integrity_detects_tampered_content_hash(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(2)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET content_hash = ? WHERE id = ?",
            ("f" * 64, rules[1]["id"]),
        )
        conn.commit()

    result = get_integrity(client)
    assert result["valid"] is False
    assert result["broken_policy_rule_id"] == rules[1]["id"]


def test_integrity_reports_first_broken_rule_in_chain_order(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(3)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET priority = 77 WHERE id = ?",
            (rules[0]["id"],),
        )
        conn.execute(
            "UPDATE policy_rules SET priority = 88 WHERE id = ?",
            (rules[2]["id"],),
        )
        conn.commit()

    result = get_integrity(client)
    assert result["valid"] is False
    assert result["checked_count"] == 3
    assert result["broken_policy_rule_id"] == rules[0]["id"]


def test_unparseable_created_at_is_counted_last_and_flagged(client):
    first = create_rule(client, action_type="action-1", priority=1).json()
    second = create_rule(client, action_type="action-2", priority=2).json()
    with sqlite3.connect(_db_path(client)) as conn:
        # Tamper with the stamp after creation: its stored content digest was
        # computed from the original value, so the content check fails.
        conn.execute(
            "UPDATE policy_rules SET created_at = 'not-a-time' WHERE id = ?",
            (second["id"],),
        )
        conn.commit()

    links = get_chain(client)
    # The damaged stamp sorts after every parseable instant without crashing.
    assert [link["id"] for link in links] == [first["id"], second["id"]]

    result = get_integrity(client)
    assert result["valid"] is False
    assert result["checked_count"] == 2
    assert result["broken_policy_rule_id"] == second["id"]


def test_unparseable_stamp_with_sound_chain_values_is_not_broken(
    tmp_path, monkeypatch
):
    # A legacy rule whose stored stamp never parses but whose chain values are
    # backfilled from exactly the stored content: the stamp decides ordering
    # (last) and the count, never the verdict on its own.
    db_path = tmp_path / "badstamp.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_pattern VARCHAR NOT NULL,
            effect VARCHAR NOT NULL,
            priority INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            UNIQUE (action_type, resource_pattern, priority)
        )
        """
    )
    connection.executemany(
        "INSERT INTO policy_rules "
        "(id, action_type, resource_pattern, effect, priority, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        [
            (rid(1), "read", "res/a", "allow", 1,
             "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
            (rid(2), "read", "res/b", "allow", 2,
             "not-a-time", "not-a-time"),
        ],
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as client:
        links = get_chain(client)
        assert [link["id"] for link in links] == [rid(1), rid(2)]
        assert links[1]["content_hash"] == canonical_content_hash(links[1])
        assert links[1]["chain_hash"] == chain_hash(
            links[0]["chain_hash"], links[1]["content_hash"]
        )
        assert get_integrity(client) == {
            "valid": True,
            "checked_count": 2,
            "broken_policy_rule_id": None,
        }


def test_unparseable_stamp_row_with_missing_chain_values_is_broken(client):
    # A raw row inserted while the app is running is never backfilled: its
    # NULL chain values fail the chain-value checks. The unparseable stamp
    # sorts it last (here: only), it is still counted, and the verdict comes
    # from the missing chain values rather than from the stamp itself.
    insert_rule_row(client, rid(1), "not-a-time", priority=1)
    assert get_integrity(client) == {
        "valid": False,
        "checked_count": 1,
        "broken_policy_rule_id": rid(1),
    }
    assert [link["id"] for link in get_chain(client)] == [rid(1)]


def test_integrity_is_read_only_and_byte_stable(client):
    rules = [
        create_rule(client, action_type=f"action-{i}", priority=i).json()
        for i in range(3)
    ]
    with sqlite3.connect(_db_path(client)) as conn:
        conn.execute(
            "UPDATE policy_rules SET effect = 'forged' WHERE id = ?",
            (rules[1]["id"],),
        )
        conn.commit()

    first = client.get(INTEGRITY_PATH).content
    second = client.get(INTEGRITY_PATH).content
    third = client.get(INTEGRITY_PATH).content
    assert first == second == third

    # The damaged stored value is left untouched.
    with sqlite3.connect(_db_path(client)) as conn:
        stored = conn.execute(
            "SELECT effect FROM policy_rules WHERE id = ?", (rules[1]["id"],)
        ).fetchone()[0]
    assert stored == "forged"


def test_chain_query_is_read_only_and_byte_stable(client):
    for i in range(3):
        create_rule(client, action_type=f"action-{i}", priority=i)
    first = client.get(CHAIN_PATH).content
    second = client.get(CHAIN_PATH).content
    assert first == second
    assert first.endswith(b"\n")


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_creations_form_one_unbroken_chain(client):
    count = 30

    def append(index):
        return create_rule(client, action_type=f"action-{index}", priority=index)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(append, range(count)))

    assert all(response.status_code == 201 for response in responses)
    links = get_chain(client)
    assert len(links) == count
    assert len({link["id"] for link in links}) == count

    ids = [link["id"] for link in links]
    previous_ids = [link["previous_rule_id"] for link in links]
    # No loss, no skipped link, and no two rules share a predecessor.
    assert previous_ids[0] is None
    assert previous_ids[1:] == ids[:-1]
    assert len({previous for previous in previous_ids if previous is not None}) == count - 1

    assert get_integrity(client) == {
        "valid": True,
        "checked_count": count,
        "broken_policy_rule_id": None,
    }


def test_concurrent_identical_creations_have_one_winner(client):
    count = 12

    def append(_index):
        return create_rule(client, action_type="same", priority=5)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(append, range(count)))

    successes = [r for r in responses if r.status_code == 201]
    conflicts = [r for r in responses if r.status_code == 409]
    assert len(successes) == 1
    assert len(conflicts) == count - 1
    assert len(get_chain(client)) == 1
    assert get_integrity(client) == {
        "valid": True,
        "checked_count": 1,
        "broken_policy_rule_id": None,
    }


# --------------------------------------------------------------------------- #
# Legacy migration, backfill, restarts
# --------------------------------------------------------------------------- #


def _create_legacy_database(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_pattern VARCHAR NOT NULL,
            effect VARCHAR NOT NULL,
            priority INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            UNIQUE (action_type, resource_pattern, priority)
        )
        """
    )
    # Deliberately unordered, with an exact/fractional same-second pair.
    connection.executemany(
        "INSERT INTO policy_rules "
        "(id, action_type, resource_pattern, effect, priority, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        [
            (rid(3), "read", "res/c", "allow", 3,
             "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"),
            (rid(2), "read", "res/b", "allow", 2,
             "2026-01-01T00:00:00.500000Z", "2026-01-01T00:00:00.500000Z"),
            (rid(1), "read", "res/a", "allow", 1,
             "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ],
    )
    connection.commit()
    connection.close()


def test_legacy_rules_are_backfilled_on_startup(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as client:
        links = get_chain(client)
        assert [link["id"] for link in links] == [rid(1), rid(2), rid(3)]

        assert links[0]["previous_rule_id"] is None
        assert links[1]["previous_rule_id"] == rid(1)
        assert links[2]["previous_rule_id"] == rid(2)
        for link in links:
            assert HEX64_RE.match(link["content_hash"])
            assert HEX64_RE.match(link["chain_hash"])
            assert link["content_hash"] == canonical_content_hash(link)
        assert links[0]["chain_hash"] == chain_hash("", links[0]["content_hash"])
        assert links[1]["chain_hash"] == chain_hash(
            links[0]["chain_hash"], links[1]["content_hash"]
        )
        assert links[2]["chain_hash"] == chain_hash(
            links[1]["chain_hash"], links[2]["content_hash"]
        )

        assert get_integrity(client) == {
            "valid": True,
            "checked_count": 3,
            "broken_policy_rule_id": None,
        }

        # A rule created after migration links to the backfilled tail.
        created = create_rule(client, action_type="read",
                              resource_pattern="res/d", priority=4).json()
        links = get_chain(client)
        assert [link["id"] for link in links] == [rid(1), rid(2), rid(3), created["id"]]
        assert links[-1]["previous_rule_id"] == rid(3)
        assert get_integrity(client)["valid"] is True


def test_complete_chain_is_not_rewritten_on_restart(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path / 'persist.db'}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        for i in range(3):
            create_rule(first, action_type=f"action-{i}", priority=i)
        expected_chain = first.get(CHAIN_PATH).content
        expected_integrity = first.get(INTEGRITY_PATH).content

    with TestClient(app) as second:
        assert second.get(CHAIN_PATH).content == expected_chain
        assert second.get(INTEGRITY_PATH).content == expected_integrity

    # A third startup over the already complete chain is still a no-op.
    with TestClient(app) as third:
        assert third.get(CHAIN_PATH).content == expected_chain
        assert third.get(INTEGRITY_PATH).content == expected_integrity


def test_chain_result_survives_restart(tmp_path, monkeypatch):
    db_path = tmp_path / "persist.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", db_url)

    with TestClient(app) as first:
        created = [
            create_rule(first, action_type=f"action-{i}", priority=i).json()
            for i in range(3)
        ]
        expected = get_chain(first)
        assert get_integrity(first) == {
            "valid": True,
            "checked_count": 3,
            "broken_policy_rule_id": None,
        }

        # Damage the middle rule after the first process checked the chain.
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE policy_rules SET priority = 99 WHERE id = ?",
                (created[1]["id"],),
            )
            conn.commit()
        expected_bad = {
            "valid": False,
            "checked_count": 3,
            "broken_policy_rule_id": created[1]["id"],
        }
        assert get_integrity(first) == expected_bad

    with TestClient(app) as second:
        assert get_integrity(second) == expected_bad
        # The startup backfill never "repairs" the damaged rule: its stored
        # priority and the chain verdict are unchanged.
        [damaged] = [
            link for link in get_chain(second) if link["id"] == created[1]["id"]
        ]
        assert damaged["priority"] == 99


def test_empty_legacy_database_can_create_and_query(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-empty.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE policy_rules (
            id VARCHAR(36) PRIMARY KEY,
            action_type VARCHAR NOT NULL,
            resource_pattern VARCHAR NOT NULL,
            effect VARCHAR NOT NULL,
            priority INTEGER NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            UNIQUE (action_type, resource_pattern, priority)
        )
        """
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("ACCOUNTABILITY_DATABASE_URL", f"sqlite:///{db_path}")

    with TestClient(app) as client:
        assert client.get(CHAIN_PATH).content == b"[]\n"
        assert get_integrity(client) == {
            "valid": True,
            "checked_count": 0,
            "broken_policy_rule_id": None,
        }
        created = create_rule(client).json()
        [link] = get_chain(client)
        assert link["id"] == created["id"]
        assert link["previous_rule_id"] is None
        assert get_integrity(client)["valid"] is True


# --------------------------------------------------------------------------- #
# Existing surfaces keep their shape and semantics
# --------------------------------------------------------------------------- #


def test_listing_keeps_only_existing_visible_fields(client):
    created = create_rule(client, priority=4).json()
    [rule] = client.get("/policy-rules").json()
    assert set(rule.keys()) == set(CONTENT_KEYS)
    assert rule == created


def test_compliance_export_keeps_only_existing_fields(client):
    created = create_rule(client, priority=4).json()
    response = client.get(
        "/policy-rules/compliance-export"
        "?from_created_at=2000-01-01T00:00:00Z&to_created_at=2100-01-01T00:00:00Z"
    )
    assert response.status_code == 200
    [rule] = response.json()["policy_rules"]
    assert set(rule.keys()) == set(CONTENT_KEYS)
    assert rule == created


def test_chain_fields_do_not_change_authorization_evaluation(client):
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

    # The chain and integrity queries never participate in evaluation.
    client.get(CHAIN_PATH)
    client.get(INTEGRITY_PATH)
    decision = client.post(
        f"/machines/{machine_id}/authorization-evaluations",
        json={"action_type": "read", "resource": "res/x"},
    ).json()
    client.get(CHAIN_PATH)
    client.get(INTEGRITY_PATH)

    assert decision == {"allowed": True, "reason": "allowed_by_policy"}


def test_chain_and_listing_stay_consistent(client):
    for i in range(3):
        create_rule(client, action_type=f"action-{i}", priority=i)

    listing = client.get("/policy-rules").json()
    links = get_chain(client)
    assert [link["id"] for link in links] == [rule["id"] for rule in listing]
    for rule, link in zip(listing, links):
        for key in CONTENT_KEYS:
            assert rule[key] == link[key]
