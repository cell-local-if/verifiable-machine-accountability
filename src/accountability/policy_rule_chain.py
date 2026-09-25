"""Global tamper-evident hash chain for policy rules.

Unlike the per-machine chains, every global policy rule belongs to one chain
spanning the whole table. Rules are ordered by the actual UTC instant of
``created_at`` and then by ``id`` (an exact-second stamp sorts before any
fractional-second stamp of the same second, which ISO text ordering alone
cannot express):

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the rule's seven visible fields: ``{id, action_type,
  resource_pattern, effect, priority, created_at, updated_at}``. The chain
  fields themselves are never part of the digest.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first rule uses the empty string as the previous chain hash.
* ``previous_rule_id`` is ``None`` for the first rule and the immediately
  preceding rule's id otherwise, so no two rules point at the same
  predecessor and the links never skip.

A new rule is appended to the chain tail inside a single locked write
transaction together with the duplicate business-identity check, so
concurrent creations cannot lose rules, fork the chain, skip a link, or point
two rules at the same predecessor. A failed creation (invalid body or a
duplicate ``(action_type, resource_pattern, priority)`` triple) writes
nothing: no rule row and no chain link.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import PolicyRule

_HASH_LEN = 64

_TABLE = PolicyRule.__table__
_CONTENT_COLUMNS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a tampered value that no longer parses sorts
    after every parseable record (its position in the chain simply moves last)
    instead of crashing the chain scan, the listing, or the backfill.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order rules by the actual UTC instant of ``created_at``, then id."""
    return sorted(
        rows,
        key=lambda row: (
            _created_instant(row._mapping["created_at"]),
            row._mapping["id"],
        ),
    )


def ordered_rules(session) -> list[Any]:
    """Load every policy rule in chain order.

    Ordering is by the actual UTC instant of ``created_at`` and then ``id`` so
    an exact-second rule sorts before any fractional-second rule of the same
    second (ISO text alone cannot express that). A stored timestamp that no
    longer parses sorts last rather than raising, so a read-only query or the
    audit never crashes on one damaged record. Read-only.
    """
    return _chain_order(list(session.execute(_TABLE.select())))


def compute_content_hash(
    *,
    id: str,
    action_type: str,
    resource_pattern: str,
    effect: str,
    priority: int,
    created_at: str,
    updated_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "action_type": action_type,
            "resource_pattern": resource_pattern,
            "effect": effect,
            "priority": priority,
            "created_at": created_at,
            "updated_at": updated_at,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def compute_chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    message = f"{previous_chain_hash}:{content_hash}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def migrate_schema(engine: Engine) -> None:
    """Add chain columns to databases created before the chain feature."""
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "previous_rule_id": "VARCHAR(36)",
        "content_hash": f"VARCHAR({_HASH_LEN})",
        "chain_hash": f"VARCHAR({_HASH_LEN})",
    }
    with engine.begin() as conn:
        for name, column_type in additions.items():
            if name not in existing:
                conn.execute(
                    text(
                        f"ALTER TABLE {_TABLE.name} ADD COLUMN {name} {column_type}"
                    )
                )


def _load_rules(conn: Connection) -> list[Any]:
    return _chain_order(list(conn.execute(_TABLE.select())))


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_rule_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_rule_id"] != previous_rule_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_rule_id": previous_rule_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_rule_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def _apply_updates(conn: Connection, rows: list[Any]) -> None:
    for values in _recompute_rows(rows):
        rule_id = values.pop("id")
        conn.execute(
            _TABLE.update().where(_TABLE.c.id == rule_id).values(**values)
        )


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on rules written before the chain feature.

    The whole global table is processed in (created_at instant, id) order
    inside one locked write transaction. The recomputation is deterministic,
    so a restart over an already complete database issues no writes and a
    concurrent rule creation can neither interleave with the backfill nor
    fork the chain.
    """

    def _work(conn: Connection) -> None:
        rows = _load_rules(conn)
        # previous_rule_id is NULL on the first rule, so completeness is
        # determined by the two hashes being present everywhere.
        if not rows or any(
            row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
            for row in rows
        ):
            _apply_updates(conn, rows)

    _run_with_lock_retry(engine, _work)


def append_policy_rule(
    engine: Engine,
    *,
    action_type: str,
    resource_pattern: str,
    effect: str,
    priority: int,
) -> dict[str, Any]:
    """Create one global policy rule and append it to the chain tail.

    The duplicate ``(action_type, resource_pattern, priority)`` check, the
    tail read, and the insert all happen inside one locked write transaction,
    so concurrent creations cannot lose rules, fork the chain, skip a link,
    or point two rules at the same predecessor. Returns a status dict:
    ``duplicate_policy_rule`` (the same business triple already exists;
    nothing is written) or ``ok`` with the new ``rule`` record including its
    ``previous_rule_id``, ``content_hash``, and ``chain_hash``.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        duplicate = conn.execute(
            _TABLE.select().where(
                _TABLE.c.action_type == action_type,
                _TABLE.c.resource_pattern == resource_pattern,
                _TABLE.c.priority == priority,
            )
        ).first()
        if duplicate is not None:
            return {"status": "duplicate_policy_rule"}

        rows = _load_rules(conn)

        # Normally the startup backfill leaves every row complete. If any row
        # is missing chain data (e.g. an external writer), rebuild the whole
        # chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
            for row in rows
        ):
            _apply_updates(conn, rows)
            rows = _load_rules(conn)

        tail = rows[-1] if rows else None

        # Regenerate (rarely) until the new key sorts strictly after the tail
        # in (created_at instant, id) chain order. Comparison is by parsed
        # instant, not text, so an exact-second and fractional spelling of the
        # same instant still tie-break on id deterministically.
        created_at = _utc_now_iso()
        rule_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_key = (_created_instant(tail_created_at), tail._mapping["id"])
            while (_created_instant(created_at), rule_id) <= tail_key:
                if _created_instant(created_at) < tail_key[0]:
                    created_at = tail_created_at
                else:
                    rule_id = str(uuid.uuid4())

        if tail is None:
            previous_rule_id = None
            previous_chain_hash = ""
        else:
            previous_rule_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        content_hash = compute_content_hash(
            id=rule_id,
            action_type=action_type,
            resource_pattern=resource_pattern,
            effect=effect,
            priority=priority,
            created_at=created_at,
            updated_at=created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        conn.execute(
            _TABLE.insert().values(
                id=rule_id,
                action_type=action_type,
                resource_pattern=resource_pattern,
                effect=effect,
                priority=priority,
                created_at=created_at,
                updated_at=created_at,
                previous_rule_id=previous_rule_id,
                content_hash=content_hash,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "rule": {
                "id": rule_id,
                "action_type": action_type,
                "resource_pattern": resource_pattern,
                "effect": effect,
                "priority": priority,
                "created_at": created_at,
                "updated_at": created_at,
                "previous_rule_id": previous_rule_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        }

    return _run_with_lock_retry(engine, _work)


def verify_chain(session) -> tuple[bool, int, str | None]:
    """Verify the global policy-rule chain in (created_at instant, id) order.

    Returns ``(valid, checked_count, broken_policy_rule_id)``. Rules are
    scanned in chain order — an exact-second stamp before any
    fractional-second stamp of the same second, and a damaged stamp after
    every parseable instant, but still counted in the total. The first rule
    whose recomputed content hash, previous-rule link, or chain hash differs
    from the stored chain values is reported by its stored id; an unparseable
    ``created_at`` alone never crashes the scan and the rule is judged by its
    chain values like every other row. An empty table is valid with a zero
    count. Read-only: nothing is normalized, repaired, or recomputed back to
    the table.
    """
    rows = ordered_rules(session)

    previous_rule_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_rule_id"] != previous_rule_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_rule_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
