"""Per-machine tamper-evident hash chains for authorization decision events.

Each machine's events form an ordered chain (ordered by ``created_at`` then
``id``):

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the event's own fields.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first event uses the empty string as the previous chain hash.
* ``previous_event_id`` is ``None`` for a machine's first event and the prior
  event's id otherwise.

New events are appended to the chain tail inside a single write transaction
that reads the tail and inserts the row, so concurrent appenders cannot lose
events, fork the chain, or break a link.
"""

import hashlib
import json
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.exc import OperationalError

from .db import (
    AuthorizationDecisionEvent,
    BehaviorDeclaration,
    Machine,
    PolicyRule,
)

_HASH_LEN = 64
_MAX_LOCK_ATTEMPTS = 20

_TABLE = AuthorizationDecisionEvent.__table__
_MACHINE_TABLE = Machine.__table__
_DECLARATION_TABLE = BehaviorDeclaration.__table__
_POLICY_TABLE = PolicyRule.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "action_type",
    "resource",
    "allowed",
    "reason",
    "created_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "action_type": action_type,
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
            "created_at": created_at,
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
        "previous_event_id": "VARCHAR(36)",
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


@contextmanager
def _locked_connection(engine: Engine) -> Iterator[Connection]:
    """A connection inside a write transaction that serializes appenders.

    SQLite's default deferred transactions only take a write lock on first
    write, which lets two appenders read the same tail and fork the chain;
    ``BEGIN IMMEDIATE`` takes the reserved lock up front. Other databases use
    SERIALIZABLE isolation, which abides by the same guarantee.
    """
    if engine.dialect.name == "sqlite":
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("BEGIN IMMEDIATE"))
        try:
            yield conn
            conn.execute(text("COMMIT"))
        except Exception:
            try:
                conn.execute(text("ROLLBACK"))
            except Exception:
                pass
            raise
        finally:
            conn.close()
    else:
        conn = engine.connect().execution_options(isolation_level="SERIALIZABLE")
        try:
            with conn.begin():
                yield conn
        finally:
            conn.close()


def _is_lock_conflict(error: OperationalError) -> bool:
    pgcode = getattr(error.orig, "pgcode", None)
    if pgcode in ("40001", "55P03"):  # serialization failure / lock unavailable
        return True
    args = getattr(error.orig, "args", ())
    if args and args[0] in (1205, 1213):  # MySQL lock wait timeout / deadlock
        return True
    message = str(error.orig).lower()
    return "database is locked" in message or "database table is locked" in message


def _run_with_lock_retry(engine: Engine, work):
    for attempt in range(_MAX_LOCK_ATTEMPTS):
        try:
            with _locked_connection(engine) as conn:
                return work(conn)
        except OperationalError as error:
            if not _is_lock_conflict(error) or attempt == _MAX_LOCK_ATTEMPTS - 1:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.2))


def _load_events(conn: Connection, machine_id: str | None = None) -> list[Any]:
    statement = _TABLE.select()
    if machine_id is not None:
        statement = statement.where(_TABLE.c.machine_id == machine_id)
    statement = statement.order_by(_TABLE.c.created_at, _TABLE.c.id)
    return list(conn.execute(statement))


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_event_id"] != previous_event_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_event_id": previous_event_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_event_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on events written before the chain feature.

    Processing is per machine in (created_at, id) order. The recomputation is
    deterministic, so a restart over an already complete database issues no
    writes. Each machine is handled inside a locked transaction so a
    concurrent appender can neither interleave with the backfill nor fork.
    """
    with engine.connect() as conn:
        machine_ids = [
            row[0]
            for row in conn.execute(
                text(
                    f"SELECT DISTINCT machine_id FROM {_TABLE.name} "
                    "ORDER BY machine_id"
                )
            )
        ]

    for machine_id in machine_ids:
        def _work(conn: Connection, machine_id=machine_id) -> None:
            rows = _load_events(conn, machine_id)
            # previous_event_id is NULL on the first event, so completeness is
            # determined by the two hashes being present everywhere.
            if not rows or any(
                row._mapping["content_hash"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                for values in _recompute_rows(rows):
                    event_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == event_id)
                        .values(**values)
                    )

        _run_with_lock_retry(engine, _work)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pattern_matches(pattern: str, value: str) -> bool:
    regex = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(regex, value, re.DOTALL) is not None


def decide_authorization(
    conn: Connection,
    machine_id: str,
    machine_status: str,
    action_type: str,
    resource: str,
) -> tuple[bool, str]:
    """Compute an authorization decision against state visible in one txn.

    A suspended machine is denied without consulting its behavior declarations or
    the policy rules. The ordering of the checks mirrors the read-only
    evaluation endpoint: suspension short-circuits, then an enabled matching
    declaration is required, then a matching policy rule whose lowest priority
    decides with deny taking precedence. Reading through ``conn`` keeps the
    decision consistent with every other row seen in the same write transaction.
    """
    if machine_status == "suspended":
        return False, "machine_suspended"

    declaration_rows = conn.execute(
        _DECLARATION_TABLE.select().where(
            _DECLARATION_TABLE.c.machine_id == machine_id,
            _DECLARATION_TABLE.c.action_type == action_type,
            _DECLARATION_TABLE.c.enabled.is_(True),
        )
    ).all()
    if not any(
        _pattern_matches(row._mapping["resource_pattern"], resource)
        for row in declaration_rows
    ):
        return False, "no_enabled_declaration"

    rule_rows = conn.execute(
        _POLICY_TABLE.select().where(_POLICY_TABLE.c.action_type == action_type)
    ).all()
    matching = [
        row for row in rule_rows if _pattern_matches(row._mapping["resource_pattern"], resource)
    ]
    if not matching:
        return False, "no_matching_policy"

    lowest = min(row._mapping["priority"] for row in matching)
    decisive = [row for row in matching if row._mapping["priority"] == lowest]
    if any(row._mapping["effect"] == "deny" for row in decisive):
        return False, "denied_by_policy"
    return True, "allowed_by_policy"


def _repair_chain_if_incomplete(conn: Connection, machine_id: str) -> list[Any]:
    """Load the machine's events, rebuilding a chain with missing link data.

    Normally the startup backfill leaves every row complete. If any row is
    missing chain data (e.g. an external writer), rebuild the whole machine
    chain before appending so the new link has a sound tail. Returns the
    reloaded rows.
    """
    rows = _load_events(conn, machine_id)
    if any(
        row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
        for row in rows
    ):
        for values in _recompute_rows(rows):
            event_id = values.pop("id")
            conn.execute(
                _TABLE.update().where(_TABLE.c.id == event_id).values(**values)
            )
        rows = _load_events(conn, machine_id)
    return rows


def _insert_event(
    conn: Connection,
    *,
    machine_id: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
) -> dict[str, Any]:
    """Read the machine's chain tail and insert one linked event.

    Must run inside the locked write transaction. The event id and timestamp
    are minted here and are guaranteed to sort after the current tail in
    (created_at, id) order, so the previous-event link always matches the order
    used by backfill and verification even under same-timestamp concurrency.
    """
    rows = _repair_chain_if_incomplete(conn, machine_id)
    tail = rows[-1] if rows else None

    # Regenerate (rarely) until the new key sorts strictly after the tail.
    created_at = _utc_now_iso()
    event_id = str(uuid.uuid4())
    if tail is not None:
        tail_created_at = tail._mapping["created_at"]
        tail_id = tail._mapping["id"]
        if created_at < tail_created_at:
            created_at = tail_created_at
        while created_at == tail_created_at and event_id <= tail_id:
            event_id = str(uuid.uuid4())

    if tail is None:
        previous_event_id = None
        previous_chain_hash = ""
    else:
        previous_event_id = tail._mapping["id"]
        previous_chain_hash = tail._mapping["chain_hash"]

    content_hash = compute_content_hash(
        id=event_id,
        machine_id=machine_id,
        action_type=action_type,
        resource=resource,
        allowed=allowed,
        reason=reason,
        created_at=created_at,
    )
    chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
    conn.execute(
        _TABLE.insert().values(
            id=event_id,
            machine_id=machine_id,
            action_type=action_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
            created_at=created_at,
            previous_event_id=previous_event_id,
            content_hash=content_hash,
            chain_hash=chain_hash,
        )
    )
    return {
        "id": event_id,
        "machine_id": machine_id,
        "action_type": action_type,
        "resource": resource,
        "allowed": allowed,
        "reason": reason,
        "created_at": created_at,
        "previous_event_id": previous_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }


def append_event(
    engine: Engine,
    *,
    machine_id: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
) -> dict[str, Any]:
    """Atomically append one event with a pre-computed result to its chain tail."""

    def _work(conn: Connection) -> dict[str, Any]:
        return _insert_event(
            conn,
            machine_id=machine_id,
            action_type=action_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
        )

    return _run_with_lock_retry(engine, _work)


def append_decision_event(
    engine: Engine,
    *,
    machine_id: str,
    action_type: str,
    resource: str,
) -> dict[str, Any]:
    """Decide one authorization request and append its event atomically.

    The machine lookup, the suspended-status check, the declaration/policy
    decision, the chain-tail read, and the event insert all run in one locked
    write transaction. The same lock serializes machine status changes, so a
    concurrent status change and decision event have a single definite serial
    order: when the status change commits first the event is decided against
    the new status, and when the event commits first a later status change can
    never rewrite it. Returns a status dict:

    * ``not_found`` — the machine does not exist (nothing is written);
    * ``ok`` — with the appended ``event`` record carrying the decision made
      against the machine state read in this same transaction.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine_row is None:
            return {"status": "not_found"}

        allowed, reason = decide_authorization(
            conn,
            machine_id,
            machine_row._mapping["status"],
            action_type,
            resource,
        )
        event = _insert_event(
            conn,
            machine_id=machine_id,
            action_type=action_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
        )
        return {"status": "ok", "event": event}

    return _run_with_lock_retry(engine, _work)


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's chain in (created_at, id) order.

    Returns ``(valid, checked_count, broken_event_id)``. The first event whose
    recomputed content hash, previous-event link, or chain hash differs from
    the stored values is reported; an empty chain is valid.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(_TABLE.c.machine_id == machine_id)
            .order_by(_TABLE.c.created_at, _TABLE.c.id)
        )
    )

    previous_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_event_id"] != previous_event_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_event_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
