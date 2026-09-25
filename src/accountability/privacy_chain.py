"""Per-machine tamper-evident hash chains for privacy access registrations.

Each machine's privacy access records form an ordered chain, ordered by the
actual UTC instant of ``accessed_at`` and then by ``id`` (an exact-second
stamp sorts before any fractional-second stamp of the same second, which ISO
text ordering alone cannot express):

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the record's seven business fields: ``{id, machine_id, accessed_at,
  window_start, window_end, result, matches_count}``. The chain fields
  themselves are never part of the digest.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first record uses the empty string as the previous chain hash.
* ``previous_access_id`` is ``None`` for a machine's first record and the
  immediately preceding record's id otherwise, so no two records of one
  machine point at the same predecessor and no record points across machines.

A new registration is inserted and linked into the machine's chain inside a
single locked write transaction, so concurrent registrations cannot lose
records, fork the chain, or break a link. Because ``accessed_at`` is
client-supplied, a new record may sort before the current tail; the machine's
links are then recomputed in chain order inside the same transaction, which
is a no-op for the common tail-append case.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry
from .db import Machine, PrivacyAccess

_HASH_LEN = 64

_TABLE = PrivacyAccess.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _accessed_instant(value: object) -> datetime:
    """Parse a stored ``accessed_at`` to its actual UTC instant.

    Registered values always satisfy the RFC 3339 ``Z`` contract, so parsing
    cannot fail for legitimately written rows; a tampered value that no
    longer parses sorts after every parseable record (its content hash will
    not verify anyway) instead of crashing the audit.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order records by the actual UTC instant of ``accessed_at``, then id."""
    return sorted(
        rows,
        key=lambda row: (_accessed_instant(row._mapping["accessed_at"]), row._mapping["id"]),
    )


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    accessed_at: str,
    window_start: str,
    window_end: str,
    result: str,
    matches_count: int,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "accessed_at": accessed_at,
            "window_start": window_start,
            "window_end": window_end,
            "result": result,
            "matches_count": matches_count,
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
        "previous_access_id": "VARCHAR(36)",
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


def _load_records(conn: Connection, machine_id: str | None = None) -> list[Any]:
    statement = _TABLE.select()
    if machine_id is not None:
        statement = statement.where(_TABLE.c.machine_id == machine_id)
    return _chain_order(list(conn.execute(statement)))


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_access_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_access_id"] != previous_access_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_access_id": previous_access_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_access_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def _apply_updates(conn: Connection, rows: list[Any]) -> None:
    for values in _recompute_rows(rows):
        access_id = values.pop("id")
        conn.execute(
            _TABLE.update().where(_TABLE.c.id == access_id).values(**values)
        )


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on records written before the chain feature.

    Processing is per machine in (accessed_at instant, id) order. The
    recomputation is deterministic, so a restart over an already complete
    database issues no writes. Each machine is handled inside a locked
    transaction so a concurrent registration can neither interleave with the
    backfill nor fork.
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
            rows = _load_records(conn, machine_id)
            # previous_access_id is NULL on the first record, so completeness
            # is determined by the two hashes being present everywhere.
            if not rows or any(
                row._mapping["content_hash"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                _apply_updates(conn, rows)

        _run_with_lock_retry(engine, _work)


def append_access(
    engine: Engine,
    *,
    machine_id: str,
    accessed_at: str,
    window_start: str,
    window_end: str,
    result: str,
    matches_count: int,
) -> dict[str, Any]:
    """Atomically register one privacy access and link it into the chain.

    The machine lookup, the duplicate check, the insert, and the chain
    linking all happen inside one locked write transaction, so concurrent
    registrations cannot lose records, fork the per-machine chain, or break
    a link. Returns a status dict: ``not_found`` (machine missing; nothing is
    written), ``duplicate_access`` (same ``(accessed_at, window_start,
    window_end, result)`` already registered for the machine; nothing is
    written), or ``ok`` with the new ``access``.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        duplicate = conn.execute(
            _TABLE.select()
            .where(
                _TABLE.c.machine_id == machine_id,
                _TABLE.c.accessed_at == accessed_at,
                _TABLE.c.window_start == window_start,
                _TABLE.c.window_end == window_end,
                _TABLE.c.result == result,
            )
            .with_only_columns(_TABLE.c.id)
        ).first()
        if duplicate is not None:
            return {"status": "duplicate_access"}

        access_id = str(uuid.uuid4())
        conn.execute(
            _TABLE.insert().values(
                id=access_id,
                machine_id=machine_id,
                accessed_at=accessed_at,
                window_start=window_start,
                window_end=window_end,
                result=result,
                matches_count=matches_count,
                previous_access_id=None,
                content_hash=None,
                chain_hash=None,
            )
        )

        # Link the new row into the machine's chain. ``accessed_at`` is
        # client-supplied, so the row may sort before the current tail;
        # recomputing the machine's links in chain order inside this same
        # transaction keeps every record pointing at its immediate
        # predecessor. When the row sorts last (the common case) the
        # recomputation touches only the new row.
        rows = _load_records(conn, machine_id)
        _apply_updates(conn, rows)

        return {
            "status": "ok",
            "access": {
                "id": access_id,
                "machine_id": machine_id,
                "accessed_at": accessed_at,
                "window_start": window_start,
                "window_end": window_end,
                "result": result,
                "matches_count": matches_count,
            },
        }

    return _run_with_lock_retry(engine, _work)


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's privacy access chain in (accessed_at instant, id)
    order.

    Returns ``(valid, checked_count, broken_access_id)``. The first record
    whose recomputed content hash, previous-access link, or chain hash
    differs from the stored values — a missing predecessor, a cross-machine
    or repeated pointer, an out-of-order link, or a corrupted hash — is
    reported; an empty chain is valid. Only the path machine's records are
    examined.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    previous_access_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_access_id"] != previous_access_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_access_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
