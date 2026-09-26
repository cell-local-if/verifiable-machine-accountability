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
import re
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


def _find_duplicate(conn: Connection, machine_id: str, item: dict[str, Any]):
    return conn.execute(
        _TABLE.select()
        .where(
            _TABLE.c.machine_id == machine_id,
            _TABLE.c.accessed_at == item["accessed_at"],
            _TABLE.c.window_start == item["window_start"],
            _TABLE.c.window_end == item["window_end"],
            _TABLE.c.result == item["result"],
        )
        .with_only_columns(_TABLE.c.id)
    ).first()


def _insert_access(
    conn: Connection, machine_id: str, item: dict[str, Any]
) -> dict[str, Any]:
    access_id = str(uuid.uuid4())
    conn.execute(
        _TABLE.insert().values(
            id=access_id,
            machine_id=machine_id,
            accessed_at=item["accessed_at"],
            window_start=item["window_start"],
            window_end=item["window_end"],
            result=item["result"],
            matches_count=item["matches_count"],
            previous_access_id=None,
            content_hash=None,
            chain_hash=None,
        )
    )
    return {
        "id": access_id,
        "machine_id": machine_id,
        "accessed_at": item["accessed_at"],
        "window_start": item["window_start"],
        "window_end": item["window_end"],
        "result": item["result"],
        "matches_count": item["matches_count"],
    }


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

        item = {
            "accessed_at": accessed_at,
            "window_start": window_start,
            "window_end": window_end,
            "result": result,
            "matches_count": matches_count,
        }
        if _find_duplicate(conn, machine_id, item) is not None:
            return {"status": "duplicate_access"}

        access = _insert_access(conn, machine_id, item)

        # Link the new row into the machine's chain. ``accessed_at`` is
        # client-supplied, so the row may sort before the current tail;
        # recomputing the machine's links in chain order inside this same
        # transaction keeps every record pointing at its immediate
        # predecessor. When the row sorts last (the common case) the
        # recomputation touches only the new row.
        rows = _load_records(conn, machine_id)
        _apply_updates(conn, rows)

        return {"status": "ok", "access": access}

    return _run_with_lock_retry(engine, _work)


def append_accesses_batch(
    engine: Engine, *, machine_id: str, items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Register a whole batch of privacy accesses in one locked transaction.

    The batch goes through the same lock primitive as :func:`append_access`,
    so a batch and single registrations are fully serialized against each
    other. The machine is looked up once; every item then reuses the existing
    duplicate check and insert in request-array order. The duplicate SELECT
    runs on the batch's own connection, so a prior item inserted earlier in
    this same transaction is visible: the first item of a given access
    identity registers and later identical items (same ``(accessed_at,
    window_start, window_end, result)``; the hit count never participates)
    come back as ``duplicate_access`` without a new row, while the other items
    still register. A duplicate never aborts the batch.

    All inserts and the chain relink commit together or not at all: a
    persistence failure rolls the single transaction back and leaves no
    partial records. Returns ``{"status": "not_found"}`` for a missing
    machine, otherwise ``{"status": "ok", "outcomes": [...]}`` with one
    ``{"status": "duplicate_access"}`` or ``{"status": "ok", "access": ...}``
    entry per item, in request order.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        outcomes: list[dict[str, Any]] = []
        for item in items:
            if _find_duplicate(conn, machine_id, item) is not None:
                outcomes.append({"status": "duplicate_access"})
                continue
            access = _insert_access(conn, machine_id, item)
            outcomes.append({"status": "ok", "access": access})

        # Link every new row into the machine's chain once, in chain order.
        # As with the single path, client-supplied ``accessed_at`` values may
        # sort before the current tail; recomputing inside this same
        # transaction keeps every record pointing at its immediate predecessor.
        rows = _load_records(conn, machine_id)
        _apply_updates(conn, rows)

        return {"status": "ok", "outcomes": outcomes}

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


# --- read-only per-record chain diagnostics ---------------------------------
#
# The seven stable per-record anomaly codes, emitted on each record in this
# fixed order (the order the codes are documented in). A sound record carries
# an empty array.
ERROR_MISSING_PREVIOUS = "missing_previous"
ERROR_BAD_PREVIOUS = "bad_previous"
ERROR_BAD_CONTENT_HASH = "bad_content_hash"
ERROR_BAD_CHAIN_HASH = "bad_chain_hash"
ERROR_BAD_TIME = "bad_time"
ERROR_BAD_ID = "bad_id"
ERROR_BAD_OWNERSHIP = "bad_ownership"

_ERROR_CODE_ORDER = (
    ERROR_MISSING_PREVIOUS,
    ERROR_BAD_PREVIOUS,
    ERROR_BAD_CONTENT_HASH,
    ERROR_BAD_CHAIN_HASH,
    ERROR_BAD_TIME,
    ERROR_BAD_ID,
    ERROR_BAD_OWNERSHIP,
)

# A stored privacy access id is a 36-character UUID string; a tampered id of
# another shape is a record-identifier anomaly rather than a crash.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# RFC 3339 date-time in UTC with a literal ``Z`` suffix.
_RFC3339_Z_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def _is_parseable_stamp(value: object) -> bool:
    """Whether a stored ``accessed_at`` is a parseable RFC 3339 ``Z`` instant."""
    if not isinstance(value, str) or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _diagnostic_order(rows: list[Any]) -> list[Any]:
    """Chain order for a diagnostic scan, tolerant of a damaged identifier.

    Same (accessed-at instant, id) ordering as :func:`_chain_order`, but a
    stored id that is no longer a string never crashes the comparison: such
    ids deterministically sort after string ids within one instant (stable
    sort keeps database order among them), and damaged stamps already sort
    after every parseable instant.
    """
    def key(row: Any) -> tuple[Any, int, str]:
        record_id = row._mapping["id"]
        instant = _accessed_instant(row._mapping["accessed_at"])
        if isinstance(record_id, str):
            return (instant, 0, record_id)
        return (instant, 1, "")

    return sorted(rows, key=key)


def diagnose_chain(session, machine_id: str) -> dict[str, Any]:
    """Diagnose one machine's privacy access chain, record by record, read-only.

    Only the path machine's records are examined, in the same chain order used
    to build and verify the chain: the actual UTC instant of ``accessed_at``
    and then ``id``. A stored ``accessed_at`` that no longer parses sorts after
    every parseable record (the tolerant audit convention) instead of crashing
    and is flagged on that record; other damaged stored values are judged as
    anomalies, never repaired or normalized.

    Every record is returned — including records after the first anomaly. The
    overall ``valid`` flag is false once any record carries at least one
    anomaly; each record's ``errors`` array keeps every anomaly that applies
    to it, in the fixed error-code order. Content and chain digests follow the
    existing chain rules exactly: the content digest is the SHA-256 of the
    compact key-sorted JSON of the seven business fields, and the chain digest
    is ``sha256("<previous expected chain digest>:<content digest>")`` running
    from the empty prefix in chain order, so a damaged record also breaks the
    expected chain continuation for its successors.

    Returns ``{"machine_id", "valid", "checked_count", "records"}``; each
    record is ``{"id", "position", "previous_access_id", "content_hash",
    "chain_hash", "errors"}`` with ``position`` numbered from one. Issues no
    writes.
    """
    rows = _diagnostic_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    records: list[dict[str, Any]] = []
    overall_valid = True
    previous_chain_hash = ""
    seen_ids: set[str] = set()

    for position, row in enumerate(rows, start=1):
        mapping = row._mapping
        record_id = mapping["id"]
        stored_previous = mapping["previous_access_id"]
        stored_content_hash = mapping["content_hash"]
        stored_chain_hash = mapping["chain_hash"]

        errors: list[str] = []

        # Ownership: a row whose stored machine id no longer names the path
        # machine does not belong to this machine's chain.
        if mapping["machine_id"] != machine_id:
            errors.append(ERROR_BAD_OWNERSHIP)

        # Record identifier: a UUID-shaped string, unique within the machine.
        if not isinstance(record_id, str) or not _UUID_RE.fullmatch(record_id):
            errors.append(ERROR_BAD_ID)
        elif record_id in seen_ids:
            errors.append(ERROR_BAD_ID)
        if isinstance(record_id, str):
            seen_ids.add(record_id)

        # Access instant: must parse as an RFC 3339 Z date-time. Damaged
        # stamps sort last via the tolerant ordering; flag them here too.
        if not _is_parseable_stamp(mapping["accessed_at"]):
            errors.append(ERROR_BAD_TIME)

        # Predecessor link: null only on the first record; every later record
        # must name exactly its immediate predecessor in chain order. A null
        # link on a non-first record is a missing predecessor; any other
        # mismatch (foreign, repeated, skipped, or malformed pointer) is a
        # wrong predecessor. The two cannot co-occur on one record.
        if position == 1:
            if stored_previous is not None:
                errors.append(ERROR_BAD_PREVIOUS)
        else:
            expected_previous_id = records[-1]["id"]
            if stored_previous is None:
                errors.append(ERROR_MISSING_PREVIOUS)
            elif stored_previous != expected_previous_id:
                errors.append(ERROR_BAD_PREVIOUS)

        # Content digest over the seven stored business fields, then the chain
        # digest running from the empty prefix in chain order — the same
        # recomputation the integrity audit uses. A damaged business field
        # (e.g. a non-string or non-integer stored value) makes the digest
        # uncomputable; that is a content-digest anomaly, never a crash, and
        # such a record also cannot extend the chain continuation.
        try:
            expected_content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
        except (TypeError, ValueError):
            expected_content_hash = None
        if (
            expected_content_hash is None
            or stored_content_hash != expected_content_hash
        ):
            errors.append(ERROR_BAD_CONTENT_HASH)
        if expected_content_hash is not None:
            expected_chain_hash = compute_chain_hash(
                previous_chain_hash, expected_content_hash
            )
        else:
            expected_chain_hash = None
        if (
            expected_chain_hash is None
            or stored_chain_hash != expected_chain_hash
        ):
            errors.append(ERROR_BAD_CHAIN_HASH)
        previous_chain_hash = (
            expected_chain_hash if expected_chain_hash is not None else ""
        )

        errors = [code for code in _ERROR_CODE_ORDER if code in errors]
        if errors:
            overall_valid = False

        records.append(
            {
                "id": record_id,
                "position": position,
                # Emitted exactly as stored (including a damaged null on a
                # non-first record); only a sound first record is null.
                "previous_access_id": stored_previous,
                "content_hash": stored_content_hash,
                "chain_hash": stored_chain_hash,
                "errors": errors,
            }
        )

    return {
        "machine_id": machine_id,
        "valid": overall_valid,
        "checked_count": len(records),
        "records": records,
    }
