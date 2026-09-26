"""Per-machine tamper-evident hash chains for incident status events.

Each machine's incident status transition records form an ordered chain,
ordered by the actual UTC instant of ``created_at`` and then by ``id``,
following the same rules as the authorization decision event chain:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the record's own fields: ``{id, machine_id, event_id, incident_id,
  from_status, to_status, created_at}`` — the record, its ownership, the
  status edge, and the creation moment.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first record uses the empty string as the previous chain hash.
* ``previous_status_event_id`` is ``None`` for a machine's first record and
  the prior record's id otherwise.

A successful incident status transition updates the incident row and appends
the linked history record inside a single locked write transaction, so the
status, the history record, and its chain link commit together or not at
all, and concurrent transitions cannot lose records, fork the chain, or
break a link.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import IncidentStatusEvent

_HASH_LEN = 64

_TABLE = IncidentStatusEvent.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "from_status",
    "to_status",
    "created_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    event_id: str,
    incident_id: str,
    from_status: str,
    to_status: str,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "event_id": event_id,
            "incident_id": incident_id,
            "from_status": from_status,
            "to_status": to_status,
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
        "previous_status_event_id": "VARCHAR(36)",
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


_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a damaged value that no longer parses sorts
    after every parseable record (its content hash cannot verify anyway)
    instead of crashing the read-only audit.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _created_at_parseable(value: object) -> bool:
    """Whether a stored ``created_at`` still parses to a UTC instant."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
            return True
        except ValueError:
            pass
    return False


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order status-event rows by the actual UTC instant of ``created_at``.

    Ordering is by the parsed UTC instant and then by ``id``, so an
    exact-second stamp sorts before any fractional-second stamp of the same
    second (ISO text alone cannot express that, since ``.`` precedes ``Z``).
    A stamp that no longer parses sorts deterministically last, and a damaged
    non-string id sorts as empty rather than crashing the comparison.
    """
    return sorted(
        rows,
        key=lambda row: (
            _created_instant(row._mapping["created_at"]),
            row._mapping["id"] if isinstance(row._mapping["id"], str) else "",
        ),
    )


def _load_records(conn: Connection, machine_id: str) -> list[Any]:
    """Load one machine's status events in chain (instant, id) order."""
    rows = list(
        conn.execute(_TABLE.select().where(_TABLE.c.machine_id == machine_id))
    )
    return _chain_order(rows)


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_status_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_status_event_id"] != previous_status_event_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_status_event_id": previous_status_event_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_status_event_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on records written before the chain feature.

    Processing is per machine in (created-at instant, id) order. The
    recomputation is deterministic, so a restart over an already complete
    database issues no writes. Each machine is handled inside a locked
    transaction so a concurrent writer can neither interleave with the
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
            # previous_status_event_id is NULL on the first record, so
            # completeness is determined by the two hashes being present
            # everywhere.
            if not rows or any(
                row._mapping["content_hash"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                for values in _recompute_rows(rows):
                    status_event_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == status_event_id)
                        .values(**values)
                    )

        _run_with_lock_retry(engine, _work)


def mint_tail_link(
    conn: Connection,
    *,
    machine_id: str,
    event_id: str,
    incident_id: str,
    from_status: str,
    to_status: str,
) -> dict[str, Any]:
    """Read the machine's chain tail and insert one linked status event.

    Must run inside the locked write transaction (the same transaction that
    updates the incident's status), so the status, the history record, and
    its chain link commit together or not at all. The record id and timestamp
    are minted here and are guaranteed to sort strictly after the current
    tail in (created-at instant, id) order, so the previous-record link
    always matches the order used by backfill and verification even under
    same-timestamp concurrency.
    """
    rows = _load_records(conn, machine_id)
    # Normally the startup backfill leaves every row complete. If any row is
    # missing chain data (e.g. an external writer), rebuild the whole machine
    # chain before appending so the new link has a sound tail.
    if any(
        row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
        for row in rows
    ):
        for values in _recompute_rows(rows):
            status_event_id = values.pop("id")
            conn.execute(
                _TABLE.update()
                .where(_TABLE.c.id == status_event_id)
                .values(**values)
            )
        rows = _load_records(conn, machine_id)

    tail = rows[-1] if rows else None

    # Regenerate (rarely) until the new key sorts strictly after the tail. A
    # damaged tail stamp cannot be parsed to an instant, so it is never
    # adopted for the new record.
    created_at = _utc_now_iso()
    status_event_id = str(uuid.uuid4())
    if tail is not None:
        tail_created_at = tail._mapping["created_at"]
        tail_id = tail._mapping["id"]
        if _created_at_parseable(tail_created_at):
            tail_instant = _created_instant(tail_created_at)
            if _created_instant(created_at) < tail_instant:
                created_at = tail_created_at
            if _created_instant(created_at) == tail_instant:
                while status_event_id <= tail_id:
                    status_event_id = str(uuid.uuid4())

    if tail is None:
        previous_status_event_id = None
        previous_chain_hash = ""
    else:
        previous_status_event_id = tail._mapping["id"]
        previous_chain_hash = tail._mapping["chain_hash"]

    content_hash = compute_content_hash(
        id=status_event_id,
        machine_id=machine_id,
        event_id=event_id,
        incident_id=incident_id,
        from_status=from_status,
        to_status=to_status,
        created_at=created_at,
    )
    chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
    conn.execute(
        _TABLE.insert().values(
            id=status_event_id,
            machine_id=machine_id,
            event_id=event_id,
            incident_id=incident_id,
            from_status=from_status,
            to_status=to_status,
            created_at=created_at,
            previous_status_event_id=previous_status_event_id,
            content_hash=content_hash,
            chain_hash=chain_hash,
        )
    )
    return {
        "id": status_event_id,
        "machine_id": machine_id,
        "event_id": event_id,
        "incident_id": incident_id,
        "from_status": from_status,
        "to_status": to_status,
        "created_at": created_at,
        "previous_status_event_id": previous_status_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }


def _first_unparseable_created_at_id(rows: list[Any]) -> str | None:
    """Id of the first row in chain order whose ``created_at`` cannot parse.

    A corrupted stamp sorts its record after every parseable one, so the
    record's chain successor would otherwise be blamed for the broken link;
    reporting the corrupted record itself keeps the audit pointed at the
    actual damage. ``None`` when every row parses.
    """
    for row in rows:
        if not _created_at_parseable(row._mapping["created_at"]):
            return row._mapping["id"]
    return None


def verify_machine_chain(
    session, machine_id: str
) -> tuple[bool, int, str | None]:
    """Read-only verification of one machine's status-event chain for audit.

    Tolerant of damaged stored values so a corrupted ``created_at``, ``id``,
    digest, or reference never crashes the query, is never repaired or
    recomputed for storage, and never removes the record from the total.
    Returns ``(valid, checked_count, broken_status_event_id)``.

    Records are examined in the order of the actual UTC instant of
    ``created_at`` and then ``id`` (an exact-second stamp precedes any
    fractional-second stamp of the same second). A record whose ``created_at``
    no longer parses still enters the total and is itself reported as the
    first broken record, instead of crashing the scan or blaming its chain
    successor. The first record's previous-status-event id must be empty;
    every later record's must be the id of the immediately preceding record,
    and its stored content and chain digests must match the digests
    recomputed under the public creation-time rules — nothing is written back
    when they do not. The first mismatch sets the broken id; later records
    cannot change it. Only rows whose stored ``machine_id`` equals the path
    machine are examined, so another machine's damaged records never change
    this result.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    corrupted_id = _first_unparseable_created_at_id(rows)
    if corrupted_id is not None:
        return False, len(rows), corrupted_id

    previous_status_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        try:
            content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
            chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        except (TypeError, ValueError):
            # A damaged stored value (e.g. a non-text field) cannot produce
            # the published digest; the record is broken, not a reason to
            # crash the read-only audit.
            return False, len(rows), mapping["id"]
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_status_event_id"] != previous_status_event_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_status_event_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
