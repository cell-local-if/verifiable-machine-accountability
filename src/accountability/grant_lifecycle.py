"""Per-machine tamper-evident audit chain of grant lifecycle events.

Every successful authorization-grant action — issue, the single consumption,
the one emergency revocation — appends exactly one immutable lifecycle event
inside the same locked write transaction that performs the action, so the
action and its audit record commit together or leave no trace. A rejected
attempt (missing machine or grant, terminal-state conflict, expiry, race
loser) writes no event, and a repeated consumption or revocation has exactly
one terminal winner, so at most one ``consumed`` or ``revoked`` event can
ever exist for a grant.

Each machine's events form an ordered chain, ordered by the actual UTC
instant of ``occurred_at`` and then by ``id``, following the same rules as
the other per-machine chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the event's own fields: ``{id, machine_id, grant_id,
  authorization_event_id, type, occurred_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first event uses the empty string as the previous chain hash.
* ``previous_event_id`` is ``None`` for a machine's first event and the prior
  event's id otherwise.

``occurred_at`` reuses the exact moment the successful response already
carries (``issued_at`` / ``consumed_at`` / ``revoked_at``); it is never
re-minted or rewritten. Events are isolated per machine: one machine's chain
never contains another machine's events. The table is created at startup on
databases that predate the feature; historical grants are never rewritten
and no events are fabricated for them.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .db import AuthorizationGrantLifecycleEvent

_HASH_LEN = 64

_TABLE = AuthorizationGrantLifecycleEvent.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "grant_id",
    "authorization_event_id",
    "type",
    "occurred_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    grant_id: str,
    authorization_event_id: str,
    type: str,
    occurred_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "grant_id": grant_id,
            "authorization_event_id": authorization_event_id,
            "type": type,
            "occurred_at": occurred_at,
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
    """Add missing chain columns to a table created by an earlier schema.

    ``Base.metadata.create_all`` already creates the table on databases that
    predate the feature; this hook only brings a pre-existing partial table
    forward. Existing rows are never rewritten, and no events are fabricated
    for historical grants.
    """
    inspector = inspect(engine)
    if _TABLE.name not in inspector.get_table_names():
        # ``create_all`` builds a current-schema table; there is nothing to
        # bring forward.
        return
    existing = {
        column["name"] for column in inspector.get_columns(_TABLE.name)
    }
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


_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def occurred_at_instant(value: object) -> datetime:
    """Parse a stored ``occurred_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a damaged value that no longer parses sorts
    after every parseable record (its content hash cannot verify anyway)
    instead of crashing the read-only audit or query.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _occurred_at_parseable(value: object) -> bool:
    """Whether a stored ``occurred_at`` still parses to a UTC instant."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
            return True
        except ValueError:
            pass
    return False


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order lifecycle rows by the actual UTC instant of ``occurred_at``.

    Ordering is by the parsed UTC instant and then by ``id``, so an
    exact-second stamp sorts before any fractional-second stamp of the same
    second (ISO text alone cannot express that, since ``.`` precedes ``Z``).
    A stamp that no longer parses sorts deterministically last, and a damaged
    non-string id sorts as empty rather than crashing the comparison.
    """
    return sorted(
        rows,
        key=lambda row: (
            occurred_at_instant(row._mapping["occurred_at"]),
            row._mapping["id"] if isinstance(row._mapping["id"], str) else "",
        ),
    )


def _load_records(conn: Connection, machine_id: str) -> list[Any]:
    """Load one machine's lifecycle events in chain (instant, id) order."""
    rows = list(
        conn.execute(_TABLE.select().where(_TABLE.c.machine_id == machine_id))
    )
    return _chain_order(rows)


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


def append_event(
    conn: Connection,
    *,
    machine_id: str,
    grant_id: str,
    authorization_event_id: str,
    type: str,
    occurred_at: str,
) -> dict[str, Any]:
    """Append one linked lifecycle event to the machine's chain tail.

    Must run inside the locked write transaction that performs the grant
    action, so the action, the audit event, and its chain link commit
    together or leave no trace. ``occurred_at`` is the exact moment the
    successful response already carries and is never adjusted: the action's
    moment is minted inside this same locked transaction, so events commit in
    non-decreasing (``occurred_at``, ``id``) order. The event id is minted
    here and is regenerated (rarely) until the new key sorts strictly after
    the current tail, so the previous-event link always matches the order
    used by verification even under same-instant actions.
    """
    rows = _load_records(conn, machine_id)
    # Every row this feature writes carries its hashes. If any row is missing
    # chain data (e.g. an external writer), rebuild the whole machine chain
    # before appending so the new link has a sound tail.
    if any(
        row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
        for row in rows
    ):
        for values in _recompute_rows(rows):
            event_id = values.pop("id")
            conn.execute(
                _TABLE.update().where(_TABLE.c.id == event_id).values(**values)
            )
        rows = _load_records(conn, machine_id)

    tail = rows[-1] if rows else None

    event_id = str(uuid.uuid4())
    if tail is not None:
        tail_id = tail._mapping["id"]
        # Same-instant actions (coarse clock) must still sort strictly after
        # the tail; the id is the only degree of freedom, since occurred_at
        # is fixed by the response contract.
        if occurred_at == tail._mapping["occurred_at"]:
            while event_id <= tail_id:
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
        grant_id=grant_id,
        authorization_event_id=authorization_event_id,
        type=type,
        occurred_at=occurred_at,
    )
    chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
    conn.execute(
        _TABLE.insert().values(
            id=event_id,
            machine_id=machine_id,
            grant_id=grant_id,
            authorization_event_id=authorization_event_id,
            type=type,
            occurred_at=occurred_at,
            previous_event_id=previous_event_id,
            content_hash=content_hash,
            chain_hash=chain_hash,
        )
    )
    return {
        "id": event_id,
        "machine_id": machine_id,
        "grant_id": grant_id,
        "authorization_event_id": authorization_event_id,
        "type": type,
        "occurred_at": occurred_at,
        "previous_event_id": previous_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }


def _first_unparseable_occurred_at_id(rows: list[Any]) -> str | None:
    """Id of the first row in chain order whose ``occurred_at`` cannot parse.

    A corrupted stamp sorts its record after every parseable one, so the
    record's chain successor would otherwise be blamed for the broken link;
    reporting the corrupted record itself keeps the audit pointed at the
    actual damage. ``None`` when every row parses.
    """
    for row in rows:
        if not _occurred_at_parseable(row._mapping["occurred_at"]):
            return row._mapping["id"]
    return None


def verify_machine_chain(
    session, machine_id: str
) -> tuple[bool, int, str | None]:
    """Read-only verification of one machine's lifecycle chain for audit.

    Tolerant of damaged stored values so a corrupted ``occurred_at``, ``id``,
    digest, or reference never crashes the query, is never repaired or
    recomputed for storage, and never removes the record from the total.
    Returns ``(valid, checked_count, broken_event_id)``.

    Records are examined in the order of the actual UTC instant of
    ``occurred_at`` and then ``id`` (an exact-second stamp precedes any
    fractional-second stamp of the same second). A record whose
    ``occurred_at`` no longer parses still enters the total and is itself
    reported as the first broken record, instead of crashing the scan or
    blaming its chain successor. The first record's previous-event id must be
    empty; every later record's must be the id of the immediately preceding
    record, and its stored content and chain digests must match the digests
    recomputed under the public append-time rules — nothing is written back
    when they do not. The first mismatch sets the broken id; later records
    cannot change it. Only rows whose stored ``machine_id`` equals the path
    machine are examined, so another machine's damaged records never change
    this result. An empty chain is valid.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    corrupted_id = _first_unparseable_occurred_at_id(rows)
    if corrupted_id is not None:
        return False, len(rows), corrupted_id

    previous_event_id: str | None = None
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
            or mapping["previous_event_id"] != previous_event_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_event_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None


# Strict RFC 3339 UTC ``Z`` stamp contract for the compliance-export window:
# a stored ``occurred_at`` that does not satisfy it exactly (missing, non-text,
# offset form, malformed, or out-of-range) is excluded from the window rather
# than admitted or repaired.
_UTC_Z_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _parse_occurred_at_utc(value: object) -> datetime | None:
    """Parse a stored ``occurred_at`` to its UTC instant, or ``None``.

    Returns the actual UTC instant only for a stamp that satisfies the RFC
    3339 ``Z`` contract; a missing, non-text, offset-form, malformed, or
    out-of-range value returns ``None`` so the read-only window export can
    exclude the row without crashing, deleting, or rewriting its stored
    text.
    """
    if isinstance(value, str) and _UTC_Z_STAMP_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            return None
    return None


def _event_id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored lifecycle event id, tolerant of damage."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


_EXPORT_FIELDS = (
    "id",
    "machine_id",
    "grant_id",
    "authorization_event_id",
    "type",
    "occurred_at",
    "previous_event_id",
    "content_hash",
    "chain_hash",
)


def export_event_window(
    session,
    machine_id: str,
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Read-only fixed-window compliance slice of one machine's events.

    Reads only lifecycle rows owned by ``machine_id`` (the ownership column
    is the sole machine boundary) and keeps a row only when its stored
    ``occurred_at`` parses to a UTC instant inside the closed interval
    ``[start, end]``. A stamp that no longer parses — non-text, missing the
    ``Z`` suffix, carrying an offset, malformed, or out-of-range — is
    excluded from the slice and left exactly as stored; the read never
    crashes on, repairs, normalizes, or recomputes it.

    The retained events are ordered by the actual UTC instant of
    ``occurred_at`` and then by event id ascending, so an exact-second stamp
    sorts before any fractional-second stamp of the same second. Each item
    carries exactly the nine stored fields in fixed order — the six content
    fields followed by ``previous_event_id``, ``content_hash``, and
    ``chain_hash`` — emitted verbatim: references, types, chain links, and
    hashes are never fixed or recomputed, and a chain-damaged event inside
    the window is exported exactly as stored. The query is strictly
    read-only: it never inserts, updates, deletes, backfills, or normalizes
    an event or any related row, so repeated reads of unchanged data return
    byte-identical results and the data survives restarts untouched.
    """
    rows = list(
        session.execute(
            _TABLE.select().where(_TABLE.c.machine_id == machine_id)
        )
    )

    in_window: list[tuple[datetime, Any]] = []
    for row in rows:
        instant = _parse_occurred_at_utc(row._mapping["occurred_at"])
        if instant is not None and start <= instant <= end:
            in_window.append((instant, row))

    in_window.sort(
        key=lambda item: (
            item[0],
            _event_id_key(item[1]._mapping["id"]),
        )
    )

    return [
        {field: row._mapping[field] for field in _EXPORT_FIELDS}
        for _, row in in_window
    ]
