"""Per-machine tamper-evident hash chains for grant lifecycle events.

Each machine's grant lifecycle audit events form an ordered chain (ordered by
``occurred_at`` then ``id``), following the same rules as the other per-machine
chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the event's own fields: ``{id, machine_id, grant_id,
  authorization_event_id, type, occurred_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first event uses the empty string as the previous chain hash.
* ``previous_event_id`` is ``None`` for a machine's first event and the prior
  event's id otherwise.

A successful grant action (issue, single consume, or emergency revocation)
appends its lifecycle event inside the action's own locked write transaction,
so the action and the audit event commit together or leave no trace, and
concurrent actions cannot lose events, fork the chain, or break a link.
Duplicate consumptions or revocations that lose to the one terminal winner
write nothing and append nothing.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .db import AuthorizationGrantLifecycleEvent, Machine

_HASH_LEN = 64

_TABLE = AuthorizationGrantLifecycleEvent.__table__
_MACHINE_TABLE = Machine.__table__
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
    """Bring a database that already has the events table up to date.

    The table is created current-schema by ``Base.metadata.create_all`` on
    databases that predate the feature, and old data is never rewritten. This
    only adds the chain columns to a table that somehow predates them.
    """
    inspector = inspect(engine)
    if _TABLE.name not in inspector.get_table_names():
        return
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
                        f"ALTER TABLE {_TABLE.name} ADD COLUMN {name} "
                        f"{column_type}"
                    )
                )


def _load_events(conn: Connection, machine_id: str) -> list[Any]:
    statement = _TABLE.select().where(_TABLE.c.machine_id == machine_id)
    return list(
        conn.execute(statement.order_by(_TABLE.c.occurred_at, _TABLE.c.id))
    )


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


def _ensure_sound_tail(conn: Connection, machine_id: str) -> list[Any]:
    """Return the machine's complete chain rows, repairing missing links.

    Normally every row is complete when appended. If any row is missing chain
    data (e.g. an external writer), rebuild the whole machine chain before
    appending so the new link has a sound tail; the read-only integrity query
    never repairs.
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


def append_lifecycle_event(
    conn: Connection,
    *,
    machine_id: str,
    grant_id: str,
    authorization_event_id: str,
    type: str,
    occurred_at: str,
) -> dict[str, Any]:
    """Append one lifecycle event to the machine's chain tail.

    Must run inside the grant action's locked write transaction, so the action
    and the event commit together. ``occurred_at`` is the successful action's
    own response timestamp (issued/consumed/revoked moment), reused verbatim.
    The event id is minted here and, when it shares the tail's instant,
    regenerated until it sorts strictly after the tail id, so the predecessor
    link always matches the (occurred_at, id) order used by verification.
    """
    rows = _ensure_sound_tail(conn, machine_id)
    tail = rows[-1] if rows else None

    event_id = str(uuid.uuid4())
    if tail is not None:
        tail_occurred_at = tail._mapping["occurred_at"]
        tail_id = tail._mapping["id"]
        while occurred_at == tail_occurred_at and event_id <= tail_id:
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


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's lifecycle chain in (occurred_at, id) order.

    Returns ``(valid, checked_count, broken_event_id)``. The first event whose
    recomputed content hash, previous-event link, or chain hash differs from
    the stored values is reported; an empty chain is valid. Only the path
    machine's rows are examined and nothing is ever written back.

    A damaged stored value (e.g. a non-text column an external writer left)
    cannot produce the published digest: it is reported as that event being
    broken rather than crashing the read-only audit, with a stable JSON-safe
    rendering of a non-text id.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(_TABLE.c.machine_id == machine_id)
            .order_by(_TABLE.c.occurred_at, _TABLE.c.id)
        )
    )

    previous_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        raw_id = mapping["id"]
        row_id = raw_id if isinstance(raw_id, str) else str(raw_id)
        try:
            content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
            chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        except (TypeError, ValueError):
            return False, len(rows), row_id
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_event_id"] != previous_event_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), row_id
        previous_event_id = row_id
        previous_chain_hash = chain_hash

    return True, len(rows), None


def machine_exists(session, machine_id: str) -> bool:
    """Whether the path machine exists (read-only, used by the read queries)."""
    return (
        session.execute(
            _MACHINE_TABLE.select()
            .where(_MACHINE_TABLE.c.id == machine_id)
            .with_only_columns(_MACHINE_TABLE.c.id)
        ).first()
        is not None
    )
