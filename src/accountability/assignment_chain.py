"""Per-incident tamper-evident hash chains for responsibility assignments.

Each incident's responsibility assignments form their own ordered chain
(ordered by ``created_at`` then ``id``), following the same rules as the
authorization decision event integrity chain:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the assignment's own fields: ``{id, machine_id, event_id,
  incident_id, party, role, created_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  incident's first assignment uses the empty string as the previous chain
  hash.
* ``previous_assignment_id`` is ``None`` for an incident's first assignment
  and the prior assignment's id otherwise.

Chains are independent per incident, so two incidents of the same machine
(or of different machines) never share a link. New assignments are appended
to their incident's chain tail inside a single write transaction, so
concurrent assignments cannot lose records, fork the chain, or break a link.
The transaction only ever reads and writes the assignments table: the
incident, its event, evidence, and the other hash chains are never modified.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import (
    AuthorizationDecisionEvent,
    AuthorizationDecisionIncident,
    IncidentResponsibilityAssignment,
    Machine,
)

_HASH_LEN = 64

_TABLE = IncidentResponsibilityAssignment.__table__
_MACHINE_TABLE = Machine.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__
_INCIDENT_TABLE = AuthorizationDecisionIncident.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "event_id",
    "incident_id",
    "party",
    "role",
    "created_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    event_id: str,
    incident_id: str,
    party: str,
    role: str,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "event_id": event_id,
            "incident_id": incident_id,
            "party": party,
            "role": role,
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
        "previous_assignment_id": "VARCHAR(36)",
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


def _load_assignments(conn: Connection, incident_id: str) -> list[Any]:
    return list(
        conn.execute(
            _TABLE.select()
            .where(_TABLE.c.incident_id == incident_id)
            .order_by(_TABLE.c.created_at, _TABLE.c.id)
        )
    )


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_assignment_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_assignment_id"] != previous_assignment_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_assignment_id": previous_assignment_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_assignment_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on assignments written before the feature.

    Processing is per incident in (created_at, id) order. The recomputation
    is deterministic, so a restart over an already complete database issues
    no writes. Each incident is handled inside a locked transaction so a
    concurrent assignment can neither interleave with the backfill nor fork.
    """
    with engine.connect() as conn:
        incident_ids = [
            row[0]
            for row in conn.execute(
                text(
                    f"SELECT DISTINCT incident_id FROM {_TABLE.name} "
                    "ORDER BY incident_id"
                )
            )
        ]

    for incident_id in incident_ids:

        def _work(conn: Connection, incident_id=incident_id) -> None:
            rows = _load_assignments(conn, incident_id)
            # previous_assignment_id is NULL on the first record, so
            # completeness is determined by the two hashes being present.
            if not rows or not all(
                row._mapping["content_hash"] is not None
                and row._mapping["chain_hash"] is not None
                for row in rows
            ):
                for values in _recompute_rows(rows):
                    assignment_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == assignment_id)
                        .values(**values)
                    )

        _run_with_lock_retry(engine, _work)


def assign_responsibility(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    incident_id: str,
    party: str,
    role: str,
) -> dict[str, Any]:
    """Validate the path and append one assignment to its incident's chain tail.

    All lookups, the duplicate check, and the insert happen inside one locked
    write transaction, so concurrent assignments to the same incident cannot
    lose records or fork the per-incident chain, and a duplicate ``(party,
    role)`` pair is observed atomically. Returns a status dict:

    * ``not_found`` — the machine, event, or incident is missing or owned by
      another machine/event;
    * ``duplicate_assignment`` — the trimmed ``(party, role)`` pair already
      exists on this incident (nothing is written);
    * ``ok`` — with the new ``assignment`` record including its chain fields.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        event = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == event_id,
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if event is None:
            return {"status": "not_found"}
        incident = conn.execute(
            _INCIDENT_TABLE.select().where(
                _INCIDENT_TABLE.c.id == incident_id,
                _INCIDENT_TABLE.c.machine_id == machine_id,
                _INCIDENT_TABLE.c.event_id == event_id,
            )
        ).first()
        if incident is None:
            return {"status": "not_found"}

        duplicate = conn.execute(
            _TABLE.select().where(
                _TABLE.c.incident_id == incident_id,
                _TABLE.c.party == party,
                _TABLE.c.role == role,
            )
        ).first()
        if duplicate is not None:
            return {"status": "duplicate_assignment"}

        rows = _load_assignments(conn, incident_id)

        # Normally the startup backfill leaves every row complete. If any row
        # is missing chain data (e.g. an external writer), rebuild the whole
        # incident chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_hash"] is None
            or row._mapping["chain_hash"] is None
            for row in rows
        ):
            for values in _recompute_rows(rows):
                stale_id = values.pop("id")
                conn.execute(
                    _TABLE.update().where(_TABLE.c.id == stale_id).values(**values)
                )
            rows = _load_assignments(conn, incident_id)

        tail = rows[-1] if rows else None

        # Regenerate (rarely) until the new key sorts strictly after the tail.
        created_at = _utc_now_iso()
        assignment_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_id = tail._mapping["id"]
            if created_at < tail_created_at:
                created_at = tail_created_at
            while created_at == tail_created_at and assignment_id <= tail_id:
                assignment_id = str(uuid.uuid4())

        if tail is None:
            previous_assignment_id = None
            previous_chain_hash = ""
        else:
            previous_assignment_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        content_hash = compute_content_hash(
            id=assignment_id,
            machine_id=machine_id,
            event_id=event_id,
            incident_id=incident_id,
            party=party,
            role=role,
            created_at=created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        conn.execute(
            _TABLE.insert().values(
                id=assignment_id,
                machine_id=machine_id,
                event_id=event_id,
                incident_id=incident_id,
                party=party,
                role=role,
                created_at=created_at,
                previous_assignment_id=previous_assignment_id,
                content_hash=content_hash,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "assignment": {
                "id": assignment_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                "party": party,
                "role": role,
                "created_at": created_at,
                "previous_assignment_id": previous_assignment_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        }

    return _run_with_lock_retry(engine, _work)


def verify_chain(session, incident_id: str) -> tuple[bool, int, str | None]:
    """Verify one incident's assignment chain in (created_at, id) order.

    Returns ``(valid, checked_count, broken_assignment_id)``. The first
    assignment whose recomputed content hash, previous-assignment link, or
    chain hash differs from the stored values is reported; an empty chain is
    valid. Only the given incident's assignments are examined. The caller is
    responsible for resolving and authorizing the machine/event/incident
    path first.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(_TABLE.c.incident_id == incident_id)
            .order_by(_TABLE.c.created_at, _TABLE.c.id)
        )
    )

    previous_assignment_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_assignment_id"] != previous_assignment_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_assignment_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
