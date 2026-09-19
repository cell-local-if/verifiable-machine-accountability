"""Atomic incident status transitions with an immutable history trail.

An incident's status moves only forward along
``open -> acknowledged -> resolved``. Each successful transition updates the
incident row and appends one history record inside a single locked write
transaction, so the two writes commit together or not at all: a rejected
transition leaves no trace, concurrent transitions cannot both succeed from
the same source status, and existing history records are never modified.
"""

import uuid
from typing import Any

from sqlalchemy import Connection, Engine

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import (
    AuthorizationDecisionIncident,
    AuthorizationDecisionIncidentStatusHistory,
)

_INCIDENT_TABLE = AuthorizationDecisionIncident.__table__
_HISTORY_TABLE = AuthorizationDecisionIncidentStatusHistory.__table__

# The only permitted (from_status, to_status) pairs.
_ALLOWED_TRANSITIONS = frozenset(
    {
        ("open", "acknowledged"),
        ("acknowledged", "resolved"),
    }
)


def transition_status(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    incident_id: str,
    new_status: str,
) -> dict[str, Any]:
    """Atomically transition one incident's status and append its history.

    The incident row is re-read under the write lock, scoped to the path
    machine and event, so a missing or foreign incident reports
    ``not_found`` and the transition is validated against the current stored
    status. The status update and the immutable history insert commit in the
    same transaction; a disallowed transition writes nothing.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        row = conn.execute(
            _INCIDENT_TABLE.select().where(
                _INCIDENT_TABLE.c.id == incident_id,
                _INCIDENT_TABLE.c.machine_id == machine_id,
                _INCIDENT_TABLE.c.event_id == event_id,
            )
        ).first()
        if row is None:
            return {"status": "not_found"}

        mapping = row._mapping
        from_status = mapping["status"]
        if (from_status, new_status) not in _ALLOWED_TRANSITIONS:
            return {"status": "invalid_status_transition"}

        conn.execute(
            _INCIDENT_TABLE.update()
            .where(_INCIDENT_TABLE.c.id == incident_id)
            .values(status=new_status)
        )
        conn.execute(
            _HISTORY_TABLE.insert().values(
                id=str(uuid.uuid4()),
                machine_id=machine_id,
                event_id=event_id,
                incident_id=incident_id,
                from_status=from_status,
                to_status=new_status,
                created_at=_utc_now_iso(),
            )
        )
        return {
            "status": "ok",
            "incident": {
                "id": mapping["id"],
                "machine_id": mapping["machine_id"],
                "event_id": mapping["event_id"],
                "incident_type": mapping["incident_type"],
                "summary": mapping["summary"],
                "status": new_status,
                "created_at": mapping["created_at"],
            },
        }

    return _run_with_lock_retry(engine, _work)
