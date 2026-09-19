"""Incident status state machine and immutable transition history.

An incident moves ``open -> acknowledged -> resolved`` only. Every accepted
transition updates the incident row and appends one immutable history record
inside a single locked write transaction, so the new status and its history
entry are committed together or not at all, and concurrent transitions cannot
both observe the same prior status. Rejected transitions write nothing.
"""

import uuid
from typing import Any

from sqlalchemy import Connection, Engine, select
from sqlalchemy.orm import Session

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import (
    AuthorizationDecisionEvent,
    AuthorizationDecisionIncident,
    IncidentStatusEvent,
    Machine,
)

_INCIDENT_TABLE = AuthorizationDecisionIncident.__table__
_HISTORY_TABLE = IncidentStatusEvent.__table__

# The only legal edges of the incident state machine.
_ALLOWED_TRANSITIONS = {
    "open": "acknowledged",
    "acknowledged": "resolved",
}


def get_machine_event_incident(
    session: Session, machine_id: str, event_id: str, incident_id: str
) -> AuthorizationDecisionIncident | None:
    """Resolve an incident that belongs to the path machine and event.

    Returns ``None`` when the machine, event, or incident is missing or when
    any of them belongs to a different owner than the path claims.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return None
    event = session.scalar(
        select(AuthorizationDecisionEvent).where(
            AuthorizationDecisionEvent.id == event_id,
            AuthorizationDecisionEvent.machine_id == machine_id,
        )
    )
    if event is None:
        return None
    return session.scalar(
        select(AuthorizationDecisionIncident).where(
            AuthorizationDecisionIncident.id == incident_id,
            AuthorizationDecisionIncident.machine_id == machine_id,
            AuthorizationDecisionIncident.event_id == event_id,
        )
    )


def change_incident_status(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    incident_id: str,
    to_status: str,
) -> dict[str, Any]:
    """Atomically advance one incident's status and append its history record.

    All lookups and the write happen in one locked transaction. Returns a
    status dict:

    * ``not_found`` — the machine, event, or incident is missing or owned by
      another machine/event;
    * ``invalid_status_transition`` — the incident's current status has no edge
      to ``to_status`` (nothing is written);
    * ``ok`` — with the updated ``incident`` and the new ``history`` record.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        event = conn.execute(
            AuthorizationDecisionEvent.__table__.select().where(
                AuthorizationDecisionEvent.__table__.c.id == event_id,
                AuthorizationDecisionEvent.__table__.c.machine_id == machine_id,
            )
        ).first()
        if event is None:
            return {"status": "not_found"}
        incident_row = conn.execute(
            _INCIDENT_TABLE.select().where(
                _INCIDENT_TABLE.c.id == incident_id,
                _INCIDENT_TABLE.c.machine_id == machine_id,
                _INCIDENT_TABLE.c.event_id == event_id,
            )
        ).first()
        if incident_row is None:
            return {"status": "not_found"}

        incident = incident_row._mapping
        from_status = incident["status"]
        if _ALLOWED_TRANSITIONS.get(from_status) != to_status:
            return {"status": "invalid_status_transition"}

        now = _utc_now_iso()
        history_id = str(uuid.uuid4())
        conn.execute(
            _INCIDENT_TABLE.update()
            .where(_INCIDENT_TABLE.c.id == incident_id)
            .values(status=to_status)
        )
        conn.execute(
            _HISTORY_TABLE.insert().values(
                id=history_id,
                machine_id=machine_id,
                event_id=event_id,
                incident_id=incident_id,
                from_status=from_status,
                to_status=to_status,
                created_at=now,
            )
        )
        return {
            "status": "ok",
            "incident": {
                "id": incident["id"],
                "machine_id": incident["machine_id"],
                "event_id": incident["event_id"],
                "incident_type": incident["incident_type"],
                "summary": incident["summary"],
                "status": to_status,
                "created_at": incident["created_at"],
            },
            "history": {
                "id": history_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "incident_id": incident_id,
                "from_status": from_status,
                "to_status": to_status,
                "created_at": now,
            },
        }

    return _run_with_lock_retry(engine, _work)


def list_status_history(
    session: Session, machine_id: str, event_id: str, incident_id: str
) -> list[IncidentStatusEvent]:
    """Return one incident's immutable transitions in (created_at, id) order."""
    return list(
        session.scalars(
            select(IncidentStatusEvent)
            .where(
                IncidentStatusEvent.machine_id == machine_id,
                IncidentStatusEvent.event_id == event_id,
                IncidentStatusEvent.incident_id == incident_id,
            )
            .order_by(IncidentStatusEvent.created_at, IncidentStatusEvent.id)
        )
    )
