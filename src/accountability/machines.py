"""Persistent machine enablement state (``active`` <-> ``suspended``).

Every accepted status change updates only the machine's ``status`` and
``updated_at`` inside a single locked write transaction; ``version``,
``public_key``, ``created_at``, and every other record are left untouched.
The lock serializes concurrent updates, so two requests for the same target
status can never both observe the prior state: at most one succeeds, and the
other sees ``invalid_status_transition`` and writes nothing. The state lives
in the ``machines`` table, so it survives restarts.
"""

from typing import Any

from sqlalchemy import Connection, Engine, func, select

from .chain import (
    JointWriteOutcome,
    _utc_now_iso,
    run_joint_write,
    verify_chain,
)
from .db import AuthorizationDecisionEvent, Machine

_MACHINE_TABLE = Machine.__table__

_MACHINE_FIELDS = (
    "id",
    "external_id",
    "display_name",
    "public_key",
    "status",
    "version",
    "created_at",
    "updated_at",
)


def change_machine_status(
    engine: Engine,
    *,
    machine_id: str,
    to_status: str,
) -> dict[str, Any]:
    """Atomically set one machine's status.

    The lookup, same-status check, and update happen in one locked joint
    write transaction (the same lock a decision-event append takes). Returns
    a status dict:

    * ``not_found`` — the machine does not exist (nothing is written);
    * ``invalid_status_transition`` — the machine already has ``to_status``
      (nothing is written); this is the concurrency-loser outcome and is
      diagnosed as a ``race`` rollback;
    * ``ok`` — with the full updated ``machine`` record. Only ``status`` and
      ``updated_at`` change; ``version``, ``public_key``, and ``created_at``
      keep their stored values.

    Every attempt records exactly one joint-write diagnostic.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine_row is None:
            raise JointWriteOutcome({"status": "not_found"})
        current_status = machine_row._mapping["status"]
        if current_status == to_status:
            # The same-target request lost the race to the request that
            # already committed this status: nothing is written.
            raise JointWriteOutcome(
                {"status": "invalid_status_transition"}, fail="race"
            )

        now = _utc_now_iso()
        # The status predicate is an optimistic guard in addition to the write
        # lock: the row only changes while it still holds the status we read.
        result = conn.execute(
            _MACHINE_TABLE.update()
            .where(
                _MACHINE_TABLE.c.id == machine_id,
                _MACHINE_TABLE.c.status == current_status,
            )
            .values(status=to_status, updated_at=now)
        )
        if result.rowcount == 0:
            # The write lock serializes status updates, so this only defends
            # against a raced external writer: reclassify from the current row.
            current = conn.execute(
                _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
            ).first()
            if current is None:
                raise JointWriteOutcome({"status": "not_found"})
            raise JointWriteOutcome(
                {"status": "invalid_status_transition"}, fail="race"
            )

        updated_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        event_count = conn.execute(
            select(func.count())
            .select_from(AuthorizationDecisionEvent.__table__)
            .where(AuthorizationDecisionEvent.__table__.c.machine_id == machine_id)
        ).scalar_one()
        # Snapshot the attempt's own terminal state inside the lock for its
        # diagnostic: the new status, the (unchanged) event count, and the
        # machine's event-chain check.
        diag_check = verify_chain(conn, machine_id)
        return {
            "status": "ok",
            "machine": {key: updated_row._mapping[key] for key in _MACHINE_FIELDS},
            "diag_status": to_status,
            "diag_count": int(event_count or 0),
            "diag_check": diag_check,
        }

    return run_joint_write(
        engine, machine_id=machine_id, op="change", work=_work
    )
