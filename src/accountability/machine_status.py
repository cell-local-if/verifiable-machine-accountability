"""Machine start/stop control: the ``active <-> suspended`` state machine.

A machine is either ``active`` (normal authorization rules apply) or
``suspended`` (every authorization evaluation and decision event is denied
with ``machine_suspended`` before declarations or policy are consulted). The
status lives on the machine row itself and therefore survives restarts.

Every status change reads the row and updates ``status`` and ``updated_at``
inside a single locked write transaction, so concurrent requests targeting
the same resulting state can never both observe the prior state: at most one
succeeds and the rest get ``invalid_status_transition`` without writing.
"""

from typing import Any

from sqlalchemy import Connection, Engine

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import Machine

_MACHINE_TABLE = Machine.__table__


def change_machine_status(
    engine: Engine,
    *,
    machine_id: str,
    to_status: str,
) -> dict[str, Any]:
    """Atomically set one machine's status.

    All work happens in one locked transaction. Returns a status dict:

    * ``not_found`` — the machine does not exist;
    * ``invalid_status_transition`` — the machine already has ``to_status``
      (nothing is written);
    * ``ok`` — with the complete updated machine record.

    Only ``status`` and ``updated_at`` change: ``version``, ``public_key``,
    ``created_at``, and every other column are left untouched.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine_row is None:
            return {"status": "not_found"}
        machine = machine_row._mapping
        if machine["status"] == to_status:
            return {"status": "invalid_status_transition"}

        now = _utc_now_iso()
        result = conn.execute(
            _MACHINE_TABLE.update()
            .where(
                _MACHINE_TABLE.c.id == machine_id,
                _MACHINE_TABLE.c.status != to_status,
            )
            .values(status=to_status, updated_at=now)
        )
        if result.rowcount == 0:
            # The write lock serializes transitions, so this is only a
            # defensive reclassification of a raced machine row.
            return {"status": "invalid_status_transition"}

        return {
            "status": "ok",
            "machine": {
                "id": machine["id"],
                "external_id": machine["external_id"],
                "display_name": machine["display_name"],
                "public_key": machine["public_key"],
                "status": to_status,
                "version": machine["version"],
                "created_at": machine["created_at"],
                "updated_at": now,
            },
        }

    return _run_with_lock_retry(engine, _work)
