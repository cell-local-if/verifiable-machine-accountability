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

from sqlalchemy import Connection, Engine

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import Machine

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

    The lookup, same-status check, and update happen in one locked
    transaction. Returns a status dict:

    * ``not_found`` — the machine does not exist (nothing is written);
    * ``invalid_status_transition`` — the machine already has ``to_status``
      (nothing is written);
    * ``ok`` — with the full updated ``machine`` record. Only ``status`` and
      ``updated_at`` change; ``version``, ``public_key``, and ``created_at``
      keep their stored values.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine_row is None:
            return {"status": "not_found"}
        current_status = machine_row._mapping["status"]
        if current_status == to_status:
            return {"status": "invalid_status_transition"}

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
                return {"status": "not_found"}
            return {"status": "invalid_status_transition"}

        updated_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        return {
            "status": "ok",
            "machine": {key: updated_row._mapping[key] for key in _MACHINE_FIELDS},
        }

    return _run_with_lock_retry(engine, _work)
