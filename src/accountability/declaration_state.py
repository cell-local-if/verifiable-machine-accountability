"""Persistent enablement state of existing behavior declarations.

An accepted toggle updates only the declaration's ``enabled`` flag and
``updated_at`` inside a single locked write transaction; ``id``,
``machine_id``, ``action_type``, ``resource_pattern``, and ``created_at``
keep their stored values, and no other record is touched. The lock
serializes concurrent toggles of the same declaration, so two requests for
the same target state can never both observe the prior value: exactly one
succeeds and every other one sees ``declaration_state_unchanged`` and
writes nothing, while requests for different targets commit in lock order
and never lose an update. The state lives in the ``behavior_declarations``
table, so it survives restarts; new authorization decisions read the
updated flag, while already-stored decision bases, events, grants,
receipts, and their hash chains are never rewritten.
"""

from typing import Any

from sqlalchemy import Connection, Engine

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import BehaviorDeclaration, Machine

_DECLARATION_TABLE = BehaviorDeclaration.__table__

_DECLARATION_FIELDS = (
    "id",
    "machine_id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)


def set_declaration_enabled(
    engine: Engine,
    *,
    machine_id: str,
    declaration_id: str,
    enabled: bool,
) -> dict[str, Any]:
    """Atomically set one existing declaration's ``enabled`` flag.

    The machine lookup, declaration lookup, same-state check, and update
    happen in one locked write transaction (the same lock the other
    internal writers take), so concurrent toggles serialize. Returns a
    status dict:

    * ``not_found`` — the machine does not exist, or no declaration with
      this id belongs to the path machine (nothing is written);
    * ``declaration_state_unchanged`` — the declaration already carries
      the target ``enabled`` value (nothing is written); this is the
      concurrency-loser outcome for same-target requests;
    * ``ok`` — with the full updated ``declaration`` record. Only
      ``enabled`` and ``updated_at`` change; ``id``, ``machine_id``,
      ``action_type``, ``resource_pattern``, and ``created_at`` keep their
      stored values.

    The toggle writes no decision event, evidence, incident, grant,
    receipt, or diagnostic record and touches no hash chain, no machine
    row, no policy rule, and no other declaration.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine_row is None:
            return {"status": "not_found"}
        declaration_row = conn.execute(
            _DECLARATION_TABLE.select().where(
                _DECLARATION_TABLE.c.id == declaration_id,
                _DECLARATION_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if declaration_row is None:
            return {"status": "not_found"}

        current_enabled = bool(declaration_row._mapping["enabled"])
        if current_enabled == enabled:
            # The same-target request lost the race to the toggle that
            # already committed this state: nothing is written.
            return {"status": "declaration_state_unchanged"}

        now = _utc_now_iso()
        # The enabled predicate is an optimistic guard in addition to the
        # write lock: the row only changes while it still holds the value
        # we read.
        result = conn.execute(
            _DECLARATION_TABLE.update()
            .where(
                _DECLARATION_TABLE.c.id == declaration_id,
                _DECLARATION_TABLE.c.enabled.is_(current_enabled),
            )
            .values(enabled=enabled, updated_at=now)
        )
        if result.rowcount == 0:
            # The write lock serializes toggles, so this only defends
            # against a raced external writer: reclassify from the current
            # row.
            current = conn.execute(
                _DECLARATION_TABLE.select().where(
                    _DECLARATION_TABLE.c.id == declaration_id,
                    _DECLARATION_TABLE.c.machine_id == machine_id,
                )
            ).first()
            if current is None:
                return {"status": "not_found"}
            # The flag is boolean: a row that no longer holds the value we
            # read necessarily already carries the target value.
            return {"status": "declaration_state_unchanged"}

        updated_row = conn.execute(
            _DECLARATION_TABLE.select().where(
                _DECLARATION_TABLE.c.id == declaration_id
            )
        ).first()
        return {
            "status": "ok",
            "declaration": {
                key: updated_row._mapping[key] for key in _DECLARATION_FIELDS
            },
        }

    return _run_with_lock_retry(engine, _work)
