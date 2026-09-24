"""Read-only diagnostics for the joint machine write transactions.

The two public mutating operations serialized by the per-database machine
write lock are a machine status change (``op = "change"``) and authorization
decision-event creation (``op = "event"``); together they form the "joint
write". Every attempt at either entry produces exactly one diagnostic
record, persisted in the append-only ``write_transaction_diagnostics``
table.

Lifecycle of one attempt:

1. A ``phase = "started"`` marker is inserted in its own short transaction
   before the locked joint transaction is taken, so the marker survives a
   crash of the process or the joint transaction.
2. The locked joint write runs (with lock waits/retries folded into this one
   record).
3. The marker is finalized once, in a separate transaction, to
   ``started-commit`` (``fail = "none"``) or ``started-rollback`` with a
   stable failure category (``race``, ``io``, ``other``).

If the process dies between steps 1 and 3, the marker is left in
``started``; on startup :func:`recover_pending` finalizes each residual from
evidence — the committed business effect is present (a decision event for an
event attempt, a fresh ``updated_at`` for a status change) or absent — so a
residual is classified as a commit or a crash rollback, never left partial.

Records carry operational metadata only: op, phase, failure category,
experienced lock-wait/retry flags, terminal status, event id, and event
count. They never contain keys, secrets, policy text, or identity material.
The read endpoint only issues SELECTs; diagnostics are never repaired,
recomputed, or deleted by a query.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, inspect, text

from .db import AuthorizationDecisionEvent, Machine, WriteTransactionDiagnostic

_TABLE = WriteTransactionDiagnostic.__table__
_MACHINE_TABLE = Machine.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__

# Terminal phases exposed by the read endpoint. The transient "started"
# phase only exists between step 1 and step 3 above and is always finalized
# (including at startup recovery) before a query can observe the table.
PHASE_STARTED = "started"
PHASE_COMMIT = "started-commit"
PHASE_ROLLBACK = "started-rollback"

FAIL_NONE = "none"
FAIL_RACE = "race"
FAIL_IO = "io"
FAIL_OTHER = "other"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_instant(value: str) -> datetime:
    """Parse a ``Z``-suffixed UTC stamp to a tz-aware instant."""
    return datetime.fromisoformat(value[:-1] + "+00:00")


def migrate_schema(engine: Engine) -> None:
    """Ensure the diagnostics table and its columns exist on older databases.

    The table is created with all columns for upgraded engines (it did not
    exist before this feature); the column additions make the migration
    self-contained against any partial, same-named table and leave an empty
    database fully usable.
    """
    _TABLE.create(bind=engine, checkfirst=True)
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "check_valid": "BOOLEAN NOT NULL DEFAULT 1",
        "check_checked_count": "INTEGER NOT NULL DEFAULT 0",
        "check_broken_event_id": "VARCHAR(36)",
        "started_at": "VARCHAR",
    }
    with engine.begin() as conn:
        for name, column_type in additions.items():
            if name not in existing:
                conn.execute(
                    text(
                        f"ALTER TABLE {_TABLE.name} ADD COLUMN {name} {column_type}"
                    )
                )


def _event_chain_check(
    conn, machine_id: str
) -> tuple[bool, int, str | None]:
    """Snapshot the machine's event hash-chain audit on this connection.

    Imported lazily to avoid the ``chain`` -> ``diagnostics`` -> ``chain``
    import cycle. The shape is the existing event-integrity audit result:
    ``{valid, checked_count, broken_event_id}``.
    """
    from . import chain

    return chain.verify_chain(conn, machine_id)


def insert_started(
    engine: Engine,
    *,
    machine_id: str,
    op: str,
    started_at: str,
    status: str,
    count: int,
) -> tuple[str, float]:
    """Insert the attempt marker before the locked joint transaction.

    Returns ``(tid, waited_seconds)``. The marker owns its own transaction,
    so it is durable independently of the joint write it observes; its
    values are best-effort pre-reads and are overwritten at finalization.
    ``waited_seconds`` reports how long the marker's own write blocked on
    another writer's lock — part of the attempt's total lock wait even
    though it happens before the joint transaction.
    """
    tid = str(uuid.uuid4())
    began = time.monotonic()
    with engine.begin() as conn:
        conn.execute(
            _TABLE.insert().values(
                id=tid,
                machine_id=machine_id,
                op=op,
                at=started_at,
                phase=PHASE_STARTED,
                fail=FAIL_NONE,
                flags=json.dumps([]),
                status=status,
                event=None,
                count=count,
                check_valid=True,
                check_checked_count=count,
                check_broken_event_id=None,
                started_at=started_at,
            )
        )
    return tid, max(0.0, time.monotonic() - began)


def finalize(
    engine: Engine,
    tid: str,
    *,
    phase: str,
    fail: str,
    flags: list[str],
    status: str,
    event: str | None,
    count: int,
    check: tuple[bool, int, str | None],
    at: str | None = None,
) -> None:
    """Finalize one marker exactly once to its terminal outcome."""
    valid, checked_count, broken_event_id = check
    with engine.begin() as conn:
        conn.execute(
            _TABLE.update()
            .where(_TABLE.c.id == tid)
            .values(
                at=at or _utc_now_iso(),
                phase=phase,
                fail=fail,
                flags=json.dumps(flags),
                status=status,
                event=event,
                count=count,
                check_valid=valid,
                check_checked_count=checked_count,
                check_broken_event_id=broken_event_id,
            )
        )


def _machine_events_ordered(conn, machine_id: str) -> list[Any]:
    return list(
        conn.execute(
            _EVENT_TABLE.select()
            .where(_EVENT_TABLE.c.machine_id == machine_id)
            .order_by(_EVENT_TABLE.c.created_at, _EVENT_TABLE.c.id)
        )
    )


def recover_pending(engine: Engine) -> None:
    """Finalize ``started`` markers left by a crashed previous process.

    Each residual is classified from evidence, in ``(started_at, id)``
    order:

    * ``event`` attempt — committed iff the machine has a decision event
      created at or after the attempt start; the earliest such event is the
      evidenced effect and ``count`` is its chain position;
    * ``change`` attempt — committed iff the machine's ``updated_at`` is at
      or after the attempt start (the status update visibly landed);
    * no evidence — the joint transaction rolled back when the process died,
      so the residual is a ``started-rollback`` record with ``fail = other``
      (the stable category for crash rollbacks).

    The recovery instant is used as the terminal ``at``; after this one
    startup pass the table is stable across restarts and queries.
    """
    with engine.connect() as conn:
        pending = list(
            conn.execute(
                _TABLE.select()
                .where(_TABLE.c.phase == PHASE_STARTED)
                .order_by(_TABLE.c.started_at, _TABLE.c.id)
            )
        )

    # A committed event is attributed to at most one attempt. Seed the claim
    # set with every event already referenced by a finalized commit record
    # (a successful live attempt that kept its own diagnostic): a crash
    # residual can never re-attribute an already-accounted-for event. The
    # earliest-starting remaining marker then claims each other event.
    with engine.connect() as conn:
        claimed_rows = conn.execute(
            _TABLE.select()
            .where(
                _TABLE.c.phase == PHASE_COMMIT,
                _TABLE.c.op == "event",
                _TABLE.c.event.is_not(None),
            )
        ).all()
    claimed_events: set[str] = {row._mapping["event"] for row in claimed_rows}

    for row in pending:
        marker = row._mapping
        machine_id = marker["machine_id"]
        op = marker["op"]
        started_at = marker["started_at"]
        started_instant = _parse_instant(started_at)
        at = _utc_now_iso()

        with engine.begin() as conn:
            machine_row = conn.execute(
                _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
            ).first()
            events = _machine_events_ordered(conn, machine_id)
            check = _event_chain_check(conn, machine_id)

            if machine_row is None:
                # Machines are never deleted through the API; keep this
                # defensive so a residual can never block startup.
                conn.execute(
                    _TABLE.update()
                    .where(_TABLE.c.id == marker["id"])
                    .values(
                        at=at,
                        phase=PHASE_ROLLBACK,
                        fail=FAIL_OTHER,
                        flags=json.dumps([]),
                        status="",
                        event=None,
                        count=len(events),
                        check_valid=check[0],
                        check_checked_count=check[1],
                        check_broken_event_id=check[2],
                    )
                )
                continue

            machine = machine_row._mapping
            status = machine["status"]

            if op == "event":
                evidenced = None
                position = 0
                for index, event_row in enumerate(events, start=1):
                    event_id = event_row._mapping["id"]
                    event_at = event_row._mapping["created_at"]
                    # Each committed event evidences at most one residual: the
                    # earliest-starting marker claims it; later markers whose
                    # only candidate was already claimed rolled back.
                    if (
                        event_id not in claimed_events
                        and _parse_instant(event_at) >= started_instant
                    ):
                        evidenced = event_id
                        position = index
                        break
                if evidenced is not None:
                    claimed_events.add(evidenced)
                conn.execute(
                    _TABLE.update()
                    .where(_TABLE.c.id == marker["id"])
                    .values(
                        at=at,
                        phase=PHASE_COMMIT if evidenced is not None else PHASE_ROLLBACK,
                        fail=FAIL_NONE if evidenced is not None else FAIL_OTHER,
                        flags=json.dumps([]),
                        status=status,
                        event=evidenced,
                        count=position if evidenced is not None else len(events),
                        check_valid=check[0],
                        check_checked_count=check[1],
                        check_broken_event_id=check[2],
                    )
                )
            else:  # op == "change"
                committed = _parse_instant(machine["updated_at"]) >= started_instant
                conn.execute(
                    _TABLE.update()
                    .where(_TABLE.c.id == marker["id"])
                    .values(
                        at=at,
                        phase=PHASE_COMMIT if committed else PHASE_ROLLBACK,
                        fail=FAIL_NONE if committed else FAIL_OTHER,
                        flags=json.dumps([]),
                        status=status,
                        event=None,
                        count=len(events),
                        check_valid=check[0],
                        check_checked_count=check[1],
                        check_broken_event_id=check[2],
                    )
                )


def fetch_records(
    session, machine_id: str, window_start: datetime, window_end: datetime
) -> list[Any]:
    """Read one machine's finalized diagnostics in a closed UTC window.

    Only terminal records of the path machine are returned, ordered by the
    actual UTC instant of ``at`` and then by ``tid`` (``id``); ISO text
    ordering is not chronological across the fractional-second boundary.
    The query issues no writes.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(
                _TABLE.c.machine_id == machine_id,
                _TABLE.c.phase != PHASE_STARTED,
            )
            .order_by(_TABLE.c.at, _TABLE.c.id)
        )
    )
    in_window = [
        row
        for row in rows
        if window_start <= _parse_instant(row._mapping["at"]) <= window_end
    ]
    return sorted(
        in_window,
        key=lambda row: (_parse_instant(row._mapping["at"]), row._mapping["id"]),
    )
