"""Per-machine tamper-evident hash chains for authorization decision events.

Each machine's events form an ordered chain (ordered by ``created_at`` then
``id``):

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the event's own fields.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first event uses the empty string as the previous chain hash.
* ``previous_event_id`` is ``None`` for a machine's first event and the prior
  event's id otherwise.

New events are appended to the chain tail inside a single write transaction
that reads the tail and inserts the row, so concurrent appenders cannot lose
events, fork the chain, or break a link. A decision-event append goes further
(:func:`append_decision_event`): the machine lookup, the authorization
decision, and the append share one locked transaction, the same lock status
changes take, so a status change and an event append have a single definite
serial order.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, func, inspect, select, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from . import diagnostics
from .db import AuthorizationDecisionEvent, Machine

_HASH_LEN = 64
_MAX_LOCK_ATTEMPTS = 20
# A ``BEGIN IMMEDIATE`` that takes longer than this visibly waited for another
# writer's lock; the attempt records ``lock_wait`` even when SQLite's own
# busy-timeout eventually grants the lock without an application-level retry.
_LOCK_WAIT_MIN_SECONDS = 0.02

_TABLE = AuthorizationDecisionEvent.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "action_type",
    "resource",
    "allowed",
    "reason",
    "created_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "action_type": action_type,
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
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


@contextmanager
def _locked_connection(
    engine: Engine, on_lock_acquired: Callable[[float], None] | None = None
) -> Iterator[Connection]:
    """A connection inside a write transaction that serializes appenders.

    SQLite's default deferred transactions only take a write lock on first
    write, which lets two appenders read the same tail and fork the chain;
    ``BEGIN IMMEDIATE`` takes the reserved lock up front. Other databases use
    SERIALIZABLE isolation, which abides by the same guarantee.

    ``on_lock_acquired``, when given, is called with the elapsed seconds the
    ``BEGIN IMMEDIATE`` blocked before the lock was granted, so the joint
    write runner can record a ``lock_wait`` flag for contended attempts.
    """
    if engine.dialect.name == "sqlite":
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        began = time.monotonic()
        try:
            conn.execute(text("BEGIN IMMEDIATE"))
        except OperationalError as error:
            # Even a rejected BEGIN blocked behind another writer's lock for
            # the busy-timeout; surface the wait for the diagnostic flags.
            try:
                error.lock_wait_seconds = time.monotonic() - began  # type: ignore[attr-defined]
            except Exception:
                pass
            conn.close()
            raise
        if on_lock_acquired is not None:
            on_lock_acquired(time.monotonic() - began)
        try:
            yield conn
            conn.execute(text("COMMIT"))
        except Exception:
            try:
                conn.execute(text("ROLLBACK"))
            except Exception:
                pass
            raise
        finally:
            conn.close()
    else:
        conn = engine.connect().execution_options(isolation_level="SERIALIZABLE")
        started = time.monotonic()
        try:
            with conn.begin():
                if on_lock_acquired is not None:
                    on_lock_acquired(time.monotonic() - started)
                yield conn
        finally:
            conn.close()


def _is_lock_conflict(error: OperationalError) -> bool:
    pgcode = getattr(error.orig, "pgcode", None)
    if pgcode in ("40001", "55P03"):  # serialization failure / lock unavailable
        return True
    args = getattr(error.orig, "args", ())
    if args and args[0] in (1205, 1213):  # MySQL lock wait timeout / deadlock
        return True
    message = str(error.orig).lower()
    return "database is locked" in message or "database table is locked" in message


def _run_with_lock_retry(engine: Engine, work):
    """Run ``work`` in the locked write transaction with conflict retries.

    This is the internal lock primitive for writers that are *not* the
    machine status-change / decision-event joint write (key rotations,
    responsibility assignments, incident status transitions, and the startup
    chain backfills). The two public joint-write entries —
    :func:`append_decision_event` (``op = "event"``) and
    :func:`machines.change_machine_status` (``op = "change"``) — must go
    through :func:`run_joint_write` instead, which wraps this locking with
    the per-attempt diagnostic. No new internal entry that changes a
    machine's status or creates a decision event should bypass that runner.
    """
    for attempt in range(_MAX_LOCK_ATTEMPTS):
        try:
            with _locked_connection(engine) as conn:
                return work(conn)
        except OperationalError as error:
            if not _is_lock_conflict(error) or attempt == _MAX_LOCK_ATTEMPTS - 1:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.2))


class JointWriteOutcome(Exception):
    """Raised inside a joint-write work callable for an early terminal result.

    A status change ends this way when the machine is missing or already
    carries the requested status (the documented 404/409 outcomes); an event
    append uses it only for the missing machine. The locked transaction is
    rolled back, leaving no business trace, and the attempt still records one
    diagnostic. ``fail`` is the stable rollback category for the rejection:
    ``race`` for the same-target status conflict (the concurrency-loser
    outcome), ``other`` for a missing machine.
    """

    def __init__(self, result: dict[str, Any], fail: str = "other"):
        self.result = result
        self.fail = fail


def _failure_category(error: BaseException) -> str:
    """Map a joint-write exception to the stable rollback failure category.

    * ``race`` — a serialization/deadlock/lock conflict between concurrent
      writers (the lock itself was not granted or the write was aborted);
    * ``io`` — a persistence failure reaching the database driver
      (``OperationalError`` that is not a concurrency conflict, e.g. a
      read-only or I/O database);
    * ``other`` — anything else, including an application crash represented
      by an unexpected error.
    """
    if isinstance(error, OperationalError):
        return "race" if _is_lock_conflict(error) else "io"
    return "other"


def _pre_read(engine: Engine, machine_id: str) -> tuple[str | None, int]:
    """Best-effort ``(current_status, event_count)`` before the attempt.

    Only used to seed the durable marker; every value is re-read inside the
    locked transaction and again at finalization, so a stale pre-read never
    changes the terminal record.
    """
    try:
        with engine.connect() as conn:
            status = conn.execute(
                Machine.__table__.select()
                .where(Machine.__table__.c.id == machine_id)
                .with_only_columns(Machine.__table__.c.status)
            ).scalar()
            count = conn.execute(
                select(func.count())
                .select_from(AuthorizationDecisionEvent.__table__)
                .where(AuthorizationDecisionEvent.__table__.c.machine_id == machine_id)
            ).scalar_one()
            return status, int(count or 0)
    except SQLAlchemyError:
        return None, 0


def _terminal_snapshot(
    engine: Engine, machine_id: str
) -> tuple[str, int, tuple[bool, int, str | None]]:
    """Read the committed terminal ``(status, event_count, chain_check)``.

    Used to finalize a marker from evidence after the business attempt
    finished. Read-only.
    """
    with engine.connect() as conn:
        status = conn.execute(
            Machine.__table__.select()
            .where(Machine.__table__.c.id == machine_id)
            .with_only_columns(Machine.__table__.c.status)
        ).scalar()
        count = conn.execute(
            select(func.count())
            .select_from(AuthorizationDecisionEvent.__table__)
            .where(AuthorizationDecisionEvent.__table__.c.machine_id == machine_id)
        ).scalar_one()
        check = verify_chain(conn, machine_id)
        return status, int(count or 0), check


def run_joint_write(
    engine: Engine,
    *,
    machine_id: str,
    op: str,
    work: Callable[[Connection], dict[str, Any]],
) -> dict[str, Any]:
    """Run one public joint-write attempt and record exactly one diagnostic.

    ``op`` is ``"change"`` for a machine status change and ``"event"`` for a
    decision-event creation. ``work`` runs inside the locked write
    transaction and either returns the ``{"status": "ok", ...}`` result or
    raises :class:`JointWriteOutcome` for the documented non-ok outcomes
    (``not_found`` / ``invalid_status_transition``), which roll the business
    transaction back.

    The whole attempt — lock waits and application-level retries folded in —
    produces one append-only diagnostic: ``started-commit`` with
    ``fail = "none"`` on success, otherwise ``started-rollback`` with the
    stable ``race`` / ``io`` / ``other`` category. Flags list ``lock_wait``
    and ``retry`` in the order actually experienced. The business result is
    returned unchanged.
    """
    pre_status, pre_count = _pre_read(engine, machine_id)
    tid, marker_wait = diagnostics.insert_started(
        engine,
        machine_id=machine_id,
        op=op,
        started_at=_utc_now_iso(),
        status=pre_status or "",
        count=pre_count,
    )

    # The attempt's lock wait includes waiting to durably insert its own
    # marker before the joint transaction and waiting to take the joint
    # write lock; either is a contended lock wait.
    flags: list[str] = []
    if marker_wait >= _LOCK_WAIT_MIN_SECONDS:
        flags.append("lock_wait")
    retried = False
    error: BaseException | None = None
    result: dict[str, Any] | None = None
    reject_fail = diagnostics.FAIL_OTHER

    for attempt in range(_MAX_LOCK_ATTEMPTS):
        wait_box = [0.0]
        try:
            with _locked_connection(
                engine, on_lock_acquired=lambda elapsed: wait_box.__setitem__(0, elapsed)
            ) as conn:
                result = work(conn)
        except JointWriteOutcome as outcome:
            # Documented early outcome: the transaction rolled back cleanly,
            # nothing was written, and the caller gets its 404/409 result.
            if wait_box[0] >= _LOCK_WAIT_MIN_SECONDS and "lock_wait" not in flags:
                flags.append("lock_wait")
            if retried and "retry" not in flags:
                flags.append("retry")
            result = outcome.result
            reject_fail = outcome.fail
            break
        except OperationalError as exc:
            begin_wait = getattr(exc, "lock_wait_seconds", 0.0) or 0.0
            if (
                max(wait_box[0], begin_wait) >= _LOCK_WAIT_MIN_SECONDS
                and "lock_wait" not in flags
            ):
                flags.append("lock_wait")
            if not _is_lock_conflict(exc) or attempt == _MAX_LOCK_ATTEMPTS - 1:
                error = exc
                break
            # The attempt lost the concurrency conflict and will be retried;
            # lock_wait (already noted above if it occurred) precedes retry.
            retried = True
            if "retry" not in flags:
                flags.append("retry")
            time.sleep(min(0.01 * (attempt + 1), 0.2))
            continue
        except Exception as exc:  # noqa: BLE001 - every failure is categorized
            error = exc
            break
        else:
            if wait_box[0] >= _LOCK_WAIT_MIN_SECONDS and "lock_wait" not in flags:
                flags.append("lock_wait")
            if retried and "retry" not in flags:
                flags.append("retry")
            break

    if error is None:
        assert result is not None
        _finalize_joint_write(
            engine, tid, machine_id, op, result, flags, reject_fail
        )
        return result

    fail = _failure_category(error)
    if retried and "retry" not in flags:
        flags.append("retry")
    status, count, check = _terminal_snapshot(engine, machine_id)
    diagnostics.finalize(
        engine,
        tid,
        phase=diagnostics.PHASE_ROLLBACK,
        fail=fail,
        flags=flags,
        status=status or "",
        event=None,
        count=count,
        check=check,
    )
    raise error


def _finalize_joint_write(
    engine: Engine,
    tid: str,
    machine_id: str,
    op: str,
    result: dict[str, Any],
    flags: list[str],
    reject_fail: str = diagnostics.FAIL_OTHER,
) -> None:
    """Finalize the marker from a completed (committed or cleanly rejected)
    business attempt.

    A committed attempt carries the status, event count, and event-chain
    check captured inside the locked transaction (``diag_*``), so the record
    reflects the attempt's own outcome even if a concurrent writer commits
    before finalization. A cleanly rejected attempt has no such snapshot and
    re-reads the terminal state read-only.
    """
    if result.get("status") == "ok":
        status = result.get("diag_status")
        count = result.get("diag_count")
        check = result.get("diag_check")
        if status is None or count is None or check is None:
            status, count, check = _terminal_snapshot(engine, machine_id)
        event_id: str | None = None
        if op == "event":
            event_id = result["event"]["id"]
        diagnostics.finalize(
            engine,
            tid,
            phase=diagnostics.PHASE_COMMIT,
            fail=diagnostics.FAIL_NONE,
            flags=flags,
            status=status,
            event=event_id,
            count=count,
            check=check,
        )
        return

    # Cleanly rejected attempt (missing machine / same-target status): the
    # business transaction rolled back and wrote nothing. Snapshot the
    # terminal state it left behind.
    status, count, check = _terminal_snapshot(engine, machine_id)
    diagnostics.finalize(
        engine,
        tid,
        phase=diagnostics.PHASE_ROLLBACK,
        fail=reject_fail,
        flags=flags,
        status=status or "",
        event=None,
        count=count,
        check=check,
    )


def _load_events(conn: Connection, machine_id: str | None = None) -> list[Any]:
    statement = _TABLE.select()
    if machine_id is not None:
        statement = statement.where(_TABLE.c.machine_id == machine_id)
    statement = statement.order_by(_TABLE.c.created_at, _TABLE.c.id)
    return list(conn.execute(statement))


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


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on events written before the chain feature.

    Processing is per machine in (created_at, id) order. The recomputation is
    deterministic, so a restart over an already complete database issues no
    writes. Each machine is handled inside a locked transaction so a
    concurrent appender can neither interleave with the backfill nor fork.
    """
    with engine.connect() as conn:
        machine_ids = [
            row[0]
            for row in conn.execute(
                text(
                    f"SELECT DISTINCT machine_id FROM {_TABLE.name} "
                    "ORDER BY machine_id"
                )
            )
        ]

    for machine_id in machine_ids:
        def _work(conn: Connection, machine_id=machine_id) -> None:
            rows = _load_events(conn, machine_id)
            # previous_event_id is NULL on the first event, so completeness is
            # determined by the two hashes being present everywhere.
            if not rows or any(
                row._mapping["content_hash"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                for values in _recompute_rows(rows):
                    event_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == event_id)
                        .values(**values)
                    )

        _run_with_lock_retry(engine, _work)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _backfill_machine_rows(conn: Connection, machine_id: str) -> list[Any]:
    """Repair a machine's missing chain data, then reload and return its rows.

    Normally the startup backfill leaves every row complete. If any row is
    missing chain data (e.g. an external writer), rebuild the whole machine
    chain before appending so the new link has a sound tail.
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


def _mint_tail_link(
    conn: Connection,
    *,
    machine_id: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
) -> dict[str, Any]:
    """Read the machine's chain tail and insert one linked event.

    Must run inside the locked write transaction. The event id and timestamp
    are minted here and are guaranteed to sort after the current tail in
    ``(created_at, id)`` order, so the previous-event link always matches the
    order used by backfill and verification even under same-timestamp
    concurrency.
    """
    rows = _backfill_machine_rows(conn, machine_id)
    tail = rows[-1] if rows else None

    # Regenerate (rarely) until the new key sorts strictly after the tail.
    created_at = _utc_now_iso()
    event_id = str(uuid.uuid4())
    if tail is not None:
        tail_created_at = tail._mapping["created_at"]
        tail_id = tail._mapping["id"]
        if created_at < tail_created_at:
            created_at = tail_created_at
        while created_at == tail_created_at and event_id <= tail_id:
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
        action_type=action_type,
        resource=resource,
        allowed=allowed,
        reason=reason,
        created_at=created_at,
    )
    chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
    conn.execute(
        _TABLE.insert().values(
            id=event_id,
            machine_id=machine_id,
            action_type=action_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
            created_at=created_at,
            previous_event_id=previous_event_id,
            content_hash=content_hash,
            chain_hash=chain_hash,
        )
    )
    event = {
        "id": event_id,
        "machine_id": machine_id,
        "action_type": action_type,
        "resource": resource,
        "allowed": allowed,
        "reason": reason,
        "created_at": created_at,
        "previous_event_id": previous_event_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }
    # The new event's 1-based position in the machine's (created_at, id)
    # chain, captured inside the locked transaction for its diagnostic.
    position = len(rows) + 1
    return event, position


def append_decision_event(
    engine: Engine,
    *,
    machine_id: str,
    action_type: str,
    resource: str,
) -> dict[str, Any]:
    """Determine one authorization decision and append its event atomically.

    The machine lookup and status read, the declaration/policy evaluation,
    and the chain-tail append all happen inside a single locked write
    transaction. The lock is the same one status changes take
    (``change_machine_status`` runs its read/check/update in an identical
    locked transaction), so a status change and an event append have a single
    definite serial order and can never interleave:

    * status change first — this transaction reads the new status, so a
      suspension denies the event with ``machine_suspended`` instead of
      persisting a result computed from the old active state;
    * event append first — the event commits its pre-change result and the
      later status change never rewrites it.

    Returns ``{"status": "not_found"}`` when the machine is missing (nothing
    is written), otherwise ``{"status": "ok", "event": {...}}``. Every
    attempt records exactly one joint-write diagnostic via
    :func:`run_joint_write`.
    """
    from . import authorization

    def _work(conn: Connection) -> dict[str, Any]:
        status = authorization.machine_status(conn, machine_id)
        if status is None:
            raise JointWriteOutcome({"status": "not_found"})
        allowed, reason = authorization.decide(
            conn,
            machine_id,
            status,
            action_type,
            resource,
        )
        event, position = _mint_tail_link(
            conn,
            machine_id=machine_id,
            action_type=action_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
        )
        # Capture the attempt's own terminal snapshot inside the lock for
        # its diagnostic: the status the decision used, the new event's chain
        # position, and the chain check including the just-committed event.
        diag_check = verify_chain(conn, machine_id)
        return {
            "status": "ok",
            "event": event,
            "diag_status": status,
            "diag_count": position,
            "diag_check": diag_check,
        }

    return run_joint_write(
        engine, machine_id=machine_id, op="event", work=_work
    )


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's chain in (created_at, id) order.

    Returns ``(valid, checked_count, broken_event_id)``. The first event whose
    recomputed content hash, previous-event link, or chain hash differs from
    the stored values is reported; an empty chain is valid.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(_TABLE.c.machine_id == machine_id)
            .order_by(_TABLE.c.created_at, _TABLE.c.id)
        )
    )

    previous_event_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_event_id"] != previous_event_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_event_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
