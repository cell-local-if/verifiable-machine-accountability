"""Instrumented runner for the joint status-change / decision-event write.

The only business writes that participate in the joint machine write are a
machine status change (``op = "change"``) and an authorization decision event
creation (``op = "event"``). Both go through :func:`run_joint_write`, which
adds a read-only observability layer around the locked write transaction
without changing the write semantics:

* every attempt that actually enters the joint write transaction leaves
  exactly one diagnostic row — ``started-commit`` for a business write that
  commits, ``started-rollback`` for one that begins and then aborts;
* a committed attempt's diagnostic is inserted in the *same* transaction as
  the business write, so the two appear together and can never diverge;
* a rolled-back attempt's diagnostic is written in an immediate follow-up
  transaction, so a rollback leaves exactly one intact record and never a
  fragment of the business write;
* lock waits and serialization retries that never enter the business work are
  coalesced into the one attempt that does, carrying the ``lock_wait`` and
  ``retry`` flags it actually experienced.

The runner never stores keys, secrets, policy text, or identity material: the
authorization outcome is summarized only by the created event id and the
machine's chain verdict as of the attempt's terminal instant.
"""

import time
import uuid
from typing import Any, Callable

from sqlalchemy import Connection, Engine
from sqlalchemy.exc import OperationalError

from . import chain
from .chain import (
    _is_lock_conflict,
    _locked_connection,
    _MAX_LOCK_ATTEMPTS,
    _utc_now_iso,
)
from .db import WriteTransactionDiagnostic

_TABLE = WriteTransactionDiagnostic.__table__

# Elapsed acquisition time above which an uncontended first attempt is treated
# as having waited on the lock. Acquisition of an idle SQLite file lock is
# sub-millisecond, so a conservative threshold avoids false positives while
# still recording a genuine busy-wait that resolves on the first attempt.
_LOCK_WAIT_MIN_SECONDS = 0.02


class NotFound(Exception):
    """The business work found no target machine.

    Raised inside the locked transaction before any business write. The
    transaction is released empty and, because the request never entered a
    joint write for a known machine, no diagnostic row is produced.
    """


class JointRollback(Exception):
    """The business work deliberately aborted after entering the transaction.

    The business write is rolled back (leaving no fragment) and one
    ``started-rollback`` diagnostic is persisted with the stable failure
    category ``fail`` (``race``, ``io``, or ``other``). ``payload`` is the
    ordinary API result dict the caller returns for this outcome.
    """

    def __init__(self, fail: str, payload: dict[str, Any]):
        super().__init__(fail)
        self.fail = fail
        self.payload = payload


def _flags(lock_wait: bool, retry: bool) -> list[str]:
    """Canonical ordered flag list: ``lock_wait`` then ``retry``."""
    flags: list[str] = []
    if lock_wait:
        flags.append("lock_wait")
    if retry:
        flags.append("retry")
    return flags


def _snapshot(conn: Connection, machine_id: str) -> tuple[bool, int, str | None]:
    """The machine's event-chain verdict and event count on this connection.

    Runs against the attempt's own connection, so a committing event attempt
    sees the row it just inserted; a rollback diagnostic written in a fresh
    transaction sees the committed state the rollback left behind.
    """
    return chain.verify_chain(conn, machine_id)


def _insert_diagnostic(
    conn: Connection,
    *,
    machine_id: str,
    op: str,
    phase: str,
    fail: str,
    flags: list[str],
    status: str,
    event_id: str | None,
) -> None:
    valid, count, broken_event_id = _snapshot(conn, machine_id)
    conn.execute(
        _TABLE.insert().values(
            id=str(uuid.uuid4()),
            machine_id=machine_id,
            at=_utc_now_iso(),
            op=op,
            phase=phase,
            fail=fail,
            flags=",".join(flags),
            status=status,
            event_id=event_id,
            event_count=count,
            chain_valid=valid,
            broken_event_id=broken_event_id,
        )
    )


def _persist_rollback_diagnostic(
    engine: Engine,
    *,
    machine_id: str,
    op: str,
    fail: str,
    flags: list[str],
) -> None:
    """Write one intact started-rollback record after a rolled-back attempt.

    Uses a fresh transaction (the business one already rolled back), so the
    diagnostic is committed on its own and never carries business state.
    """
    with engine.begin() as conn:
        _insert_diagnostic(
            conn,
            machine_id=machine_id,
            op=op,
            phase="started-rollback",
            fail=fail,
            flags=flags,
            status="rolled_back",
            event_id=None,
        )


def _classify_unexpected(error: Exception) -> str:
    """Map an unexpected in-transaction failure to a stable category.

    A persistence fault (a database ``OperationalError`` that is not a
    serialization/lock conflict) is ``io``; every other failure or crash is
    ``other``. Concurrency conflicts never reach here — they are retried.
    """
    if isinstance(error, OperationalError):
        return "io"
    return "other"


def run_joint_write(
    engine: Engine,
    *,
    machine_id: str,
    op: str,
    work: Callable[[Connection], dict[str, Any]],
) -> dict[str, Any]:
    """Run one joint business write and record a diagnostic per real attempt.

    ``work`` runs on the locked connection and must:

    * raise :class:`NotFound` when the machine is missing (nothing is written
      and no diagnostic is produced);
    * raise :class:`JointRollback` for a deliberate business abort (a
      same-target conflict is ``fail="race"``);
    * return the API payload on success. For an event attempt the payload's
      ``event["id"]`` is the created event id; for a status change there is no
      event.

    Lock/serialization failures are coalesced: failed lock acquisitions never
    enter the business work and are retried as one logical attempt, surfacing
    as ``lock_wait``/``retry`` flags on the attempt that finally runs. Any
    other failure rolls back, records a ``started-rollback`` diagnostic
    (``io`` or ``other``), and is re-raised so existing error behavior is
    unchanged.
    """
    lock_wait = False
    for attempt in range(_MAX_LOCK_ATTEMPTS):
        retry = attempt > 0
        acquired_at = time.monotonic()
        try:
            with _locked_connection(engine) as conn:
                if time.monotonic() - acquired_at >= _LOCK_WAIT_MIN_SECONDS:
                    lock_wait = True

                payload = work(conn)

                event_id = None
                if op == "event":
                    event_id = payload.get("event", {}).get("id")
                # The diagnostic commits in the same transaction as the
                # business write, so a committed attempt always has its record
                # and the two can never diverge.
                _insert_diagnostic(
                    conn,
                    machine_id=machine_id,
                    op=op,
                    phase="started-commit",
                    fail="none",
                    flags=_flags(lock_wait, retry),
                    status="committed",
                    event_id=event_id,
                )
                return payload

        except NotFound:
            # Missing target: no joint write for a known machine began, so no
            # diagnostic; the empty transaction is released by the context.
            return {"status": "not_found"}

        except JointRollback as rolled_back:
            # The business write began and deliberately aborted; the locked
            # context has already rolled it back. Record one intact rollback
            # diagnostic and return the ordinary business outcome.
            _persist_rollback_diagnostic(
                engine,
                machine_id=machine_id,
                op=op,
                fail=rolled_back.fail,
                flags=_flags(lock_wait, retry),
            )
            return rolled_back.payload

        except OperationalError as error:
            if _is_lock_conflict(error) and attempt < _MAX_LOCK_ATTEMPTS - 1:
                # The attempt never got far enough to enter business work (or
                # was aborted by serialization). Coalesce it into the attempt
                # that does, and retry.
                lock_wait = True
                time.sleep(min(0.01 * (attempt + 1), 0.2))
                continue
            if _is_lock_conflict(error):
                # Retries exhausted without entering a durable business write.
                # Leave one rollback record describing the conflict, then keep
                # the historical behavior of propagating the database error.
                _persist_rollback_diagnostic(
                    engine,
                    machine_id=machine_id,
                    op=op,
                    fail="race",
                    flags=_flags(lock_wait, True),
                )
                raise
            # A genuine persistence fault inside or at commit: rollback is
            # already handled by the locked context; record and re-raise.
            _persist_rollback_diagnostic(
                engine,
                machine_id=machine_id,
                op=op,
                fail="io",
                flags=_flags(lock_wait, retry),
            )
            raise

        except Exception as error:  # noqa: BLE001 - diagnose, then preserve behavior
            _persist_rollback_diagnostic(
                engine,
                machine_id=machine_id,
                op=op,
                fail=_classify_unexpected(error),
                flags=_flags(lock_wait, retry),
            )
            raise
