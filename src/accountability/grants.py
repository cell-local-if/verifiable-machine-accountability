"""One-time, short-lived authorization grants and their single consumption.

A grant converts exactly one already-audited allowed decision event into a
ticket that can be consumed at most once:

* issuance accepts only an event owned by the path machine with
  ``allowed = true`` and ``reason = "allowed_by_policy"`` whose stored
  decision-basis snapshot exists and passes the same read-only integrity
  audit the public ``decision-basis/integrity`` endpoint performs;
* each event can back at most one grant (a database-level unique
  constraint), so concurrent issuance against one event has exactly one
  winner and every losing request answers ``grant_already_exists``;
* a grant is born ``active`` with ``expires_at = issued_at + ttl_seconds``;
  consumption atomically flips it to ``consumed`` and inserts its sole use
  record in the same locked write transaction, so concurrent consumption
  succeeds exactly once and a failed attempt leaves neither the status flip
  nor a use row;
* grants and uses are never revoked, renewed, or transferred; expiry is
  derived from ``expires_at`` at consumption time without a sweeper, and
  every row lives in its own table that is created automatically on both
  empty and pre-existing databases without touching any existing table.

Both operations run on the same internal locked-connection primitive the
other non-joint writers use (``BEGIN IMMEDIATE`` on SQLite, SERIALIZABLE
elsewhere), so the read-validate-write sequence is serialized against
concurrent grant issuance and consumption.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from sqlalchemy import Connection, Engine, select
from sqlalchemy.exc import IntegrityError

from . import decision_basis_integrity
from .chain import _run_with_lock_retry
from .db import (
    AuthorizationDecisionBasis,
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantUse,
    Machine,
)

_GRANT_TABLE = AuthorizationGrant.__table__
_USE_TABLE = AuthorizationGrantUse.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__
_BASIS_TABLE = AuthorizationDecisionBasis.__table__
_MACHINE_TABLE = Machine.__table__


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_z(value: datetime) -> str:
    """Render a UTC instant as the codebase's RFC 3339 ``...Z`` text."""
    return value.isoformat().replace("+00:00", "Z")


def _parse_z(value: str) -> datetime:
    """Parse an RFC 3339 ``...Z`` stamp emitted here back into a UTC instant."""
    return datetime.fromisoformat(value[:-1] + "+00:00")


def issue_grant(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Mint the one grant for one audited allowed event.

    All reads and the insert happen in one locked write transaction. Returns
    a status dict:

    * ``not_found`` — the path machine is missing, or the event is missing or
      owned by another machine (nothing is written);
    * ``event_not_allowed`` — the event is not an ``allowed_by_policy`` allow
      (nothing is written);
    * ``decision_basis_unavailable`` — the event has no stored basis snapshot
      (nothing is written);
    * ``decision_basis_invalid`` — the stored snapshot fails the read-only
      integrity audit (nothing is written);
    * ``grant_already_exists`` — a grant for this event already exists; this
      is the concurrency-loser outcome (nothing is written);
    * ``ok`` — with the full stored ``grant`` record.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        # Scoped to the path machine: an event owned by another machine is a
        # missing object, never grantable.
        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == event_id,
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if event_row is None:
            return {"status": "not_found"}
        event = event_row._mapping

        if not bool(event["allowed"]) or event["reason"] != "allowed_by_policy":
            return {"status": "event_not_allowed"}

        # The one-per-event rule is enforced both here (the locked read sees
        # any earlier committed grant) and by the unique constraint below
        # (which rejects a same-instant racing insert on every backend).
        existing = conn.execute(
            select(_GRANT_TABLE.c.id).where(_GRANT_TABLE.c.event_id == event_id)
        ).first()
        if existing is not None:
            return {"status": "grant_already_exists"}

        document = conn.execute(
            select(_BASIS_TABLE.c.document).where(
                _BASIS_TABLE.c.event_id == event_id,
                _BASIS_TABLE.c.machine_id == machine_id,
            )
        ).scalar()
        if document is None:
            return {"status": "decision_basis_unavailable"}

        # The same read-only audit the decision-basis integrity endpoint runs;
        # verify() only issues reads and accepts a connection, so it runs
        # inside this locked transaction on the exact snapshot the grant
        # signs. A shim row exposes the event attributes verify() reads.
        event_view = SimpleNamespace(
            **{name: event[name] for name in event.keys()}
        )
        conclusion = decision_basis_integrity.verify(
            conn,
            machine_id=machine_id,
            event=event_view,
            document=document,
        )
        if not conclusion["valid"]:
            return {"status": "decision_basis_invalid"}

        now = _utc_now()
        issued_at = _to_z(now)
        expires_at = _to_z(now + timedelta(seconds=ttl_seconds))
        grant_id = str(uuid.uuid4())
        grant_values = {
            "id": grant_id,
            "machine_id": machine_id,
            "event_id": event_id,
            "ttl_seconds": ttl_seconds,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "status": "active",
        }
        # The unique(event_id) constraint is the final guarantee for a racing
        # pair whose backends let both transactions read no existing grant
        # (e.g. SERIALIZABLE on PostgreSQL); the violation propagates out of
        # the locked transaction (rolled back cleanly) and is classified by
        # issue_grant() below. On SQLite the locked pre-check already covers
        # it, so this insert never races.
        conn.execute(_GRANT_TABLE.insert().values(**grant_values))
        return {"status": "ok", "grant": grant_values}

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The only constraint this insert can hit is the one-grant-per-event
        # unique key; this request is the documented concurrency loser and
        # its transaction has already been rolled back with nothing written.
        return {"status": "grant_already_exists"}


def consume_grant(
    engine: Engine,
    *,
    machine_id: str,
    grant_id: str,
) -> dict[str, Any]:
    """Atomically consume one active, unexpired grant exactly once.

    All reads, the status flip, and the use-record insert happen in one
    locked write transaction. Returns a status dict:

    * ``not_found`` — the path machine is missing, or the grant is missing or
      owned by another machine (nothing is written);
    * ``grant_consumed`` — the grant's single consumption already happened
      (nothing is written); consumed takes precedence over expired;
    * ``grant_expired`` — the grant is still active but the current UTC
      instant is at or past ``expires_at`` (nothing is written);
    * ``ok`` — with ``{grant_id, use_id, consumed_at}`` for the one use row.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        grant_row = conn.execute(
            _GRANT_TABLE.select().where(_GRANT_TABLE.c.id == grant_id)
        ).first()
        if grant_row is None or grant_row._mapping["machine_id"] != machine_id:
            return {"status": "not_found"}
        grant = grant_row._mapping

        if grant["status"] == "consumed":
            return {"status": "grant_consumed"}

        now = _utc_now()
        if now >= _parse_z(grant["expires_at"]):
            return {"status": "grant_expired"}

        consumed_at = _to_z(now)
        use_id = str(uuid.uuid4())
        # Optimistic guard in addition to the write lock: the flip only lands
        # while the row is still active, so even a raced external writer
        # cannot produce a second consumption.
        result = conn.execute(
            _GRANT_TABLE.update()
            .where(
                _GRANT_TABLE.c.id == grant_id,
                _GRANT_TABLE.c.status == "active",
            )
            .values(status="consumed")
        )
        if result.rowcount == 0:
            return {"status": "grant_consumed"}
        # The sole use row joins the same transaction: the flip and the record
        # commit together or leave no trace (no half consumption).
        conn.execute(
            _USE_TABLE.insert().values(
                id=use_id,
                grant_id=grant_id,
                machine_id=machine_id,
                event_id=grant["event_id"],
                consumed_at=consumed_at,
            )
        )
        return {
            "status": "ok",
            "use": {
                "grant_id": grant_id,
                "use_id": use_id,
                "consumed_at": consumed_at,
            },
        }

    return _run_with_lock_retry(engine, _work)
