"""One-time, short-lived authorization grants and their single consumption.

A grant turns one already-committed, audit-passing *allow* decision event into
a short-lived credential that can be consumed exactly once:

* :func:`issue_grant` accepts only an event of the path machine that committed
  ``allowed = true`` with ``reason = "allowed_by_policy"`` and whose immutable
  decision-basis snapshot passes the read-only consistency audit. The event,
  its basis, and the audit are read inside one locked write transaction, which
  also inserts the grant, so a concurrent burst signing for the same event has
  exactly one winner — a database-level unique constraint on ``event_id`` is
  the final backstop. Grants never modify the event, the basis, or any chain.
* :func:`consume_grant` atomically flips one unused, unexpired grant to
  ``consumed`` and inserts its single use record in the same locked
  transaction, so a concurrent burst of consumptions has exactly one success;
  already-consumed and expired grants are rejected and nothing is written.

Grants and use records live in their own tables and are never revoked,
renewed, or transferred. A failed attempt writes neither the grant state
change nor a use record: the whole operation either commits together or
leaves no trace.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError

from . import decision_basis, decision_basis_integrity
from .chain import _run_with_lock_retry
from .db import (
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantUse,
    Machine,
)

_EVENT_TABLE = AuthorizationDecisionEvent.__table__
_GRANT_TABLE = AuthorizationGrant.__table__
_USE_TABLE = AuthorizationGrantUse.__table__


def _utc_iso(instant: datetime) -> str:
    """Format an aware UTC instant as RFC 3339 text ending in ``Z``."""
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    """Parse a stored ``Z``-suffixed UTC stamp back to an aware instant."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def issue_grant(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Mint one one-time grant for one audit-passing allow event.

    All reads, the eligibility checks, and the insert run in one locked write
    transaction. Returns a status dict:

    * ``not_found`` — the path machine or the (path-machine scoped) event is
      missing or owned by another machine;
    * ``event_not_allowed`` — the event exists but did not commit
      ``allowed = true`` / ``reason = "allowed_by_policy"``;
    * ``decision_basis_unavailable`` — the event has no historical
      decision-basis snapshot (it predates the feature);
    * ``decision_basis_invalid`` — the snapshot exists but the read-only
      consistency audit rejects it;
    * ``grant_already_exists`` — the event already has a grant;
    * ``ok`` — with the new ``grant`` dict.

    Nothing is written on any non-ok outcome.
    """

    def _work(conn) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == event_id,
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if event_row is None:
            return {"status": "not_found"}

        event = event_row._mapping
        # Only a committed policy allow may ever be signed into a grant.
        if not event["allowed"] or event["reason"] != "allowed_by_policy":
            return {"status": "event_not_allowed"}

        document = decision_basis.load_document(
            conn, machine_id=machine_id, event_id=event_id
        )
        if document is None:
            # An event committed before the basis feature has no historical
            # basis: never reconstruct one from current data.
            return {"status": "decision_basis_unavailable"}

        conclusion = decision_basis_integrity.verify(
            conn, machine_id=machine_id, event=event_row, document=document
        )
        if not conclusion["valid"]:
            return {"status": "decision_basis_invalid"}

        existing = conn.execute(
            select(_GRANT_TABLE.c.id).where(_GRANT_TABLE.c.event_id == event_id)
        ).first()
        if existing is not None:
            return {"status": "grant_already_exists"}

        issued = datetime.now(timezone.utc)
        issued_at = _utc_iso(issued)
        expires_at = _utc_iso(issued + timedelta(seconds=ttl_seconds))
        grant_id = str(uuid.uuid4())
        conn.execute(
            _GRANT_TABLE.insert().values(
                id=grant_id,
                machine_id=machine_id,
                event_id=event_id,
                issued_at=issued_at,
                expires_at=expires_at,
                status="active",
                consumed_at=None,
            )
        )
        return {
            "status": "ok",
            "grant": {
                "id": grant_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "status": "active",
            },
        }

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique event_id constraint is the cross-backstop race guard: a
        # concurrent transaction that committed the first grant first wins.
        return {"status": "grant_already_exists"}


def consume_grant(
    engine: Engine,
    *,
    machine_id: str,
    grant_id: str,
) -> dict[str, Any]:
    """Atomically consume one unused, unexpired grant exactly once.

    The grant lookup, state checks, the state flip, and the use-record insert
    run in one locked write transaction. Returns a status dict:

    * ``not_found`` — the path machine is missing or the grant does not exist
      under it (a grant owned by another machine is indistinguishable from a
      missing one);
    * ``grant_consumed`` — the grant was already consumed;
    * ``grant_expired`` — the grant is past its ``expires_at``;
    * ``ok`` — with ``{grant_id, use_id, consumed_at}``.

    A rejection writes nothing; the status change and the use record commit
    together or leave no trace.
    """

    def _work(conn) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        grant_row = conn.execute(
            _GRANT_TABLE.select().where(
                _GRANT_TABLE.c.id == grant_id,
                _GRANT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if grant_row is None:
            return {"status": "not_found"}

        grant = grant_row._mapping
        if grant["status"] == "consumed":
            return {"status": "grant_consumed"}

        now = datetime.now(timezone.utc)
        if now >= _parse_utc(grant["expires_at"]):
            # Expiry is derived from the immutable expires_at; an expired
            # grant is never updated, renewed, or rewritten.
            return {"status": "grant_expired"}

        consumed_at = _utc_iso(now)
        use_id = str(uuid.uuid4())
        conn.execute(
            _GRANT_TABLE.update()
            .where(_GRANT_TABLE.c.id == grant_id)
            .values(status="consumed", consumed_at=consumed_at)
        )
        # The unique grant_id makes a second use row impossible even if the
        # state flip were ever raced; both statements commit together.
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

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        return {"status": "grant_consumed"}
