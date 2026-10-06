"""One-time, short-lived authorization grants and their single consumption.

A grant turns one already-committed, audit-passing *allow* decision event into
a short-lived credential that can be consumed exactly once:

* :func:`issue_grant` accepts only an event of the path machine that committed
  ``allowed = true`` with ``reason = "allowed_by_policy"`` and whose immutable
  decision-basis snapshot passes the read-only consistency audit. The event,
  its basis, and the audit are read inside one locked write transaction, which
  also inserts the grant, so a concurrent burst signing for the same event has
  exactly one winner — a database-level unique constraint on ``event_id`` is
  the final backstop. Once every historical check has passed, the machine's
  current status is read inside the same locked transaction before the
  insert: a suspended machine is rejected with ``machine_suspended`` and no
  grant is created, so a status change and an issue racing each other have
  one definite serial order. Grants never modify the event, the basis, or
  any chain.
* :func:`issue_grants_batch` signs several qualified allow events into
  independent grants in one locked write transaction: every item is checked
  in input order against exactly the single-issue eligibility rules, the
  first failing item decides the whole batch's outcome, and a failure leaves
  no grant behind at all. A successful batch stamps every grant with the same
  UTC issue moment and its own ``ttl_seconds`` expiry.
* :func:`consume_grant` atomically flips one unused, unexpired, unrevoked
  grant to ``consumed`` and inserts its single use record in the same locked
  transaction, so a concurrent burst of consumptions has exactly one success;
  already-consumed, revoked, and expired grants are rejected and nothing is
  written. A grant that is still usable but belongs to a suspended machine is
  rejected with ``machine_suspended`` — after the not-found, consumed,
  revoked, and expired outcomes — and that rejection likewise writes nothing:
  the machine status change and the consumption take the same write lock, so
  whichever commits first decides, and the grant stays consumable again once
  the machine returns to ``active``.
* :func:`consume_grants_batch` confirms the use of several grants in one
  locked write transaction: every grant is checked in input order against
  exactly the single-consume rules, the first failing grant decides the
  whole batch's outcome and nothing is written, and a successful batch
  flips every grant to ``consumed``, inserts exactly one use record per
  grant, and appends one ``consumed`` lifecycle event per grant — all
  stamped with one shared UTC consume moment. A concurrent single consume
  or another batch racing for any of these grants has exactly one winner;
  the loser observes the winner's committed state in its ordered checks and
  rolls back entirely.
* :func:`revoke_grant` performs the emergency revocation: it atomically flips
  one unused, unexpired, unrevoked grant to ``revoked`` and stamps
  ``revoked_at`` in the same kind of locked transaction, so a concurrent burst
  of revocations has exactly one winner and a revocation racing a consumption
  has one definite terminal winner — consume first makes the revocation answer
  ``grant_consumed``; revoke first makes the consumption answer
  ``grant_revoked``.

Grants and use records live in their own tables and are never renewed,
transferred, or deleted. Revocation never touches the decision event, its
immutable basis, any hash chain, evidence, incident, diagnostic, or export. A
failed attempt writes neither the grant state change nor a use record: the
whole operation either commits together or leaves no trace.

Every successful action also appends exactly one immutable lifecycle audit
event (``issued`` / ``consumed`` / ``revoked``) to the machine's
tamper-evident grant-lifecycle chain inside the same locked transaction
(:mod:`.grant_lifecycle`), reusing the response's own moment as the event's
``occurred_at``; a failed attempt appends no event.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError

from . import decision_basis, decision_basis_integrity, grant_lifecycle
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


def migrate_schema(engine: Engine) -> None:
    """Add the revocation column to databases created before the feature."""
    inspector = inspect(engine)
    if _GRANT_TABLE.name not in inspector.get_table_names():
        # ``create_all`` builds a current-schema table; there is nothing to
        # bring forward.
        return
    existing = {
        column["name"]
        for column in inspector.get_columns(_GRANT_TABLE.name)
    }
    if "revoked_at" not in existing:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"ALTER TABLE {_GRANT_TABLE.name} "
                    "ADD COLUMN revoked_at VARCHAR"
                )
            )


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
    * ``machine_suspended`` — every historical check passed but the
      machine's current status is suspended; the not-found, event, basis,
      and existing-grant outcomes above keep their precedence and are
      answered even while the machine is suspended;
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

        if machine._mapping["status"] == "suspended":
            # The current-state gate runs last, after every historical
            # check, inside the same locked transaction the status change
            # takes: a suspension that committed first is already visible
            # here and no grant is created, while an issue that commits
            # first keeps its success. Nothing is written on this outcome.
            return {"status": "machine_suspended"}

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
        # The audit event commits in the same locked transaction as the
        # grant, reusing the response's issued_at as its occurred_at.
        grant_lifecycle.append_event(
            conn,
            machine_id=machine_id,
            grant_id=grant_id,
            authorization_event_id=event_id,
            type="issued",
            occurred_at=issued_at,
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


def issue_grants_batch(
    engine: Engine,
    *,
    machine_id: str,
    items: list[tuple[str, int]],
) -> dict[str, Any]:
    """Mint one one-time grant per ``(event_id, ttl_seconds)`` item, or none.

    Every item is checked in input order against exactly the eligibility
    rules of :func:`issue_grant` — path-machine ownership, committed policy
    allow, historical basis snapshot present, read-only consistency audit
    passing, no existing grant, machine not currently suspended — inside one
    locked write transaction that also inserts every grant. The first failing
    item in input order decides the whole batch's outcome; the status dict
    uses the same status values as :func:`issue_grant` (``not_found``,
    ``event_not_allowed``, ``decision_basis_unavailable``,
    ``decision_basis_invalid``, ``grant_already_exists``,
    ``machine_suspended``, or ``ok`` with the ordered ``grants`` list).

    A rejected batch writes nothing: no partial issue is ever committed. A
    successful batch stamps every grant with one shared UTC issue moment and
    an expiry of that moment plus the item's own ``ttl_seconds``. The unique
    ``event_id`` constraint remains the cross-request backstop: a concurrent
    single or batch issue racing for any of these events has exactly one
    winner, and the loser's whole transaction rolls back.
    """

    def _work(conn) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        # Eligibility is decided per item in input order before anything is
        # written: the first failing item's outcome is the batch's outcome.
        for event_id, _ttl in items:
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
                # An event committed before the basis feature has no
                # historical basis: never reconstruct one from current data.
                return {"status": "decision_basis_unavailable"}

            conclusion = decision_basis_integrity.verify(
                conn, machine_id=machine_id, event=event_row, document=document
            )
            if not conclusion["valid"]:
                return {"status": "decision_basis_invalid"}

            existing = conn.execute(
                select(_GRANT_TABLE.c.id).where(
                    _GRANT_TABLE.c.event_id == event_id
                )
            ).first()
            if existing is not None:
                return {"status": "grant_already_exists"}

            if machine._mapping["status"] == "suspended":
                # The current-state gate runs last for every item, after
                # that item's historical checks, inside the same locked
                # transaction the status change takes: a suspended machine
                # rejects the whole batch here and nothing is written.
                return {"status": "machine_suspended"}

        # Every item qualified: one shared issue moment for the whole batch,
        # each grant expiring that moment plus its own TTL.
        issued = datetime.now(timezone.utc)
        issued_at = _utc_iso(issued)
        issued_grants = []
        for event_id, ttl_seconds in items:
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
            # The audit event commits in the same locked transaction as the
            # grant, reusing the response's issued_at as its occurred_at.
            grant_lifecycle.append_event(
                conn,
                machine_id=machine_id,
                grant_id=grant_id,
                authorization_event_id=event_id,
                type="issued",
                occurred_at=issued_at,
            )
            issued_grants.append(
                {
                    "id": grant_id,
                    "machine_id": machine_id,
                    "event_id": event_id,
                    "issued_at": issued_at,
                    "expires_at": expires_at,
                    "status": "active",
                }
            )
        return {"status": "ok", "grants": issued_grants}

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique event_id constraint is the cross-backstop race guard: a
        # concurrent transaction that committed a grant for any of these
        # events first wins, and this batch's partial inserts roll back with
        # the failed transaction.
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
    * ``grant_revoked`` — the grant was emergency-revoked;
    * ``grant_expired`` — the grant is past its ``expires_at``;
    * ``machine_suspended`` — the grant is still usable but its machine is
      suspended; the terminal and derived grant outcomes above keep their
      precedence and are answered even while the machine is suspended;
    * ``ok`` — with ``{grant_id, use_id, consumed_at}``.

    A rejection writes nothing; the status change and the use record commit
    together or leave no trace. The machine status is read inside this same
    locked transaction — the lock status changes take — so a suspension that
    commits first makes the consumption answer ``machine_suspended`` while a
    consumption that commits first keeps its unique success; reactivating the
    machine makes the same unconsumed, unexpired grant consumable again.
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
        # The two terminal states are checked before the derived expiry: a
        # consumed or revoked grant keeps answering its own outcome even once
        # its TTL has elapsed.
        if grant["status"] == "consumed":
            return {"status": "grant_consumed"}
        if grant["status"] == "revoked":
            return {"status": "grant_revoked"}

        now = datetime.now(timezone.utc)
        if now >= _parse_utc(grant["expires_at"]):
            # Expiry is derived from the immutable expires_at; an expired
            # grant is never updated, renewed, or rewritten.
            return {"status": "grant_expired"}

        if machine._mapping["status"] == "suspended":
            # A suspended machine may start no new authorization action. Only
            # a still-usable grant reaches this gate: the grant's state and
            # stamps are untouched, no use record or lifecycle event is
            # written, and the grant is consumable again after reactivation.
            return {"status": "machine_suspended"}

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
        # The audit event commits in the same locked transaction as the state
        # flip and the use record, reusing the response's consumed_at.
        grant_lifecycle.append_event(
            conn,
            machine_id=machine_id,
            grant_id=grant_id,
            authorization_event_id=grant["event_id"],
            type="consumed",
            occurred_at=consumed_at,
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


def consume_grants_batch(
    engine: Engine,
    *,
    machine_id: str,
    grant_ids: list[str],
) -> dict[str, Any]:
    """Atomically consume several unused, unexpired grants, or none.

    Every grant is checked in input order against exactly the eligibility
    rules of :func:`consume_grant` — path-machine ownership, not consumed,
    not revoked, not expired, machine not suspended — inside one locked
    write transaction that also performs every state flip, use-record
    insert, and lifecycle event. The first failing grant in input order
    decides the whole batch's outcome; the status dict uses the same
    status values as :func:`consume_grant` (``not_found``,
    ``grant_consumed``, ``grant_revoked``, ``grant_expired``,
    ``machine_suspended``, or ``ok`` with the ordered ``uses`` list of
    ``{grant_id, use_id, consumed_at}`` dicts).

    A rejected batch writes nothing: no partial consumption is ever
    committed. A successful batch stamps every use record, every grant's
    ``consumed_at``, and every lifecycle event with one shared UTC consume
    moment. The lock is the same one the single consume, the revocation,
    and the machine status change take, so a concurrent single consume or
    another batch racing for any of these grants has exactly one winner
    and the loser's whole transaction rolls back; the unique ``grant_id``
    on the use table remains the cross-request backstop.
    """

    def _work(conn) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        now = datetime.now(timezone.utc)
        # Eligibility is decided per grant in input order before anything
        # is written: the first failing grant's outcome is the batch's
        # outcome. The two terminal states keep their precedence over the
        # derived expiry, and all three keep precedence over the suspended
        # gate, exactly as in the single consume.
        grant_rows = []
        for grant_id in grant_ids:
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
            if grant["status"] == "revoked":
                return {"status": "grant_revoked"}
            if now >= _parse_utc(grant["expires_at"]):
                # Expiry is derived from the immutable expires_at; an
                # expired grant is never updated, renewed, or rewritten.
                return {"status": "grant_expired"}
            if machine._mapping["status"] == "suspended":
                # Only a still-usable grant reaches this gate: nothing is
                # written and the grant is consumable again after
                # reactivation.
                return {"status": "machine_suspended"}
            grant_rows.append(grant)

        # Every grant qualified: one shared consume moment for the whole
        # batch, one state flip, one use record, and one lifecycle event
        # per grant, all committed together.
        consumed_at = _utc_iso(now)
        uses = []
        for grant_id, grant in zip(grant_ids, grant_rows):
            use_id = str(uuid.uuid4())
            conn.execute(
                _GRANT_TABLE.update()
                .where(_GRANT_TABLE.c.id == grant_id)
                .values(status="consumed", consumed_at=consumed_at)
            )
            # The unique grant_id makes a second use row impossible even
            # if the state flip were ever raced; all statements of the
            # batch commit together.
            conn.execute(
                _USE_TABLE.insert().values(
                    id=use_id,
                    grant_id=grant_id,
                    machine_id=machine_id,
                    event_id=grant["event_id"],
                    consumed_at=consumed_at,
                )
            )
            # The audit event commits in the same locked transaction as
            # the state flip and the use record, reusing the response's
            # consumed_at.
            grant_lifecycle.append_event(
                conn,
                machine_id=machine_id,
                grant_id=grant_id,
                authorization_event_id=grant["event_id"],
                type="consumed",
                occurred_at=consumed_at,
            )
            uses.append(
                {
                    "grant_id": grant_id,
                    "use_id": use_id,
                    "consumed_at": consumed_at,
                }
            )
        return {"status": "ok", "uses": uses}

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique grant_id constraint is the cross-backstop race guard:
        # a concurrent transaction that consumed any of these grants first
        # wins, and this batch's partial writes roll back with the failed
        # transaction.
        return {"status": "grant_consumed"}


def revoke_grant(
    engine: Engine,
    *,
    machine_id: str,
    grant_id: str,
) -> dict[str, Any]:
    """Emergency-revoke one unconsumed, unrevoked, unexpired grant.

    The grant lookup, state checks, and the state flip with ``revoked_at``
    run in one locked write transaction, the same lock
    :func:`consume_grant` takes, so revocation and consumption have one
    definite serial order. Returns a status dict:

    * ``not_found`` — the path machine is missing or the grant does not exist
      under it (a grant owned by another machine is indistinguishable from a
      missing one);
    * ``grant_consumed`` — the grant was already consumed;
    * ``grant_revoked`` — the grant was already revoked;
    * ``grant_expired`` — the grant is past its ``expires_at``;
    * ``ok`` — with ``{grant_id, revoked_at, status: "revoked"}``.

    A rejection writes nothing. Revocation never touches the use table, the
    decision event, the basis, or any chain: only the grant's own
    ``status`` / ``revoked_at`` change, and an old grant's ``issued_at``,
    ``expires_at``, ``consumed_at`` are never rewritten.
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
        if grant["status"] == "revoked":
            return {"status": "grant_revoked"}

        now = datetime.now(timezone.utc)
        if now >= _parse_utc(grant["expires_at"]):
            # Expiry is derived from the immutable expires_at; an expired
            # grant is never updated, renewed, or rewritten.
            return {"status": "grant_expired"}

        revoked_at = _utc_iso(now)
        # The lock is the same one consume takes: a concurrent consumption or
        # revocation that committed first is already visible in the status
        # check above and cannot interleave with this flip.
        conn.execute(
            _GRANT_TABLE.update()
            .where(_GRANT_TABLE.c.id == grant_id)
            .values(status="revoked", revoked_at=revoked_at)
        )
        # The audit event commits in the same locked transaction as the state
        # flip, reusing the response's revoked_at.
        grant_lifecycle.append_event(
            conn,
            machine_id=machine_id,
            grant_id=grant_id,
            authorization_event_id=grant["event_id"],
            type="revoked",
            occurred_at=revoked_at,
        )
        return {
            "status": "ok",
            "revocation": {
                "grant_id": grant_id,
                "revoked_at": revoked_at,
                "status": "revoked",
            },
        }

    return _run_with_lock_retry(engine, _work)
