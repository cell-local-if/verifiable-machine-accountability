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
* :func:`consume_grant` atomically flips one unused, unexpired, unrevoked
  grant to ``consumed`` and inserts its single use record in the same locked
  transaction, so a concurrent burst of consumptions has exactly one success;
  already-consumed, revoked, and expired grants are rejected and nothing is
  written.
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


# --------------------------------------------------------------------------- #
# Read-only audit listing
# --------------------------------------------------------------------------- #

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _issued_at_instant(value: object) -> datetime:
    """Parse a stored ``issued_at`` to its actual UTC instant for ordering.

    Legitimately written stamps always satisfy the RFC 3339 ``Z`` contract,
    so parsing cannot fail for them; a damaged value that no longer parses
    sorts after every parseable stamp (instead of crashing the read-only
    audit), and locating the cursor on stored text keeps such a row pageable.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _grant_id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored grant identifier.

    Legitimately written ids are strings; a damaged non-string id never
    crashes the ordering — it sorts after string ids within one instant, and
    the stored value is still emitted verbatim.
    """
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def encode_grant_cursor(issued_at: str, grant_id: str) -> str:
    """Encode the opaque keyset position ``<issued_at original text>|<id>``."""
    return f"{issued_at}|{grant_id}"


def _parse_utc_or_none(value: object) -> datetime | None:
    """Parse a stored ``Z`` UTC stamp, or ``None`` when it no longer parses."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _grant_to_item(grant: Any, use: Any | None, now: datetime) -> dict[str, Any]:
    """One stored grant as the fixed ten-field audit item.

    The four stored stamps (``issued_at``, ``expires_at``, ``consumed_at``,
    ``revoked_at``) are emitted exactly as stored. The two terminal states
    are sticky: a consumed or revoked grant keeps that status even once its
    TTL has elapsed. For any other grant the status is derived read-only from
    the immutable ``expires_at`` — ``expired`` once the current UTC instant
    reaches it, ``active`` before — and this derivation never rewrites the
    stored status. A non-terminal grant whose ``expires_at`` no longer parses
    cannot be shown to still be unexpired, so it answers ``expired``
    fail-closed rather than crashing or fabricating an instant.
    ``use_id``/``use_at`` name the grant's unique consumption record and are
    both ``None`` when no such record exists.
    """
    stored_status = grant["status"]
    if stored_status == "consumed":
        status = "consumed"
    elif stored_status == "revoked":
        status = "revoked"
    else:
        expires_at = _parse_utc_or_none(grant["expires_at"])
        # Expiry is derived from the immutable expires_at; an expired grant
        # is never updated, renewed, or rewritten.
        if expires_at is None or now >= expires_at:
            status = "expired"
        else:
            status = "active"

    use_id = use["id"] if use is not None else None
    use_at = use["consumed_at"] if use is not None else None

    return {
        "id": grant["id"],
        "machine_id": grant["machine_id"],
        "event_id": grant["event_id"],
        "issued_at": grant["issued_at"],
        "expires_at": grant["expires_at"],
        "status": status,
        "consumed_at": grant["consumed_at"],
        "revoked_at": grant["revoked_at"],
        "use_id": use_id,
        "use_at": use_at,
    }


def list_grants(
    engine: Engine,
    *,
    machine_id: str,
    limit: int,
    cursor: str | None,
) -> dict[str, Any]:
    """Read one keyset page of one machine's grants without writing anything.

    Issues only ``SELECT`` statements in a plain read transaction: it never
    creates, repairs, updates, or deletes a grant, use record, lifecycle
    event, or chain hash, and expiry is derived rather than persisted.
    Returns a status dict:

    * ``not_found`` — the path machine is missing;
    * ``invalid_cursor`` — a well-shaped cursor whose ``(issued_at, id)``
      position names no stored grant of the path machine (including a
      position built from another machine's grant);
    * ``ok`` — with ``items`` (the page, oldest ``issued_at`` first, id
      ascending for a tie) and ``next_cursor`` (the opaque position pointing
      at the next page's first item, or ``None`` on the last page).

    The position is resolved while the path machine's grants are read,
    before the machine existence check, so a parameter error always takes
    priority over ``not_found``.
    """
    with engine.connect() as conn:
        grant_rows = conn.execute(
            _GRANT_TABLE.select().where(
                _GRANT_TABLE.c.machine_id == machine_id
            )
        ).all()
        use_rows = conn.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.machine_id == machine_id)
        ).all()

        ordered = sorted(
            grant_rows,
            key=lambda row: (
                _issued_at_instant(row._mapping["issued_at"]),
                _grant_id_key(row._mapping["id"]),
            ),
        )

        start = 0
        if cursor is not None:
            cursor_issued_at, cursor_id = cursor.rsplit("|", 1)
            # Exclusive keyset position. A cursor this endpoint issued always
            # names a stored row, so locate it by its exact stored
            # ``(issued_at text, id)`` pair; a well-shaped cursor that no row
            # of the path machine matches (a deleted grant, another machine's
            # position, drifted text) cannot be positioned and is rejected.
            # The grant read is filtered by the path machine, so a cursor can
            # never page into another machine's grants.
            positions = [
                index
                for index, row in enumerate(ordered)
                if row._mapping["issued_at"] == cursor_issued_at
                and row._mapping["id"] == cursor_id
            ]
            if not positions:
                return {"status": "invalid_cursor"}
            start = positions[0] + 1

        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        uses_by_grant = {
            row._mapping["grant_id"]: row._mapping for row in use_rows
        }

        remaining = ordered[start:]
        page = remaining[:limit]
        now = datetime.now(timezone.utc)
        items = [
            _grant_to_item(
                row._mapping, uses_by_grant.get(row._mapping["id"]), now
            )
            for row in page
        ]
        has_more = len(remaining) > len(page)
        next_cursor = (
            encode_grant_cursor(
                page[-1]._mapping["issued_at"], page[-1]._mapping["id"]
            )
            if page and has_more
            else None
        )
        return {"status": "ok", "items": items, "next_cursor": next_cursor}
