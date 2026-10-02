"""Immutable execution receipts bound to one consumed authorization grant.

A receipt closes the loop between a one-time grant and what actually happened:
after a grant has been consumed exactly once, its single consumption can be
bound to exactly one immutable execution receipt recording the cited evidence,
the action actually performed (``action_type`` / ``resource``), its
``outcome``, and the ``matches_authorization`` verdict comparing the actual
action with the original authorization decision event.

* :func:`append_receipt` runs the machine, grant, consumption, evidence, and
  duplicate lookups and the insert in one locked write transaction — the same
  lock primitive the other per-machine writers use — so a concurrent burst of
  receipts for one grant has exactly one winner; database-level unique
  constraints on ``grant_id`` and ``use_id`` are the final backstop. A rejected
  attempt writes nothing and never modifies the grant, its use row, the
  decision event, evidence, or any other record.
* :func:`load_receipt` is the strictly read-only single-receipt lookup.
* :func:`verify_machine_chain` is the read-only integrity audit of one
  machine's receipt chain.

Each machine's receipts form an ordered chain, ordered by the actual UTC
instant of ``created_at`` and then by ``id``, following the same rules as the
other per-machine chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the receipt's own content fields: ``{id, machine_id, grant_id, use_id,
  authorization_event_id, evidence_id, action_type, resource, outcome,
  matches_authorization, created_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first receipt uses the empty string as the previous chain hash.
* ``previous_receipt_id`` is ``None`` for a machine's first receipt and the
  prior receipt's id otherwise.

Receipts are isolated per machine: one machine's chain never contains another
machine's receipts. The table is created automatically at startup on
databases that predate the feature; existing rows are never rewritten.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import (
    AuthorizationDecisionEvidence,
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantExecutionReceipt,
    AuthorizationGrantUse,
    Machine,
)

_HASH_LEN = 64

_TABLE = AuthorizationGrantExecutionReceipt.__table__
_EVIDENCE_TABLE = AuthorizationDecisionEvidence.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__
_GRANT_TABLE = AuthorizationGrant.__table__
_USE_TABLE = AuthorizationGrantUse.__table__

_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "grant_id",
    "use_id",
    "authorization_event_id",
    "evidence_id",
    "action_type",
    "resource",
    "outcome",
    "matches_authorization",
    "created_at",
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    grant_id: str,
    use_id: str,
    authorization_event_id: str,
    evidence_id: str,
    action_type: str,
    resource: str,
    outcome: str,
    matches_authorization: bool,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "grant_id": grant_id,
            "use_id": use_id,
            "authorization_event_id": authorization_event_id,
            "evidence_id": evidence_id,
            "action_type": action_type,
            "resource": resource,
            "outcome": outcome,
            "matches_authorization": matches_authorization,
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
    """Add missing columns to a receipt table created by an earlier schema.

    ``Base.metadata.create_all`` already creates the table on databases that
    predate the feature; this hook only brings a pre-existing partial table
    forward. Existing rows are never rewritten.
    """
    inspector = inspect(engine)
    if _TABLE.name not in inspector.get_table_names():
        # ``create_all`` builds a current-schema table; there is nothing to
        # bring forward.
        return
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "previous_receipt_id": "VARCHAR(36)",
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


def _created_at_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a damaged value that no longer parses sorts
    after every parseable record (its content hash cannot verify anyway)
    instead of crashing the read-only audit or query.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _created_at_parseable(value: object) -> bool:
    """Whether a stored ``created_at`` still parses to a UTC instant."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
            return True
        except ValueError:
            pass
    return False


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order receipt rows by the actual UTC instant of ``created_at``.

    Ordering is by the parsed UTC instant and then by ``id``, so an
    exact-second stamp sorts before any fractional-second stamp of the same
    second (ISO text alone cannot express that, since ``.`` precedes ``Z``).
    A stamp that no longer parses sorts deterministically last, and a damaged
    non-string id sorts as empty rather than crashing the comparison.
    """
    return sorted(
        rows,
        key=lambda row: (
            _created_at_instant(row._mapping["created_at"]),
            row._mapping["id"] if isinstance(row._mapping["id"], str) else "",
        ),
    )


def _load_records(conn, machine_id: str) -> list[Any]:
    """Load one machine's receipts in chain (instant, id) order."""
    rows = list(
        conn.execute(_TABLE.select().where(_TABLE.c.machine_id == machine_id))
    )
    return _chain_order(rows)


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_receipt_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_receipt_id"] != previous_receipt_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_receipt_id": previous_receipt_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_receipt_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def append_receipt(
    engine: Engine,
    *,
    machine_id: str,
    grant_id: str,
    evidence_id: str,
    action_type: str,
    resource: str,
    outcome: str,
) -> dict[str, Any]:
    """Bind one consumed grant's single execution receipt, exactly once.

    The machine, grant, consumption, evidence, and duplicate lookups and the
    insert run in one locked write transaction. Returns a status dict:

    * ``not_found`` — the path machine is missing; the grant does not exist
      under the path machine (a grant owned by another machine is
      indistinguishable from a missing one); or the grant's consumption record
      is missing despite the consumed state or owned by another machine;
    * ``grant_not_consumed`` — the grant exists but has never been consumed;
    * ``evidence_not_found`` — the evidence does not exist, or is not attached
      to the grant's own authorization event under the path machine;
    * ``duplicate_receipt`` — the grant's consumption already has a receipt;
    * ``ok`` — with the new ``receipt`` dict.

    A rejected attempt writes nothing; a successful one inserts the receipt
    row only and never modifies the grant, its use record, the decision event,
    the evidence, or any chain.
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

        # A consumed grant must carry its consumption record on the path
        # machine bound to the grant's own event: a consumed grant with no use
        # row (or a cross-machine/cross-event one) is a missing consumption and
        # answers not_found before any further check. A grant that has never
        # been consumed has no use row by construction; that is distinguished
        # below as grant_not_consumed.
        if grant["status"] == "consumed":
            use_row = conn.execute(
                _USE_TABLE.select().where(_USE_TABLE.c.grant_id == grant_id)
            ).first()
            if use_row is None:
                return {"status": "not_found"}
            use = use_row._mapping
            if (
                use["machine_id"] != machine_id
                or use["event_id"] != grant["event_id"]
            ):
                return {"status": "not_found"}

        # The evidence must exist on the grant's own authorization event under
        # the path machine: a missing id, another event's evidence, or another
        # machine's evidence are all the same evidence_not_found outcome. This
        # check precedes the not-consumed/duplicate conflict checks.
        evidence = conn.execute(
            _EVIDENCE_TABLE.select().where(
                _EVIDENCE_TABLE.c.id == evidence_id,
                _EVIDENCE_TABLE.c.event_id == grant["event_id"],
                _EVIDENCE_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if evidence is None:
            return {"status": "evidence_not_found"}

        # An active (including expired) or emergency-revoked grant has never
        # been consumed, so the receipt has nothing to bind to.
        if grant["status"] != "consumed":
            return {"status": "grant_not_consumed"}

        # Each consumed authorization carries at most one receipt.
        existing = conn.execute(
            select(_TABLE.c.id).where(_TABLE.c.grant_id == grant_id)
        ).first()
        if existing is not None:
            return {"status": "duplicate_receipt"}

        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == grant["event_id"],
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        # The grant's own event cannot be missing here (a grant always names a
        # path-machine event); treat such a damaged state like a missing grant
        # rather than fabricating a verdict.
        if event_row is None:
            return {"status": "not_found"}
        event = event_row._mapping
        # The verdict compares the actual action (the claimed action/resource
        # pair) with the original authorization event; the execution outcome
        # is recorded but never changes the verdict. A mismatch stores the
        # real values exactly as executed alongside ``matches_authorization =
        # false`` — nothing is coerced to the authorized action.
        matches_authorization = (
            action_type == event["action_type"] and resource == event["resource"]
        )

        rows = _load_records(conn, machine_id)
        # Every row this feature writes carries its hashes. If any row is
        # missing chain data (e.g. an external writer), rebuild the whole
        # machine chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_hash"] is None or row._mapping["chain_hash"] is None
            for row in rows
        ):
            for values in _recompute_rows(rows):
                receipt_id = values.pop("id")
                conn.execute(
                    _TABLE.update().where(_TABLE.c.id == receipt_id).values(**values)
                )
            rows = _load_records(conn, machine_id)

        tail = rows[-1] if rows else None

        created_at = _utc_now_iso()
        receipt_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_key = (_created_at_instant(tail_created_at), tail._mapping["id"])
            while (_created_at_instant(created_at), receipt_id) <= tail_key:
                if _created_at_instant(created_at) < tail_key[0]:
                    created_at = tail_created_at
                else:
                    receipt_id = str(uuid.uuid4())

        if tail is None:
            previous_receipt_id = None
            previous_chain_hash = ""
        else:
            previous_receipt_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        content_hash = compute_content_hash(
            id=receipt_id,
            machine_id=machine_id,
            grant_id=grant_id,
            use_id=use["id"],
            authorization_event_id=grant["event_id"],
            evidence_id=evidence_id,
            action_type=action_type,
            resource=resource,
            outcome=outcome,
            matches_authorization=matches_authorization,
            created_at=created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        conn.execute(
            _TABLE.insert().values(
                id=receipt_id,
                machine_id=machine_id,
                grant_id=grant_id,
                use_id=use["id"],
                authorization_event_id=grant["event_id"],
                evidence_id=evidence_id,
                action_type=action_type,
                resource=resource,
                outcome=outcome,
                matches_authorization=matches_authorization,
                created_at=created_at,
                previous_receipt_id=previous_receipt_id,
                content_hash=content_hash,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "receipt": {
                "evidence_id": evidence_id,
                "action_type": action_type,
                "resource": resource,
                "outcome": outcome,
                "id": receipt_id,
                "grant_id": grant_id,
                "use_id": use["id"],
                "matches_authorization": matches_authorization,
                "created_at": created_at,
                "previous_receipt_id": previous_receipt_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        }

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique grant_id/use_id constraints are the cross-backstop race
        # guards: a concurrent transaction that committed the first receipt
        # first wins.
        return {"status": "duplicate_receipt"}


def load_receipt(
    session: Session, *, machine_id: str, grant_id: str
) -> AuthorizationGrantExecutionReceipt | None:
    """Read one machine's unique receipt for one grant, or ``None``.

    Strictly read-only; scoped to the path machine so another machine's
    receipt can never be returned.
    """
    return session.scalar(
        select(AuthorizationGrantExecutionReceipt).where(
            AuthorizationGrantExecutionReceipt.machine_id == machine_id,
            AuthorizationGrantExecutionReceipt.grant_id == grant_id,
        )
    )


def receipt_to_dict(record: AuthorizationGrantExecutionReceipt) -> dict[str, Any]:
    """One stored receipt as the fixed twelve-field view, in fixed order."""
    return {
        "evidence_id": record.evidence_id,
        "action_type": record.action_type,
        "resource": record.resource,
        "outcome": record.outcome,
        "id": record.id,
        "grant_id": record.grant_id,
        "use_id": record.use_id,
        "matches_authorization": record.matches_authorization,
        "created_at": record.created_at,
        "previous_receipt_id": record.previous_receipt_id,
        "content_hash": record.content_hash,
        "chain_hash": record.chain_hash,
    }


def verify_machine_chain(
    session: Session, machine_id: str
) -> tuple[bool, int, str | None]:
    """Read-only verification of one machine's receipt chain for audit.

    Tolerant of damaged stored values so a corrupted ``created_at``, ``id``,
    digest, or reference never crashes the query, is never repaired or
    recomputed for storage, and never removes the record from the total.
    Returns ``(valid, checked_count, broken_receipt_id)``.

    Records are examined in the order of the actual UTC instant of
    ``created_at`` and then ``id`` (an exact-second stamp precedes any
    fractional-second stamp of the same second). A record whose
    ``created_at`` no longer parses still enters the total and is itself
    reported as the first broken record, instead of crashing the scan or
    blaming its chain successor. The first record's previous-receipt id must
    be empty; every later record's must be the id of the immediately preceding
    record, and its stored content and chain digests must match the digests
    recomputed under the public append-time rules — nothing is written back
    when they do not. The first mismatch sets the broken id; later records
    cannot change it. Only rows whose stored ``machine_id`` equals the path
    machine are examined, so another machine's damaged records never change
    this result. An empty chain is valid.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    for row in rows:
        if not _created_at_parseable(row._mapping["created_at"]):
            return False, len(rows), row._mapping["id"]

    previous_receipt_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        try:
            content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
            chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        except (TypeError, ValueError):
            # A damaged stored value (e.g. a non-text field) cannot produce
            # the published digest; the record is broken, not a reason to
            # crash the read-only audit.
            return False, len(rows), mapping["id"]
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_receipt_id"] != previous_receipt_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_receipt_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
