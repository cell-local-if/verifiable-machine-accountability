"""Immutable execution receipts bound to one consumed grant's use record.

A receipt records what actually happened when a one-time authorization grant
was exercised. It can be written only after the grant has been consumed, and
each consumption can carry at most one receipt: the unique ``use_id`` (and the
unique ``grant_id``) are database-level backstops so a concurrent burst of
submissions has exactly one winner. The receipt never modifies the grant, its
use record, the decision event, its basis, evidence, or any other chain.

The actual executed action (``action_type`` / ``resource``) is stored verbatim
even when it differs from the original authorization event; the comparison
result is recorded separately as ``matches_authorization`` (``true`` only when
both the action type and the resource equal the event's values), so a
non-conforming execution is preserved as it happened rather than masked.

Each machine's receipts form an ordered chain, ordered by the actual UTC
instant of ``created_at`` and then by ``id``, following the same rules as the
other per-machine chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the receipt's own content fields: ``{id, machine_id, grant_id, use_id,
  event_id, evidence_id, action_type, resource, outcome,
  matches_authorization, created_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first receipt uses the empty string as the previous chain hash.
* ``previous_receipt_id`` is ``None`` for a machine's first receipt and the
  immediately preceding receipt's id otherwise.

The receipt insert and its chain-tail link commit inside one locked write
transaction — the same lock primitive the other per-machine chains use — so
concurrent submissions cannot lose records, fork the chain, skip a link, or
point two records at the same predecessor. The table is created at startup on
databases that predate the feature; historical data is never rewritten and no
receipts are fabricated.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError

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
_GRANT_TABLE = AuthorizationGrant.__table__
_USE_TABLE = AuthorizationGrantUse.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__
_EVIDENCE_TABLE = AuthorizationDecisionEvidence.__table__

_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "grant_id",
    "use_id",
    "event_id",
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
    event_id: str,
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
            "event_id": event_id,
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
    """Add missing chain columns to a table created by an earlier schema.

    ``Base.metadata.create_all`` already creates the table on databases that
    predate the feature; this hook only brings a pre-existing partial table
    forward. Existing rows are never rewritten, and no receipts are
    fabricated.
    """
    inspector = inspect(engine)
    if _TABLE.name not in inspector.get_table_names():
        # ``create_all`` builds a current-schema table; there is nothing to
        # bring forward.
        return
    existing = {
        column["name"] for column in inspector.get_columns(_TABLE.name)
    }
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


def created_at_instant(value: object) -> datetime:
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
            created_at_instant(row._mapping["created_at"]),
            row._mapping["id"] if isinstance(row._mapping["id"], str) else "",
        ),
    )


def _load_records(conn: Connection, machine_id: str) -> list[Any]:
    """Load one machine's receipts in chain (instant, id) order."""
    return _chain_order(
        list(conn.execute(_TABLE.select().where(_TABLE.c.machine_id == machine_id)))
    )


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


def create_receipt(
    engine: Engine,
    *,
    machine_id: str,
    grant_id: str,
    evidence_id: str,
    action_type: str,
    resource: str,
    outcome: str,
) -> dict[str, Any]:
    """Write the single execution receipt for one consumed grant.

    All lookups, the comparison, the chain-tail append, and the insert run in
    one locked write transaction. Returns a status dict:

    * ``not_found`` — the path machine is missing, the grant does not exist
      under it, or its consumption record cannot be resolved within the path
      scope;
    * ``grant_not_consumed`` — the grant exists but has not been consumed (it
      is still active, was revoked, or expired unconsumed);
    * ``evidence_not_found`` — the evidence id does not name an evidence
      record of the path machine that belongs to the grant's own
      authorization event;
    * ``duplicate_receipt`` — the grant's consumption already carries a
      receipt;
    * ``ok`` — with the new ``receipt`` dict.

    Nothing is written on any non-ok outcome: the receipt insert and its chain
    link commit together or leave no trace, and no other record is ever
    modified.
    """

    def _work(conn: Connection) -> dict[str, Any]:
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

        use_row = conn.execute(
            _USE_TABLE.select().where(
                _USE_TABLE.c.grant_id == grant_id,
                _USE_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if use_row is None:
            # A resolvable grant without its consumption record is the
            # not-consumed conflict when the grant itself is not in the
            # consumed terminal state; a consumed grant without its use row
            # cannot resolve within the path scope and is reported not_found.
            if grant["status"] != "consumed":
                return {"status": "grant_not_consumed"}
            return {"status": "not_found"}
        use = use_row._mapping

        evidence_row = conn.execute(
            _EVIDENCE_TABLE.select().where(
                _EVIDENCE_TABLE.c.id == evidence_id,
                _EVIDENCE_TABLE.c.machine_id == machine_id,
                _EVIDENCE_TABLE.c.event_id == grant["event_id"],
            )
        ).first()
        if evidence_row is None:
            return {"status": "evidence_not_found"}

        existing = conn.execute(
            select(_TABLE.c.id).where(_TABLE.c.use_id == use["id"])
        ).first()
        if existing is not None:
            return {"status": "duplicate_receipt"}

        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == grant["event_id"],
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        # The grant was minted from this event of this machine, so the event
        # cannot be missing; treat an unresolvable event as an out-of-scope
        # lookup rather than fabricating a comparison.
        if event_row is None:
            return {"status": "not_found"}
        event = event_row._mapping
        matches_authorization = (
            action_type == event["action_type"]
            and resource == event["resource"]
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
                    _TABLE.update()
                    .where(_TABLE.c.id == receipt_id)
                    .values(**values)
                )
            rows = _load_records(conn, machine_id)

        tail = rows[-1] if rows else None

        created_at = _utc_now_iso()
        receipt_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_key = (created_at_instant(tail_created_at), tail._mapping["id"])
            # Same-instant submissions (coarse clock) must still sort strictly
            # after the tail; the id is the only degree of freedom, since
            # created_at is minted inside the same locked transaction.
            while (created_at_instant(created_at), receipt_id) <= tail_key:
                if created_at_instant(created_at) < tail_key[0]:
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
            event_id=grant["event_id"],
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
                event_id=grant["event_id"],
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
                "id": receipt_id,
                "grant_id": grant_id,
                "use_id": use["id"],
                "evidence_id": evidence_id,
                "action_type": action_type,
                "resource": resource,
                "outcome": outcome,
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
        # The unique use_id/grant_id constraints are the cross-backstop race
        # guard: a concurrent transaction that committed the first receipt
        # first wins.
        return {"status": "duplicate_receipt"}


def get_receipt(
    session, *, machine_id: str, grant_id: str
) -> dict[str, Any] | None:
    """Read the one receipt of one grant, scoped to the path machine.

    Returns ``None`` when the grant has no receipt. Read-only.
    """
    row = session.execute(
        _TABLE.select().where(
            _TABLE.c.grant_id == grant_id,
            _TABLE.c.machine_id == machine_id,
        )
    ).first()
    if row is None:
        return None
    mapping = row._mapping
    return {
        "id": mapping["id"],
        "grant_id": mapping["grant_id"],
        "use_id": mapping["use_id"],
        "evidence_id": mapping["evidence_id"],
        "action_type": mapping["action_type"],
        "resource": mapping["resource"],
        "outcome": mapping["outcome"],
        "matches_authorization": mapping["matches_authorization"],
        "created_at": mapping["created_at"],
        "previous_receipt_id": mapping["previous_receipt_id"],
        "content_hash": mapping["content_hash"],
        "chain_hash": mapping["chain_hash"],
    }


def _first_unparseable_created_at_id(rows: list[Any]) -> str | None:
    """Id of the first row in chain order whose ``created_at`` cannot parse.

    A corrupted stamp sorts its record after every parseable one, so the
    record's chain successor would otherwise be blamed for the broken link;
    reporting the corrupted record itself keeps the audit pointed at the
    actual damage. ``None`` when every row parses.
    """
    for row in rows:
        if not _created_at_parseable(row._mapping["created_at"]):
            return row._mapping["id"]
    return None


def verify_machine_chain(
    session, machine_id: str
) -> tuple[bool, int, str | None]:
    """Read-only verification of one machine's receipt chain for audit.

    Tolerant of damaged stored values so a corrupted ``created_at``, ``id``,
    digest, or reference never crashes the query, is never repaired or
    recomputed for storage, and never removes the record from the total.
    Returns ``(valid, checked_count, broken_receipt_id)``.

    Records are examined in the order of the actual UTC instant of
    ``created_at`` and then ``id``. The first record's previous-receipt id
    must be empty; every later record's must be the id of the immediately
    preceding record, and its stored content and chain digests must match the
    digests recomputed under the public append-time rules — nothing is
    written back when they do not. The first mismatch sets the broken id;
    later records cannot change it. Only rows whose stored ``machine_id``
    equals the path machine are examined. An empty chain is valid.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    corrupted_id = _first_unparseable_created_at_id(rows)
    if corrupted_id is not None:
        return False, len(rows), corrupted_id

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
            # A damaged stored value cannot produce the published digest; the
            # record is broken, not a reason to crash the read-only audit.
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
