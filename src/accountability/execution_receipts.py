"""Execution-completion receipts bound to consumed authorization grant uses.

A receipt closes the accountability loop: after a consumed grant has been
exercised exactly once, the caller records one execution-completion receipt
that binds the consumed :class:`~.db.AuthorizationGrantUse` to the execution
result and to the per-machine tamper-evident receipt chain.

* :func:`create_receipt` accepts only a use that belongs to the path machine
  and has been consumed, whose source decision event committed
  ``allowed = true`` with ``reason = "allowed_by_policy"``; the receipt's
  ``action_type`` and ``resource`` must match the source event verbatim. The
  lookup, eligibility checks, and insert run in one locked write transaction
  — the same lock the grant actions take — and ``use_id`` carries a
  database-level unique constraint, so a concurrent burst for one use has
  exactly one winner; the losers answer ``receipt_already_exists`` and write
  nothing. Old uses never receive a backfilled receipt.
* :func:`verify_machine_receipts` is the independent, strictly read-only
  audit of one machine's receipt chain. Receipts are examined by the actual
  UTC instant of ``occurred_at`` and then by ``id``; a stored stamp that no
  longer parses sorts its row deterministically last. The first anomaly
  found is reported with one stable category — ``timestamp_unparseable``,
  ``chain_break``, ``ownership_mismatch``, ``use_mismatch``,
  ``scope_mismatch``, or ``digest_mismatch`` — and damaged stored values are
  reported, never crashed on, repaired, rewritten, or recomputed for
  storage.
* :func:`coverage_report` is a strictly read-only use-to-receipt coverage
  report for one machine: every consumed use is matched against the
  machine's receipts by ``use_id`` alone, reporting both the consumed uses
  that carry no receipt and the receipts whose ``use_id`` is not one of the
  machine's consumptions. It never rebuilds, backfills, repairs, or mutates
  a use, a receipt, or any chain; chain, scope, and digest soundness stay
  the separate concern of :func:`verify_machine_receipts`.

Each machine's receipts form an ordered chain following the same rules as
the other per-machine chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the receipt's ten content fields: ``{id, machine_id, use_id,
  grant_id, authorization_event_id, action_type, resource, outcome,
  result_digest, occurred_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first receipt uses the empty string as the previous chain hash.
* ``previous_receipt_id`` is ``None`` for a machine's first receipt and the
  prior receipt's id otherwise.

The table is created at startup on databases that predate the feature;
historical rows are never rewritten and no receipts are fabricated for
historical uses.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError

from .chain import _run_with_lock_retry
from .db import (
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantUse,
    ExecutionReceipt,
    Machine,
)

_HASH_LEN = 64
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")

_TABLE = ExecutionReceipt.__table__
_USE_TABLE = AuthorizationGrantUse.__table__
_GRANT_TABLE = AuthorizationGrant.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__

_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "use_id",
    "grant_id",
    "authorization_event_id",
    "action_type",
    "resource",
    "outcome",
    "result_digest",
    "occurred_at",
)

_OUTCOMES = ("succeeded", "failed")

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    use_id: str,
    grant_id: str,
    authorization_event_id: str,
    action_type: str,
    resource: str,
    outcome: str,
    result_digest: str,
    occurred_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "use_id": use_id,
            "grant_id": grant_id,
            "authorization_event_id": authorization_event_id,
            "action_type": action_type,
            "resource": resource,
            "outcome": outcome,
            "result_digest": result_digest,
            "occurred_at": occurred_at,
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
    fabricated for historical uses.
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
                        f"ALTER TABLE {_TABLE.name} ADD COLUMN {name} "
                        f"{column_type}"
                    )
                )


def _utc_iso(instant: datetime) -> str:
    """Format an aware UTC instant as RFC 3339 text ending in ``Z``."""
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def occurred_at_instant(value: object) -> datetime:
    """Parse a stored ``occurred_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract,
    so parsing cannot fail for them; a damaged value that no longer parses
    sorts after every parseable record (its content hash cannot verify
    anyway) instead of crashing the read-only audit.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _receipt_id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored receipt id, tolerant of a damaged value."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order receipt rows by the actual UTC instant of ``occurred_at``.

    Ordering is by the parsed UTC instant and then by ``id``, so an
    exact-second stamp sorts before any fractional-second stamp of the same
    second. A stamp that no longer parses sorts deterministically last, and a
    damaged non-string id sorts as empty rather than crashing the comparison.
    """
    return sorted(
        rows,
        key=lambda row: (
            occurred_at_instant(row._mapping["occurred_at"]),
            _receipt_id_key(row._mapping["id"]),
        ),
    )


def _load_records(conn: Connection, machine_id: str) -> list[Any]:
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


def create_receipt(
    engine: Engine,
    *,
    machine_id: str,
    use_id: str,
    action_type: str,
    resource: str,
    outcome: str,
    result_digest: str,
) -> dict[str, Any]:
    """Record one execution-completion receipt for one consumed grant use.

    The machine lookup, the use/grant/event binding reads, the source-event
    eligibility and verbatim scope checks, the duplicate-use check, and the
    chain-tail insert all run in one locked write transaction, so a
    concurrent burst for the same use has exactly one winner and concurrent
    receipts for different uses cannot fork the per-machine chain. Returns a
    status dict:

    * ``not_found`` — the path machine is missing, the use does not exist,
      or it is owned by another machine;
    * ``receipt_already_exists`` — the use already carries a receipt;
    * ``authorization_not_allowed`` — the use's source event did not commit
      ``allowed = true`` / ``reason = "allowed_by_policy"``;
    * ``execution_scope_mismatch`` — the receipt's ``action_type`` or
      ``resource`` does not match the source event verbatim;
    * ``ok`` — with the new ``receipt`` dict.

    Nothing is written on any non-ok outcome: the receipt insert and its
    chain link commit together or leave no trace. Creating a receipt never
    modifies the use, the grant, the source event, or any other chain.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        # The use must belong to the path machine: a use owned by another
        # machine is indistinguishable from a missing one.
        use_row = conn.execute(
            _USE_TABLE.select().where(
                _USE_TABLE.c.id == use_id,
                _USE_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if use_row is None:
            return {"status": "not_found"}
        use = use_row._mapping

        grant_row = conn.execute(
            _GRANT_TABLE.select().where(
                _GRANT_TABLE.c.id == use["grant_id"],
                _GRANT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if grant_row is None:
            return {"status": "not_found"}
        grant = grant_row._mapping

        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == use["event_id"],
                _EVENT_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if event_row is None:
            return {"status": "not_found"}
        event = event_row._mapping

        # The use only exists for a consumed grant whose source event passed
        # the grant-issue gate, but the audit never trusts that history: the
        # source must still be a committed policy allow.
        if not event["allowed"] or event["reason"] != "allowed_by_policy":
            return {"status": "authorization_not_allowed"}

        # The receipt attests the exact action/resource the authorization
        # was issued for: compared verbatim, with no trimming or folding.
        if action_type != event["action_type"] or resource != event["resource"]:
            return {"status": "execution_scope_mismatch"}

        # The semantic checks above take precedence; once they pass, one
        # receipt per consumed use is a hard, database-enforced invariant —
        # a repeated or racing request that names the same use gets the
        # terminal conflict.
        existing = conn.execute(
            select(_TABLE.c.id).where(_TABLE.c.use_id == use_id)
        ).first()
        if existing is not None:
            return {"status": "receipt_already_exists"}

        rows = _load_records(conn, machine_id)
        # Every row this feature writes carries its hashes. If any row is
        # missing chain data (e.g. an external writer), rebuild the whole
        # machine chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_hash"] is None
            or row._mapping["chain_hash"] is None
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

        occurred_at = _utc_iso(datetime.now(timezone.utc))
        receipt_id = str(uuid.uuid4())
        if tail is not None:
            tail_id = tail._mapping["id"]
            # Same-instant receipts must still sort strictly after the tail;
            # the id is the only degree of freedom, since occurred_at is the
            # commit moment shared by the response.
            if occurred_at == tail._mapping["occurred_at"]:
                while receipt_id <= tail_id:
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
            use_id=use_id,
            grant_id=grant["id"],
            authorization_event_id=event["id"],
            action_type=action_type,
            resource=resource,
            outcome=outcome,
            result_digest=result_digest,
            occurred_at=occurred_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        conn.execute(
            _TABLE.insert().values(
                id=receipt_id,
                machine_id=machine_id,
                use_id=use_id,
                grant_id=grant["id"],
                authorization_event_id=event["id"],
                action_type=action_type,
                resource=resource,
                outcome=outcome,
                result_digest=result_digest,
                occurred_at=occurred_at,
                previous_receipt_id=previous_receipt_id,
                content_hash=content_hash,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "receipt": {
                "id": receipt_id,
                "machine_id": machine_id,
                "use_id": use_id,
                "grant_id": grant["id"],
                "authorization_event_id": event["id"],
                "occurred_at": occurred_at,
                "previous_receipt_id": previous_receipt_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        }

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique use_id constraint is the cross-backstop race guard: a
        # concurrent transaction that inserted the first receipt wins.
        return {"status": "receipt_already_exists"}


def _is_lower_hex_64(value: object) -> bool:
    return isinstance(value, str) and _LOWER_HEX_64.fullmatch(value) is not None


def verify_machine_receipts(
    session, machine_id: str
) -> dict[str, Any]:
    """Read-only verification of one machine's execution receipts.

    Returns ``{valid, checked_count, broken_receipt_id, anomaly}``. The
    receipts are examined by the actual UTC instant of ``occurred_at`` and
    then by ``id``; a stamp that no longer parses sorts its row last and,
    when reached, is the first anomaly (``timestamp_unparseable``) rather
    than crashing the scan. The first anomaly found — in scan order, and for
    one receipt in the fixed category order ``chain_break``,
    ``ownership_mismatch``, ``use_mismatch``, ``scope_mismatch``,
    ``digest_mismatch`` — sets ``broken_receipt_id`` and ``anomaly``; later
    receipts cannot change it. ``checked_count`` is always the machine's
    total receipt count, damaged rows included, and ``anomaly`` is ``None``
    on a fully sound (including empty) chain.

    Only the path machine's receipts are examined, so another machine's
    damaged records never change this result. The audit is strictly
    read-only: it never writes, repairs, deletes, recomputes, or normalizes
    a receipt, use, grant, or event.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )
    checked_count = len(rows)

    # A corrupted stamp sorts its record after every parseable one, so the
    # record's chain successor would otherwise be blamed for the broken
    # link; pre-scan and report the corrupted record itself first, matching
    # the other per-machine chain audits.
    for row in rows:
        if occurred_at_instant(row._mapping["occurred_at"]) == _FAR_FUTURE:
            return _anomaly(
                checked_count, row._mapping["id"], "timestamp_unparseable"
            )

    uses = {
        row._mapping["id"]: row._mapping
        for row in session.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.machine_id == machine_id)
        )
    }
    grants = {
        row._mapping["id"]: row._mapping
        for row in session.execute(
            _GRANT_TABLE.select().where(_GRANT_TABLE.c.machine_id == machine_id)
        )
    }
    events = {
        row._mapping["id"]: row._mapping
        for row in session.execute(
            _EVENT_TABLE.select().where(_EVENT_TABLE.c.machine_id == machine_id)
        )
    }

    previous_receipt_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        receipt_id = mapping["id"]

        # 1. The link structure and the chain digest over the stored
        #    content hash: a wrong predecessor or a chain digest that does
        #    not continue the predecessor's chain breaks the chain.
        stored_content_hash = mapping["content_hash"]
        stored_chain_hash = mapping["chain_hash"]
        try:
            expected_chain_hash = compute_chain_hash(
                previous_chain_hash, stored_content_hash
            )
        except TypeError:
            # A damaged (non-text) stored content hash cannot continue the
            # chain digest.
            return _anomaly(checked_count, receipt_id, "chain_break")
        if (
            mapping["previous_receipt_id"] != previous_receipt_id
            or not _is_lower_hex_64(stored_chain_hash)
            or stored_chain_hash != expected_chain_hash
        ):
            return _anomaly(checked_count, receipt_id, "chain_break")

        # 2. Every referenced record must exist under the path machine.
        use = uses.get(mapping["use_id"])
        grant = grants.get(mapping["grant_id"])
        event = events.get(mapping["authorization_event_id"])
        if use is None or grant is None or event is None:
            return _anomaly(checked_count, receipt_id, "ownership_mismatch")

        # 3. The receipt must agree with its consumed use: the receipt names
        #    exactly the grant and source event the use bound together.
        if (
            use["grant_id"] != mapping["grant_id"]
            or use["event_id"] != mapping["authorization_event_id"]
            or grant["event_id"] != use["event_id"]
        ):
            return _anomaly(checked_count, receipt_id, "use_mismatch")

        # 4. The source must still be a committed policy allow for the
        #    receipt's verbatim action and resource.
        if (
            not event["allowed"]
            or event["reason"] != "allowed_by_policy"
            or mapping["action_type"] != event["action_type"]
            or mapping["resource"] != event["resource"]
        ):
            return _anomaly(checked_count, receipt_id, "scope_mismatch")

        # 5. The result fields and the content digest must be sound: a legal
        #    outcome, a 64 lowercase-hex result fingerprint, and a stored
        #    content hash equal to the digest of the ten covered fields as
        #    stored — nothing is recomputed for storage when they disagree.
        if mapping["outcome"] not in _OUTCOMES or not _is_lower_hex_64(
            mapping["result_digest"]
        ):
            return _anomaly(checked_count, receipt_id, "digest_mismatch")
        try:
            expected_content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
        except TypeError:
            # A damaged (non-text) content field cannot produce the
            # published digest.
            return _anomaly(checked_count, receipt_id, "digest_mismatch")
        if (
            not _is_lower_hex_64(stored_content_hash)
            or stored_content_hash != expected_content_hash
        ):
            return _anomaly(checked_count, receipt_id, "digest_mismatch")

        previous_receipt_id = receipt_id
        previous_chain_hash = stored_chain_hash

    return {
        "valid": True,
        "checked_count": checked_count,
        "broken_receipt_id": None,
        "anomaly": None,
    }


def _anomaly(
    checked_count: int, broken_receipt_id: str | None, anomaly: str
) -> dict[str, Any]:
    return {
        "valid": False,
        "checked_count": checked_count,
        "broken_receipt_id": broken_receipt_id,
        "anomaly": anomaly,
    }


def _consumed_at_instant(value: object) -> datetime:
    """Parse a stored use ``consumed_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract,
    so parsing cannot fail for them; a damaged value that no longer parses
    sorts after every parseable record instead of crashing the read-only
    report — it is reported, never repaired or rewritten.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _use_id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored use id, tolerant of a damaged value."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def coverage_report(session, machine_id: str) -> dict[str, Any]:
    """Read-only use-to-receipt coverage report for one machine.

    Reads only the path machine's consumption records (rows of
    ``authorization_grant_uses`` owned by the machine) and its receipts
    (rows of ``execution_receipts`` owned by the machine), then matches the
    two sets by ``use_id`` one to one — nothing about the chain, the
    authorization scope, or any content digest is adjudicated here; those
    stay the independent concern of :func:`verify_machine_receipts`.

    Returns a dict with, in this fixed order: ``consumed_count`` (the total
    number of the machine's consumption records), ``receipt_count`` (the
    total number of the machine's receipts), ``covered_count`` (the number
    of consumed uses whose id is named by one of the machine's receipts),
    ``missing_count``/``missing_use_ids`` (consumed uses with no receipt —
    ordered by the actual UTC instant of ``consumed_at`` and then by id, a
    stamp that no longer parses sorting last), and
    ``orphan_count``/``orphan_receipt_ids`` (the machine's receipts whose
    ``use_id`` is not one of the machine's consumption record ids — ordered
    by the actual UTC instant of ``occurred_at`` and then by receipt id,
    again with an unparseable stamp last). Every list is always present,
    even when empty, and counts are plain integers.

    Matching is by set membership over stored ids, so it never depends on
    receipt chain order or soundness: a chain-damaged receipt that still
    names one of the machine's consumed uses covers it, and another
    machine's uses or receipts never enter either set. The report is
    strictly read-only: it never creates, rebuilds, backfills, updates,
    deletes, repairs, recomputes, or normalizes a use, a receipt, or any
    chain, so a later legitimate receipt simply improves the next report.
    """
    use_rows = list(
        session.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.machine_id == machine_id)
        )
    )
    receipt_rows = list(
        session.execute(
            _TABLE.select().where(_TABLE.c.machine_id == machine_id)
        )
    )

    consumed_count = len(use_rows)
    receipt_count = len(receipt_rows)

    # The machine's consumption set is defined by stored use id; a damaged
    # non-string id still occupies a set slot (membership only), and the
    # same value below reads it verbatim for ordering and output.
    use_ids = {row._mapping["id"] for row in use_rows}
    receipt_use_ids = {row._mapping["use_id"] for row in receipt_rows}

    covered_count = sum(1 for use_id in use_ids if use_id in receipt_use_ids)

    missing_rows = sorted(
        (row for row in use_rows if row._mapping["id"] not in receipt_use_ids),
        key=lambda row: (
            _consumed_at_instant(row._mapping["consumed_at"]),
            _use_id_key(row._mapping["id"]),
        ),
    )
    orphan_rows = sorted(
        (row for row in receipt_rows if row._mapping["use_id"] not in use_ids),
        key=lambda row: (
            occurred_at_instant(row._mapping["occurred_at"]),
            _receipt_id_key(row._mapping["id"]),
        ),
    )

    missing_use_ids = [row._mapping["id"] for row in missing_rows]
    orphan_receipt_ids = [row._mapping["id"] for row in orphan_rows]

    return {
        "consumed_count": consumed_count,
        "receipt_count": receipt_count,
        "covered_count": covered_count,
        "missing_count": len(missing_use_ids),
        "missing_use_ids": missing_use_ids,
        "orphan_count": len(orphan_receipt_ids),
        "orphan_receipt_ids": orphan_receipt_ids,
        "valid": not missing_use_ids and not orphan_receipt_ids,
    }
