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
* :func:`create_receipts_batch` applies the same binding semantics to a
  whole batch of records in one locked write transaction: every record is
  resolved and checked in submission order, the first refusal rejects the
  whole batch with no partial receipts, and a committed batch shares one
  UTC commit instant with chain links and response order following the
  submission order. Concurrent batches serialize on the same lock and the
  ``use_id`` unique constraint, so two batches naming one use have exactly
  one winner and the chain never forks.
* :func:`verify_machine_receipts` is the independent, strictly read-only
  audit of one machine's receipt chain. Receipts are examined by the actual
  UTC instant of ``occurred_at`` and then by ``id``; a stored stamp that no
  longer parses sorts its row deterministically last. The first anomaly
  found is reported with one stable category — ``timestamp_unparseable``,
  ``chain_break``, ``ownership_mismatch``, ``use_mismatch``,
  ``chronology_mismatch``, ``scope_mismatch``, or ``digest_mismatch`` — and
  damaged stored values are reported, never crashed on, repaired,
  rewritten, or recomputed for storage.
* :func:`coverage_report` is a strictly read-only use-to-receipt coverage
  report for one machine: every consumed use is matched against the
  machine's receipts by ``use_id`` alone, reporting both the consumed uses
  that carry no receipt and the receipts whose ``use_id`` is not one of the
  machine's consumptions. It never rebuilds, backfills, repairs, or mutates
  a use, a receipt, or any chain; chain, scope, and digest soundness stay
  the separate concern of :func:`verify_machine_receipts`.
* :func:`execution_receipt_summary` is the strictly read-only roll-up that
  lets a caller see the registered results alongside the consumed
  authorizations still awaiting a receipt: one machine's receipt rows are
  counted as stored — ``outcome`` verbatim ``succeeded``/``failed`` versus
  any other (damaged) value, and ``result_digest`` exactly 64 lowercase
  hexadecimal characters versus any other value — and the machine's
  consumption records with no matching receipt are counted as missing.
  Damaged values are counted, never hidden, repaired, recomputed, or folded
  into success or failure.
* :func:`export_receipt_window` is the strictly read-only fixed-window
  compliance slice: it returns one machine's receipts whose ``occurred_at``
  is a parseable UTC ``Z`` instant inside a closed ``[start, end]``
  interval, ordered by that instant and then by id, with every stored field
  emitted verbatim. A receipt whose stamp no longer parses is excluded, not
  repaired, and no reference, chain field, or hash is ever recomputed.

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

_UUID_MAX_INT = (1 << 128) - 1

# Strict shape of a moment the chain audit compares chronologically: an RFC
# 3339 date-time in UTC with a literal ``Z`` suffix and optional fractional
# seconds. A missing, non-text, offset-form, unsuffixed, malformed, or
# out-of-range value is ``timestamp_unparseable`` — never folded into a
# chronology verdict — whereas the tolerant ``occurred_at_instant`` parser
# only needs a total ordering and lets such a stamp sort last.
_UTC_Z_MOMENT_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


def _parse_utc_z_moment(value: object) -> datetime | None:
    """Parse a stored moment to its UTC instant only under strict ``Z`` rules.

    Returns the actual UTC instant for text that has the RFC 3339 ``Z``
    shape and range-checks; a missing, non-text, offset-form, malformed, or
    out-of-range value returns ``None`` so the read-only audit can report
    ``timestamp_unparseable`` instead of crashing, repairing, or normalizing
    the stored text.
    """
    if isinstance(value, str) and _UTC_Z_MOMENT_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            return None
    return None


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


def create_receipts_batch(
    engine: Engine,
    *,
    machine_id: str,
    records: list[dict[str, str]],
) -> dict[str, Any]:
    """Atomically record one execution-completion receipt per record.

    Every record carries the already-validated ``use_id``, ``action_type``,
    ``resource``, ``outcome``, and ``result_digest`` of one consumed grant
    use; the batch applies the single-receipt binding semantics to each item
    in ``records`` order. The machine lookup, the per-record use/grant/event
    binding reads, the source-event eligibility and verbatim scope checks,
    the duplicate-use checks, and all chain-tail inserts run in one locked
    write transaction — the same lock the single-receipt create takes — so
    concurrent batches serialize, never fork the per-machine chain, and a
    batch racing another writer for one use has exactly one winner. Returns
    a status dict:

    * ``not_found`` — the path machine is missing, or any record's use,
      grant, or source event is missing or owned by another machine;
    * ``authorization_not_allowed`` — the first record (in ``records``
      order) whose source event did not commit ``allowed = true`` /
      ``reason = "allowed_by_policy"``;
    * ``execution_scope_mismatch`` — the first record whose ``action_type``
      or ``resource`` does not match its source event verbatim;
    * ``receipt_already_exists`` — the first record whose use already
      carries a receipt (also the cross-backstop race-loser outcome);
    * ``ok`` — with the ``receipts`` list in ``records`` order.

    Every receipt of one batch shares one UTC commit instant, links to its
    predecessor in ``records`` order, and carries an id that sorts after its
    same-instant predecessor's. Nothing is written on any non-ok outcome:
    the whole batch commits together or leaves no trace, and a rejected
    batch never modifies a use, a grant, an event, or any other chain.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}

        # Binding pass: every record's use, grant, and source event must
        # exist under the path machine before any business check runs — a
        # record owned by another machine is indistinguishable from a
        # missing one.
        resolved: list[tuple[dict[str, str], Any, Any, Any]] = []
        for record in records:
            use_row = conn.execute(
                _USE_TABLE.select().where(
                    _USE_TABLE.c.id == record["use_id"],
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
            resolved.append((record, use, grant, event_row._mapping))

        # Business pass in records order: the first refusal decides the
        # whole batch, and nothing has been written yet.
        for record, use, grant, event in resolved:
            # The audit never trusts history: the source must still be a
            # committed policy allow.
            if not event["allowed"] or event["reason"] != "allowed_by_policy":
                return {"status": "authorization_not_allowed"}
            # The receipt attests the exact action/resource the
            # authorization was issued for: verbatim, no trimming or folding.
            if (
                record["action_type"] != event["action_type"]
                or record["resource"] != event["resource"]
            ):
                return {"status": "execution_scope_mismatch"}
            # One receipt per consumed use is a hard, database-enforced
            # invariant; a repeated or racing batch naming the same use gets
            # the terminal conflict.
            existing = conn.execute(
                select(_TABLE.c.id).where(_TABLE.c.use_id == record["use_id"])
            ).first()
            if existing is not None:
                return {"status": "receipt_already_exists"}

        rows = _load_records(conn, machine_id)
        # Every row this feature writes carries its hashes. If any row is
        # missing chain data (e.g. an external writer), rebuild the whole
        # machine chain before appending so the new links have a sound tail.
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

        # One commit instant shared by the whole batch; the chain links and
        # the response follow records order, and same-instant ids sort
        # strictly after their predecessor's.
        occurred_at = _utc_iso(datetime.now(timezone.utc))
        previous_receipt_id: str | None = None
        previous_chain_hash = ""
        same_instant_floor: str | None = None
        if tail is not None:
            previous_receipt_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]
            # Same-instant receipts must still sort strictly after the
            # tail; the id is the only degree of freedom, since
            # occurred_at is the commit moment shared by the batch.
            if tail._mapping["occurred_at"] == occurred_at:
                same_instant_floor = tail._mapping["id"]

        # Canonical UUID strings order like their integers, so one random
        # base (lifted above a same-instant tail) plus one increment per
        # record gives the whole batch strictly increasing ids without a
        # per-record redraw — redrawing against a rising floor would shrink
        # the remaining id space exponentially with the batch size.
        start_int: int | None = None
        if same_instant_floor is None:
            start_int = uuid.uuid4().int
        else:
            try:
                start_int = max(
                    uuid.uuid4().int, uuid.UUID(same_instant_floor).int + 1
                )
            except (ValueError, TypeError, AttributeError):
                # A damaged non-UUID tail id has no integer floor.
                start_int = None
        if start_int is not None and (
            start_int + len(resolved) - 1 > _UUID_MAX_INT
        ):
            # No room for the whole batch above the base — only reachable
            # with an externally damaged near-maximum tail id.
            start_int = None
        if start_int is not None:
            receipt_ids = [
                str(uuid.UUID(int=start_int + offset))
                for offset in range(len(resolved))
            ]
        else:
            # Fall back to the single-receipt redraw discipline.
            receipt_ids = []
            floor = same_instant_floor
            for _ in resolved:
                candidate = str(uuid.uuid4())
                if floor is not None:
                    while candidate <= floor:
                        candidate = str(uuid.uuid4())
                receipt_ids.append(candidate)
                floor = candidate

        receipts: list[dict[str, Any]] = []
        for (record, use, grant, event), receipt_id in zip(resolved, receipt_ids):

            content_hash = compute_content_hash(
                id=receipt_id,
                machine_id=machine_id,
                use_id=record["use_id"],
                grant_id=grant["id"],
                authorization_event_id=event["id"],
                action_type=record["action_type"],
                resource=record["resource"],
                outcome=record["outcome"],
                result_digest=record["result_digest"],
                occurred_at=occurred_at,
            )
            chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
            conn.execute(
                _TABLE.insert().values(
                    id=receipt_id,
                    machine_id=machine_id,
                    use_id=record["use_id"],
                    grant_id=grant["id"],
                    authorization_event_id=event["id"],
                    action_type=record["action_type"],
                    resource=record["resource"],
                    outcome=record["outcome"],
                    result_digest=record["result_digest"],
                    occurred_at=occurred_at,
                    previous_receipt_id=previous_receipt_id,
                    content_hash=content_hash,
                    chain_hash=chain_hash,
                )
            )
            receipts.append(
                {
                    "id": receipt_id,
                    "machine_id": machine_id,
                    "use_id": record["use_id"],
                    "grant_id": grant["id"],
                    "authorization_event_id": event["id"],
                    "occurred_at": occurred_at,
                    "previous_receipt_id": previous_receipt_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
            previous_receipt_id = receipt_id
            previous_chain_hash = chain_hash

        return {"status": "ok", "receipts": receipts}

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique use_id constraint is the cross-backstop race guard: a
        # concurrent transaction that inserted the first receipt wins, and
        # the losing batch leaves no partial records.
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
    ``ownership_mismatch``, ``use_mismatch``, ``chronology_mismatch``,
    ``scope_mismatch``, ``digest_mismatch`` — sets ``broken_receipt_id`` and
    ``anomaly``; later receipts cannot change it. ``checked_count`` is
    always the machine's total receipt count, damaged rows included, and
    ``anomaly`` is ``None`` on a fully sound (including empty) chain.

    The time-consistency checks bind the receipt to the moments of the
    authorization it closes: every related moment — the receipt's
    ``occurred_at`` and its consumed use's ``consumed_at`` plus the named
    grant's ``issued_at`` and ``expires_at`` — must be strict RFC 3339 UTC
    text with a literal ``Z`` (fractional seconds allowed); a missing,
    non-text, offset-form, unsuffixed, malformed, or out-of-range value is
    ``timestamp_unparseable``, never a chronology verdict. When all four
    parse, a receipt whose ``occurred_at`` precedes its use's
    ``consumed_at``, whose ``consumed_at`` precedes ``issued_at``, or whose
    ``consumed_at`` is at or after ``expires_at`` is
    ``chronology_mismatch``. Equality at either lower boundary
    (``occurred_at == consumed_at`` or ``consumed_at == issued_at``) is
    legal, and a receipt registered after ``expires_at`` for a use consumed
    while the grant was valid is legal and never misreported. An
    ``occurred_at`` that does not parse is pre-scanned across the whole
    chain and precedes every per-receipt check; the use/grant moments are
    examined after the chain, ownership, and binding checks and before the
    chronology, scope, and digest checks.

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
    # the other per-machine chain audits. The verdict uses the strict
    # RFC 3339 ``Z`` contract (an offset, missing ``Z``, or non-extended
    # shape is unparseable even though the ordering parser tolerates some
    # such text solely to keep the scan total and deterministic).
    for row in rows:
        if _parse_utc_z_moment(row._mapping["occurred_at"]) is None:
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

        # 4. Time consistency needs every related moment under the strict
        #    RFC 3339 ``Z`` contract. The receipt's own stamp was verified
        #    in the pre-scan; the consumed use's ``consumed_at`` and the
        #    named grant's ``issued_at``/``expires_at`` are checked here so
        #    that a damaged moment is reported, not crashed on, and is never
        #    folded into a chronology verdict.
        occurred_instant = _parse_utc_z_moment(mapping["occurred_at"])
        consumed_instant = _parse_utc_z_moment(use["consumed_at"])
        issued_instant = _parse_utc_z_moment(grant["issued_at"])
        expires_instant = _parse_utc_z_moment(grant["expires_at"])
        if (
            occurred_instant is None
            or consumed_instant is None
            or issued_instant is None
            or expires_instant is None
        ):
            return _anomaly(checked_count, receipt_id, "timestamp_unparseable")

        # 5. Chronology: execution cannot predate the consumption that
        #    authorized it, consumption cannot predate issue, and the single
        #    consumption must land inside the grant's validity window — at
        #    the expiry instant the window is already closed, so equality
        #    there is a mismatch. The lower boundaries are inclusive
        #    (consumption at the issue instant and registration at the
        #    consumption instant are legal), and registering the receipt
        #    after ``expires_at`` for a timely consumption is legal.
        if (
            occurred_instant < consumed_instant
            or consumed_instant < issued_instant
            or consumed_instant >= expires_instant
        ):
            return _anomaly(checked_count, receipt_id, "chronology_mismatch")

        # 6. The source must still be a committed policy allow for the
        #    receipt's verbatim action and resource.
        if (
            not event["allowed"]
            or event["reason"] != "allowed_by_policy"
            or mapping["action_type"] != event["action_type"]
            or mapping["resource"] != event["resource"]
        ):
            return _anomaly(checked_count, receipt_id, "scope_mismatch")

        # 7. The result fields and the content digest must be sound: a legal
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


def execution_receipt_summary(session, machine_id: str) -> dict[str, Any]:
    """Read-only execution-receipt roll-up for one machine.

    Reads only the path machine's receipt rows (rows of
    ``execution_receipts`` owned by the machine) and its consumption records
    (rows of ``authorization_grant_uses`` owned by the machine) and counts
    stored values exactly as found:

    * ``total_receipts`` — the machine's receipt-row count;
    * ``succeeded_count`` / ``failed_count`` — receipts whose stored
      ``outcome`` is verbatim ``"succeeded"`` / ``"failed"``;
    * ``invalid_outcome_count`` — every other stored ``outcome`` value,
      including a missing or non-text (damaged) one;
    * ``valid_result_digest_count`` — receipts whose stored
      ``result_digest`` is exactly 64 lowercase hexadecimal characters;
    * ``invalid_result_digest_count`` — every other stored value;
    * ``missing_receipt_count`` — the number of the machine's distinct
      consumption records whose id is not named by any of the machine's
      receipts.

    A damaged value occupies its own invalid bucket rather than being
    hidden, repaired, recomputed, or folded into success or failure, and it
    still counts toward ``total_receipts``. Matching for the missing count
    is set membership over stored use ids (one consumption counts at most
    once), independent of receipt chain order or soundness; another
    machine's receipts or consumptions never enter either set. The summary
    is strictly read-only: it never creates, rebuilds, backfills, updates,
    deletes, repairs, recomputes, or normalizes a use, a receipt, or any
    chain, so repeated reads of unchanged data return identical counts and
    the counts survive restarts exactly as stored.
    """
    receipt_rows = list(
        session.execute(
            _TABLE.select().where(_TABLE.c.machine_id == machine_id)
        )
    )
    use_rows = list(
        session.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.machine_id == machine_id)
        )
    )

    total_receipts = len(receipt_rows)
    succeeded_count = 0
    failed_count = 0
    invalid_outcome_count = 0
    valid_result_digest_count = 0
    invalid_result_digest_count = 0
    receipt_use_ids: set[Any] = set()

    for row in receipt_rows:
        mapping = row._mapping
        outcome = mapping["outcome"]
        if outcome == "succeeded":
            succeeded_count += 1
        elif outcome == "failed":
            failed_count += 1
        else:
            # A missing, non-text, or otherwise damaged outcome is counted
            # as invalid, never hidden or treated as a success or failure.
            invalid_outcome_count += 1
        if _is_lower_hex_64(mapping["result_digest"]):
            valid_result_digest_count += 1
        else:
            invalid_result_digest_count += 1
        receipt_use_ids.add(mapping["use_id"])

    missing_receipt_count = sum(
        1 for row in use_rows if row._mapping["id"] not in receipt_use_ids
    )

    return {
        "total_receipts": total_receipts,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "invalid_outcome_count": invalid_outcome_count,
        "valid_result_digest_count": valid_result_digest_count,
        "invalid_result_digest_count": invalid_result_digest_count,
        "missing_receipt_count": missing_receipt_count,
    }


# The window export admits a row only when its stored stamp has the strict
# RFC 3339 ``Z`` shape (see ``_parse_utc_z_moment``) and range-checks, unlike
# the chain scan whose tolerant parser only needs a total ordering; offset
# forms and damaged text are excluded from the window rather than admitted
# or repaired.
_EXPORT_FIELDS = (
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
    "previous_receipt_id",
    "content_hash",
    "chain_hash",
)


def export_receipt_window(
    session,
    machine_id: str,
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Read-only fixed-window compliance slice of one machine's receipts.

    Reads only rows of ``execution_receipts`` owned by ``machine_id`` (the
    ownership column is the sole machine boundary) and keeps a row only when
    its stored ``occurred_at`` parses to a UTC instant inside the closed
    interval ``[start, end]``. A stamp that no longer parses — non-text,
    missing the ``Z`` suffix, carrying an offset, malformed, or
    out-of-range — is excluded from the slice and left exactly as stored;
    the read never crashes on, repairs, normalizes, or recomputes it.

    The retained receipts are ordered by the actual UTC instant of
    ``occurred_at`` and then by receipt id ascending, so an exact-second
    stamp sorts before any fractional-second stamp of the same second. Each
    item carries exactly the thirteen stored fields in fixed order — the ten
    content fields followed by ``previous_receipt_id``, ``content_hash``,
    and ``chain_hash`` — emitted verbatim: references, chain links, and
    hashes are never fixed or recomputed, and a chain-damaged receipt inside
    the window is exported exactly as stored. The query is strictly
    read-only: it never inserts, updates, deletes, backfills, or normalizes
    a receipt or any related row, so repeated reads of unchanged data return
    byte-identical results and the data survives restarts untouched.
    """
    rows = list(
        session.execute(
            _TABLE.select().where(_TABLE.c.machine_id == machine_id)
        )
    )

    in_window: list[tuple[datetime, Any]] = []
    for row in rows:
        instant = _parse_utc_z_moment(row._mapping["occurred_at"])
        if instant is not None and start <= instant <= end:
            in_window.append((instant, row))

    in_window.sort(
        key=lambda item: (
            item[0],
            _receipt_id_key(item[1]._mapping["id"]),
        )
    )

    return [
        {field: row._mapping[field] for field in _EXPORT_FIELDS}
        for _, row in in_window
    ]
