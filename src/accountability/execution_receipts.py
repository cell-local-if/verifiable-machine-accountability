"""Per-machine tamper-evident chain of execution completion receipts.

An execution completion receipt binds one already-consumed authorization —
the single use record of a one-time grant — to the execution result it
produced, closing the grant → use → lifecycle → accountability-tracing loop
with a verifiable outcome record:

* :func:`create_receipt` accepts only a use that belongs to the path machine
  (a use row exists only for a consumed grant), whose source decision event
  committed ``allowed = true`` with ``reason = "allowed_by_policy"``, and
  whose ``action_type`` / ``resource`` match the source event verbatim. The
  use lookup, the source-event checks, the one-receipt-per-use check, and the
  insert run in one locked write transaction, so a concurrent burst of
  completions for the same use has exactly one winner — a database-level
  unique constraint on ``use_id`` is the final backstop. Receipts never
  modify the use, the grant, the event, or any other chain, and a rejected
  attempt writes nothing.
* :func:`verify_machine_chain` re-checks one machine's receipts read-only for
  audit, reporting the first anomaly by category.

Each machine's receipts form an ordered chain, ordered by the actual UTC
instant of ``occurred_at`` and then by ``id``, following the same rules as
the other per-machine chains:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the receipt's own fields: ``{id, machine_id, use_id, grant_id,
  authorization_event_id, action_type, resource, outcome, result_digest,
  occurred_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first receipt uses the empty string as the previous chain hash.
* ``previous_receipt_id`` is ``None`` for a machine's first receipt and the
  prior receipt's id otherwise.

``occurred_at`` is minted inside the same locked write transaction that
inserts the row, so receipts commit in non-decreasing (``occurred_at``,
``id``) order. Receipts are isolated per machine: one machine's chain never
contains another machine's receipts. The table is created at startup on
databases that predate the feature; uses recorded before the feature existed
are never backfilled, and historical rows are never rewritten.
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
    AuthorizationExecutionReceipt,
    AuthorizationGrantUse,
    Machine,
)

_HASH_LEN = 64

# A result digest is exactly 64 lowercase hexadecimal characters; uppercase
# or non-hex text is a format error, never case-folded.
RESULT_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

_TABLE = AuthorizationExecutionReceipt.__table__
_USE_TABLE = AuthorizationGrantUse.__table__
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
                        f"ALTER TABLE {_TABLE.name} ADD COLUMN {name} {column_type}"
                    )
                )


def _utc_iso(instant: datetime) -> str:
    """Format an aware UTC instant as RFC 3339 text ending in ``Z``."""
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def occurred_at_instant(value: object) -> datetime:
    """Parse a stored ``occurred_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a damaged value that no longer parses sorts
    after every parseable record (its content hash cannot verify anyway)
    instead of crashing the read-only audit.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _occurred_at_parseable(value: object) -> bool:
    """Whether a stored ``occurred_at`` still parses to a UTC instant."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
            return True
        except ValueError:
            pass
    return False


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order receipt rows by the actual UTC instant of ``occurred_at``.

    Ordering is by the parsed UTC instant and then by ``id``, so an
    exact-second stamp sorts before any fractional-second stamp of the same
    second (ISO text alone cannot express that, since ``.`` precedes ``Z``).
    A stamp that no longer parses sorts deterministically last, and a damaged
    non-string id sorts as empty rather than crashing the comparison.
    """
    return sorted(
        rows,
        key=lambda row: (
            occurred_at_instant(row._mapping["occurred_at"]),
            row._mapping["id"] if isinstance(row._mapping["id"], str) else "",
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


def _append_receipt(
    conn: Connection,
    *,
    machine_id: str,
    use_id: str,
    grant_id: str,
    authorization_event_id: str,
    action_type: str,
    resource: str,
    outcome: str,
    result_digest: str,
    occurred_at: str,
) -> dict[str, Any]:
    """Append one linked receipt to the machine's chain tail.

    Must run inside the locked write transaction that performs the
    completion, so the receipt and its chain link commit together or leave no
    trace. ``occurred_at`` is minted inside this same locked transaction, so
    receipts commit in non-decreasing (``occurred_at``, ``id``) order. The
    receipt id is minted here and is regenerated (rarely) until the new key
    sorts strictly after the current tail, so the previous-receipt link
    always matches the order used by verification even under same-instant
    completions.
    """
    rows = _load_records(conn, machine_id)
    # Every row this feature writes carries its hashes. If any row is missing
    # chain data (e.g. an external writer), rebuild the whole machine chain
    # before appending so the new link has a sound tail.
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

    receipt_id = str(uuid.uuid4())
    if tail is not None:
        tail_id = tail._mapping["id"]
        # Same-instant completions (coarse clock) must still sort strictly
        # after the tail; the id is the only degree of freedom, since
        # occurred_at is fixed by the response contract.
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
        grant_id=grant_id,
        authorization_event_id=authorization_event_id,
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
            grant_id=grant_id,
            authorization_event_id=authorization_event_id,
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
        "id": receipt_id,
        "machine_id": machine_id,
        "use_id": use_id,
        "grant_id": grant_id,
        "authorization_event_id": authorization_event_id,
        "occurred_at": occurred_at,
        "previous_receipt_id": previous_receipt_id,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
    }


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
    """Record the execution completion of one consumed authorization.

    The machine and use lookups, the source-event checks, the
    one-receipt-per-use check, and the chained insert run in one locked write
    transaction. Returns a status dict:

    * ``not_found`` — the path machine is missing, or the use does not exist
      under it (a use owned by another machine is indistinguishable from a
      missing one);
    * ``authorization_not_allowed`` — the source decision event is not a
      committed policy allow (``allowed = true`` with
      ``reason = "allowed_by_policy"``);
    * ``execution_scope_mismatch`` — ``action_type`` or ``resource`` does not
      match the source event verbatim;
    * ``receipt_already_exists`` — the use already has its single receipt;
    * ``ok`` — with the new ``receipt`` dict.

    Nothing is written on any non-ok outcome.
    """

    def _work(conn) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        use_row = conn.execute(
            _USE_TABLE.select().where(
                _USE_TABLE.c.id == use_id,
                _USE_TABLE.c.machine_id == machine_id,
            )
        ).first()
        if use_row is None:
            return {"status": "not_found"}

        use = use_row._mapping
        event_row = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == use["event_id"],
            )
        ).first()
        if event_row is None:
            # A use always references its source event; a dangling reference
            # means the consumed authorization cannot be verified at all.
            return {"status": "not_found"}

        event = event_row._mapping
        # Only a committed policy allow may ever back an execution receipt.
        if not event["allowed"] or event["reason"] != "allowed_by_policy":
            return {"status": "authorization_not_allowed"}

        # The executed action and resource must match the authorized scope
        # verbatim — no normalization, trimming, or case folding.
        if event["action_type"] != action_type or event["resource"] != resource:
            return {"status": "execution_scope_mismatch"}

        existing = conn.execute(
            select(_TABLE.c.id).where(_TABLE.c.use_id == use_id)
        ).first()
        if existing is not None:
            return {"status": "receipt_already_exists"}

        occurred_at = _utc_iso(datetime.now(timezone.utc))
        receipt = _append_receipt(
            conn,
            machine_id=machine_id,
            use_id=use_id,
            grant_id=use["grant_id"],
            authorization_event_id=use["event_id"],
            action_type=action_type,
            resource=resource,
            outcome=outcome,
            result_digest=result_digest,
            occurred_at=occurred_at,
        )
        return {"status": "ok", "receipt": receipt}

    try:
        return _run_with_lock_retry(engine, _work)
    except IntegrityError:
        # The unique use_id constraint is the cross-backstop race guard: a
        # concurrent transaction that committed the first receipt first wins.
        return {"status": "receipt_already_exists"}


def _first_unparseable_occurred_at_id(rows: list[Any]) -> str | None:
    """Id of the first row in chain order whose ``occurred_at`` cannot parse.

    A corrupted stamp sorts its record after every parseable one, so the
    record's chain successor would otherwise be blamed for the broken link;
    reporting the corrupted record itself keeps the audit pointed at the
    actual damage. ``None`` when every row parses.
    """
    for row in rows:
        if not _occurred_at_parseable(row._mapping["occurred_at"]):
            return row._mapping["id"]
    return None


def verify_machine_chain(
    session, machine_id: str
) -> tuple[bool, int, str | None, str | None]:
    """Read-only verification of one machine's receipt chain for audit.

    Tolerant of damaged stored values so a corrupted ``occurred_at``, ``id``,
    digest, or reference never crashes the query, is never repaired or
    recomputed for storage, and never removes the record from the total.
    Returns ``(valid, checked_count, broken_receipt_id, anomaly)`` where
    ``anomaly`` is the category of the first broken record: one of
    ``timestamp_unparseable``, ``chain_break``, ``ownership_mismatch``,
    ``use_mismatch``, ``scope_mismatch``, or ``digest_mismatch`` (``None``
    when the chain is sound).

    Records are examined in the order of the actual UTC instant of
    ``occurred_at`` and then ``id`` (an exact-second stamp precedes any
    fractional-second stamp of the same second). A record whose
    ``occurred_at`` no longer parses still enters the total and is itself
    reported as the first broken record (``timestamp_unparseable``), instead
    of crashing the scan or blaming its chain successor. For each record, in
    order: the previous-receipt link and the chain digest must match the
    append-time rules applied to the stored content digest (``chain_break``);
    the referenced use and source event must belong to the receipt's own
    machine (``ownership_mismatch``); the use must exist and name the
    receipt's grant and decision event (``use_mismatch``); the receipt's
    action and resource must match the source event verbatim
    (``scope_mismatch``); and the stored content digest must recompute from
    the record's own fields, with a well-formed stored result digest
    (``digest_mismatch``). The first mismatch sets the broken id and the
    anomaly; later records cannot change them. Only rows whose stored
    ``machine_id`` equals the path machine are examined, so another machine's
    damaged records never change this result. An empty chain is valid.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    corrupted_id = _first_unparseable_occurred_at_id(rows)
    if corrupted_id is not None:
        return False, len(rows), corrupted_id, "timestamp_unparseable"

    previous_receipt_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        receipt_id = mapping["id"]

        # Chain linkage, verified against the stored content digest: the
        # digest's own correctness is the separate digest_mismatch check
        # below, so a tampered content field is not misreported as a broken
        # link.
        if mapping["previous_receipt_id"] != previous_receipt_id:
            return False, len(rows), receipt_id, "chain_break"
        stored_content_hash = mapping["content_hash"]
        if not isinstance(stored_content_hash, str):
            return False, len(rows), receipt_id, "digest_mismatch"
        expected_chain_hash = compute_chain_hash(
            previous_chain_hash, stored_content_hash
        )
        if mapping["chain_hash"] != expected_chain_hash:
            return False, len(rows), receipt_id, "chain_break"

        use_row = session.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.id == mapping["use_id"])
        ).first()
        if use_row is None:
            return False, len(rows), receipt_id, "use_mismatch"
        use = use_row._mapping
        if use["machine_id"] != mapping["machine_id"]:
            return False, len(rows), receipt_id, "ownership_mismatch"
        if (
            use["grant_id"] != mapping["grant_id"]
            or use["event_id"] != mapping["authorization_event_id"]
        ):
            return False, len(rows), receipt_id, "use_mismatch"

        event_row = session.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == mapping["authorization_event_id"]
            )
        ).first()
        if event_row is None:
            return False, len(rows), receipt_id, "use_mismatch"
        event = event_row._mapping
        if event["machine_id"] != mapping["machine_id"]:
            return False, len(rows), receipt_id, "ownership_mismatch"
        if (
            event["action_type"] != mapping["action_type"]
            or event["resource"] != mapping["resource"]
        ):
            return False, len(rows), receipt_id, "scope_mismatch"

        try:
            content_hash = compute_content_hash(
                **{key: mapping[key] for key in _CONTENT_COLUMNS}
            )
        except (TypeError, ValueError):
            # A damaged stored value (e.g. a non-text field) cannot produce
            # the published digest; the record is broken, not a reason to
            # crash the read-only audit.
            return False, len(rows), receipt_id, "digest_mismatch"
        if mapping["content_hash"] != content_hash:
            return False, len(rows), receipt_id, "digest_mismatch"
        result_digest = mapping["result_digest"]
        if not isinstance(result_digest, str) or not RESULT_DIGEST_RE.fullmatch(
            result_digest
        ):
            return False, len(rows), receipt_id, "digest_mismatch"

        previous_receipt_id = receipt_id
        previous_chain_hash = expected_chain_hash

    return True, len(rows), None, None
