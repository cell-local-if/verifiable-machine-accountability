"""Per-machine tamper-evident hash chains for authorization decision evidence.

Each machine's evidence records form an ordered chain, ordered by the actual
UTC instant of ``created_at`` and then by ``id`` (an exact-second stamp sorts
before any fractional-second stamp of the same second, which ISO text ordering
alone cannot express):

* the per-record content digest (stored as ``content_digest``) is SHA-256 of
  the compact, key-sorted JSON document built from the record's six content
  fields: ``{id, machine_id, event_id, evidence_type, content_hash,
  created_at}``. ``content_hash`` here is the client-supplied evidence
  fingerprint; the chain fields themselves are never part of the digest.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_digest>``; the
  first record uses the empty string as the previous chain hash.
* ``previous_evidence_id`` is ``None`` for a machine's first record and the
  immediately preceding record's id otherwise, so no two records of one
  machine point at the same predecessor and no record points across machines.

A new record is appended to the machine's chain tail inside a single locked
write transaction — the same lock primitive the other per-machine chains use —
so concurrent registrations cannot lose records, fork the chain, skip a link,
or point two records at the same predecessor.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, select, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import AuthorizationDecisionEvidence, AuthorizationDecisionEvent, Machine

_HASH_LEN = 64
# Evidence fingerprints are checked exactly as stored: 64 lowercase hex chars.
_LOWER_HEX_HASH_RE = re.compile(r"[0-9a-f]{64}")

_TABLE = AuthorizationDecisionEvidence.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "event_id",
    "evidence_type",
    "content_hash",
    "created_at",
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract, so
    parsing cannot fail for them; a tampered value that no longer parses sorts
    after every parseable record (its content digest will not verify anyway)
    instead of crashing the audit.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _created_at_parseable(value: object) -> bool:
    """Whether a stored ``created_at`` still parses to a UTC instant.

    Mirrors :func:`_created_instant`: anything that would sort to the
    far-future sentinel is unparseable.
    """
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
            return True
        except ValueError:
            pass
    return False


def _first_unparseable_created_at_id(rows: list[Any]) -> str | None:
    """Id of the first row in chain order whose ``created_at`` cannot parse.

    A corrupted stamp sorts its record after every parseable one, so the
    record's chain successor would otherwise be blamed for the broken link;
    reporting the corrupted record itself keeps the audit pointed at the
    actual damage. ``None`` when every row parses.
    """
    for row in rows:
        mapping = row._mapping
        if not _created_at_parseable(mapping["created_at"]):
            return mapping["id"]
    return None


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order records by the actual UTC instant of ``created_at``, then id."""
    return sorted(
        rows,
        key=lambda row: (_created_instant(row._mapping["created_at"]), row._mapping["id"]),
    )


def ordered_records(session, machine_id: str) -> list[Any]:
    """Load one machine's evidence rows in chain order.

    Ordering is by the actual UTC instant of ``created_at`` and then ``id`` so
    an exact-second record sorts before any fractional-second record of the
    same second (ISO text alone cannot express that). A stored timestamp that
    no longer parses sorts last instead of raising, so a tampered record stays
    in the result for the audit to flag. Read-only.
    """
    return _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )


def order_evidence_rows(rows: list[Any]) -> list[Any]:
    """Order ORM evidence objects in chain order.

    Same ordering as :func:`ordered_records`, for rows already loaded as ORM
    instances. A timestamp that no longer parses sorts last rather than
    raising, so a read-only listing never crashes on one damaged record.
    """
    return sorted(
        rows,
        key=lambda row: (_created_instant(getattr(row, "created_at")),
                         getattr(row, "id")),
    )


def compute_content_digest(
    *,
    id: str,
    machine_id: str,
    event_id: str,
    evidence_type: str,
    content_hash: str,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "event_id": event_id,
            "evidence_type": evidence_type,
            "content_hash": content_hash,
            "created_at": created_at,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def compute_chain_hash(previous_chain_hash: str, content_digest: str) -> str:
    message = f"{previous_chain_hash}:{content_digest}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def migrate_schema(engine: Engine) -> None:
    """Add evidence-chain columns to databases created before the feature."""
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "previous_evidence_id": "VARCHAR(36)",
        "content_digest": f"VARCHAR({_HASH_LEN})",
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


def _load_records(conn: Connection, machine_id: str | None = None) -> list[Any]:
    statement = _TABLE.select()
    if machine_id is not None:
        statement = statement.where(_TABLE.c.machine_id == machine_id)
    return _chain_order(list(conn.execute(statement)))


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_evidence_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_digest = compute_content_digest(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        if (
            mapping["previous_evidence_id"] != previous_evidence_id
            or mapping["content_digest"] != content_digest
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_evidence_id": previous_evidence_id,
                    "content_digest": content_digest,
                    "chain_hash": chain_hash,
                }
            )
        previous_evidence_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def _apply_updates(conn: Connection, rows: list[Any]) -> None:
    for values in _recompute_rows(rows):
        evidence_id = values.pop("id")
        conn.execute(
            _TABLE.update().where(_TABLE.c.id == evidence_id).values(**values)
        )


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on evidence written before the chain feature.

    Processing is per machine in (created_at instant, id) order. The
    recomputation is deterministic, so a restart over an already complete
    database issues no writes. Each machine is handled inside a locked
    transaction so a concurrent registration can neither interleave with the
    backfill nor fork.
    """
    with engine.connect() as conn:
        machine_ids = [
            row[0]
            for row in conn.execute(
                text(
                    f"SELECT DISTINCT machine_id FROM {_TABLE.name} "
                    "ORDER BY machine_id"
                )
            )
        ]

    for machine_id in machine_ids:

        def _work(conn: Connection, machine_id=machine_id) -> None:
            rows = _load_records(conn, machine_id)
            # previous_evidence_id is NULL on the first record, so
            # completeness is determined by the two hashes being present
            # everywhere.
            if not rows or any(
                row._mapping["content_digest"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                _apply_updates(conn, rows)

        _run_with_lock_retry(engine, _work)


def append_evidence(
    engine: Engine,
    *,
    machine_id: str,
    event_id: str,
    evidence_type: str,
    content_hash: str,
) -> dict[str, Any]:
    """Atomically register one evidence record and link it into the chain.

    The machine lookup, the event-ownership lookup, the duplicate fingerprint
    check, the insert, and the chain-tail append all happen inside one locked
    write transaction, so concurrent registrations cannot lose records, fork
    the per-machine chain, skip a link, or point two records at the same
    predecessor. Returns a status dict: ``not_found`` (machine or event
    missing, or the event belongs to another machine; nothing is written),
    ``duplicate_evidence`` (the same fingerprint already exists on the event;
    nothing is written), or ``ok`` with the new ``evidence`` record.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        event = conn.execute(
            AuthorizationDecisionEvent.__table__.select().where(
                AuthorizationDecisionEvent.__table__.c.id == event_id,
                AuthorizationDecisionEvent.__table__.c.machine_id == machine_id,
            )
        ).first()
        if event is None:
            return {"status": "not_found"}

        duplicate = conn.execute(
            _TABLE.select().where(
                _TABLE.c.event_id == event_id,
                _TABLE.c.content_hash == content_hash,
            )
        ).first()
        if duplicate is not None:
            return {"status": "duplicate_evidence"}

        rows = _load_records(conn, machine_id)

        # Normally the startup backfill leaves every row complete. If any row
        # is missing chain data (e.g. an external writer), rebuild the whole
        # machine chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_digest"] is None
            or row._mapping["chain_hash"] is None
            for row in rows
        ):
            _apply_updates(conn, rows)
            rows = _load_records(conn, machine_id)

        tail = rows[-1] if rows else None

        # Regenerate (rarely) until the new key sorts strictly after the tail
        # in (created_at instant, id) chain order. Comparison is by parsed
        # instant, not text, so an exact-second and fractional spelling of the
        # same instant still tie-break on id deterministically.
        created_at = _utc_now_iso()
        evidence_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_key = (_created_instant(tail_created_at), tail._mapping["id"])
            while (_created_instant(created_at), evidence_id) <= tail_key:
                if _created_instant(created_at) < tail_key[0]:
                    created_at = tail_created_at
                else:
                    evidence_id = str(uuid.uuid4())

        if tail is None:
            previous_evidence_id = None
            previous_chain_hash = ""
        else:
            previous_evidence_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        content_digest = compute_content_digest(
            id=evidence_id,
            machine_id=machine_id,
            event_id=event_id,
            evidence_type=evidence_type,
            content_hash=content_hash,
            created_at=created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        conn.execute(
            _TABLE.insert().values(
                id=evidence_id,
                machine_id=machine_id,
                event_id=event_id,
                evidence_type=evidence_type,
                content_hash=content_hash,
                created_at=created_at,
                previous_evidence_id=previous_evidence_id,
                content_digest=content_digest,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "evidence": {
                "id": evidence_id,
                "machine_id": machine_id,
                "event_id": event_id,
                "evidence_type": evidence_type,
                "content_hash": content_hash,
                "created_at": created_at,
                "previous_evidence_id": previous_evidence_id,
                "chain_hash": chain_hash,
            },
        }

    return _run_with_lock_retry(engine, _work)


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's evidence chain in (created_at instant, id) order.

    Returns ``(valid, checked_count, broken_evidence_id)``. The first record
    whose recomputed content digest, previous-evidence link, or chain hash
    differs from the stored values is reported; an empty chain is valid. A
    record whose ``created_at`` no longer parses still enters the total count
    and is itself reported as the first broken record (its digest cannot
    verify), instead of crashing the scan or blaming its chain successor.
    Only the path machine's records are examined. Read-only.
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

    previous_evidence_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_digest = compute_content_digest(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        if (
            mapping["content_digest"] != content_digest
            or mapping["previous_evidence_id"] != previous_evidence_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_evidence_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None


def verify_full_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Audit one machine's evidence records and their hash chain together.

    Returns ``(valid, checked_count, broken_evidence_id)`` in the same shape
    as the existing evidence audit. Records are scanned in (created_at
    instant, id) chain order. A record is broken when it fails any existing
    evidence-audit check — its ``event_id`` does not resolve to an existing
    decision event owned by the path machine, ``evidence_type`` is not a
    non-blank string, or the stored ``content_hash`` fingerprint is not
    exactly 64 lowercase hexadecimal characters compared as stored — or when
    its recomputed content digest, previous-evidence link, or chain hash does
    not match. A record whose ``created_at`` is corrupted is still counted and
    is itself reported as the first broken record (its digest cannot verify),
    never crashing the scan or blaming its chain successor; a missing
    associated event does not remove the record. Only the path machine's
    records are examined, the first broken record only is reported, and the
    function is strictly read-only: it never writes, repairs, deletes, or
    normalizes anything.
    """
    rows = ordered_records(session, machine_id)

    corrupted_id = _first_unparseable_created_at_id(rows)
    if corrupted_id is not None:
        return False, len(rows), corrupted_id

    machine_event_ids = set(
        session.scalars(
            select(AuthorizationDecisionEvent.id).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    )

    previous_evidence_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_digest = compute_content_digest(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        broken = (
            mapping["event_id"] not in machine_event_ids
            or not isinstance(mapping["evidence_type"], str)
            or not mapping["evidence_type"].strip()
            or not isinstance(mapping["content_hash"], str)
            or _LOWER_HEX_HASH_RE.fullmatch(mapping["content_hash"]) is None
            or mapping["content_digest"] != content_digest
            or mapping["previous_evidence_id"] != previous_evidence_id
            or mapping["chain_hash"] != chain_hash
        )
        if broken:
            return False, len(rows), mapping["id"]
        previous_evidence_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
