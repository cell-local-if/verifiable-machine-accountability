"""Per-machine tamper-evident hash chains for authorization decision evidence.

Each machine's evidence records form an ordered chain (ordered by the commit
``created_at`` then ``id``), following the same rules as the authorization
decision event chain:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the record's own fields: ``{id, machine_id, event_id, evidence_type,
  content_hash, created_at}`` — the stored evidence fingerprint participates
  in the digest under its existing ``content_hash`` name; the chain fields
  themselves are never part of the digest.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first record uses the empty string as the previous chain hash.
* ``previous_evidence_id`` is ``None`` for a machine's first record and the
  immediately preceding record's id otherwise.

A new evidence record is registered (machine/event lookup, duplicate-fingerprint
check, insert) and linked into the machine's chain tail inside a single locked
write transaction, so concurrent registrations cannot lose records, fork the
chain, point two records at the same predecessor, or break a link.
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import AuthorizationDecisionEvent, AuthorizationDecisionEvidence, Machine

_HASH_LEN = 64
_LOWER_HEX_HASH_RE = re.compile(r"[0-9a-f]{64}")
# RFC 3339 date-time in UTC ending in ``Z`` with optional fractional seconds;
# offset forms, a missing suffix, surrounding whitespace, and non-``Z``
# suffixes are rejected. The calendar/time fields are range-checked by parsing.
_RFC3339_Z_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

_TABLE = AuthorizationDecisionEvidence.__table__
_EVENT_TABLE = AuthorizationDecisionEvent.__table__
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

    Registered values always satisfy the RFC 3339 ``Z`` contract, so parsing
    cannot fail for legitimately written rows; a tampered value that no
    longer parses sorts after every parseable record (its content digest and
    chain digest will not verify and the malformed stamp is an anomaly of its
    own) instead of crashing the audit.
    """
    if isinstance(value, str) and _RFC3339_Z_DATETIME_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _is_valid_created_at(value: object) -> bool:
    """Whether a stored stamp is a well-formed, in-range UTC ``Z`` date-time."""
    if not isinstance(value, str) or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _chain_order(rows: list[Any]) -> list[Any]:
    """Order records by the actual UTC instant of ``created_at``, then id.

    ISO text alone is not chronological across the fractional-second boundary
    (within one second ``...:00.5Z`` precedes ``...:00Z`` lexicographically
    even though the instant is later), so stamps are parsed first.
    """
    return sorted(
        rows,
        key=lambda row: (
            _created_instant(row._mapping["created_at"]),
            row._mapping["id"],
        ),
    )


def compute_content_hash(
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


def compute_chain_hash(previous_chain_hash: str, content_hash: str) -> str:
    message = f"{previous_chain_hash}:{content_hash}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def migrate_schema(engine: Engine) -> None:
    """Add chain columns to databases created before the chain feature.

    The evidence fingerprint column (``content_hash``) predates the chain and
    is untouched; only the predecessor link and the chain digest are added.
    """
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "previous_evidence_id": "VARCHAR(36)",
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
        content_digest = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        if (
            mapping["previous_evidence_id"] != previous_evidence_id
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_evidence_id": previous_evidence_id,
                    "chain_hash": chain_hash,
                }
            )
        previous_evidence_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on evidence written before the chain feature.

    Processing is per machine in (created_at, id) order. The recomputation is
    deterministic, so a restart over an already complete database issues no
    writes. Each machine is handled inside a locked transaction so a
    concurrent registration can neither interleave with the backfill nor fork.
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
            # previous_evidence_id is NULL on the first record, so completeness
            # is determined by the chain digest being present everywhere.
            if not rows or any(
                row._mapping["chain_hash"] is None for row in rows
            ):
                for values in _recompute_rows(rows):
                    evidence_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == evidence_id)
                        .values(**values)
                    )

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

    The machine/event ownership lookup, the duplicate-fingerprint check, the
    insert, and the chain-tail append all happen inside one locked write
    transaction, so concurrent registrations cannot lose records, fork the
    per-machine chain, repeat a predecessor pointer, or break a link. Returns
    a status dict: ``not_found`` (machine or event missing, or the event
    belongs to another machine; nothing is written), ``duplicate_evidence``
    (the same fingerprint is already registered on the event; nothing is
    written), or ``ok`` with the new evidence ``record``.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine = conn.execute(
            Machine.__table__.select().where(Machine.__table__.c.id == machine_id)
        ).first()
        if machine is None:
            return {"status": "not_found"}
        event = conn.execute(
            _EVENT_TABLE.select().where(
                _EVENT_TABLE.c.id == event_id,
                _EVENT_TABLE.c.machine_id == machine_id,
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
        if any(row._mapping["chain_hash"] is None for row in rows):
            for values in _recompute_rows(rows):
                evidence_id = values.pop("id")
                conn.execute(
                    _TABLE.update()
                    .where(_TABLE.c.id == evidence_id).values(**values)
                )
            rows = _load_records(conn, machine_id)

        tail = rows[-1] if rows else None

        # Regenerate (rarely) until the new key sorts strictly after the tail.
        created_at = _utc_now_iso()
        evidence_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_id = tail._mapping["id"]
            if created_at < tail_created_at:
                created_at = tail_created_at
            while created_at == tail_created_at and evidence_id <= tail_id:
                evidence_id = str(uuid.uuid4())

        if tail is None:
            previous_evidence_id = None
            previous_chain_hash = ""
        else:
            previous_evidence_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        content_digest = compute_content_hash(
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
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "record": {
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


def _passes_existing_evidence_audit(mapping: Any, machine_event_ids: set[str]) -> bool:
    """The three conclusions of the pre-existing read-only evidence audit.

    A record passes only when its event resolves to an existing decision event
    owned by the same machine (missing or foreign-owned events fail), its
    ``evidence_type`` is a string that stays non-empty after trimming
    surrounding whitespace, and its stored ``content_hash`` fingerprint is
    exactly 64 lowercase hexadecimal characters compared as stored with no
    case folding.
    """
    fingerprint = mapping["content_hash"]
    return (
        mapping["event_id"] in machine_event_ids
        and isinstance(mapping["evidence_type"], str)
        and bool(mapping["evidence_type"].strip())
        and isinstance(fingerprint, str)
        and _LOWER_HEX_HASH_RE.fullmatch(fingerprint) is not None
    )


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify one machine's evidence chain in (created-at instant, id) order.

    Returns ``(valid, checked_count, broken_evidence_id)`` combining the
    pre-existing evidence audit conclusions (event ownership, non-blank
    evidence type, exact lowercase-hex fingerprint) with the chain checks: a
    well-formed ``created_at`` stamp, the recomputed compact content digest,
    the previous-evidence link, and the chain digest. The first record failing
    any check is reported; an empty chain is valid. Only the path machine's
    records are examined. Read-only: it never writes, repairs, or deletes
    anything; it issues no writes, so repeated calls return byte-identical
    conclusions.
    """
    rows = _chain_order(
        list(
            session.execute(
                _TABLE.select().where(_TABLE.c.machine_id == machine_id)
            )
        )
    )

    machine_event_ids = set(
        session.execute(
            _EVENT_TABLE.select()
            .where(_EVENT_TABLE.c.machine_id == machine_id)
            .with_only_columns(_EVENT_TABLE.c.id)
        )
        .scalars()
        .all()
    )

    previous_evidence_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        # The compact content digest is not a stored column: the stored
        # fingerprint (``content_hash``) is one of its inputs. Tampering with
        # any content field changes the recomputed digest and therefore the
        # expected chain digest. A malformed ``created_at`` is itself an
        # anomaly, counted in the total, and still feeds the digest check.
        content_digest = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_digest)
        if (
            not _passes_existing_evidence_audit(mapping, machine_event_ids)
            or not _is_valid_created_at(mapping["created_at"])
            or mapping["previous_evidence_id"] != previous_evidence_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_evidence_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
