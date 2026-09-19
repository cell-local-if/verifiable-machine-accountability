"""Per-machine tamper-evident hash chains for key rotation events.

Each machine's rotation records form an ordered chain (ordered by
``created_at`` then ``id``), following the same rules as the authorization
decision event chain:

* ``content_hash`` = SHA-256 of the compact, key-sorted JSON document built
  from the record's own fields: ``{id, machine_id, old_public_key,
  new_public_key, version, created_at}``.
* ``chain_hash`` = SHA-256 of ``<previous chain_hash>:<content_hash>``; the
  first record uses the empty string as the previous chain hash.
* ``previous_rotation_id`` is ``None`` for a machine's first record and the
  prior record's id otherwise.

A successful rotation updates the machine row and appends the chain tail
inside a single write transaction, so concurrent rotations cannot lose
records, fork the chain, or break a link.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import Connection, Engine, inspect, text

from .chain import _run_with_lock_retry, _utc_now_iso
from .db import KeyRotationEvent, Machine

_HASH_LEN = 64

_TABLE = KeyRotationEvent.__table__
_MACHINE_TABLE = Machine.__table__
_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "old_public_key",
    "new_public_key",
    "version",
    "created_at",
)


def compute_content_hash(
    *,
    id: str,
    machine_id: str,
    old_public_key: str,
    new_public_key: str,
    version: int,
    created_at: str,
) -> str:
    document = json.dumps(
        {
            "id": id,
            "machine_id": machine_id,
            "old_public_key": old_public_key,
            "new_public_key": new_public_key,
            "version": version,
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
    """Add chain columns to databases created before the chain feature."""
    inspector = inspect(engine)
    existing = {column["name"] for column in inspector.get_columns(_TABLE.name)}
    additions = {
        "previous_rotation_id": "VARCHAR(36)",
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


def _load_events(conn: Connection, machine_id: str | None = None) -> list[Any]:
    statement = _TABLE.select()
    if machine_id is not None:
        statement = statement.where(_TABLE.c.machine_id == machine_id)
    statement = statement.order_by(_TABLE.c.created_at, _TABLE.c.id)
    return list(conn.execute(statement))


def _recompute_rows(rows: list[Any]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    previous_rotation_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["previous_rotation_id"] != previous_rotation_id
            or mapping["content_hash"] != content_hash
            or mapping["chain_hash"] != chain_hash
        ):
            updates.append(
                {
                    "id": mapping["id"],
                    "previous_rotation_id": previous_rotation_id,
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                }
            )
        previous_rotation_id = mapping["id"]
        previous_chain_hash = chain_hash
    return updates


def backfill_chains(engine: Engine) -> None:
    """Fill missing chain data on records written before the chain feature.

    Processing is per machine in (created_at, id) order. The recomputation is
    deterministic, so a restart over an already complete database issues no
    writes. Each machine is handled inside a locked transaction so a
    concurrent rotation can neither interleave with the backfill nor fork.
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
            rows = _load_events(conn, machine_id)
            # previous_rotation_id is NULL on the first record, so completeness
            # is determined by the two hashes being present everywhere.
            if not rows or any(
                row._mapping["content_hash"] is None
                or row._mapping["chain_hash"] is None
                for row in rows
            ):
                for values in _recompute_rows(rows):
                    rotation_id = values.pop("id")
                    conn.execute(
                        _TABLE.update()
                        .where(_TABLE.c.id == rotation_id)
                        .values(**values)
                    )

        _run_with_lock_retry(engine, _work)


def rotate_key(
    engine: Engine,
    *,
    machine_id: str,
    new_public_key: str,
    expected_version: int,
) -> dict[str, Any]:
    """Rotate a machine's key and append the rotation to its chain tail.

    The machine update and the chain append happen inside one locked write
    transaction, so a successful rotation always leaves exactly one audit
    record linked to the tail and concurrent rotations cannot fork the chain.
    Returns a status dict: ``not_found``, ``same_public_key``,
    ``version_conflict``, or ``ok`` with the updated machine and the new
    rotation record.
    """

    def _work(conn: Connection) -> dict[str, Any]:
        machine_row = conn.execute(
            _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
        ).first()
        if machine_row is None:
            return {"status": "not_found"}
        machine = machine_row._mapping
        if new_public_key == machine["public_key"]:
            return {"status": "same_public_key"}
        if expected_version != machine["version"]:
            return {"status": "version_conflict"}

        now = _utc_now_iso()
        old_public_key = machine["public_key"]
        result = conn.execute(
            _MACHINE_TABLE.update()
            .where(
                _MACHINE_TABLE.c.id == machine_id,
                _MACHINE_TABLE.c.version == expected_version,
            )
            .values(
                public_key=new_public_key,
                version=expected_version + 1,
                updated_at=now,
            )
        )
        if result.rowcount == 0:
            # The write lock serializes rotations, so this is only a defensive
            # reclassification of a raced machine row.
            current = conn.execute(
                _MACHINE_TABLE.select().where(_MACHINE_TABLE.c.id == machine_id)
            ).first()
            if current is None:
                return {"status": "not_found"}
            if current._mapping["public_key"] == new_public_key:
                return {"status": "same_public_key"}
            return {"status": "version_conflict"}

        rows = _load_events(conn, machine_id)

        # Normally the startup backfill leaves every row complete. If any row
        # is missing chain data (e.g. an external writer), rebuild the whole
        # machine chain before appending so the new link has a sound tail.
        if any(
            row._mapping["content_hash"] is None
            or row._mapping["chain_hash"] is None
            for row in rows
        ):
            for values in _recompute_rows(rows):
                rotation_id = values.pop("id")
                conn.execute(
                    _TABLE.update().where(_TABLE.c.id == rotation_id).values(**values)
                )
            rows = _load_events(conn, machine_id)

        tail = rows[-1] if rows else None

        # Regenerate (rarely) until the new key sorts strictly after the tail.
        created_at = now
        rotation_id = str(uuid.uuid4())
        if tail is not None:
            tail_created_at = tail._mapping["created_at"]
            tail_id = tail._mapping["id"]
            if created_at < tail_created_at:
                created_at = tail_created_at
            while created_at == tail_created_at and rotation_id <= tail_id:
                rotation_id = str(uuid.uuid4())

        if tail is None:
            previous_rotation_id = None
            previous_chain_hash = ""
        else:
            previous_rotation_id = tail._mapping["id"]
            previous_chain_hash = tail._mapping["chain_hash"]

        version = expected_version + 1
        content_hash = compute_content_hash(
            id=rotation_id,
            machine_id=machine_id,
            old_public_key=old_public_key,
            new_public_key=new_public_key,
            version=version,
            created_at=created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        conn.execute(
            _TABLE.insert().values(
                id=rotation_id,
                machine_id=machine_id,
                old_public_key=old_public_key,
                new_public_key=new_public_key,
                version=version,
                created_at=created_at,
                previous_rotation_id=previous_rotation_id,
                content_hash=content_hash,
                chain_hash=chain_hash,
            )
        )
        return {
            "status": "ok",
            "machine": {
                "id": machine["id"],
                "external_id": machine["external_id"],
                "display_name": machine["display_name"],
                "public_key": new_public_key,
                "status": machine["status"],
                "version": version,
                "created_at": machine["created_at"],
                "updated_at": now,
            },
            "event": {
                "id": rotation_id,
                "machine_id": machine_id,
                "old_public_key": old_public_key,
                "new_public_key": new_public_key,
                "version": version,
                "created_at": created_at,
                "previous_rotation_id": previous_rotation_id,
                "content_hash": content_hash,
                "chain_hash": chain_hash,
            },
        }

    return _run_with_lock_retry(engine, _work)


def verify_chain(session, machine_id: str) -> tuple[bool, int, str | None]:
    """Verify a machine's rotation chain in (created_at, id) order.

    Returns ``(valid, checked_count, broken_rotation_id)``. The first record
    whose recomputed content hash, previous-rotation link, or chain hash
    differs from the stored values is reported; an empty chain is valid.
    """
    rows = list(
        session.execute(
            _TABLE.select()
            .where(_TABLE.c.machine_id == machine_id)
            .order_by(_TABLE.c.created_at, _TABLE.c.id)
        )
    )

    previous_rotation_id: str | None = None
    previous_chain_hash = ""
    for row in rows:
        mapping = row._mapping
        content_hash = compute_content_hash(
            **{key: mapping[key] for key in _CONTENT_COLUMNS}
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            mapping["content_hash"] != content_hash
            or mapping["previous_rotation_id"] != previous_rotation_id
            or mapping["chain_hash"] != chain_hash
        ):
            return False, len(rows), mapping["id"]
        previous_rotation_id = mapping["id"]
        previous_chain_hash = chain_hash

    return True, len(rows), None
