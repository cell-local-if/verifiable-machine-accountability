"""Read-only reconciliation of key rotation history against current state.

On top of the per-machine rotation hash chain (:mod:`.rotation_chain`), this
module cross-checks — for one machine, strictly read-only — whether the
machine's current ``public_key`` and ``version`` can be explained by its
complete rotation history. The records are checked in (created-at UTC
instant, id) order, and every record must satisfy four invariants:

* its ``created_at`` still parses to a UTC instant under the RFC 3339 ``Z``
  contract every writer commits to;
* its ``previous_rotation_id`` link, ``content_hash``, and ``chain_hash``
  match the recomputed chain values (the first record links to ``None``
  and chains from the empty string);
* its ``version`` continues the sequence — the first record carries
  ``2`` (a rotation always lifts the machine off its initial version
  ``1``) and every later record exactly its predecessor plus one;
* its ``old_public_key`` equals the previous record's ``new_public_key``
  (the first record's old key is the machine's initial key, which the
  history itself does not record and is therefore not judged).

The chain tail must then line up with the machine's current row: the tail's
``new_public_key`` is the current ``public_key`` and the tail's ``version``
is the current ``version``. An empty history explains exactly one state —
``version`` ``1`` — so a machine with no rotation records is consistent
only while its version is still ``1``.

The conclusion is ``{machine_id, valid, checked_count, latest_rotation_id,
broken_rotation_id, anomaly}``. ``checked_count`` is the machine's total
rotation count and ``latest_rotation_id`` the chain tail — ``0`` and
``None`` for an empty history. When everything reconciles, ``valid`` is
``True`` and ``broken_rotation_id`` and ``anomaly`` are both ``None``.
Otherwise the first anomaly in check order is reported: ``broken_rotation_id``
names the record carrying it and ``anomaly`` is one of
``timestamp_unparseable`` (a ``created_at`` that no longer parses),
``chain_mismatch`` (a previous-rotation link, content hash, or chain hash
that does not verify), ``version_transition_mismatch`` (a version that
breaks the 2-then-plus-one sequence), ``key_transition_mismatch`` (an
``old_public_key`` that does not continue the previous record's
``new_public_key``), or ``current_state_mismatch`` (the history, sound in
itself, does not explain the machine's current key or version — including
a version that drifted off an empty history). A current-state break is a
drift of the machine row, not of any one rotation record, so
``broken_rotation_id`` is ``None`` for it.

The query never inserts, updates, deletes, repairs, recomputes, or rewrites
any record — damaged stored values are reported, never fixed — so repeated
reads of unchanged data agree byte-for-byte and survive restarts, and only
rows owned by the path machine are examined: another machine's rotations,
damaged or not, never change the outcome.
"""

from typing import Any

from sqlalchemy import select

from .db import KeyRotationEvent, Machine
from .grant_reconciliation import _id_key, _parse_moment, _tolerant_instant
from .rotation_chain import compute_chain_hash, compute_content_hash

TIMESTAMP_UNPARSEABLE = "timestamp_unparseable"
CHAIN_MISMATCH = "chain_mismatch"
VERSION_TRANSITION_MISMATCH = "version_transition_mismatch"
KEY_TRANSITION_MISMATCH = "key_transition_mismatch"
CURRENT_STATE_MISMATCH = "current_state_mismatch"

_CONTENT_COLUMNS = (
    "id",
    "machine_id",
    "old_public_key",
    "new_public_key",
    "version",
    "created_at",
)


def _content_hash_of(record: KeyRotationEvent) -> str | None:
    """Recomputed content hash, or ``None`` when a damaged field cannot
    produce the published digest (reported, never repaired)."""
    try:
        return compute_content_hash(
            **{key: getattr(record, key) for key in _CONTENT_COLUMNS}
        )
    except (TypeError, ValueError):
        return None


def reconcile(session, machine: Machine) -> dict[str, Any]:
    """Read-only rotation-history reconciliation for one machine.

    Reads only the rotation records owned by the machine and never writes,
    repairs, recomputes, or fabricates anything. Returns the fixed-shape
    conclusion ``{machine_id, valid, checked_count, latest_rotation_id,
    broken_rotation_id, anomaly}`` described in the module docstring.
    """
    rows = list(
        session.scalars(
            select(KeyRotationEvent).where(
                KeyRotationEvent.machine_id == machine.id
            )
        ).all()
    )
    # Check order: the actual UTC instant of ``created_at`` then id, so an
    # exact-second record sorts before any fractional-second record of the
    # same second; a stamp that no longer parses sorts after every
    # parseable record, with ties among damaged stamps broken by id.
    ordered = sorted(
        rows,
        key=lambda record: (
            _tolerant_instant(record.created_at),
            _id_key(record.id),
        ),
    )

    broken_rotation_id: Any = None
    anomaly: str | None = None

    previous_rotation_id: Any = None
    previous_chain_hash = ""
    previous_new_public_key: Any = None
    expected_version = 2
    for record in ordered:
        # 1. Moment: the stamp the check order relies on must still parse.
        if _parse_moment(record.created_at) is None:
            broken_rotation_id, anomaly = record.id, TIMESTAMP_UNPARSEABLE
            break
        # 2. Chain: the previous-rotation link, the content hash, and the
        # chain hash must all match the recomputed values.
        content_hash = _content_hash_of(record)
        chain_hash = (
            compute_chain_hash(previous_chain_hash, content_hash)
            if content_hash is not None
            else None
        )
        if (
            record.previous_rotation_id != previous_rotation_id
            or content_hash is None
            or record.content_hash != content_hash
            or record.chain_hash != chain_hash
        ):
            broken_rotation_id, anomaly = record.id, CHAIN_MISMATCH
            break
        # 3. Version: 2 on the first record, then exactly plus one.
        if record.version != expected_version:
            broken_rotation_id, anomaly = record.id, VERSION_TRANSITION_MISMATCH
            break
        # 4. Key: every record after the first continues its predecessor's
        # new key as its old key.
        if (
            previous_rotation_id is not None
            and record.old_public_key != previous_new_public_key
        ):
            broken_rotation_id, anomaly = record.id, KEY_TRANSITION_MISMATCH
            break
        previous_rotation_id = record.id
        previous_chain_hash = chain_hash
        previous_new_public_key = record.new_public_key
        expected_version += 1
    else:
        # Current state: a sound history must explain the machine's current
        # key and version. The drift is in the machine row, not in any one
        # rotation record, so no record is named.
        if ordered:
            tail = ordered[-1]
            if (
                tail.new_public_key != machine.public_key
                or tail.version != machine.version
            ):
                anomaly = CURRENT_STATE_MISMATCH
        elif machine.version != 1:
            anomaly = CURRENT_STATE_MISMATCH

    return {
        "machine_id": machine.id,
        "valid": anomaly is None,
        "checked_count": len(ordered),
        "latest_rotation_id": ordered[-1].id if ordered else None,
        "broken_rotation_id": broken_rotation_id,
        "anomaly": anomaly,
    }
