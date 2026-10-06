"""Read-only reconciliation of key rotation history against current state.

On top of the per-machine rotation hash chain (:mod:`.rotation_chain`), this
module cross-checks — for one machine, strictly read-only — that the
machine's stored rotation records fully explain its current key state:

* every record's ``created_at`` must still parse to a UTC instant;
* the chain links (``previous_rotation_id``), content hashes, and chain
  hashes must verify in check order;
* the first record carries ``version`` 2 and every later record increments
  it by exactly one;
* every record after the first continues the key hand-off: its
  ``old_public_key`` is the previous record's ``new_public_key``;
* the chain tail explains the machine row: the last record's
  ``new_public_key`` and ``version`` equal the machine's current
  ``public_key`` and ``version``. An empty history explains only a machine
  still at ``version`` 1.

Records are checked ordered by the actual UTC instant of ``created_at`` and
then by id — within one second an exact-second stamp sorts before any
fractional stamp of that second, and a record whose ``created_at`` no longer
parses sorts last. The query never inserts, updates, deletes, repairs,
recomputes, or normalizes any record — a damaged stored value is reported,
never fixed — so repeated reads of unchanged data agree and survive
restarts, and only rows owned by the path machine are examined.

The conclusion is ``{machine_id, valid, checked_count, latest_rotation_id,
broken_rotation_id, anomaly}``. ``checked_count`` is the machine's total
rotation count and ``latest_rotation_id`` the chain tail (the last record in
check order); an empty history reports ``0`` and ``None``. When anything is
inconsistent, the first anomalous record in check order is named in
``broken_rotation_id`` — within one record the checks run in the order
moments, chain link and hashes, version transition, key transition — and
``anomaly`` is one of ``timestamp_unparseable`` (a ``created_at`` that no
longer parses), ``chain_mismatch`` (a previous-rotation link, content hash,
or chain hash that does not verify), ``version_transition_mismatch`` (a
version that is not the expected successor), ``key_transition_mismatch`` (an
``old_public_key`` that does not continue the previous record's key), or
``current_state_mismatch`` (the tail does not explain the machine's current
``public_key``/``version``, or the history is empty while the machine has
moved past ``version`` 1 — a drift with no record to blame, reported with a
``None`` ``broken_rotation_id``). When everything reconciles, ``valid`` is
``True`` and both ``broken_rotation_id`` and ``anomaly`` are ``None``.
"""

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .db import KeyRotationEvent, Machine
from .rotation_chain import compute_chain_hash, compute_content_hash

TIMESTAMP_UNPARSEABLE = "timestamp_unparseable"
CHAIN_MISMATCH = "chain_mismatch"
VERSION_TRANSITION_MISMATCH = "version_transition_mismatch"
KEY_TRANSITION_MISMATCH = "key_transition_mismatch"
CURRENT_STATE_MISMATCH = "current_state_mismatch"

# A stored moment only means something under the RFC 3339 ``Z`` contract
# every writer of these tables commits to; fractional seconds are optional
# and offset forms or a missing suffix never parse.
_UTC_Z_STAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

# A record whose ``created_at`` no longer parses cannot be ordered and
# sorts after every parseable one in the check order.
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _parse_moment(value: object) -> datetime | None:
    """Parse a stored moment to its UTC instant, or ``None`` when damaged.

    Only text satisfying the RFC 3339 ``Z`` contract parses; a missing,
    non-text, offset-form, malformed, or out-of-range value returns ``None``
    so the read-only reconciliation can report it instead of crashing,
    repairing, or normalizing it.
    """
    if isinstance(value, str) and _UTC_Z_STAMP_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            return None
    return None


def _id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored identifier, tolerant of a damaged value."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def _check_order_key(record: KeyRotationEvent) -> tuple:
    """Order records by created-at instant, then id; broken moments last.

    Parsing the stamp compares true instants, so within one second an
    exact-second stamp (no fractional part) sorts before any fractional
    stamp of that second — text ordering would invert them (``.`` precedes
    ``Z``).
    """
    instant = _parse_moment(record.created_at)
    return (
        instant if instant is not None else _FAR_FUTURE,
        _id_key(record.id),
    )


def reconcile(session, machine: Machine) -> dict[str, Any]:
    """Read-only reconciliation conclusion for one machine's rotations.

    Reads only rows owned by the path machine — its rotation records and
    the already-loaded machine row — and never writes, repairs, recomputes,
    or fabricates anything. Returns the fixed-shape conclusion
    ``{machine_id, valid, checked_count, latest_rotation_id,
    broken_rotation_id, anomaly}``; when the full history explains the
    machine's current key state, ``valid`` is ``True`` and both
    ``broken_rotation_id`` and ``anomaly`` are ``None``.
    """
    records = sorted(
        session.scalars(
            select(KeyRotationEvent).where(
                KeyRotationEvent.machine_id == machine.id
            )
        ).all(),
        key=_check_order_key,
    )

    checked_count = len(records)
    latest_rotation_id = records[-1].id if records else None

    def conclusion(broken_rotation_id, anomaly) -> dict[str, Any]:
        return {
            "machine_id": machine.id,
            "valid": anomaly is None,
            "checked_count": checked_count,
            "latest_rotation_id": latest_rotation_id,
            "broken_rotation_id": broken_rotation_id,
            "anomaly": anomaly,
        }

    if not records:
        # An empty history explains only a machine still at version 1; a
        # machine that has moved past it has drifted with no record to
        # blame, so the break is reported with a null broken_rotation_id.
        if machine.version == 1:
            return conclusion(None, None)
        return conclusion(None, CURRENT_STATE_MISMATCH)

    previous_rotation_id: str | None = None
    previous_chain_hash = ""
    previous_new_public_key: str | None = None
    expected_version = 2
    for record in records:
        # Moments: the record's created_at must still parse to a UTC instant.
        if _parse_moment(record.created_at) is None:
            return conclusion(record.id, TIMESTAMP_UNPARSEABLE)

        # Chain: the previous-rotation link, the content hash, and the
        # chain hash must verify against the running chain in check order.
        content_hash = compute_content_hash(
            id=record.id,
            machine_id=record.machine_id,
            old_public_key=record.old_public_key,
            new_public_key=record.new_public_key,
            version=record.version,
            created_at=record.created_at,
        )
        chain_hash = compute_chain_hash(previous_chain_hash, content_hash)
        if (
            record.previous_rotation_id != previous_rotation_id
            or record.content_hash != content_hash
            or record.chain_hash != chain_hash
        ):
            return conclusion(record.id, CHAIN_MISMATCH)

        # Version transition: the first record is version 2 and every later
        # record increments the previous version by exactly one.
        if record.version != expected_version:
            return conclusion(record.id, VERSION_TRANSITION_MISMATCH)

        # Key transition: every record after the first continues the
        # hand-off — its old_public_key is the previous record's new key.
        if (
            previous_rotation_id is not None
            and record.old_public_key != previous_new_public_key
        ):
            return conclusion(record.id, KEY_TRANSITION_MISMATCH)

        previous_rotation_id = record.id
        previous_chain_hash = chain_hash
        previous_new_public_key = record.new_public_key
        expected_version += 1

    # Current state: the chain tail must explain the machine row — the last
    # record's new_public_key and version are the machine's current ones.
    tail = records[-1]
    if (
        tail.new_public_key != machine.public_key
        or tail.version != machine.version
    ):
        return conclusion(tail.id, CURRENT_STATE_MISMATCH)

    return conclusion(None, None)
