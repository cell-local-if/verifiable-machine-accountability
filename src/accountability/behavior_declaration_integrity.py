"""Read-only completeness audit of one machine's behavior declarations.

Every declaration owned by the path machine is examined in a fixed order —
the actual UTC instant of ``created_at`` and then ``id`` ascending — against
the stored-record contract:

* ``machine_id`` equals the path machine;
* ``id`` is the canonical textual form of a UUID;
* ``action_type`` and ``resource_pattern`` are strings that stay non-empty
  after surrounding whitespace is stripped;
* ``enabled`` is a stored boolean (SQLite persists it as exactly ``0``/``1``);
* ``created_at`` and ``updated_at`` are UTC RFC 3339 date-times ending in
  ``Z``;
* the whitespace-stripped ``(action_type, resource_pattern)`` pair is unique
  within the machine; when a pair repeats, the earliest record of the
  duplicate group (in audit order) is the anomalous one.

Rows are read with raw SQL (not the ORM) so a tampered value is seen exactly
as stored: an ORM boolean read coerces ``2`` or ``'yes'`` into ``True`` and
would hide damage, whereas the raw bucket shows the real value. The audit is
a pure function of the stored rows — it never creates, updates, deletes,
repairs, recomputes, or normalizes anything — so repeat calls against
unchanged data are byte-identical and declarations persisted across restarts
are audited unchanged.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .db import BehaviorDeclaration

_TABLE = BehaviorDeclaration.__table__

_AUDIT_COLUMNS = (
    "id",
    "machine_id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)

# RFC 3339 date-time expressed in UTC with a literal ``Z`` suffix. Fractional
# seconds are optional; offset forms and a missing suffix are rejected.
_UTC_Z_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _is_utc_z_datetime(value: object) -> bool:
    """Whether ``value`` is a Z-terminated UTC RFC 3339 date-time.

    The regex enforces the textual shape (including the trailing ``Z``) and
    parsing the calendar/time fields rejects out-of-range values such as month
    13 or hour 24 that the regex alone would admit.
    """
    if not isinstance(value, str) or _UTC_Z_DATETIME_RE.fullmatch(value) is None:
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


def _is_uuid(value: object) -> bool:
    """Whether ``value`` is the canonical textual form of a UUID."""
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    # ``uuid.UUID`` also accepts bare hex and surrounding braces; the stored
    # contract is the canonical dashed lowercase representation.
    return str(parsed) == value


def _is_stored_boolean(value: object) -> bool:
    """Whether ``value`` is a persisted boolean.

    SQLite has no native boolean type: the ORM writes booleans as the integers
    ``0``/``1`` and a raw read returns them as ``int``. A tampered bucket such
    as ``2``, ``'yes'``, a blob, or ``NULL`` must not pass, so exactly the
    integer values 0 and 1 are accepted (``bool`` is excluded for symmetry,
    although the raw SQLite driver never returns one).
    """
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the Z contract; a damaged value
    sorts deterministically after every parseable record instead of crashing
    the read-only audit (the record is flagged anyway by the timestamp check).
    """
    if isinstance(value, str) and _UTC_Z_DATETIME_RE.fullmatch(value) is not None:
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _stripped_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    """The whitespace-stripped uniqueness key for one row, if well-formed.

    Only string values that stay non-empty after trimming can participate in
    the uniqueness condition; a field-damaged row already fails on its own and
    contributes no key.
    """
    action = row.get("action_type")
    resource = row.get("resource_pattern")
    if not isinstance(action, str) or not isinstance(resource, str):
        return None
    action = action.strip()
    resource = resource.strip()
    if not action or not resource:
        return None
    return action, resource


def load_declarations(session: Session, machine_id: str) -> list[dict[str, Any]]:
    """Load one machine's declaration rows as plain, verbatim stored values.

    Only rows stored under the path machine are returned; another machine's
    rows can never enter the audit. Issues reads only and never normalizes a
    value.
    """
    columns = ", ".join(_AUDIT_COLUMNS)
    rows = session.execute(
        text(
            f"SELECT {columns} FROM {_TABLE.name} WHERE machine_id = :machine_id"
        ),
        {"machine_id": machine_id},
    ).all()
    return [dict(row._mapping) for row in rows]


def audit_declarations(
    stored_rows: list[dict[str, Any]], machine_id: str
) -> tuple[bool, int, str | None]:
    """Audit already-loaded declaration rows.

    Pure: no database access and no writes. Returns
    ``(valid, checked_count, broken_declaration_id)`` where ``checked_count``
    is the total number of rows stored under the path machine (every row is
    counted, including broken ones) and ``broken_declaration_id`` is the
    stored id of the first record, in ``(created_at instant, id)`` order, that
    fails a field or uniqueness condition; it is ``None`` when every record is
    sound.
    """
    checked_count = len(stored_rows)

    # A total, deterministic tie-break keeps even a damaged non-string id at a
    # definite position instead of relying on the database's return order, so
    # the first-anomaly choice is byte-stable across repeat calls.
    def id_key(row: dict[str, Any]) -> tuple[int, str]:
        value = row.get("id")
        if isinstance(value, str):
            return (0, value)
        return (1, str(value))

    ordered = sorted(
        stored_rows,
        key=lambda row: (_created_instant(row.get("created_at")), id_key(row)),
    )

    # Uniqueness over the whitespace-stripped (action_type, resource_pattern)
    # pair. The first record in audit order owns each pair; when the same pair
    # appears again, the pair's earliest record becomes the duplicate-group
    # anomaly. Field-damaged rows contribute no key.
    first_row_for_pair: dict[tuple[str, str], Any] = {}
    duplicate_anomaly_ids: set[Any] = set()
    for row in ordered:
        pair = _stripped_pair(row)
        if pair is None:
            continue
        if pair in first_row_for_pair:
            duplicate_anomaly_ids.add(first_row_for_pair[pair])
        else:
            first_row_for_pair[pair] = row.get("id")

    for row in ordered:
        action = row.get("action_type")
        resource = row.get("resource_pattern")
        if (
            row.get("machine_id") != machine_id
            or not _is_uuid(row.get("id"))
            or not isinstance(action, str)
            or not action.strip()
            or not isinstance(resource, str)
            or not resource.strip()
            or not _is_stored_boolean(row.get("enabled"))
            or not _is_utc_z_datetime(row.get("created_at"))
            or not _is_utc_z_datetime(row.get("updated_at"))
            or row.get("id") in duplicate_anomaly_ids
        ):
            return False, checked_count, row.get("id")

    return True, checked_count, None
