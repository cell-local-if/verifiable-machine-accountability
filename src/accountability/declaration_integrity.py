"""Read-only integrity audit of one machine's behavior declarations.

Only the ``behavior_declarations`` rows owned by the path machine are read,
and the audit is a pure function of the stored rows: it never creates,
updates, deletes, repairs, recomputes, or normalizes a declaration, and it
never participates in authorization evaluation. Another machine's
declarations — sound or damaged — never enter the checked set and can never
change this machine's conclusion.

Each stored declaration must satisfy every condition:

* ``machine_id`` equals the path machine (guaranteed by the scoped read);
* ``id`` is a UUID string;
* ``action_type`` and ``resource_pattern`` are strings that stay non-empty
  after surrounding whitespace is stripped;
* ``enabled`` is a boolean (persisted as the SQLite boolean integers 0/1);
* ``created_at`` and ``updated_at`` are UTC RFC 3339 date-times ending in
  ``Z``;
* the stripped ``(action_type, resource_pattern)`` combination is unique
  among the machine's declarations; inside a duplicate group the record
  sorting first is the one judged broken.

Records are examined in the order the audit uses to name the first broken
one: the actual UTC instant of ``created_at`` (a damaged stamp sorts after
every parseable instant instead of crashing the read-only audit) and then
``id`` ascending. The conclusion is ``(valid, checked_count,
broken_declaration_id)``: ``checked_count`` always counts every declaration
of the path machine, and the broken id is the record's stored identifier
emitted verbatim, ``None`` when every declaration verifies.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

_FIELDS = (
    "id",
    "machine_id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)

# RFC 3339 date-time expressed in UTC with a literal ``Z`` suffix. Fractional
# seconds are optional; offset forms (``+00:00``) and a missing suffix are
# rejected. The calendar/time fields are range-checked by ``datetime``.
_RFC3339_Z_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

# A stored ``created_at`` that no longer parses is ordered after every
# parseable record (the same tolerant convention the other read-only audits
# use), so a damaged stamp sorts deterministically last instead of crashing
# the query; such a record is itself reported broken by the field checks.
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def load_declarations(session: Session, machine_id: str) -> list[dict[str, Any]]:
    """Load the path machine's declarations as plain stored values.

    Issues reads only, scoped to the path machine; values are passed through
    verbatim with no normalization or repair, so a damaged stored value is
    seen exactly as persisted.
    """
    columns = ", ".join(_FIELDS)
    rows = session.execute(
        text(
            f"SELECT {columns} FROM behavior_declarations"
            " WHERE machine_id = :machine_id"
        ),
        {"machine_id": machine_id},
    ).all()
    return [dict(row._mapping) for row in rows]


def _is_uuid(value: Any) -> bool:
    """Whether the stored id is a UUID string."""
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _is_utc_z_datetime(value: Any) -> bool:
    """Whether the stored value is a UTC RFC 3339 date-time ending in ``Z``."""
    if not isinstance(value, str) or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        # Regex passed but the calendar/time values are out of range.
        return False
    return True


def _is_boolean(value: Any) -> bool:
    """Whether the stored value is a boolean.

    SQLite persists booleans as the integers 0/1, so a legitimately written
    row reads back as one of those; anything else (another number, text, or
    null) is a damaged value.
    """
    if isinstance(value, bool):
        return True
    return isinstance(value, int) and value in (0, 1)


def _created_instant(value: Any) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract; a
    damaged value sorts after every parseable record instead of raising, so
    the read-only audit neither crashes nor repairs the stored text.
    """
    if isinstance(value, str) and _RFC3339_Z_DATETIME_RE.fullmatch(value):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return _FAR_FUTURE


def _sort_id(value: Any) -> tuple[int, str]:
    # Stored ids are always strings; a damaged non-string id still needs a
    # total, deterministic ordering key instead of crashing the audit.
    if isinstance(value, str):
        return (0, value)
    return (1, str(value))


def _json_safe_stored_value(value: Any) -> Any:
    """Surface one stored identifier in the audit conclusion.

    Stored string, integer, boolean, and ``None`` values are emitted exactly
    as stored. The response contract contains no floating-point values, so a
    damaged identifier holding a float (including non-finite or negative
    zero) is surfaced in a deterministic textual form. Raw bytes are decoded
    with replacement and anything else uses ``str``, so a value JSON cannot
    represent neither crashes the read-only query nor is repaired in storage.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def verify(
    session: Session, machine_id: str
) -> tuple[bool, int, Any]:
    """Verify the path machine's behavior declarations.

    Returns ``(valid, checked_count, broken_declaration_id)`` where
    ``checked_count`` is the machine's total declaration count and
    ``broken_declaration_id`` is the stored id of the first declaration — in
    (``created_at`` instant, ``id``) order — that violates a field or
    uniqueness condition, or ``None`` when every declaration verifies.
    """
    rows = load_declarations(session, machine_id)
    checked_count = len(rows)

    # Duplicate groups are computed over the stripped (action_type,
    # resource_pattern) combination. Records whose stored fields are not
    # strings cannot form a combination; they are already broken on the field
    # checks and take no part in the uniqueness judgement.
    combination_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        action_type = row.get("action_type")
        resource_pattern = row.get("resource_pattern")
        if isinstance(action_type, str) and isinstance(resource_pattern, str):
            combination = (action_type.strip(), resource_pattern.strip())
            combination_counts[combination] = (
                combination_counts.get(combination, 0) + 1
            )

    def _is_broken(row: dict[str, Any]) -> bool:
        if row.get("machine_id") != machine_id:
            return True
        if not _is_uuid(row.get("id")):
            return True
        action_type = row.get("action_type")
        if not isinstance(action_type, str) or not action_type.strip():
            return True
        resource_pattern = row.get("resource_pattern")
        if not isinstance(resource_pattern, str) or not resource_pattern.strip():
            return True
        if not _is_boolean(row.get("enabled")):
            return True
        if not _is_utc_z_datetime(row.get("created_at")):
            return True
        if not _is_utc_z_datetime(row.get("updated_at")):
            return True
        combination = (action_type.strip(), resource_pattern.strip())
        return combination_counts.get(combination, 0) > 1

    ordered = sorted(
        rows,
        key=lambda row: (
            _created_instant(row.get("created_at")),
            _sort_id(row.get("id")),
        ),
    )
    for row in ordered:
        if _is_broken(row):
            return False, checked_count, _json_safe_stored_value(row.get("id"))
    return True, checked_count, None
