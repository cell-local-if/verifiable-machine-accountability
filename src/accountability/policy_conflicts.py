"""Read-only conflict and override analysis over the global policy rules.

Only the global ``policy_rules`` table is read; the analysis never consults
machines, behavior declarations, authorization events, or the rule hash
chain and it never writes. Every stored row is retained in the rule details
exactly as stored (only values JSON cannot represent are surfaced in a
deterministic textual form), annotated with its relation to the other rules:

* ``invalid``    — the stored action type or resource pattern is not a
  string, the effect is not exactly ``allow``/``deny``, or the priority is a
  boolean, a non-integer, or negative. Invalid rules take no part in any
  matching decision, so they can neither conflict nor override;
* ``unmatched``  — a valid rule that shares no intersecting scope with any
  other valid rule under the relation rules below;
* ``conflict``   — a member of a conflict group: same action type,
  intersecting resource patterns, equal priority, opposite effects;
* ``overridden`` — a valid rule with a larger numeric priority than another
  valid rule over the same action type and an intersecting resource scope.

Two valid rules form a candidate pair only when their action types are equal
and their resource patterns have a common matching scope under the existing
``*`` glob semantics (a star matches any text, every other fragment matches
literally). Equal priority plus opposite effects is a conflict; different
priorities make the smaller-priority rule cover the larger-priority one,
regardless of effect.
"""

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .authorization import intersection_pattern

_ALLOW = "allow"
_DENY = "deny"

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

_DETAIL_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)

CONFLICT_REASON = "same_priority_opposite_effect"
OVERRIDE_REASON = "lower_priority_overrides"


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract; a
    damaged stamp never crashes the analysis — it deterministically sorts
    after every parseable instant (the same tolerant ordering the rule
    listing, window export, chain, and decision preview use).
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _is_valid_rule(stored: dict[str, Any]) -> bool:
    """Whether the fields matching depends on are usable as stored.

    Only the action type, resource pattern, effect, and priority gate
    matching; a damaged ``id`` or timestamp is still reported (and a damaged
    timestamp only changes detail ordering), never repaired. Booleans are
    deliberately not accepted as strings or integers.
    """
    action_type = stored.get("action_type")
    resource_pattern = stored.get("resource_pattern")
    effect = stored.get("effect")
    priority = stored.get("priority")
    if not isinstance(action_type, str):
        return False
    if not isinstance(resource_pattern, str):
        return False
    if effect not in (_ALLOW, "deny"):
        return False
    if not isinstance(priority, int) or isinstance(priority, bool):
        return False
    return priority >= 0


def json_safe_value(value: Any) -> Any:
    """Surface a stored value as finite, JSON-only content, never repairing.

    Strings, booleans, integers, and ``None`` pass through untouched. Any
    other stored shape (raw bytes from a TEXT column, a float a damaged
    writer left in the INTEGER priority column) is rendered to deterministic
    text, so the response contains no floating-point, no ``-0.0``, and no
    non-finite value while the stored row itself is left exactly as it is.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_rules(session: Session) -> list[dict[str, Any]]:
    """Load every global policy rule as plain stored values.

    Issues reads only; values are passed through verbatim with no
    normalization or repair.
    """
    columns = ", ".join(_DETAIL_FIELDS)
    rows = session.execute(text(f"SELECT {columns} FROM policy_rules")).all()
    return [dict(row._mapping) for row in rows]


def build_analysis(stored_rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the conflict/override payload from already-loaded rule rows.

    Pure: no database access, no writes.
    """
    # Details are ordered by the actual UTC instant of created_at, then by id
    # ascending: an exact-second stamp sorts before a fractional stamp of the
    # same second, and a damaged stamp sorts after every parseable one.
    ordered = sorted(
        stored_rules,
        key=lambda stored: (
            _created_instant(stored.get("created_at")),
            stored.get("id") if isinstance(stored.get("id"), str) else "",
        ),
    )

    valid_flags = [_is_valid_rule(stored) for stored in ordered]
    matchable_indices = [
        index for index, valid in enumerate(valid_flags) if valid
    ]

    conflicts: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    # Membership is tracked by detail position (not by stored id), so even a
    # rule with a damaged id is annotated correctly.
    conflict_indices: set[int] = set()
    overridden_indices: set[int] = set()

    for position, left_index in enumerate(matchable_indices):
        left = ordered[left_index]
        for right_index in matchable_indices[position + 1 :]:
            right = ordered[right_index]
            if left["action_type"] != right["action_type"]:
                continue
            intersection = intersection_pattern(
                left["resource_pattern"], right["resource_pattern"]
            )
            if intersection is None:
                continue
            if left["priority"] == right["priority"]:
                if left["effect"] != right["effect"]:
                    left_id = json_safe_value(left["id"])
                    right_id = json_safe_value(right["id"])
                    conflicts.append(
                        {
                            "rule_ids": sorted((left_id, right_id)),
                            "intersection": intersection,
                            "reason": CONFLICT_REASON,
                        }
                    )
                    conflict_indices.add(left_index)
                    conflict_indices.add(right_index)
            else:
                if left["priority"] < right["priority"]:
                    covering, covered = left, right
                    covering_index, covered_index = left_index, right_index
                else:
                    covering, covered = right, left
                    covering_index, covered_index = right_index, left_index
                overrides.append(
                    {
                        "covering_rule_id": json_safe_value(covering["id"]),
                        "covered_rule_id": json_safe_value(covered["id"]),
                        "intersection": intersection,
                        "reason": OVERRIDE_REASON,
                    }
                )
                overridden_indices.add(covered_index)

    # Conflict groups sort by the two rule ids ascending; override relations
    # sort by the covering id first and the covered id second.
    conflicts.sort(key=lambda entry: (entry["rule_ids"][0], entry["rule_ids"][1]))
    overrides.sort(
        key=lambda entry: (
            entry["covering_rule_id"],
            entry["covered_rule_id"],
        )
    )

    details: list[dict[str, Any]] = []
    for index, stored in enumerate(ordered):
        if not valid_flags[index]:
            relation = "invalid"
        elif index in conflict_indices:
            relation = "conflict"
        elif index in overridden_indices:
            relation = "overridden"
        else:
            relation = "unmatched"
        detail = {
            field: json_safe_value(stored.get(field)) for field in _DETAIL_FIELDS
        }
        detail["relation"] = relation
        details.append(detail)

    return {"rules": details, "conflicts": conflicts, "overrides": overrides}
