"""Read-only preview of the global policy decision for one action/resource.

The preview never consults machine status, behavior declarations, or
authorization decision events: only the global ``policy_rules`` table is read.
Every stored rule is kept in the detail list and annotated with its relation
to the hypothetical request:

* ``invalid``   — a stored field is missing or of an illegal shape/value, so
  the rule cannot participate in the decision;
* ``unmatched`` — syntactically valid, but its action differs or its resource
  pattern does not match the requested resource;
* ``winner``    — one of the decisive minimum-priority candidates;
* ``overridden``— a syntactically valid matching rule at a larger (numeric)
  priority than the decisive minimum;
* ``conflict``  — a decisive minimum-priority candidate when the minimum tier
  mixes allows and denies; the mixed tier is decided as a denial.

Invalid fields never participate in the decision: a rule with any illegal
stored field is ``invalid`` even when some other field happens to match.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .authorization import pattern_matches

_ALLOW = "allow"
_DENY = "deny"

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

# The seven visible rule columns plus their table types; a stored value that
# is missing (external writers) or has the wrong runtime type makes the rule
# invalid. Booleans are deliberately *not* accepted as integers.
_FIELD_TYPES: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("id", str),
    ("action_type", str),
    ("resource_pattern", str),
    ("effect", str),
    ("priority", int),
    ("created_at", str),
    ("updated_at", str),
)

_DETAIL_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)
_REFERENCE_FIELDS = ("id", "effect", "priority", "created_at")


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract; a
    damaged stamp never crashes the preview — it deterministically sorts after
    every parseable instant (the same tolerant ordering the rule listing,
    window export, and chain use).
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _is_valid_rule(stored: dict[str, Any]) -> bool:
    """Whether every field the decision depends on is usable as stored."""
    for name, expected_type in _FIELD_TYPES:
        value = stored.get(name)
        if not isinstance(value, expected_type):
            return False
        if isinstance(value, bool):
            return False
    if stored["effect"] not in (_ALLOW, _DENY):
        return False
    if stored["priority"] < 0:
        return False
    return True


def load_rules(session: Session) -> list[dict[str, Any]]:
    """Load every global policy rule as plain stored values.

    Issues reads only; values are passed through verbatim with no
    normalization or repair.
    """
    columns = ", ".join(_DETAIL_FIELDS)
    rows = session.execute(text(f"SELECT {columns} FROM policy_rules")).all()
    return [dict(row._mapping) for row in rows]


def build_preview(action: str, resource: str, stored_rules: list[dict[str, Any]]):
    """Compute the preview payload from already-loaded stored rule rows.

    Pure: no database access, no writes. ``action`` and ``resource`` must be
    the caller-trimmed, non-empty request values.
    """
    valid_rules: list[dict[str, Any]] = []
    for stored in stored_rules:
        rule = dict(stored) if _is_valid_rule(stored) else None
        if rule is not None:
            valid_rules.append(rule)

    matching: list[dict[str, Any]] = [
        rule
        for rule in valid_rules
        if rule["action_type"] == action
        and pattern_matches(rule["resource_pattern"], resource)
    ]

    decisive: list[dict[str, Any]] = []
    higher: list[dict[str, Any]] = []
    if matching:
        lowest_priority = min(rule["priority"] for rule in matching)
        decisive = [
            rule for rule in matching if rule["priority"] == lowest_priority
        ]
        higher = [rule for rule in matching if rule["priority"] > lowest_priority]
        has_deny = any(rule["effect"] == _DENY for rule in decisive)
        has_allow = any(rule["effect"] == _ALLOW for rule in decisive)
        conflict = has_deny and has_allow

        if conflict:
            allowed = False
            reason = "denied_by_policy"
            decisive_relation = "conflict"
        elif has_deny:
            allowed = False
            reason = "denied_by_policy"
            decisive_relation = "winner"
        else:
            allowed = True
            reason = "allowed_by_policy"
            decisive_relation = "winner"

        decisive_ids = {rule["id"] for rule in decisive}
        higher_ids = {rule["id"] for rule in higher}
    else:
        allowed = False
        reason = "no_matching_policy"
        conflict = False
        decisive_relation = "winner"
        decisive_ids = set()
        higher_ids = set()

    # Details are presented in priority order, then the actual UTC instant of
    # created_at (an exact-second stamp sorts before a fractional stamp of the
    # same second; a damaged stamp sorts after every parseable one), then id.
    # A rule whose stored priority is not an integer cannot be placed on the
    # numeric axis and sorts after every numeric priority, matching the
    # last-on-damage convention used for unparseable timestamps.
    def _priority_key(value: object) -> tuple[Any, ...]:
        if isinstance(value, int) and not isinstance(value, bool):
            return (0, value)
        return (1,)

    ordered = sorted(
        stored_rules,
        key=lambda stored: (
            _priority_key(stored.get("priority")),
            _created_instant(stored.get("created_at")),
            stored.get("id") if isinstance(stored.get("id"), str) else "",
        ),
    )

    details: list[dict[str, Any]] = []
    for stored in ordered:
        rule_id = stored.get("id")
        valid = _is_valid_rule(stored)
        if not valid:
            relation = "invalid"
        elif rule_id in decisive_ids:
            relation = decisive_relation
        elif rule_id in higher_ids:
            relation = "overridden"
        else:
            relation = "unmatched"
        details.append(_detail(stored, relation))

    # The conflict group reports the mixed minimum tier as unordered pairs
    # (lexicographically smaller id first), the whole list ordered by first
    # then second id, so a tier of any size serializes deterministically.
    conflicts: list[list[str]] = []
    if conflict:
        conflict_ids = sorted(rule["id"] for rule in decisive)
        for index, left in enumerate(conflict_ids):
            for right in conflict_ids[index + 1 :]:
                conflicts.append([left, right])

    if conflict:
        winners: list[dict[str, Any]] = []
    else:
        winners = [
            _reference(rule)
            for rule in sorted(
                decisive,
                key=lambda rule: (
                    _created_instant(rule["created_at"]),
                    rule["id"],
                ),
            )
        ]

    return {
        "action": action,
        "resource": resource,
        "rules": details,
        "conflicts": conflicts,
        "winners": winners,
        "decision": {"allowed": allowed, "reason": reason},
    }


def build_batch(
    items: list[tuple[str, str]], stored_rules: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute the batch audit payload from one shared rule snapshot.

    Pure: no database access, no writes. Every item is decided against the
    same ``stored_rules`` list — one read-time snapshot — so no input can
    influence a later result, and per-item semantics are exactly those of
    :func:`build_preview`. ``items`` holds the caller-trimmed, non-empty
    ``(action, resource)`` pairs in request order.

    The summary counts inputs, not rules: ``no_match``/``allow``/``deny``
    partition the batch by decision reason, ``conflict`` counts inputs whose
    preview carries a conflict group, and ``override`` counts inputs whose
    preview lists at least one overridden candidate. ``decisions`` counts the
    three decision reasons directly; every counter is always present, even
    when zero.
    """
    analyses: list[dict[str, Any]] = []
    summary = {
        "no_match": 0,
        "allow": 0,
        "deny": 0,
        "conflict": 0,
        "override": 0,
    }
    decisions = {
        "allowed_by_policy": 0,
        "denied_by_policy": 0,
        "no_matching_policy": 0,
    }

    for action, resource in items:
        result = build_preview(action, resource, stored_rules)
        reason = result["decision"]["reason"]
        decisions[reason] += 1
        if reason == "no_matching_policy":
            summary["no_match"] += 1
        elif reason == "allowed_by_policy":
            summary["allow"] += 1
        else:
            summary["deny"] += 1
        if result["conflicts"]:
            summary["conflict"] += 1
        if any(rule["relation"] == "overridden" for rule in result["rules"]):
            summary["override"] += 1
        analyses.append(
            {
                "input": {"action": action, "resource": resource},
                "result": result,
            }
        )

    return {
        "batch_count": len(items),
        "analyses": analyses,
        "summary": summary,
        "decisions": decisions,
    }


def _detail(stored: dict[str, Any], relation: str) -> dict[str, Any]:
    """Emit one stored rule verbatim with its preview relation appended."""
    detail = {field: stored.get(field) for field in _DETAIL_FIELDS}
    detail["relation"] = relation
    return detail


def _reference(rule: dict[str, Any]) -> dict[str, Any]:
    """The stable identity of a decisive rule kept on an overridden/winner."""
    return {field: rule[field] for field in _REFERENCE_FIELDS}
