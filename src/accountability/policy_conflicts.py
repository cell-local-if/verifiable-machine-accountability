"""Read-only conflict and override analysis over the global policy rules.

Only the global ``policy_rules`` table is read, and the analysis is a pure
function of the stored rows: it never writes, repairs, recomputes, or
normalizes anything. Every stored rule is kept in the detail list with its
seven visible fields emitted verbatim and a ``relation`` annotation:

* ``invalid``   — a stored action type, resource pattern, effect, or priority
  is missing or has an illegal shape/value, so the rule never takes part in
  matching;
* ``unmatched`` — a valid rule whose pattern shares no common match with any
  other valid rule for the same action;
* ``conflict``  — a valid rule taking part in a same-action, intersecting,
  equal-priority, opposite-effect pair;
* ``override``  — a valid rule taking part in a same-action intersecting pair
  whose priorities differ (the direction is carried by the ``overrides``
  array, which names the covering and the covered rule separately).

Resource-pattern overlap uses the existing glob semantics (a ``*`` matches any
text, every other segment is literal) via
``authorization.intersection_pattern``: two rules are candidates only when
their patterns can both match some common resource, and the reported
intersection is the glob describing exactly that shared scope.
"""

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

_CONFLICT_REASON = "same_priority_opposite_effect"
_OVERRIDE_REASON = "lower_priority_overrides"


def load_rules(session: Session) -> list[dict[str, Any]]:
    """Load every global policy rule as plain stored values.

    Issues reads only; values are passed through verbatim with no
    normalization or repair.
    """
    columns = ", ".join(_DETAIL_FIELDS)
    rows = session.execute(text(f"SELECT {columns} FROM policy_rules")).all()
    return [dict(row._mapping) for row in rows]


def _json_safe_stored_value(value: Any) -> Any:
    """Surface one stored field value in the analysis details.

    Stored string, integer, boolean, and ``None`` values are emitted exactly
    as stored. The response contract contains no floating-point values, so a
    damaged record holding a float (including non-finite or negative zero) is
    surfaced in a deterministic textual form. Raw bytes are decoded with
    replacement and anything else uses ``str``, so a value JSON cannot
    represent neither crashes the read-only query nor is repaired in storage.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` to its actual UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract; a
    damaged stamp never crashes the analysis — it deterministically sorts
    after every parseable instant (the same tolerant ordering the rule
    listing, window export, chain, and preview use) and its stored text is
    still emitted untouched.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _is_matchable(stored: dict[str, Any]) -> bool:
    """Whether the stored rule may take part in any matching judgement.

    Only the four decision-bearing fields named by the audit are tested: the
    action type and resource pattern must be strings, the effect must be
    exactly ``allow``/``deny``, and the priority must be a non-boolean,
    non-negative integer. A damaged ``id`` or timestamp neither invalidates
    the rule nor is repaired (matching never depends on it); such a rule still
    takes part in matching while its stored values are emitted verbatim and
    its damaged stamp sorts last.
    """
    action_type = stored.get("action_type")
    resource_pattern = stored.get("resource_pattern")
    effect = stored.get("effect")
    priority = stored.get("priority")
    return (
        isinstance(action_type, str)
        and isinstance(resource_pattern, str)
        and effect in (_ALLOW, _DENY)
        and isinstance(priority, int)
        and not isinstance(priority, bool)
        and priority >= 0
    )


def _sort_id(value: object) -> tuple[int, str]:
    # Stored ids are always strings; a damaged non-string id still needs a
    # total, deterministic ordering key instead of crashing the audit.
    if isinstance(value, str):
        return (0, value)
    return (1, str(value))


def _surfaced_key(value: Any) -> tuple[int, Any]:
    """Total ascending key over the *surfaced* (json-safe) form of an id.

    The pair ids and relation lists are emitted through
    ``_json_safe_stored_value`` and must be ordered by exactly the identifiers
    the response shows — never by the raw stored bucket (which would put every
    string id ahead of a blob id regardless of the text it surfaces as,
    emitting a pair such as ``["z", "a"]``). String ids therefore compare by
    their shown text; the other json-safe surface types get their own stable
    buckets so a damaged id still has a definite, comparable position.
    """
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, str(value))


def build_analysis(stored_rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the conflict/override analysis from already-loaded rule rows.

    Pure: no database access, no writes.
    """
    # Matchable rules are examined in a canonical order rather than storage
    # order, so every pair outcome (candidate or not, and the reported
    # intersection) is independent of how the rows happened to be returned and
    # of the direction the two rules are compared in.
    matchable = sorted(
        (dict(stored) for stored in stored_rules if _is_matchable(stored)),
        key=lambda rule: (
            rule.get("action_type"),
            _sort_id(rule.get("id")),
        ),
    )

    conflicts: list[dict[str, Any]] = []
    overrides: list[dict[str, Any]] = []
    # Relation memberships keyed by stored id; a rule can take part in several
    # pairs. Conflict marks outrank an override relation, matching the order
    # in which the audit names the two relations.
    conflict_ids: set[Any] = set()
    override_ids: set[Any] = set()

    for index, first in enumerate(matchable):
        for second in matchable[index + 1 :]:
            if first["action_type"] != second["action_type"]:
                continue
            # Candidates form only when the patterns have a common matching
            # scope; the intersection glob describes it under the same
            # ``*`` semantics.
            intersection = intersection_pattern(
                first["resource_pattern"], second["resource_pattern"]
            )
            if intersection is None:
                continue
            first_id, second_id = first["id"], second["id"]
            # Membership keys keep the raw stored id, but the emitted pair and
            # every ordering are over the surfaced identifiers: a matchable
            # rule whose id itself is damaged (for example a stored blob) is
            # surfaced through the same value conversion the details use and
            # still sorts by the text it is shown as, so the pair is always
            # emitted in ascending order instead of crashing serialization or
            # leaking the raw string-before-blob bucket order.
            safe_first = _json_safe_stored_value(first_id)
            safe_second = _json_safe_stored_value(second_id)
            if _surfaced_key(safe_first) <= _surfaced_key(safe_second):
                safe_left, safe_right = safe_first, safe_second
            else:
                safe_left, safe_right = safe_second, safe_first
            if first["priority"] == second["priority"]:
                if first["effect"] != second["effect"]:
                    conflicts.append(
                        {
                            "rule_ids": [safe_left, safe_right],
                            "intersection": intersection,
                            "reason": _CONFLICT_REASON,
                        }
                    )
                    conflict_ids.update((first_id, second_id))
            else:
                # The smaller numeric priority covers the same intersecting
                # scope; direction is reported in the override entry.
                if first["priority"] < second["priority"]:
                    covering, covered = first, second
                else:
                    covering, covered = second, first
                overrides.append(
                    {
                        "overriding_rule_id": _json_safe_stored_value(
                            covering["id"]
                        ),
                        "overridden_rule_id": _json_safe_stored_value(
                            covered["id"]
                        ),
                        "intersection": intersection,
                        "reason": _OVERRIDE_REASON,
                    }
                )
                override_ids.update((first_id, second_id))

    # Conflict groups and override relations are ordered by both party rule
    # ids ascending — by the identifiers actually emitted — so repeat queries
    # over the same rows are byte-identical even when a damaged id surfaces as
    # text that sorts among the string ids.
    conflicts.sort(
        key=lambda entry: tuple(_surfaced_key(value) for value in entry["rule_ids"])
    )
    overrides.sort(
        key=lambda entry: (
            _surfaced_key(entry["overriding_rule_id"]),
            _surfaced_key(entry["overridden_rule_id"]),
        ),
    )

    # Rule details order by the actual UTC instant of created_at and then by
    # the surfaced id ascending, so an exact-second record sorts before a
    # fractional-second record of the same second; a damaged stamp sorts after
    # every parseable one.
    ordered = sorted(
        stored_rules,
        key=lambda stored: (
            _created_instant(stored.get("created_at")),
            _surfaced_key(_json_safe_stored_value(stored.get("id"))),
        ),
    )

    details: list[dict[str, Any]] = []
    for stored in ordered:
        rule_id = stored.get("id")
        if not _is_matchable(stored):
            relation = "invalid"
        elif rule_id in conflict_ids:
            relation = "conflict"
        elif rule_id in override_ids:
            relation = "override"
        else:
            relation = "unmatched"
        detail = {
            field: _json_safe_stored_value(stored.get(field))
            for field in _DETAIL_FIELDS
        }
        detail["relation"] = relation
        details.append(detail)

    return {"rules": details, "conflicts": conflicts, "overrides": overrides}
