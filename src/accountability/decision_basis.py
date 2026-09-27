"""Immutable decision-basis snapshots for authorization decision events.

Every new authorization decision event freezes, inside the same locked write
transaction that appends it, the exact basis the decision was reached on:

* the machine's status at the decision instant and whether declarations and
  policy rules were read at all (a suspended machine is decided without either
  read, so its snapshot records that explicitly rather than silently showing
  empty inputs);
* for an active machine, the enabled behavior declarations that participated
  in the action/resource scope check, each with its visible business fields
  and whether its stored scope matched the requested resource;
* the global policy rules that participated (rules for the requested action),
  each annotated with the priority-semantics relation it had in THIS decision:
  ``winner``, ``overridden``, ``conflict``, or ``not_adopted``;
* the event result and reason and the capture moment.

The classification uses exactly the same predicates and priority semantics as
``authorization.decide`` over rows read at the same instant in the same locked
transaction, so the recorded ``decision`` can never diverge from the event's
committed ``allowed``/``reason``. Key material and policy rawtext beyond the
rules' visible fields are never stored. The snapshot document is rendered once
as compact UTF-8 JSON and served byte-for-byte afterwards: it is never
recomputed, so later declaration/rule/status changes never rewrite history and
repeat queries are byte-identical.
"""

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection

from .authorization import pattern_matches
from .db import BehaviorDeclaration, PolicyRule

_DECLARATION_TABLE = BehaviorDeclaration.__table__
_RULE_TABLE = PolicyRule.__table__

# The visible business fields frozen from each participating declaration, in
# fixed output order.
_DECLARATION_FIELDS = (
    "id",
    "machine_id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)

# The seven visible business fields frozen from each participating policy rule,
# in the same fixed order the rule listing/preview use. No chain field, key, or
# policy rawtext beyond these columns is stored.
_RULE_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
)

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _created_instant(value: object) -> datetime:
    """Parse a stored ``created_at`` tolerantly; a damaged stamp sorts last.

    Every row written through the service carries a valid RFC 3339 ``Z``
    stamp; the tolerant branch only guards externally damaged rows and keeps
    the same last-on-damage ordering convention as the other read-only views.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _id_key(value: object) -> tuple[int, str]:
    return (0, value) if isinstance(value, str) else (1, "")


def _priority_key(value: object) -> tuple[Any, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    return (1,)


def capture(
    conn: Connection,
    *,
    machine_id: str,
    status: str,
    action_type: str,
    resource: str,
    allowed: bool,
    reason: str,
) -> dict[str, Any]:
    """Read the decision inputs on ``conn`` and classify them for a snapshot.

    Must run inside the event's locked write transaction, after the decision
    itself was computed with :func:`authorization.decide` on the same
    connection; ``allowed``/``reason`` are that committed decision and are
    recorded verbatim — the snapshot never derives a second, possibly
    divergent result. The reads mirror ``decide``'s read order exactly, so
    the rows frozen here are the rows that decided the event (the write lock
    makes the two reads identical anyway): a suspended machine reads neither
    declarations nor rules; an active machine reads its enabled declarations
    for the action and, only when one matches the requested scope, reads the
    global rules for the action (a ``no_enabled_declaration`` decision
    therefore records the policy read as not performed).
    """
    captured_at = _utc_now_iso()

    if status == "suspended":
        return {
            "captured_at": captured_at,
            "status": status,
            "declarations_read": False,
            "policies_read": False,
            "allowed": allowed,
            "reason": reason,
            "declarations": [],
            "candidates": [],
        }

    declaration_rows = conn.execute(
        _DECLARATION_TABLE.select()
        .where(
            _DECLARATION_TABLE.c.machine_id == machine_id,
            _DECLARATION_TABLE.c.action_type == action_type,
            _DECLARATION_TABLE.c.enabled.is_(True),
        )
    ).all()
    declarations: list[dict[str, Any]] = []
    any_scope_match = False
    for row in declaration_rows:
        mapping = row._mapping
        pattern = mapping["resource_pattern"]
        matched = pattern_matches(pattern, resource)
        any_scope_match = any_scope_match or matched
        entry = {field: mapping[field] for field in _DECLARATION_FIELDS}
        entry["matched"] = matched
        declarations.append(entry)
    declarations.sort(
        key=lambda entry: (
            _created_instant(entry.get("created_at")),
            _id_key(entry.get("id")),
        )
    )

    if not any_scope_match:
        # Mirror the decision short-circuit: with no matching enabled
        # declaration the policy rules are never read, so the snapshot records
        # that policies were not consulted and keeps an empty candidate set.
        return {
            "captured_at": captured_at,
            "status": status,
            "declarations_read": True,
            "policies_read": False,
            "allowed": allowed,
            "reason": reason,
            "declarations": declarations,
            "candidates": [],
        }

    rule_rows = conn.execute(
        _RULE_TABLE.select().where(_RULE_TABLE.c.action_type == action_type)
    ).all()
    rules = [
        {field: row._mapping[field] for field in _RULE_FIELDS} for row in rule_rows
    ]

    return {
        "captured_at": captured_at,
        "status": status,
        "declarations_read": True,
        "policies_read": True,
        "allowed": allowed,
        "reason": reason,
        "declarations": declarations,
        "candidates": _order_candidates(rules, resource),
    }


def _order_candidates(
    rules: list[dict[str, Any]], resource: str
) -> list[dict[str, Any]]:
    """Annotate each participating rule with its decision relation, ordered.

    Relations follow the existing priority semantics:

    * ``winner``    — a decisive minimum-priority matching rule when the
      decisive tier is one effect;
    * ``conflict``  — a decisive minimum-priority rule when that tier mixes
      allows and denies (the tier is then decided as a denial);
    * ``overridden``— a matching rule at a larger numeric priority than the
      decisive minimum;
    * ``not_adopted``— a read rule whose stored scope does not match the
      requested resource, so it never entered the priority judgement.

    The stable order is priority ascending, then the actual UTC instant of
    ``created_at``, then id — the same order the policy preview uses.
    """
    matching = [
        rule
        for rule in rules
        if pattern_matches(rule["resource_pattern"], resource)
    ]
    decisive_ids: set[str] = set()
    overridden_ids: set[str] = set()
    mixed_tier = False
    if matching:
        lowest = min(rule["priority"] for rule in matching)
        decisive = [rule for rule in matching if rule["priority"] == lowest]
        decisive_ids = {rule["id"] for rule in decisive}
        overridden_ids = {
            rule["id"] for rule in matching if rule["priority"] > lowest
        }
        has_deny = any(rule["effect"] == "deny" for rule in decisive)
        has_allow = any(rule["effect"] == "allow" for rule in decisive)
        mixed_tier = has_deny and has_allow

    ordered = sorted(
        rules,
        key=lambda rule: (
            _priority_key(rule.get("priority")),
            _created_instant(rule.get("created_at")),
            _id_key(rule.get("id")),
        ),
    )
    candidates: list[dict[str, Any]] = []
    for rule in ordered:
        rule_id = rule.get("id")
        if rule_id in decisive_ids:
            relation = "conflict" if mixed_tier else "winner"
        elif rule_id in overridden_ids:
            relation = "overridden"
        else:
            relation = "not_adopted"
        entry = {field: rule[field] for field in _RULE_FIELDS}
        entry["relation"] = relation
        candidates.append(entry)
    return candidates


def build_document(event: dict[str, Any], basis: dict[str, Any]) -> str:
    """Render the five-group snapshot document (without trailing newline).

    The fixed group order is ``event_summary``, ``status_basis``,
    ``declaration_basis``, ``policy_candidates``, ``decision``; every
    collection is present even when empty. The output is compact UTF-8 JSON in
    fixed field order, with no floating-point or non-finite value; the query
    appends a single newline when serving this exact text.
    """
    document = {
        "event_summary": {
            "id": event["id"],
            "machine_id": event["machine_id"],
            "action_type": event["action_type"],
            "resource": event["resource"],
            "allowed": event["allowed"],
            "reason": event["reason"],
            "created_at": event["created_at"],
            "captured_at": basis["captured_at"],
        },
        "status_basis": {
            "machine_id": event["machine_id"],
            "status": basis["status"],
            "declarations_read": basis["declarations_read"],
            "policies_read": basis["policies_read"],
        },
        "declaration_basis": basis["declarations"],
        "policy_candidates": basis["candidates"],
        "decision": {
            "allowed": basis["allowed"],
            "reason": basis["reason"],
        },
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
