"""Immutable decision-basis snapshots for authorization decision events.

Every time the public authorization-decision-event creation entry appends an
event, the same locked write transaction captures the basis the decision was
made on and stores it next to the event:

* the machine status the decision used and whether declarations and policy
  rules were read at all (a suspended machine reads neither, and an active
  machine whose enabled declarations do not match stops before the rules);
* the enabled behavior declarations participating in the decision and the
  scope that matched the requested resource;
* the policy candidate rules under the existing priority-first semantics,
  annotated as winner / overridden / conflict / unmatched, with the winner
  and conflict groups;
* the committed result (``allowed``/``reason``), identical to the event.

Only visible business fields are snapshotted — never machine keys or policy
text beyond the rule's existing visible columns. The captured document is
serialized once, at capture time, as compact UTF-8 JSON; the read-only query
re-emits that exact text, so repeated queries are byte-identical without ever
recomputing the decision. Snapshots are append-only: they are never updated,
deleted, repaired, or reconstructed from current data, so an event written
before this feature existed simply has no row.
"""

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Connection, select
from sqlalchemy.orm import Session

from . import authorization, policy_preview
from .db import AuthorizationDecisionBasis, BehaviorDeclaration

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

# The visible declaration fields the snapshot keeps (no key material).
_DECLARATION_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)

_TABLE = AuthorizationDecisionBasis.__table__
_DECLARATION_TABLE = BehaviorDeclaration.__table__


def _created_instant(value: object) -> datetime:
    """Parse a visible ``created_at`` to its UTC instant for stable ordering.

    Captured values always satisfy the RFC 3339 ``Z`` contract; a damaged
    value sorts after every parseable instant instead of raising, matching
    the tolerant ordering convention used by the read-only audit queries.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _enabled_action_declarations(
    conn: Connection, machine_id: str, action_type: str
) -> list[dict[str, Any]]:
    """Visible fields of the machine's enabled declarations for the action.

    These are exactly the rows the single-event decision reads before testing
    the requested resource against their patterns.
    """
    rows = conn.execute(
        select(
            *[_DECLARATION_TABLE.c[name] for name in _DECLARATION_FIELDS]
        ).where(
            _DECLARATION_TABLE.c.machine_id == machine_id,
            _DECLARATION_TABLE.c.action_type == action_type,
            _DECLARATION_TABLE.c.enabled.is_(True),
        )
    ).all()
    return [dict(row._mapping) for row in rows]


def capture(
    conn: Connection,
    *,
    event: dict[str, Any],
    status: str,
    action_type: str,
    resource: str,
) -> dict[str, Any]:
    """Build the immutable basis row for an event inside its write tx.

    Must run on the same locked connection that just appended ``event``, so
    every read here sees exactly the status/declaration/rule snapshot the
    committed decision was based on. Returns the table row values
    ``{event_id, machine_id, created_at, document}``; the caller inserts them
    on the same connection so the event and snapshot commit atomically.

    A suspended machine reads neither declarations nor rules, mirroring the
    decision short-circuit. An active machine always reads its enabled
    declarations for the action; policy rules are read only when at least one
    declaration pattern matches (the decision stops at
    ``no_enabled_declaration`` before that read otherwise).
    """
    machine_id = event["machine_id"]
    captured_at = event["created_at"]

    declarations_read = status != "suspended"
    policies_read = False

    declaration_items: list[dict[str, Any]] = []
    candidate_items: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    conflicts: list[list[str]] = []

    if declarations_read:
        stored_declarations = _enabled_action_declarations(
            conn, machine_id, action_type
        )
        matched_any = False
        for stored in sorted(
            stored_declarations,
            key=lambda row: (_created_instant(row.get("created_at")), row.get("id")),
        ):
            pattern = stored.get("resource_pattern")
            matched = isinstance(pattern, str) and authorization.pattern_matches(
                pattern, resource
            )
            matched_any = matched_any or matched
            item = {field: stored.get(field) for field in _DECLARATION_FIELDS}
            item["matched"] = matched
            declaration_items.append(item)

        if matched_any:
            # The decision reads global rules for this action only; annotate
            # them with the established preview priority semantics
            # (winner/overridden/conflict/unmatched, plus invalid for a
            # damaged row). Rows for other actions never participate, so they
            # are not part of the basis.
            policies_read = True
            stored_rules = [
                row
                for row in policy_preview.load_rules(conn)
                if row.get("action_type") == action_type
            ]
            preview = policy_preview.build_preview(action_type, resource, stored_rules)
            candidate_items = preview["rules"]
            winners = preview["winners"]
            conflicts = preview["conflicts"]

    document = {
        "event_summary": {
            "id": event["id"],
            "machine_id": event["machine_id"],
            "action_type": event["action_type"],
            "resource": event["resource"],
            "allowed": event["allowed"],
            "reason": event["reason"],
            "created_at": event["created_at"],
            "previous_event_id": event["previous_event_id"],
            "content_hash": event["content_hash"],
            "chain_hash": event["chain_hash"],
        },
        "status_basis": {
            "machine_id": machine_id,
            "status": status,
            "captured_at": captured_at,
            "declarations_read": declarations_read,
            "policies_read": policies_read,
        },
        "declaration_basis": {
            "read": declarations_read,
            "declarations": declaration_items,
        },
        "policy_candidates": {
            "read": policies_read,
            "candidates": candidate_items,
            "winners": winners,
            "conflicts": conflicts,
        },
        "decision": {
            "allowed": event["allowed"],
            "reason": event["reason"],
        },
    }
    # Serialize exactly once, here: the read endpoint re-emits this text
    # verbatim. Compact UTF-8 JSON, key order fixed by insertion, and no
    # floating-point or non-finite values possible in the captured fields.
    document_text = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return {
        "event_id": event["id"],
        "machine_id": machine_id,
        "created_at": captured_at,
        "document": document_text,
    }


def insert(conn: Connection, row: dict[str, Any]) -> None:
    """Insert one basis row on the caller's locked write connection."""
    conn.execute(_TABLE.insert().values(**row))


def load_document(
    session: Session | Connection, *, machine_id: str, event_id: str
) -> str | None:
    """Return the stored basis document text for one path-machine event.

    Read-only; returns ``None`` when no snapshot exists (an event written
    before the feature, or a foreign/unknown pair). The stored text is the
    exact compact JSON the query re-emits — never re-parsed or recomputed.
    """
    return session.execute(
        select(_TABLE.c.document).where(
            _TABLE.c.event_id == event_id,
            _TABLE.c.machine_id == machine_id,
        )
    ).scalar()
