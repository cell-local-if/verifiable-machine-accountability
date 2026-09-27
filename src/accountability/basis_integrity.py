"""Read-only consistency audit of one authorization decision-basis snapshot.

The immutable basis snapshot (see :mod:`decision_basis`) is read back and
checked against the event it was captured for and the accountable records
that existed at capture time — machine status history, behavior
declarations, and global policy rules. The audit is strictly read-only: it
never creates, updates, deletes, repairs, recomputes, or normalizes an
event, snapshot, declaration, rule, or any other accountability record, and
it never recomputes or back-fills the decision from current data. Another
machine's records can never enter the check.

The audit answers four fixed conclusions:

* ``valid`` — ``true`` only when the event summary, status basis,
  declaration basis, policy candidates, and final decision are mutually
  consistent;
* ``checked_count`` — the number of snapshots counted for this event: ``0``
  when no snapshot exists, ``1`` for the single present snapshot (even a
  damaged one). Other machines' snapshots are never counted;
* ``broken_basis_id`` — the event id when the snapshot is missing or the
  present snapshot is inconsistent, reported verbatim as submitted on the
  path (the event the snapshot belongs to);
* ``reason`` — ``null`` when valid, otherwise the fixed code
  ``snapshot_not_found`` for a missing snapshot or the first concrete
  inconsistency category for a present-but-broken one.

Verification order is fixed so the first anomaly is stable:

1. ``document_structure`` — the stored text is one JSON object carrying
   exactly the five known groups with the expected container shapes;
2. ``event_summary_mismatch`` — the summary corresponds verbatim to the
   stored event record (result, reason, creation moment, chain fields,
   action/resource/ownership), never recomputed or completed;
3. ``status_basis_mismatch`` — the status at capture time and the
   declaration/policy read flags match the decision short-circuit rules;
4. ``declaration_basis_mismatch`` — exactly the machine's enabled
   declarations for the event action participate, in the captured
   (creation-instant, id) order, with the scope each pattern had against
   the requested resource;
5. ``policy_candidates_mismatch`` — the candidate rules and their
   winner/overridden/conflict/unmatched relations follow the priority
   semantics of the moment, in the captured order, and the winner/conflict
   groups support the final judgement;
6. ``decision_mismatch`` — the recorded decision is the one the recorded
   basis entails and equals the event's committed allowed flag and reason.
"""

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import policy_preview
from .authorization import pattern_matches
from .db import (
    AuthorizationDecisionBasis,
    AuthorizationDecisionEvent,
    BehaviorDeclaration,
    MachineStatusEvent,
)

_BASIS_TABLE = AuthorizationDecisionBasis.__table__
_DECLARATION_TABLE = BehaviorDeclaration.__table__
_STATUS_TABLE = MachineStatusEvent.__table__

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

_DECLARATION_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
)

# Stable first-anomaly categories (see module docstring).
_DOCUMENT_STRUCTURE = "document_structure"
_EVENT_SUMMARY_MISMATCH = "event_summary_mismatch"
_STATUS_BASIS_MISMATCH = "status_basis_mismatch"
_DECLARATION_BASIS_MISMATCH = "declaration_basis_mismatch"
_POLICY_CANDIDATES_MISMATCH = "policy_candidates_mismatch"
_DECISION_MISMATCH = "decision_mismatch"


def load_snapshot(
    session: Session, *, machine_id: str, event_id: str
) -> str | None:
    """Return the stored basis document text for one path-machine event.

    Read-only and machine scoped; another machine's snapshot can never be
    returned. ``None`` means this event has no snapshot row.
    """
    return session.execute(
        select(_BASIS_TABLE.c.document).where(
            _BASIS_TABLE.c.event_id == event_id,
            _BASIS_TABLE.c.machine_id == machine_id,
        )
    ).scalar()


def verify(
    session: Session,
    *,
    machine_id: str,
    event_id: str,
    event: AuthorizationDecisionEvent,
) -> tuple[bool, int, str | None, str | None]:
    """Audit one known, path-machine event's basis snapshot.

    The caller has already established that the machine and the event exist
    and that the event is owned by the path machine; this function performs
    only read-only snapshot/accountability access. Returns
    ``(valid, checked_count, broken_basis_id, reason)``.
    """
    document_text = load_snapshot(session, machine_id=machine_id, event_id=event_id)
    if document_text is None:
        # An event without a snapshot (written before the feature) is never
        # healed: no basis is fabricated from current declarations or rules.
        return False, 0, event_id, "snapshot_not_found"

    try:
        document = json.loads(document_text)
    except (ValueError, TypeError):
        return _broken(event_id, _DOCUMENT_STRUCTURE)

    if not _has_document_structure(document):
        return _broken(event_id, _DOCUMENT_STRUCTURE)

    if not _event_summary_matches(document["event_summary"], event, machine_id):
        return _broken(event_id, _EVENT_SUMMARY_MISMATCH)

    action_type = event.action_type
    resource = event.resource
    captured_at = event.created_at

    status = _status_at(session, machine_id, captured_at)
    stored_declarations = _enabled_action_declarations(
        session, machine_id, action_type, captured_at
    )
    expected_items = _expected_declaration_items(stored_declarations, resource)
    matched_any = any(item["matched"] for item in expected_items)

    if not _status_basis_matches(
        document["status_basis"],
        machine_id=machine_id,
        captured_at=captured_at,
        status=status,
        matched_any=matched_any,
    ):
        return _broken(event_id, _STATUS_BASIS_MISMATCH)

    if not _declaration_basis_matches(
        document["declaration_basis"],
        suspended=status == "suspended",
        expected_items=expected_items,
    ):
        return _broken(event_id, _DECLARATION_BASIS_MISMATCH)

    # The decision the recorded basis itself entails is derived only from
    # the snapshot's own recorded status/declarations/candidates — never
    # recomputed from live data — then compared with the stored decision
    # section and the committed event.
    if status == "suspended" or not matched_any:
        if not _policy_section_unread(document["policy_candidates"]):
            return _broken(event_id, _POLICY_CANDIDATES_MISMATCH)
        expected_decision = (
            (False, "machine_suspended")
            if status == "suspended"
            else (False, "no_enabled_declaration")
        )
    else:
        expected_decision = _policy_outcome(
            document["policy_candidates"],
            action_type=action_type,
            resource=resource,
            stored_rules=_rules_existing_at(session, captured_at),
        )
        if expected_decision is None:
            return _broken(event_id, _POLICY_CANDIDATES_MISMATCH)

    decision = document["decision"]
    if not (
        isinstance(decision, dict)
        and set(decision) == {"allowed", "reason"}
        and (decision["allowed"], decision["reason"]) == expected_decision
        and decision["allowed"] == event.allowed
        and decision["reason"] == event.reason
    ):
        return _broken(event_id, _DECISION_MISMATCH)

    return True, 1, None, None


def _broken(
    broken_id: str, reason: str
) -> tuple[bool, int, str | None, str | None]:
    """Build the fixed four-field conclusion for a present, broken snapshot."""
    return False, 1, broken_id, reason


def _has_document_structure(value: Any) -> bool:
    """Whether the parsed document carries exactly the five known groups.

    Only the container shapes the subsequent checks navigate are enforced
    here: exactly the five group keys, the two single objects, the fixed
    inner keys, and list-typed collections. Field-level consistency is each
    later check's responsibility, so a damaged value is reported with that
    check's specific category rather than as a generic structural fault.
    """
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        return False
    if set(value) != {
        "event_summary",
        "status_basis",
        "declaration_basis",
        "policy_candidates",
        "decision",
    }:
        return False
    if not isinstance(value["event_summary"], dict) or not isinstance(
        value["decision"], dict
    ):
        return False
    status_basis = value["status_basis"]
    if not isinstance(status_basis, dict) or set(status_basis) != {
        "machine_id",
        "status",
        "captured_at",
        "declarations_read",
        "policies_read",
    }:
        return False
    declaration_basis = value["declaration_basis"]
    if not isinstance(declaration_basis, dict) or set(declaration_basis) != {
        "read",
        "declarations",
    } or not isinstance(declaration_basis["declarations"], list):
        return False
    policy_candidates = value["policy_candidates"]
    if not isinstance(policy_candidates, dict) or set(policy_candidates) != {
        "read",
        "candidates",
        "winners",
        "conflicts",
    }:
        return False
    return (
        isinstance(policy_candidates["candidates"], list)
        and isinstance(policy_candidates["winners"], list)
        and isinstance(policy_candidates["conflicts"], list)
    )


def _event_summary_matches(
    summary: Any, event: AuthorizationDecisionEvent, machine_id: str
) -> bool:
    """Whether the summary repeats the stored event record verbatim.

    Every committed field — including the chain fields — must correspond
    exactly to the stored row; nothing is recomputed, tolerated, or filled
    in, and an unexpected summary key is itself a mismatch.
    """
    if not isinstance(summary, dict) or set(summary) != {
        "id",
        "machine_id",
        "action_type",
        "resource",
        "allowed",
        "reason",
        "created_at",
        "previous_event_id",
        "content_hash",
        "chain_hash",
    }:
        return False
    return (
        summary["id"] == event.id
        and summary["machine_id"] == machine_id
        and summary["action_type"] == event.action_type
        and summary["resource"] == event.resource
        and summary["allowed"] == event.allowed
        and summary["reason"] == event.reason
        and summary["created_at"] == event.created_at
        and summary["previous_event_id"] == event.previous_event_id
        and summary["content_hash"] == event.content_hash
        and summary["chain_hash"] == event.chain_hash
    )


def _created_instant(value: object) -> datetime:
    """Parse a visible ``created_at`` to its UTC instant for stable ordering.

    Captured values always satisfy the RFC 3339 ``Z`` contract; a damaged
    value sorts after every parseable instant instead of raising, matching
    the tolerant ordering convention used by the other read-only audits.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _status_at(session: Session, machine_id: str, captured_at: str) -> str | None:
    """The machine status in effect at the capture moment.

    Status changes are append-only (``machine_status_events``); each
    transition applies at its own ``created_at`` instant, ordered by the
    actual UTC instant (an exact-second stamp precedes a fractional stamp of
    the same second, which text ordering gets backwards). With no prior
    transition the machine has never changed state, so the status is its
    creation-time default (``active``). Returns ``None`` when the history is
    damaged or internally contradictory — an uncertainty the audit must not
    paper over by assuming a status.
    """
    rows = session.execute(
        select(
            _STATUS_TABLE.c.from_status,
            _STATUS_TABLE.c.to_status,
            _STATUS_TABLE.c.created_at,
        ).where(_STATUS_TABLE.c.machine_id == machine_id)
    ).all()
    transitions = [
        (row._mapping["from_status"], row._mapping["to_status"], row._mapping["created_at"])
        for row in rows
    ]
    if any(
        not isinstance(at, str)
        or not isinstance(frm, str)
        or not isinstance(to, str)
        or frm not in ("active", "suspended")
        or to not in ("active", "suspended")
        or frm == to
        for frm, to, at in transitions
    ):
        return None

    try:
        capture_instant = datetime.fromisoformat(captured_at[:-1] + "+00:00")
    except (ValueError, OverflowError, TypeError):
        return None

    # Every legitimate history stamp is a parseable UTC RFC 3339 value; a
    # damaged stamp cannot be placed relative to the capture moment, so the
    # status at that moment is undecidable rather than guessed.
    instants: list[datetime] = []
    for _, _, at in transitions:
        try:
            instants.append(datetime.fromisoformat(at[:-1] + "+00:00"))
        except (ValueError, OverflowError):
            return None

    status = "active"
    for (frm, to, _), at in sorted(
        zip(transitions, instants, strict=True), key=lambda item: item[1]
    ):
        if frm != status:
            # Contradictory history (a transition that does not continue the
            # status established by the preceding ones).
            return None
        if at <= capture_instant:
            status = to
        else:
            break
    return status


def _status_basis_matches(
    status_basis: dict[str, Any],
    *,
    machine_id: str,
    captured_at: str,
    status: str | None,
    matched_any: bool,
) -> bool:
    """Whether the captured status and read flags match the moment.

    A suspended machine is denied before any declaration or policy read, so
    both flags are false. An active machine always reads its enabled
    declarations, and reads the global rules exactly when at least one
    declaration pattern matches the requested resource. Policy is therefore
    never read without declarations, and read precisely on a scope match.
    """
    if status is None:
        return False
    if not isinstance(status_basis["declarations_read"], bool) or not isinstance(
        status_basis["policies_read"], bool
    ):
        return False
    if status_basis["machine_id"] != machine_id:
        return False
    if status_basis["status"] != status:
        return False
    if status_basis["captured_at"] != captured_at:
        return False
    if status == "suspended":
        return (
            status_basis["declarations_read"] is False
            and status_basis["policies_read"] is False
        )
    return (
        status_basis["declarations_read"] is True
        and status_basis["policies_read"] is matched_any
    )


def _enabled_action_declarations(
    session: Session, machine_id: str, action_type: str, captured_at: str
) -> list[dict[str, Any]]:
    """The machine's enabled action declarations existing at capture time.

    Declarations are append-only and their visible fields never change, so
    the rows that existed at the capture moment are exactly the rows the
    decision read; a declaration inserted afterwards never participated and
    must never make a historical snapshot inconsistent. Existence is judged
    by the actual UTC instant of ``created_at`` (text ordering gets the
    exact-second/fractional-second boundary backwards). A stamp that does
    not parse sorts after every finite instant, so an externally damaged
    row cannot be shown to have existed at capture time and never enters
    the expected basis; a snapshot that nevertheless carries it is reported
    inconsistent rather than silently matched. Disabled rows, other
    actions, and other machines never participate.
    """
    rows = session.execute(
        select(
            *[_DECLARATION_TABLE.c[name] for name in _DECLARATION_FIELDS]
        ).where(
            _DECLARATION_TABLE.c.machine_id == machine_id,
            _DECLARATION_TABLE.c.action_type == action_type,
            _DECLARATION_TABLE.c.enabled.is_(True),
        )
    ).all()
    capture_instant = _created_instant(captured_at)
    return [
        dict(row._mapping)
        for row in rows
        if _created_instant(row._mapping.get("created_at")) <= capture_instant
    ]


def _expected_declaration_items(
    stored_declarations: list[dict[str, Any]], resource: str
) -> list[dict[str, Any]]:
    """The declaration items the capture must record, in its capture order.

    The order is the capture order — creation instant (a damaged stamp
    sorts last), then id — so an anomalous ordering in the snapshot is
    detectable, not just membership.
    """
    items: list[dict[str, Any]] = []
    for stored in sorted(
        stored_declarations,
        key=lambda row: (_created_instant(row.get("created_at")), row.get("id")),
    ):
        item = {field: stored.get(field) for field in _DECLARATION_FIELDS}
        pattern = stored.get("resource_pattern")
        item["matched"] = (
            isinstance(pattern, str) and pattern_matches(pattern, resource)
        )
        items.append(item)
    return items


def _declaration_basis_matches(
    declaration_basis: dict[str, Any],
    *,
    suspended: bool,
    expected_items: list[dict[str, Any]],
) -> bool:
    """Whether exactly the participating enabled declarations are recorded.

    Only enabled declarations for the event action may appear: a missing
    record, a disabled/other-action/foreign row, or any extra record is
    inconsistent. Each item must carry exactly the six visible fields plus
    the boolean ``matched`` scope flag with its captured value, and the
    item sequence must follow the captured ordering.
    """
    read = declaration_basis["read"]
    items = declaration_basis["declarations"]
    if suspended:
        return read is False and items == []
    if read is not True:
        return False
    if len(items) != len(expected_items):
        return False
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            *_DECLARATION_FIELDS,
            "matched",
        } or not isinstance(item.get("matched"), bool):
            return False
    # Ordered, field-for-field equality catches missing, extra, damaged,
    # foreign/disabled/other-action, wrongly-flagged, and mis-ordered items.
    return items == expected_items


def _rules_existing_at(session: Session, captured_at: str) -> list[dict[str, Any]]:
    """All global policy rules that existed at the capture moment.

    Rules are loaded with the same raw-text loader the capture used
    (:func:`policy_preview.load_rules`), so a row with a damaged stored
    field is passed through verbatim and surfaces as an ``invalid``
    candidate exactly as it did at capture time — never coerced or dropped
    by typed column access. The preview builder itself filters to the event
    action. A rule inserted after the event never participated in its
    judgement and must never make the historical snapshot inconsistent, so
    existence is judged by the actual UTC instant of ``created_at`` rather
    than text order (an unparseable stamp sorts after every finite instant
    and so never enters the historical candidate set).
    """
    capture_instant = _created_instant(captured_at)
    return [
        row
        for row in policy_preview.load_rules(session)
        if _created_instant(row.get("created_at")) <= capture_instant
    ]


def _policy_section_unread(policy_candidates: dict[str, Any]) -> bool:
    """Whether the policy section is the fixed empty shape for no read.

    A suspended machine and an active machine whose enabled declarations
    never match the resource never consult policy rules, so the captured
    section must record no read and no candidates, winners, or conflicts.
    """
    return (
        policy_candidates["read"] is False
        and policy_candidates["candidates"] == []
        and policy_candidates["winners"] == []
        and policy_candidates["conflicts"] == []
    )


def _policy_outcome(
    policy_candidates: dict[str, Any],
    *,
    action_type: str,
    resource: str,
    stored_rules: list[dict[str, Any]],
) -> tuple[bool, str] | None:
    """Validate the read policy section and return the judgement it entails.

    Called only when the status basis records that policy was read. The
    recorded section must equal the preview the established priority
    semantics produce for the action/resource — candidate order and
    relations (winner/overridden/conflict/unmatched, plus invalid for a
    damaged row), winner references, and conflict pairs. It must then be
    internally coherent (conflict groups and winners never coexist) and its
    structure must entail one definite judgement:

    * a conflict group denies (``denied_by_policy``) with no winner;
    * a denying winner denies, an all-allow winner allows;
    * no winners and no conflicts means no matching rule
      (``no_matching_policy``).

    Returns ``None`` for any structural or ordering inconsistency.
    """
    if policy_candidates["read"] is not True:
        return None

    action_rules = [
        row for row in stored_rules if row.get("action_type") == action_type
    ]
    preview = policy_preview.build_preview(action_type, resource, action_rules)

    if policy_candidates["candidates"] != preview["rules"]:
        return None
    if policy_candidates["winners"] != preview["winners"]:
        return None
    if policy_candidates["conflicts"] != preview["conflicts"]:
        return None

    winners = preview["winners"]
    conflicts = preview["conflicts"]
    relations = [candidate.get("relation") for candidate in preview["rules"]]

    if conflicts:
        # A mixed minimum tier is recorded as conflict pairs with no winner
        # and is decided as a denial.
        if winners:
            return None
        if not any(relation == "conflict" for relation in relations):
            return None
        return False, "denied_by_policy"

    if winners:
        if any(relation == "conflict" for relation in relations):
            return None
        # Overridden rules may accompany a winner; a deny at the decisive
        # tier denies, an all-allow tier allows.
        if any(winner.get("effect") == "deny" for winner in winners):
            return False, "denied_by_policy"
        if all(winner.get("effect") == "allow" for winner in winners):
            return True, "allowed_by_policy"
        return None

    # No winner and no conflict: either no rule matched or a damaged tier
    # left only overridden rules (which would be contradictory).
    if any(
        relation in ("winner", "overridden", "conflict") for relation in relations
    ):
        return None
    return False, "no_matching_policy"
