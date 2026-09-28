"""Read-only consistency audit of one authorization decision-basis snapshot.

The baseline ``decision-basis`` query only re-emits the stored snapshot
byte-for-byte; this module independently checks that snapshot against the
committed event record without ever creating, updating, deleting, repairing,
recomputing, or normalizing the event, the snapshot, declarations, rules, or
any other accountability record. The audit issues reads only and is a pure
function of stored data: the same stored rows always produce the same
conclusion, and another machine's snapshot can never enter the checked set
(the read is scoped to the path machine and event).

The conclusion is fixed at four fields:

* ``valid`` — true only when the event summary, status basis, declaration
  basis, policy candidates, and final decision all agree with one another and
  with the committed event;
* ``checked_count`` — ``0`` when the event has no snapshot row, ``1`` when
  one snapshot row exists (even if that snapshot is damaged);
* ``broken_basis_id`` — the path event id whenever the conclusion is false,
  ``None`` on success;
* ``reason`` — the fixed ``snapshot_not_found`` category for a missing row,
  otherwise the stable category of the first anomaly found.

The audit never fabricates a basis: an event without a snapshot row is
``(False, 0, event_id, "snapshot_not_found")`` regardless of the current
declarations or rules. A damaged document is
``(False, 1, event_id, <category>)`` — the first category in the fixed
examination order, so repeated queries are stable. The damaged-content
categories are:

* ``malformed_document`` — the document does not parse, does not parse to an
  object, or has a document-level shape anomaly: a missing, extra, or
  reordered top-level group, a wrong field type on any group (including the
  event summary, status basis, declaration basis, and policy candidates), or
  an anomalous collection shape or group/record ordering. Every structural
  anomaly collapses to this one stable category rather than naming the group;
* ``malformed_decision`` — the structure is complete but the final decision
  field is illegal or contradicts the event's committed result (including a
  decision the rebuilt basis cannot support).

The policy relations are audited against the rules *as they existed when the
event committed*: winner/overridden/conflict/unmatched/invalid are rebuilt
with the capture's own pure preview builder over the rules already created
by then (both declarations and rules are append-only, so a row present then
persists byte-identically now; later rows are filtered out by ``created_at``
and other-action rows exactly as capture filters them). A later rule or
declaration therefore never retroactively breaks a faithful historical
snapshot. The declaration basis is checked against the machine's enabled
declarations for the action that already existed at the event instant: the
stored set is only allowed to name the declarations participating in this
action/resource judgement, so a missing, disabled, other-action, or
superfluous record is an inconsistency.

The status basis is audited against the machine's status *as of the event*,
not its current row. The audit independently reads the machine's own
``machine_status_events`` transition history and folds it to the capture
instant: only a transition whose actual creation instant is not later than
the capture instant (``created_at <= capture``) forms the state then, equal
instants apply in ascending ``id`` order, and with no qualifying transition
the machine's creation default ``active`` carries forward unchanged. A
snapshot that merely matches the machine's current status while
contradicting this event-time reconstruction is rejected. Every stored
transition of the machine is counted in the judgement, including one
committed after the capture, because the history is one append-only chain
whose edges must continue one another. The fixed status categories are:

* ``status_history_invalid`` — the history is misattributed, illegal, or not
  rebuildable: a non-text/unparseable ``created_at`` (the as-of boundary is
  undecidable), a non-text ``id`` (same-instant order unstable), a status
  outside ``active``/``suspended``, a self edge, or a ``from_status`` that
  does not continue the carried state;
* ``status_state_mismatch`` — the history rebuilds soundly but the rebuilt
  event-time status differs from the snapshot's recorded ``status``;
* ``read_flags_mismatch`` — the state agrees but the recorded
  ``declarations_read``/``policies_read`` flags (and the declaration group's
  own ``read`` flag) contradict the gate in force then: a suspended machine
  reads neither, an active machine reads declarations, and policy is read
  only when at least one participating declaration matched.

These three are examined in that order (history, state, read flags), after
the event summary and before the declaration content, policy relations, and
final decision. A damaged history record is reported as the stable category,
never crashed on, repaired, rewritten, or recomputed; a real failure while
reading the history is a ``500 internal_error`` carrying no conclusion.

The damaged-content categories are deliberately two stable strings instead
of one per group: every document-level anomaly — unparseable JSON, a
missing/extra/reordered top-level group, a wrong field type anywhere, or an
anomalous collection shape or ordering — is ``malformed_document``, while a
structurally complete document whose final decision field is illegal or
contradicts the event's committed result is ``malformed_decision``.
Content-level correspondence checks that are neither structural nor the
final decision (the event summary, the status/declaration/policy content)
keep their own mismatch categories.
"""

from datetime import datetime, timezone
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .authorization import pattern_matches
from . import policy_preview
from .db import BehaviorDeclaration, MachineStatusEvent

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

# The five top-level groups, in the fixed order the document carries them;
# the audit reports the first anomaly in this same order.
_GROUPS = (
    "event_summary",
    "status_basis",
    "declaration_basis",
    "policy_candidates",
    "decision",
)

_SUMMARY_FIELDS = (
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
)
_STATUS_FIELDS = (
    "machine_id",
    "status",
    "captured_at",
    "declarations_read",
    "policies_read",
)
_DECLARATION_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "enabled",
    "created_at",
    "updated_at",
    "matched",
)
_CANDIDATE_FIELDS = (
    "id",
    "action_type",
    "resource_pattern",
    "effect",
    "priority",
    "created_at",
    "updated_at",
    "relation",
)
_REFERENCE_FIELDS = ("id", "effect", "priority", "created_at")
_RELATIONS = frozenset(("winner", "overridden", "conflict", "unmatched", "invalid"))

# Outcome reasons for a false conclusion, kept as stable fixed strings. The
# snapshot-missing case is fixed by contract; every other category names the
# first damaged/contradictory part in examination order. Damaged document
# content collapses to two stable buckets: every structural/ordering anomaly
# is ``malformed_document``, and only a structurally complete document whose
# final decision field is illegal or contradicts the committed result is
# ``malformed_decision``.
SNAPSHOT_NOT_FOUND = "snapshot_not_found"
_MALFORMED_DOCUMENT = "malformed_document"
_MALFORMED_DECISION = "malformed_decision"
_EVENT_SUMMARY_MISMATCH = "event_summary_mismatch"
_STATUS_HISTORY_INVALID = "status_history_invalid"
_STATUS_STATE_MISMATCH = "status_state_mismatch"
_READ_FLAGS_MISMATCH = "read_flags_mismatch"
_STATUS_BASIS_MISMATCH = "status_basis_mismatch"
_DECLARATION_BASIS_MISMATCH = "declaration_basis_mismatch"
_DECLARATION_MATCH_MISMATCH = "declaration_match_mismatch"
_POLICY_CANDIDATE_MISMATCH = "policy_candidate_mismatch"
_POLICY_WINNER_MISMATCH = "policy_winner_mismatch"
_POLICY_CONFLICT_MISMATCH = "policy_conflict_mismatch"

_DECLARATION_TABLE = BehaviorDeclaration.__table__
_HISTORY_TABLE = MachineStatusEvent.__table__


def _created_instant(value: object) -> datetime:
    """Parse a visible ``created_at`` to its UTC instant for order checks.

    Must use the same tolerant convention as the snapshot capture: a damaged
    stamp sorts after every parseable instant instead of raising.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except (ValueError, OverflowError):
            pass
    return _FAR_FUTURE


def _is_int(value: object) -> bool:
    """A JSON integer field; booleans are never accepted as integers."""
    return isinstance(value, int) and not isinstance(value, bool)


def _result(
    valid: bool, checked_count: int, broken_basis_id: str | None, reason: str | None
) -> dict[str, Any]:
    return {
        "valid": valid,
        "checked_count": checked_count,
        "broken_basis_id": broken_basis_id,
        "reason": reason,
    }


def verify(
    session: Session,
    *,
    machine_id: str,
    event: Any,
    document: str | None,
) -> dict[str, Any]:
    """Audit one path-machine event's stored basis document.

    ``event`` is the committed ORM row (already scoped to the path machine by
    the caller); ``document`` is the stored snapshot text or ``None`` when no
    snapshot row exists. Read-only throughout. A real storage read failure
    propagates to the caller, which answers ``500 internal_error``; a damaged
    document content never raises — it is reported as a false conclusion.
    """
    event_id = event.id

    # No snapshot row: never fabricate a basis from current data. The missing
    # row counts as zero checked snapshots even though the event exists.
    if document is None:
        return _result(False, 0, event_id, SNAPSHOT_NOT_FOUND)

    # One stored snapshot exists, so it counts as one whether sound or broken.
    try:
        parsed = json_loads(document)
    except (ValueError, TypeError):
        return _result(False, 1, event_id, _MALFORMED_DOCUMENT)
    if not isinstance(parsed, dict):
        return _result(False, 1, event_id, _MALFORMED_DOCUMENT)

    structural = _structural_problem(parsed)
    if structural is not None:
        return _result(False, 1, event_id, structural)

    summary = parsed["event_summary"]
    status_basis = parsed["status_basis"]
    declaration_basis = parsed["declaration_basis"]
    policy_basis = parsed["policy_candidates"]
    decision = parsed["decision"]

    # --- event summary: verbatim correspondence with the committed event ---
    expected_summary = {
        "id": event.id,
        "machine_id": event.machine_id,
        "action_type": event.action_type,
        "resource": event.resource,
        "allowed": event.allowed,
        "reason": event.reason,
        "created_at": event.created_at,
        "previous_event_id": event.previous_event_id,
        "content_hash": event.content_hash,
        "chain_hash": event.chain_hash,
    }
    if any(summary.get(name) != expected_summary[name] for name in _SUMMARY_FIELDS):
        return _result(False, 1, event_id, _EVENT_SUMMARY_MISMATCH)

    # --- status basis: status value, capture moment, ownership, read flags ---
    status_problem, status, matched_any = _check_status_and_declarations(
        session,
        machine_id=machine_id,
        event=event,
        status_basis=status_basis,
        declaration_basis=declaration_basis,
    )
    if status_problem is not None:
        return _result(False, 1, event_id, status_problem)

    # --- policy candidates: relations self-consistent with recorded data ----
    policy_problem, derived_decision = _check_policy_candidates(
        session,
        event=event,
        status=status,
        matched_any=matched_any,
        status_basis=status_basis,
        policy_basis=policy_basis,
    )
    if policy_problem is not None:
        return _result(False, 1, event_id, policy_problem)

    # --- final decision: identical to the committed event and supported -----
    # The structure is complete and the decision group passed its field-type
    # shape check, so an illegal value here or one that contradicts the
    # committed event / the rebuilt basis is a malformed decision.
    if decision.get("allowed") != event.allowed or decision.get("reason") != event.reason:
        return _result(False, 1, event_id, _MALFORMED_DECISION)
    if status == "suspended":
        supported = (False, "machine_suspended")
    elif not matched_any:
        supported = (False, "no_enabled_declaration")
    else:
        supported = derived_decision
    if (event.allowed, event.reason) != supported:
        return _result(False, 1, event_id, _MALFORMED_DECISION)

    return _result(True, 1, None, None)


def json_loads(text: str) -> Any:
    """Parse stored snapshot text, rejecting non-finite constants.

    The capture serializes with ``allow_nan=False``; a document carrying
    ``NaN``/``Infinity`` is damaged content rather than a parseable value.
    """
    return json.loads(text, parse_constant=_raise_on_constant)


def _raise_on_constant(value: str) -> Any:  # pragma: no cover - raises always
    raise ValueError(f"invalid JSON constant: {value}")


# --- structural shape --------------------------------------------------------


def _structural_problem(document: dict[str, Any]) -> str | None:
    """First structural anomaly across the five groups, or ``None`` when sound.

    Only shape is judged here (groups present, exact key sets, value types,
    collection shapes); cross-group and event correspondence are checked
    afterwards. Every anomaly collapses to the single stable
    ``malformed_document`` category rather than naming the affected group.
    """
    if tuple(document.keys()) != _GROUPS:
        # Covers a missing group, a superfluous group, and wrong group order.
        return _MALFORMED_DOCUMENT

    summary = document["event_summary"]
    summary_problem = _check_object_shape(
        summary,
        _SUMMARY_FIELDS,
        str_fields=(
            "id",
            "machine_id",
            "action_type",
            "resource",
            "reason",
            "created_at",
            "content_hash",
            "chain_hash",
        ),
        bool_fields=("allowed",),
        nullable_str_fields=("previous_event_id",),
    )
    if summary_problem:
        return _MALFORMED_DOCUMENT

    status_problem = _check_object_shape(
        document["status_basis"],
        _STATUS_FIELDS,
        str_fields=("machine_id", "status", "captured_at"),
        bool_fields=("declarations_read", "policies_read"),
    )
    if status_problem:
        return _MALFORMED_DOCUMENT

    declaration_basis = document["declaration_basis"]
    if not isinstance(declaration_basis, dict):
        return _MALFORMED_DOCUMENT
    if tuple(declaration_basis.keys()) != ("read", "declarations"):
        return _MALFORMED_DOCUMENT
    if not isinstance(declaration_basis["read"], bool):
        return _MALFORMED_DOCUMENT
    items = declaration_basis["declarations"]
    if not isinstance(items, list):
        return _MALFORMED_DOCUMENT
    for item in items:
        if _check_object_shape(
            item,
            _DECLARATION_FIELDS,
            str_fields=(
                "id",
                "action_type",
                "resource_pattern",
                "created_at",
                "updated_at",
            ),
            bool_fields=("enabled", "matched"),
        ):
            return _MALFORMED_DOCUMENT

    policy_basis = document["policy_candidates"]
    if not isinstance(policy_basis, dict):
        return _MALFORMED_DOCUMENT
    if tuple(policy_basis.keys()) != ("read", "candidates", "winners", "conflicts"):
        return _MALFORMED_DOCUMENT
    if not isinstance(policy_basis["read"], bool):
        return _MALFORMED_DOCUMENT
    if not isinstance(policy_basis["candidates"], list):
        return _MALFORMED_DOCUMENT
    for candidate in policy_basis["candidates"]:
        problem = _candidate_shape_problem(candidate)
        if problem is not None:
            return problem
    if not isinstance(policy_basis["winners"], list):
        return _MALFORMED_DOCUMENT
    for winner in policy_basis["winners"]:
        if _check_object_shape(
            winner,
            _REFERENCE_FIELDS,
            str_fields=("id", "effect", "created_at"),
            int_fields=("priority",),
        ):
            return _MALFORMED_DOCUMENT
    conflicts = policy_basis["conflicts"]
    if not isinstance(conflicts, list) or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(side, str) for side in pair)
        for pair in conflicts
    ):
        return _MALFORMED_DOCUMENT

    if _check_object_shape(
        document["decision"],
        ("allowed", "reason"),
        str_fields=("reason",),
        bool_fields=("allowed",),
    ):
        return _MALFORMED_DOCUMENT
    return None


def _check_object_shape(
    value: Any,
    fields: tuple[str, ...],
    *,
    str_fields: tuple[str, ...] = (),
    bool_fields: tuple[str, ...] = (),
    int_fields: tuple[str, ...] = (),
    nullable_str_fields: tuple[str, ...] = (),
) -> bool:
    """Whether ``value`` is an object with exactly ``fields`` and value types."""
    if not isinstance(value, dict) or tuple(value.keys()) != fields:
        return True
    for name in str_fields:
        if not isinstance(value[name], str):
            return True
    for name in bool_fields:
        if not isinstance(value[name], bool):
            return True
    for name in int_fields:
        if not _is_int(value[name]):
            return True
    for name in nullable_str_fields:
        if value[name] is not None and not isinstance(value[name], str):
            return True
    return False


def _candidate_shape_problem(candidate: Any) -> str | None:
    """Shape of one policy candidate.

    A candidate marked ``invalid`` may carry damaged stored field values (it
    records that the row could not participate), so its seven rule fields are
    only type-checked for the four usable relations. The relation marker
    itself and the exact eight-key set must always be sound.
    """
    if not isinstance(candidate, dict) or tuple(candidate.keys()) != _CANDIDATE_FIELDS:
        return _MALFORMED_DOCUMENT
    relation = candidate["relation"]
    if relation not in _RELATIONS:
        return _MALFORMED_DOCUMENT
    if relation == "invalid":
        return None
    if not all(
        isinstance(candidate[name], str)
        for name in ("id", "action_type", "resource_pattern", "effect", "created_at", "updated_at")
    ):
        return _MALFORMED_DOCUMENT
    if not _is_int(candidate["priority"]):
        return _MALFORMED_DOCUMENT
    return None


# --- status and declaration basis -------------------------------------------


def _enabled_action_declarations(
    session: Session, machine_id: str, action_type: str
) -> list[dict[str, Any]]:
    """Visible fields of the machine's enabled declarations for the action.

    Must define the same participating set as the snapshot capture: enabled
    rows for this machine and action only; disabled and other-action rows
    never participate. Read-only, values passed through verbatim.
    """
    rows = session.execute(
        select(
            *[_DECLARATION_TABLE.c[name] for name in (
                "id",
                "action_type",
                "resource_pattern",
                "enabled",
                "created_at",
                "updated_at",
            )]
        ).where(
            _DECLARATION_TABLE.c.machine_id == machine_id,
            _DECLARATION_TABLE.c.action_type == action_type,
            _DECLARATION_TABLE.c.enabled.is_(True),
        )
    ).all()
    return [dict(row._mapping) for row in rows]


def _declarations_existing_at(
    rows: list[dict[str, Any]], event_instant: datetime
) -> list[dict[str, Any]]:
    """The participating declarations already created when the event committed.

    Declarations are append-only, but new ones may have been created after
    the event; such rows did not participate in its judgement and must not be
    required of the historical snapshot. Comparison uses the actual UTC
    instant (a damaged stamp sorts far-future, so it is treated as later and
    cannot be confused with a row the decision saw).
    """
    return [
        row
        for row in rows
        if _created_instant(row.get("created_at")) <= event_instant
    ]


def _load_machine_status_history(
    session: Session, machine_id: str
) -> list[dict[str, Any]]:
    """Read the path machine's own status-transition history, unordered.

    This is the audit's independent read of ``machine_status_events``: it
    never consults the machine's current ``status`` column (the snapshot
    could agree with today's state while contradicting the event instant).
    Read-only and scoped to the path machine, so another machine's
    transitions never enter. A real storage failure propagates to the
    caller (``500 internal_error``); damaged row *content* never raises —
    the reconstruction classifies it.
    """
    rows = session.execute(
        select(
            _HISTORY_TABLE.c.id,
            _HISTORY_TABLE.c.from_status,
            _HISTORY_TABLE.c.to_status,
            _HISTORY_TABLE.c.created_at,
        ).where(_HISTORY_TABLE.c.machine_id == machine_id)
    ).all()
    return [dict(row._mapping) for row in rows]


def _reconstruct_status_at(
    rows: list[dict[str, Any]], capture_instant: datetime
) -> tuple[str | None, str | None]:
    """Rebuild the machine status in effect at the snapshot capture instant.

    Independent fold over the machine's own transition history: only a
    transition whose actual creation instant is not later than the capture
    instant forms the state then (``created_at <= capture``), same-instant
    transitions apply in ascending ``id`` order, and with no qualifying
    transition the machine's creation default ``active`` carries forward
    unchanged. This reconstructs the state *as of the event*, never the
    current machine row.

    Returns ``(status, None)`` on success or ``(None, reason)`` when the
    history is misattributed, illegal, or not rebuildable
    (``status_history_invalid``). Every stored record is counted in the
    judgement, even one committed after the capture: the history is one
    append-only transition chain whose edges must continue one another, so a
    damaged record cannot hide behind the as-of boundary. A non-string stamp
    or a stamp that no longer parses makes ordering and the as-of prefix
    boundary undecidable; a non-string id prevents stable same-instant
    ordering; a status outside the ``active``/``suspended`` pair, a self
    edge, or a ``from_status`` that does not continue the carried state is a
    misattributed/illegal transition. Such a record is reported, never
    crashed on, repaired, rewritten, or recomputed.

    The as-of state itself is formed only by transitions whose instant is not
    later than the capture instant; later transitions are validated for chain
    legality but do not change the state returned for the event.
    """
    # An unparseable (or non-string) stamp anywhere leaves ordering and the
    # record's place relative to the capture instant undecidable: the tolerant
    # far-future convention could push an actually earlier suspension past the
    # boundary, which would risk accepting a state that contradicts the event.
    # The history therefore cannot be rebuilt soundly.
    for row in rows:
        stamp = row.get("created_at")
        if not isinstance(stamp, str) or _created_instant(stamp) == _FAR_FUTURE:
            return None, _STATUS_HISTORY_INVALID

    # Actual UTC instant first, then id ascending; a damaged non-string id
    # sorts as empty for the ordering pass and is rejected below.
    ordered = sorted(
        rows,
        key=lambda row: (
            _created_instant(row.get("created_at")),
            row.get("id") if isinstance(row.get("id"), str) else "",
        ),
    )

    # A machine is created active and each accepted transition alternates the
    # carried state; the as-of state starts at that active default and moves
    # only on a not-later transition. ``carried`` validates the whole append-
    # only chain; ``state_at_capture`` is the state for the event.
    carried = "active"
    state_at_capture = "active"
    for row in ordered:
        row_id = row.get("id")
        from_status = row.get("from_status")
        to_status = row.get("to_status")
        if not isinstance(row_id, str):
            return None, _STATUS_HISTORY_INVALID
        if from_status not in ("active", "suspended") or to_status not in (
            "active",
            "suspended",
        ):
            return None, _STATUS_HISTORY_INVALID
        if from_status == to_status:
            # A self edge is never written by the status-change entry.
            return None, _STATUS_HISTORY_INVALID
        if from_status != carried:
            # Misattributed/illegal edge: it does not continue the state the
            # earlier transitions establish (e.g. a second active->suspended
            # record once the machine is already suspended).
            return None, _STATUS_HISTORY_INVALID
        carried = to_status
        if _created_instant(row.get("created_at")) <= capture_instant:
            state_at_capture = to_status
    return state_at_capture, None


def _check_status_and_declarations(
    session: Session,
    *,
    machine_id: str,
    event: Any,
    status_basis: dict[str, Any],
    declaration_basis: dict[str, Any],
) -> tuple[str | None, str, bool]:
    """Validate status and declaration basis; return (problem, status, matched).

    The status basis is judged in the fixed order history, state, then read
    flags:

    1. the snapshot-carried facts (status value, ownership, capture moment);
    2. the machine's own transition history, independently rebuilt as of the
       capture instant — ``status_history_invalid`` when misattributed,
       illegal, or not rebuildable;
    3. the rebuilt event-time state against the recorded status —
       ``status_state_mismatch`` when they differ;
    4. the read flags against the gate in force then — a suspended machine
       reads neither declarations nor policy, an active machine reads
       declarations and reads policy only once a declaration matched —
       ``read_flags_mismatch`` on any disagreement.

    ``matched`` says whether at least one participating enabled declaration
    matched the event resource — the gate for reading policy rules. A problem
    category ends the audit; on success ``problem`` is ``None``.
    """
    status = status_basis["status"]
    if status not in ("active", "suspended"):
        return _STATUS_BASIS_MISMATCH, status, False
    if status_basis["machine_id"] != machine_id:
        return _STATUS_BASIS_MISMATCH, status, False
    if status_basis["captured_at"] != event.created_at:
        return _STATUS_BASIS_MISMATCH, status, False

    # Rebuild the status in effect when the snapshot was taken from the
    # transition history — never from today's machine row, which could match
    # the snapshot while contradicting the event instant.
    history_rows = _load_machine_status_history(session, machine_id)
    rebuilt, history_problem = _reconstruct_status_at(
        history_rows, _created_instant(event.created_at)
    )
    if history_problem is not None:
        return history_problem, status, False
    if rebuilt != status:
        return _STATUS_STATE_MISMATCH, status, False

    declarations_read = status == "active"
    policies_read = status_basis["policies_read"]
    if status_basis["declarations_read"] is not declarations_read:
        return _READ_FLAGS_MISMATCH, status, False
    if declaration_basis["read"] is not declarations_read:
        return _READ_FLAGS_MISMATCH, status, False

    if status == "suspended":
        # A suspended machine reads neither declarations nor policy: both
        # groups must explicitly record no read and an empty collection.
        if policies_read is not False:
            return _READ_FLAGS_MISMATCH, status, False
        if declaration_basis["declarations"]:
            return _DECLARATION_BASIS_MISMATCH, status, False
        return None, status, False

    # Active: derive the declaration gate independently from the stored
    # participating declarations as of the event, then judge the policy read
    # flag against that gate before auditing the recorded declaration items,
    # keeping every read-flag check ahead of declaration content.
    stored = _enabled_action_declarations(session, machine_id, event.action_type)
    event_instant = _created_instant(event.created_at)
    existing = _declarations_existing_at(stored, event_instant)
    expected_order = sorted(
        existing,
        key=lambda row: (_created_instant(row.get("created_at")), row.get("id")),
    )
    expected_items: list[dict[str, Any]] = []
    matched_any = False
    for row in expected_order:
        pattern = row.get("resource_pattern")
        matched = isinstance(pattern, str) and pattern_matches(pattern, event.resource)
        matched_any = matched_any or matched
        expected_items.append(
            {
                "id": row.get("id"),
                "action_type": row.get("action_type"),
                "resource_pattern": row.get("resource_pattern"),
                "enabled": row.get("enabled"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "matched": matched,
            }
        )

    # policies_read must equal the declaration gate the decision used: policy
    # is read only when at least one participating declaration matched.
    if policies_read is not matched_any:
        return _READ_FLAGS_MISMATCH, status, False

    # Active: the recorded set must name exactly the participating enabled
    # declarations — no missing, disabled, other-action, duplicated, or
    # superfluous row. The six stored identity/value fields are compared first
    # (ignoring the derived ``matched`` flag), so a tampered match flag on an
    # otherwise exact record keeps its own ``declaration_match_mismatch``
    # category instead of collapsing into the set mismatch.
    items = declaration_basis["declarations"]
    identity_fields = (
        "id",
        "action_type",
        "resource_pattern",
        "enabled",
        "created_at",
        "updated_at",
    )
    if len({item["id"] for item in items}) != len(items):
        return _DECLARATION_BASIS_MISMATCH, status, False
    actual_identity = sorted(
        _canonical({field: item[field] for field in identity_fields})
        for item in items
    )
    expected_identity = sorted(
        _canonical({field: item[field] for field in identity_fields})
        for item in expected_items
    )
    if actual_identity != expected_identity:
        return _DECLARATION_BASIS_MISMATCH, status, False
    actual_by_id = {item["id"]: item for item in items}
    if any(
        actual_by_id[item["id"]]["matched"] != item["matched"]
        for item in expected_items
    ):
        return _DECLARATION_MATCH_MISMATCH, status, False
    expected_keys = [
        (_created_instant(item["created_at"]), item["id"]) for item in expected_items
    ]
    actual_keys = [(_created_instant(item["created_at"]), item["id"]) for item in items]
    if actual_keys != expected_keys:
        return _MALFORMED_DOCUMENT, status, False

    return None, status, matched_any


# --- policy candidates -------------------------------------------------------


def _canonical(value: Any) -> str:
    """Canonical JSON text of one candidate/reference for set/order compares."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _check_policy_candidates(
    session: Session,
    *,
    event: Any,
    status: str,
    matched_any: bool,
    status_basis: dict[str, Any],
    policy_basis: dict[str, Any],
) -> tuple[str | None, tuple[bool, str] | None]:
    """Validate the candidate set, relations, winners, and conflict groups.

    The expected basis is reconstructed with the *same pure builder* the
    capture used (:func:`policy_preview.build_preview`) over the global rules
    that already existed when the event committed (rules are append-only, so a
    row present then is byte-identical now; later rules are filtered out by
    ``created_at`` and rows for other actions are filtered exactly as capture
    filters them). This independently reproduces the winner/overridden/
    conflict/unmatched/invalid relations, the winner group, the conflict
    pairs, and the priority-first ordering, without ever consulting today's
    decision state or recomputing the stored event. Returns the decision the
    candidates support (``None`` when policy was not read).
    """
    read = policy_basis["read"]
    if read is not status_basis["policies_read"]:
        return _POLICY_CANDIDATE_MISMATCH, None

    candidates = policy_basis["candidates"]
    winners = policy_basis["winners"]
    conflicts = policy_basis["conflicts"]

    if read is False:
        if candidates or winners or conflicts:
            return _POLICY_CANDIDATE_MISMATCH, None
        return None, None

    # Policy is read only by an active machine whose declaration gate passed.
    if status != "active" or not matched_any:
        return _POLICY_CANDIDATE_MISMATCH, None

    event_instant = _created_instant(event.created_at)
    stored_rules = [
        row
        for row in policy_preview.load_rules(session)
        if _created_instant(row.get("created_at")) <= event_instant
        and row.get("action_type") == event.action_type
    ]
    preview = policy_preview.build_preview(
        event.action_type, event.resource, stored_rules
    )
    expected_candidates = preview["rules"]
    expected_winners = preview["winners"]
    expected_conflicts = preview["conflicts"]

    # Content first (missing/extra/superfluous candidate, a wrong stored
    # field, or a wrong winner/overridden/conflict/unmatched/invalid
    # relation), order second so an ordering-only anomaly keeps its own
    # category even when the recorded set is otherwise complete.
    if sorted(_canonical(candidate) for candidate in candidates) != sorted(
        _canonical(candidate) for candidate in expected_candidates
    ):
        return _POLICY_CANDIDATE_MISMATCH, None
    if [_canonical(candidate) for candidate in candidates] != [
        _canonical(candidate) for candidate in expected_candidates
    ]:
        return _MALFORMED_DOCUMENT, None
    if winners != expected_winners:
        return _POLICY_WINNER_MISMATCH, None
    if conflicts != expected_conflicts:
        return _POLICY_CONFLICT_MISMATCH, None

    preview_decision = preview["decision"]
    return None, (preview_decision["allowed"], preview_decision["reason"])
