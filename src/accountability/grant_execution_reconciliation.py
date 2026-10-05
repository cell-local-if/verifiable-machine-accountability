"""Read-only reconciliation of grants, their uses, lifecycle events, and receipts.

On top of the grant/use/lifecycle reconciliation
(:mod:`.grant_reconciliation`) and the execution receipts
(:mod:`.execution_receipts`), this module cross-checks — for one machine,
strictly read-only — the full life of every non-historical grant:

* a grant that has any lifecycle event at all must begin with a unique,
  first ``issued`` event, and every event's ``authorization_event_id`` and
  occurrence moment must bind to the grant it belongs to;
* an ``active`` grant has no terminal event, no use record, and no
  receipt; a ``revoked`` grant has exactly one ``revoked`` event and
  neither a use nor a receipt;
* a ``consumed`` grant has exactly one ``consumed`` event, exactly one use
  record, and exactly one execution receipt bound to that same use, that
  same grant, and the same source allow decision, with the receipt's
  ``action_type`` and ``resource`` matching the source event verbatim;
* a receipt's ``outcome``, ``result_digest``, and ``content_hash`` must
  satisfy the established receipt format — a legal outcome, a 64
  lowercase-hex result fingerprint, and a content hash equal to the
  digest of the ten content fields as stored.

Grants created before the lifecycle-event feature have no events at all:
they are counted as historical (old-database compatibility) and are never
judged, repaired, or given fabricated events or receipts. The query never
inserts, updates, deletes, recomputes, or rewrites any record, so repeated
reads of unchanged data agree byte-for-byte and survive restarts, and only
rows owned by the path machine are examined — another machine's records,
damaged or not, never change the outcome.

The conclusion is ``{valid, checked_grant_count, historical_grant_count,
completed_count, broken_grant_id, broken_record_id, anomaly}``.
``completed_count`` counts only the consumed grants that reconcile with no
anomaly at all; historical grants are counted in the first two counts and
never judged, so they never complete. When anything is inconsistent, the
reported grant is the first problem grant ordered by the actual UTC
instant of ``issued_at`` and then by grant id; a grant whose ``issued_at``
no longer parses cannot be ordered and is reported first, with the anomaly
``timestamp_unparseable``. A lifecycle event, use record, or receipt whose
stored ``grant_id`` names no grant of the machine is an orphan and is
located by that stored ``grant_id``. The anomaly is one of
``timestamp_unparseable`` (a moment no longer parses),
``grant_binding_mismatch`` (a lifecycle event's decision-event reference
or moment does not bind to its grant, or an event's grant reference
dangles), ``grant_state_mismatch`` (the event sequence or the stored grant
state is impossible), ``use_mismatch`` (the use count does not match the
grant's terminal state), ``receipt_missing`` (a consumed grant's use
carries no receipt), ``receipt_mismatch`` (a receipt is not bound to the
same use, grant, and source allow decision, or its action or resource
disagrees, or a receipt exists where none may), or
``receipt_content_invalid`` (the outcome, result fingerprint, or content
hash violates the established receipt format). Within one grant the checks
run in that order, and ``broken_record_id`` names the concrete record the
first failure was found on — the grant, a lifecycle event, a use, or a
receipt.
"""

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .db import (
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantLifecycleEvent,
    AuthorizationGrantUse,
    ExecutionReceipt,
)
from .execution_receipts import compute_content_hash

TIMESTAMP_UNPARSEABLE = "timestamp_unparseable"
GRANT_BINDING_MISMATCH = "grant_binding_mismatch"
GRANT_STATE_MISMATCH = "grant_state_mismatch"
USE_MISMATCH = "use_mismatch"
RECEIPT_MISSING = "receipt_missing"
RECEIPT_MISMATCH = "receipt_mismatch"
RECEIPT_CONTENT_INVALID = "receipt_content_invalid"

# Check precedence inside one grant: a moment that cannot be parsed is
# reported before a binding break, which is reported before an impossible
# sequence or state, which is reported before a use-count inconsistency,
# which is reported before a missing, mismatched, or content-invalid
# receipt.
_ANOMALY_PRIORITY = {
    TIMESTAMP_UNPARSEABLE: 0,
    GRANT_BINDING_MISMATCH: 1,
    GRANT_STATE_MISMATCH: 2,
    USE_MISMATCH: 3,
    RECEIPT_MISSING: 4,
    RECEIPT_MISMATCH: 5,
    RECEIPT_CONTENT_INVALID: 6,
}

# A stored moment only means something under the RFC 3339 ``Z`` contract
# every writer of these tables commits to; fractional seconds are optional
# and offset forms or a missing suffix never parse.
_UTC_Z_STAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

# The established receipt format: a legal outcome, a 64 lowercase-hex
# result fingerprint, and a 64 lowercase-hex content hash equal to the
# digest of the ten content fields as stored.
_LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OUTCOMES = ("succeeded", "failed")

# Sentinels keep damaged values orderable without ever raising: an
# unparseable stamp sorts after every parseable one inside its group, and
# candidates without an orderable instant sort by tier and id alone.
_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)
_FAR_PAST = datetime.min.replace(tzinfo=timezone.utc)

# The exact event-type sequence each stored grant status demands, in
# (occurred-at instant, id) order: a unique first ``issued``, then at most
# the one terminal event matching the terminal status.
_REQUIRED_SEQUENCE = {
    "active": ["issued"],
    "consumed": ["issued", "consumed"],
    "revoked": ["issued", "revoked"],
}


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


def _tolerant_instant(value: object) -> datetime:
    """Ordering instant for a stored stamp; damaged text sorts last."""
    instant = _parse_moment(value)
    return instant if instant is not None else _FAR_FUTURE


def _id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored identifier, tolerant of a damaged value."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def _event_order_key(event: AuthorizationGrantLifecycleEvent) -> tuple:
    """Order one grant's events by occurred-at instant, then event id."""
    return (
        _tolerant_instant(event.occurred_at),
        _id_key(event.id),
    )


def _receipt_order_key(receipt: ExecutionReceipt) -> tuple:
    """Order receipts by occurred-at instant, then receipt id."""
    return (
        _tolerant_instant(receipt.occurred_at),
        _id_key(receipt.id),
    )


def _is_lower_hex_64(value: object) -> bool:
    return isinstance(value, str) and _LOWER_HEX_64.fullmatch(value) is not None


def _receipt_content_valid(receipt: ExecutionReceipt) -> bool:
    """Whether a receipt's result fields and content hash are well-formed.

    The established receipt format demands a verbatim
    ``succeeded``/``failed`` outcome, a 64 lowercase-hex result
    fingerprint, and a stored content hash equal to the digest of the ten
    content fields exactly as stored. A damaged value is reported, never
    repaired, recomputed for storage, or folded into validity.
    """
    if receipt.outcome not in _OUTCOMES:
        return False
    if not _is_lower_hex_64(receipt.result_digest):
        return False
    if not _is_lower_hex_64(receipt.content_hash):
        return False
    try:
        expected = compute_content_hash(
            id=receipt.id,
            machine_id=receipt.machine_id,
            use_id=receipt.use_id,
            grant_id=receipt.grant_id,
            authorization_event_id=receipt.authorization_event_id,
            action_type=receipt.action_type,
            resource=receipt.resource,
            outcome=receipt.outcome,
            result_digest=receipt.result_digest,
            occurred_at=receipt.occurred_at,
        )
    except TypeError:
        # A damaged (non-text) content field cannot produce the digest.
        return False
    return receipt.content_hash == expected


def _grant_anomaly(
    grant: AuthorizationGrant,
    events: list[AuthorizationGrantLifecycleEvent],
    uses: list[AuthorizationGrantUse],
    receipts: list[ExecutionReceipt],
    decision_events: dict[Any, AuthorizationDecisionEvent],
) -> tuple[str, Any] | None:
    """First anomaly of one grant that has at least one lifecycle event.

    The checks run in a fixed order — moments parse, lifecycle reference
    binding, event sequence and stored state, the moments of the verified
    sequence binding to the grant, the use count, then the receipt's
    presence, binding, and content — and the first failure decides the
    grant's anomaly and the concrete record it was found on; ``None``
    means the grant, its events, its use, and its receipt are fully
    consistent.
    """
    # Moments: every stamp the reconciliation relies on must still parse to
    # a UTC instant — the grant's issue moment, its terminal moment when the
    # stored status is terminal, and every event's occurrence moment.
    if _parse_moment(grant.issued_at) is None:
        return TIMESTAMP_UNPARSEABLE, grant.id
    if grant.status == "consumed" and _parse_moment(grant.consumed_at) is None:
        return TIMESTAMP_UNPARSEABLE, grant.id
    if grant.status == "revoked" and _parse_moment(grant.revoked_at) is None:
        return TIMESTAMP_UNPARSEABLE, grant.id
    for event in events:
        if _parse_moment(event.occurred_at) is None:
            return TIMESTAMP_UNPARSEABLE, event.id

    # Lifecycle reference binding: every event must name the grant's own
    # decision event. (The event's ``grant_id`` binds by construction:
    # events are grouped under the grant they name, and a ``grant_id``
    # naming no grant of the machine is reported as an orphan.)
    for event in events:
        if event.authorization_event_id != grant.event_id:
            return GRANT_BINDING_MISMATCH, event.id

    # Sequence and state: ordered by (occurred-at instant, id), the events
    # must be exactly the unique first ``issued`` plus the one terminal
    # event the stored status demands — an unknown status, a missing or
    # duplicated issued, an unknown type, a terminal event on an active
    # grant, or a missing terminal event on a terminal grant all break it.
    required = _REQUIRED_SEQUENCE.get(grant.status)
    actual = [event.type for event in sorted(events, key=_event_order_key)]
    if required is None or actual != required:
        return GRANT_STATE_MISMATCH, grant.id

    # Moment binding of the verified sequence: each event's occurrence
    # moment must be the exact stored moment the grant carries for that
    # action — the issue moment for ``issued``, the terminal moment for the
    # one terminal event.
    for event in events:
        if event.type == "issued" and event.occurred_at != grant.issued_at:
            return GRANT_BINDING_MISMATCH, event.id
        if event.type == "consumed" and event.occurred_at != grant.consumed_at:
            return GRANT_BINDING_MISMATCH, event.id
        if event.type == "revoked" and event.occurred_at != grant.revoked_at:
            return GRANT_BINDING_MISMATCH, event.id

    # Use count: exactly one use for a consumed grant; none otherwise.
    ordered_uses = sorted(uses, key=lambda use: _id_key(use.id))
    if grant.status == "consumed":
        if len(uses) != 1:
            return USE_MISMATCH, (
                ordered_uses[0].id if ordered_uses else grant.id
            )
    elif uses:
        return USE_MISMATCH, ordered_uses[0].id

    # Receipt: a consumed grant carries exactly one receipt bound to its
    # use, itself, and the source allow decision, with well-formed content;
    # every other state carries none.
    if grant.status == "consumed":
        use = uses[0]
        candidates = sorted(receipts, key=_receipt_order_key)
        if not candidates:
            return RECEIPT_MISSING, use.id
        if len(candidates) != 1:
            return RECEIPT_MISMATCH, candidates[0].id
        receipt = candidates[0]
        source = decision_events.get(grant.event_id)
        if (
            receipt.use_id != use.id
            or receipt.grant_id != grant.id
            or receipt.authorization_event_id != grant.event_id
            or source is None
            or not source.allowed
            or source.reason != "allowed_by_policy"
            or receipt.action_type != source.action_type
            or receipt.resource != source.resource
        ):
            return RECEIPT_MISMATCH, receipt.id
        if not _receipt_content_valid(receipt):
            return RECEIPT_CONTENT_INVALID, receipt.id
    elif receipts:
        stray = sorted(receipts, key=_receipt_order_key)[0]
        return RECEIPT_MISMATCH, stray.id

    return None


def reconcile(session, machine_id: str) -> dict[str, Any]:
    """Read-only execution reconciliation conclusion for one machine.

    Reads only rows owned by the path machine — its grants, its use
    records, its lifecycle events, its receipts, and its decision events —
    and never writes, repairs, recomputes, or fabricates anything. Grants
    with no lifecycle event are historical (old databases predate the
    feature): they are counted, never judged, and never given events or
    receipts. Returns the fixed-shape conclusion ``{valid,
    checked_grant_count, historical_grant_count, completed_count,
    broken_grant_id, broken_record_id, anomaly}``; when every checked grant
    reconciles, ``valid`` is ``True`` and ``broken_grant_id``,
    ``broken_record_id``, and ``anomaly`` are all ``None``.
    """
    grants = list(
        session.scalars(
            select(AuthorizationGrant).where(
                AuthorizationGrant.machine_id == machine_id
            )
        ).all()
    )
    uses = list(
        session.scalars(
            select(AuthorizationGrantUse).where(
                AuthorizationGrantUse.machine_id == machine_id
            )
        ).all()
    )
    events = list(
        session.scalars(
            select(AuthorizationGrantLifecycleEvent).where(
                AuthorizationGrantLifecycleEvent.machine_id == machine_id
            )
        ).all()
    )
    receipts = list(
        session.scalars(
            select(ExecutionReceipt).where(
                ExecutionReceipt.machine_id == machine_id
            )
        ).all()
    )
    decision_events = {
        event.id: event
        for event in session.scalars(
            select(AuthorizationDecisionEvent).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    }

    events_by_grant: dict[Any, list[AuthorizationGrantLifecycleEvent]] = {}
    for event in events:
        events_by_grant.setdefault(event.grant_id, []).append(event)
    uses_by_grant: dict[Any, list[AuthorizationGrantUse]] = {}
    for use in uses:
        uses_by_grant.setdefault(use.grant_id, []).append(use)
    receipts_by_grant: dict[Any, list[ExecutionReceipt]] = {}
    receipts_by_use: dict[Any, list[ExecutionReceipt]] = {}
    for receipt in receipts:
        receipts_by_grant.setdefault(receipt.grant_id, []).append(receipt)
        receipts_by_use.setdefault(receipt.use_id, []).append(receipt)

    grant_ids = {grant.id for grant in grants}
    use_ids = {use.id for use in uses}
    historical_grant_count = 0
    completed_count = 0
    # Each candidate: (sort key, broken grant id, broken record id,
    # anomaly). The sort key is tiered: a grant whose ``issued_at`` no
    # longer parses cannot be ordered and is reported first (tier 0);
    # other problem grants order by their actual ``issued_at`` UTC instant
    # and then grant id (tier 1); orphan events, uses, and receipts have no
    # grant to order by and follow, located by their stored ``grant_id``
    # (tier 2). The record id breaks any remaining tie deterministically.
    candidates: list[tuple[tuple, Any, Any, str]] = []
    for grant in grants:
        grant_events = events_by_grant.get(grant.id, [])
        if not grant_events:
            # Historical grant from before the lifecycle feature: counted,
            # never judged, and never given fabricated events or receipts.
            historical_grant_count += 1
            continue
        grant_uses = uses_by_grant.get(grant.id, [])
        # The receipt set of one grant: every receipt naming the grant,
        # plus every receipt naming one of its uses — a cross-wired
        # receipt (right use, wrong grant, or vice versa) is a mismatch of
        # this grant, not another grant's missing receipt.
        grant_receipts: dict[Any, ExecutionReceipt] = {}
        for receipt in receipts_by_grant.get(grant.id, []):
            grant_receipts.setdefault(receipt.id, receipt)
        for use in grant_uses:
            for receipt in receipts_by_use.get(use.id, []):
                grant_receipts.setdefault(receipt.id, receipt)
        result = _grant_anomaly(
            grant,
            grant_events,
            grant_uses,
            list(grant_receipts.values()),
            decision_events,
        )
        if result is None:
            if grant.status == "consumed":
                # Only an anomaly-free consumed grant completes.
                completed_count += 1
            continue
        anomaly, record_id = result
        instant = _parse_moment(grant.issued_at)
        if instant is None:
            key = (
                0,
                _FAR_PAST,
                _id_key(grant.id),
                _ANOMALY_PRIORITY[anomaly],
                _id_key(record_id),
            )
        else:
            key = (
                1,
                instant,
                _id_key(grant.id),
                _ANOMALY_PRIORITY[anomaly],
                _id_key(record_id),
            )
        candidates.append((key, grant.id, record_id, anomaly))

    # Orphan records: a lifecycle event, use, or receipt whose stored
    # ``grant_id`` names no grant of this machine (a receipt that still
    # names one of the machine's uses is judged with that use's grant
    # instead). The dangling reference is located by that ``grant_id``; an
    # orphan event is a binding break, an orphan use a use-count
    # inconsistency, an orphan receipt a receipt mismatch.
    for orphan_grant_id, orphan_events in events_by_grant.items():
        if orphan_grant_id in grant_ids:
            continue
        record_id = sorted(orphan_events, key=_event_order_key)[0].id
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(orphan_grant_id),
                    _ANOMALY_PRIORITY[GRANT_BINDING_MISMATCH],
                    _id_key(record_id),
                ),
                orphan_grant_id,
                record_id,
                GRANT_BINDING_MISMATCH,
            )
        )
    for orphan_grant_id, orphan_uses in uses_by_grant.items():
        if orphan_grant_id in grant_ids:
            continue
        record_id = sorted(orphan_uses, key=lambda use: _id_key(use.id))[0].id
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(orphan_grant_id),
                    _ANOMALY_PRIORITY[USE_MISMATCH],
                    _id_key(record_id),
                ),
                orphan_grant_id,
                record_id,
                USE_MISMATCH,
            )
        )
    orphan_receipts: dict[Any, list[ExecutionReceipt]] = {}
    for receipt in receipts:
        if receipt.grant_id in grant_ids or receipt.use_id in use_ids:
            continue
        orphan_receipts.setdefault(receipt.grant_id, []).append(receipt)
    for orphan_grant_id, group in orphan_receipts.items():
        record_id = sorted(group, key=_receipt_order_key)[0].id
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(orphan_grant_id),
                    _ANOMALY_PRIORITY[RECEIPT_MISMATCH],
                    _id_key(record_id),
                ),
                orphan_grant_id,
                record_id,
                RECEIPT_MISMATCH,
            )
        )

    broken_grant_id: Any = None
    broken_record_id: Any = None
    anomaly: str | None = None
    if candidates:
        candidates.sort(key=lambda candidate: candidate[0])
        _, broken_grant_id, broken_record_id, anomaly = candidates[0]

    return {
        "valid": not candidates,
        "checked_grant_count": len(grants),
        "historical_grant_count": historical_grant_count,
        "completed_count": completed_count,
        "broken_grant_id": broken_grant_id,
        "broken_record_id": broken_record_id,
        "anomaly": anomaly,
    }
