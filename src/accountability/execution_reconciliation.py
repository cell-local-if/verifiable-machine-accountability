"""Read-only reconciliation of grants, lifecycle events, uses, and receipts.

On top of the grant/use/lifecycle reconciliation (:mod:`.grant_reconciliation`)
and the execution receipts (:mod:`.execution_receipts`), this module
cross-checks — for one machine, strictly read-only — the full life of every
grant: the grant row itself, its immutable lifecycle events, its single use
record, and the execution-completion receipt that must close a consumed
grant. On top of the grant-side invariants of the plainer reconciliation
(every event binding to its grant, the event sequence matching the stored
status, the use record matching a consumed grant), a ``consumed`` grant must
carry exactly one receipt bound to the same use, the same grant, and the
same source allow decision, with the receipt's action and resource matching
the source event verbatim and its ``outcome``, ``result_digest``, and
``content_hash`` satisfying the established receipt format; a grant of any
other status must carry neither a use nor a receipt.

Grants created before the lifecycle-event feature have no events at all:
they are counted as historical (old-database compatibility) and are never
judged, repaired, or given fabricated events, uses, or receipts. The query
never inserts, updates, deletes, recomputes, or rewrites any record, so
repeated reads of unchanged data agree byte-for-byte and survive restarts,
and only rows owned by the path machine are examined — another machine's
records, damaged or not, never change the outcome.

The conclusion is ``{valid, checked_grant_count, historical_grant_count,
completed_count, broken_grant_id, broken_record_id, anomaly}``.
``completed_count`` tallies only the consumed grants that reconcile without
any anomaly. When anything is inconsistent, the reported grant is the first
problem grant ordered by the actual UTC instant of ``issued_at`` and then by
grant id; a grant whose ``issued_at`` no longer parses cannot be ordered and
is reported first, with the anomaly ``timestamp_unparseable``.
``broken_record_id`` names the one stored record that evidences the anomaly
— the record carrying the unparseable moment, the lifecycle event whose
reference or sequence position breaks, the use record, or the receipt —
falling back to the grant id itself when the anomaly is the *absence* of a
record (a missing terminal event, a missing use). A lifecycle event, use,
or receipt whose ``grant_id`` names no grant of the machine is an orphan and
is located by that stored ``grant_id``. The anomaly is one of
``timestamp_unparseable`` (a moment no longer parses),
``grant_binding_mismatch`` (an event's decision-event reference or moment
does not bind to its grant), ``grant_state_mismatch`` (the event sequence
or the stored grant state is impossible), ``use_mismatch`` (the use records
do not match the grant's terminal state), ``receipt_missing`` (a consumed
grant with its one use carries no receipt), ``receipt_mismatch`` (a receipt
is not bound to the same use, grant, and source allow decision, or its
action or resource disagrees with the source event, or a receipt exists
that the grant's state forbids), or ``receipt_content_invalid`` (the
receipt's ``outcome``, ``result_digest``, or ``content_hash`` does not
satisfy the established receipt format). Within one grant the checks run in
that order — moments, event binding, sequence and state, the use, then the
receipt — with the moment binding of each event verified against the grant
right after its sequence is confirmed, and the receipt's content format
checked only after its binding is confirmed.
"""

from typing import Any

from sqlalchemy import select

from .db import (
    AuthorizationDecisionEvent,
    AuthorizationGrant,
    AuthorizationGrantLifecycleEvent,
    AuthorizationGrantUse,
    ExecutionReceipt,
)
from .execution_receipts import (
    _CONTENT_COLUMNS,
    _OUTCOMES,
    _is_lower_hex_64,
    compute_content_hash,
)
from .grant_reconciliation import (
    _FAR_PAST,
    _REQUIRED_SEQUENCE,
    _event_order_key,
    _id_key,
    _parse_moment,
    _tolerant_instant,
)

TIMESTAMP_UNPARSEABLE = "timestamp_unparseable"
GRANT_BINDING_MISMATCH = "grant_binding_mismatch"
GRANT_STATE_MISMATCH = "grant_state_mismatch"
USE_MISMATCH = "use_mismatch"
RECEIPT_MISSING = "receipt_missing"
RECEIPT_MISMATCH = "receipt_mismatch"
RECEIPT_CONTENT_INVALID = "receipt_content_invalid"

# Check precedence: a moment that cannot be parsed is reported before an
# event-binding break, which is reported before an impossible sequence or
# state, which is reported before a use inconsistency, which is reported
# before a missing, misbound, or content-invalid receipt.
_ANOMALY_PRIORITY = {
    TIMESTAMP_UNPARSEABLE: 0,
    GRANT_BINDING_MISMATCH: 1,
    GRANT_STATE_MISMATCH: 2,
    USE_MISMATCH: 3,
    RECEIPT_MISSING: 4,
    RECEIPT_MISMATCH: 5,
    RECEIPT_CONTENT_INVALID: 6,
}


def _receipt_order_key(receipt: ExecutionReceipt) -> tuple:
    """Order receipts by occurred-at instant, then receipt id."""
    return (
        _tolerant_instant(receipt.occurred_at),
        _id_key(receipt.id),
    )


def _first_use_id(uses: list[AuthorizationGrantUse]) -> Any:
    """The deterministically first use id, tolerant of a damaged value."""
    return sorted(uses, key=lambda use: _id_key(use.id))[0].id


def _grant_anomaly(
    grant: AuthorizationGrant,
    events: list[AuthorizationGrantLifecycleEvent],
    uses: list[AuthorizationGrantUse],
    receipts_by_use: dict[Any, list[ExecutionReceipt]],
    receipts_by_grant: dict[Any, list[ExecutionReceipt]],
    decision_events: dict[Any, AuthorizationDecisionEvent],
) -> tuple[str, Any] | None:
    """First anomaly of one grant that has at least one lifecycle event.

    The checks run in a fixed order — moments parse, event reference
    binding, event sequence and stored state, the moment binding of the
    verified sequence, the use record, then the receipt — and the first
    failure decides the grant's anomaly and the record that evidences it;
    ``None`` means the grant, its events, its use, and its receipt are
    fully consistent.
    """
    # Moments: every stamp the reconciliation relies on must still parse to
    # a UTC instant — the grant's issue moment, its terminal moment when the
    # stored status is terminal, and every event's occurrence moment.
    if _parse_moment(grant.issued_at) is None:
        return (TIMESTAMP_UNPARSEABLE, grant.id)
    if grant.status == "consumed" and _parse_moment(grant.consumed_at) is None:
        return (TIMESTAMP_UNPARSEABLE, grant.id)
    if grant.status == "revoked" and _parse_moment(grant.revoked_at) is None:
        return (TIMESTAMP_UNPARSEABLE, grant.id)
    for event in events:
        if _parse_moment(event.occurred_at) is None:
            return (TIMESTAMP_UNPARSEABLE, event.id)

    # Event reference binding: every event must name the grant's own
    # decision event. (The event's ``grant_id`` binds by construction:
    # events are grouped under the grant they name, and a ``grant_id``
    # naming no grant of the machine is reported as an orphan.)
    for event in events:
        if event.authorization_event_id != grant.event_id:
            return (GRANT_BINDING_MISMATCH, event.id)

    # Sequence and state: ordered by (occurred-at instant, id), the events
    # must be exactly the unique first ``issued`` plus the one terminal
    # event the stored status demands. The evidence record is the first
    # event whose position breaks the required sequence, or the grant
    # itself when an expected event is absent or the status is unknown.
    required = _REQUIRED_SEQUENCE.get(grant.status)
    ordered_events = sorted(events, key=_event_order_key)
    actual = [event.type for event in ordered_events]
    if required is None:
        return (GRANT_STATE_MISMATCH, grant.id)
    if actual != required:
        record_id: Any = grant.id
        for index, event in enumerate(ordered_events):
            if index >= len(required) or event.type != required[index]:
                record_id = event.id
                break
        return (GRANT_STATE_MISMATCH, record_id)

    # Moment binding of the verified sequence: each event's occurrence
    # moment must be the exact stored moment the grant carries for that
    # action — the issue moment for ``issued``, the terminal moment for the
    # one terminal event.
    for event in events:
        if event.type == "issued" and event.occurred_at != grant.issued_at:
            return (GRANT_BINDING_MISMATCH, event.id)
        if event.type == "consumed" and event.occurred_at != grant.consumed_at:
            return (GRANT_BINDING_MISMATCH, event.id)
        if event.type == "revoked" and event.occurred_at != grant.revoked_at:
            return (GRANT_BINDING_MISMATCH, event.id)

    # Use record: exactly one use of the same ownership, decision event,
    # and consumption moment for a consumed grant; none otherwise.
    if grant.status == "consumed":
        if len(uses) != 1:
            return (USE_MISMATCH, _first_use_id(uses) if uses else grant.id)
        use = uses[0]
        if _parse_moment(use.consumed_at) is None:
            return (TIMESTAMP_UNPARSEABLE, use.id)
        if (
            use.machine_id != grant.machine_id
            or use.event_id != grant.event_id
            or use.consumed_at != grant.consumed_at
        ):
            return (USE_MISMATCH, use.id)
    elif uses:
        return (USE_MISMATCH, _first_use_id(uses))

    # Receipt: a consumed grant carries exactly one receipt bound to its
    # one use, its own id, and its source allow decision, with the action
    # and resource of the source event and a sound content format; a grant
    # of any other status carries no receipt at all.
    if grant.status == "consumed":
        use = uses[0]
        found: dict[Any, ExecutionReceipt] = {}
        for receipt in receipts_by_use.get(use.id, []):
            found.setdefault(receipt.id, receipt)
        for receipt in receipts_by_grant.get(grant.id, []):
            found.setdefault(receipt.id, receipt)
        ordered_receipts = sorted(found.values(), key=_receipt_order_key)
        if not ordered_receipts:
            return (RECEIPT_MISSING, use.id)
        if len(ordered_receipts) > 1:
            return (RECEIPT_MISMATCH, ordered_receipts[0].id)
        receipt = ordered_receipts[0]
        if _parse_moment(receipt.occurred_at) is None:
            return (TIMESTAMP_UNPARSEABLE, receipt.id)
        event = decision_events.get(grant.event_id)
        if (
            receipt.use_id != use.id
            or receipt.grant_id != grant.id
            or receipt.authorization_event_id != grant.event_id
            or event is None
            or not event.allowed
            or event.reason != "allowed_by_policy"
            or receipt.action_type != event.action_type
            or receipt.resource != event.resource
        ):
            return (RECEIPT_MISMATCH, receipt.id)
        if receipt.outcome not in _OUTCOMES or not _is_lower_hex_64(
            receipt.result_digest
        ):
            return (RECEIPT_CONTENT_INVALID, receipt.id)
        try:
            expected_content_hash = compute_content_hash(
                **{key: getattr(receipt, key) for key in _CONTENT_COLUMNS}
            )
        except TypeError:
            # A damaged (non-text) content field cannot produce the
            # published digest.
            return (RECEIPT_CONTENT_INVALID, receipt.id)
        if (
            not _is_lower_hex_64(receipt.content_hash)
            or receipt.content_hash != expected_content_hash
        ):
            return (RECEIPT_CONTENT_INVALID, receipt.id)
    else:
        stray = receipts_by_grant.get(grant.id, [])
        if stray:
            return (
                RECEIPT_MISMATCH,
                sorted(stray, key=_receipt_order_key)[0].id,
            )

    return None


def reconcile(session, machine_id: str) -> dict[str, Any]:
    """Read-only execution reconciliation conclusion for one machine.

    Reads only rows owned by the path machine — its grants, its lifecycle
    events, its use records, its receipts, and its decision events — and
    never writes, repairs, recomputes, or fabricates anything. Grants with
    no lifecycle event are historical (old databases predate the feature):
    they are counted, never judged, and never given fabricated events,
    uses, or receipts. Returns the fixed-shape conclusion ``{valid,
    checked_grant_count, historical_grant_count, completed_count,
    broken_grant_id, broken_record_id, anomaly}``; ``completed_count``
    tallies only the consumed grants that reconcile without any anomaly,
    and when every checked grant is consistent ``valid`` is ``True`` with
    ``broken_grant_id``, ``broken_record_id``, and ``anomaly`` all ``None``.
    """
    grants = list(
        session.scalars(
            select(AuthorizationGrant).where(
                AuthorizationGrant.machine_id == machine_id
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
    uses = list(
        session.scalars(
            select(AuthorizationGrantUse).where(
                AuthorizationGrantUse.machine_id == machine_id
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
    receipts_by_use: dict[Any, list[ExecutionReceipt]] = {}
    receipts_by_grant: dict[Any, list[ExecutionReceipt]] = {}
    for receipt in receipts:
        receipts_by_use.setdefault(receipt.use_id, []).append(receipt)
        receipts_by_grant.setdefault(receipt.grant_id, []).append(receipt)

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
    # (tier 2).
    candidates: list[tuple[tuple, Any, Any, str]] = []
    for grant in grants:
        grant_events = events_by_grant.get(grant.id, [])
        if not grant_events:
            # Historical grant from before the lifecycle feature: counted,
            # never judged, and never given fabricated records.
            historical_grant_count += 1
            continue
        anomaly = _grant_anomaly(
            grant,
            grant_events,
            uses_by_grant.get(grant.id, []),
            receipts_by_use,
            receipts_by_grant,
            decision_events,
        )
        if anomaly is None:
            if grant.status == "consumed":
                completed_count += 1
            continue
        category, record_id = anomaly
        instant = _parse_moment(grant.issued_at)
        if instant is None:
            key = (0, _FAR_PAST, _id_key(grant.id), _ANOMALY_PRIORITY[category])
        else:
            key = (1, instant, _id_key(grant.id), _ANOMALY_PRIORITY[category])
        candidates.append((key, grant.id, record_id, category))

    # Orphan records: a lifecycle event, use, or receipt whose stored
    # ``grant_id`` names no grant of this machine. The dangling reference
    # is located by that ``grant_id``; an orphan event is a binding break,
    # an orphan use a use inconsistency, and an orphan receipt a receipt
    # that no grant of the machine can account for. A receipt is only an
    # orphan when neither its grant nor its use belongs to the machine —
    # otherwise the grant it references by use or by id reports it.
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
                ),
                orphan_grant_id,
                record_id,
                GRANT_BINDING_MISMATCH,
            )
        )
    for orphan_grant_id, orphan_uses in uses_by_grant.items():
        if orphan_grant_id in grant_ids:
            continue
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(orphan_grant_id),
                    _ANOMALY_PRIORITY[USE_MISMATCH],
                ),
                orphan_grant_id,
                _first_use_id(orphan_uses),
                USE_MISMATCH,
            )
        )
    for receipt in receipts:
        if receipt.grant_id in grant_ids or receipt.use_id in use_ids:
            continue
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(receipt.grant_id),
                    _ANOMALY_PRIORITY[RECEIPT_MISMATCH],
                ),
                receipt.grant_id,
                receipt.id,
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
