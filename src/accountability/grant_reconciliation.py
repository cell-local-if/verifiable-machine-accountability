"""Read-only reconciliation of grants, their uses, and lifecycle events.

On top of the grant lifecycle (:mod:`.grants`, :mod:`.grant_lifecycle`) and
the execution receipts, this module cross-checks — for one machine, strictly
read-only — the three record families that a grant's life produces:

* every successful grant action appends exactly one immutable lifecycle
  event, so a grant that has any lifecycle event at all must begin with a
  unique, first ``issued`` event;
* an ``active`` grant has no terminal (``consumed``/``revoked``) event and
  no use record;
* a ``consumed`` grant has exactly one ``consumed`` event and exactly one
  use record of the same ownership (machine) and the same moment;
* a ``revoked`` grant has exactly one ``revoked`` event and no use record;
* every lifecycle event's ``authorization_event_id``, ``grant_id``, and
  terminal moment must agree with the grant it belongs to.

Grants created before the lifecycle-event feature have no events at all:
they are counted as historical (old-database compatibility) and are never
judged, repaired, or given fabricated events. The query never inserts,
updates, deletes, recomputes, or rewrites any record, so repeated reads of
unchanged data agree and survive restarts, and only rows owned by the path
machine are examined — another machine's records, damaged or not, never
change the outcome.

The conclusion is ``{valid, checked_grant_count, historical_grant_count,
broken_grant_id, anomaly}``. When anything is inconsistent, the reported
grant is the first problem grant ordered by the actual UTC instant of
``issued_at`` and then by grant id; a grant whose ``issued_at`` no longer
parses cannot be ordered and is reported first, with the anomaly
``timestamp_unparseable``. A lifecycle event or use record whose
``grant_id`` names no grant of the machine is an orphan and is located by
that stored ``grant_id``. The anomaly is one of
``timestamp_unparseable`` (a moment no longer parses),
``reference_mismatch`` (an event's decision-event reference or moment does
not bind to its grant, or the grant reference dangles),
``sequence_or_state_mismatch`` (the event sequence or the stored grant
state is impossible), or ``use_mismatch`` (the use record does not match
the grant's terminal state). Within one grant the checks run in that
order — moments, reference binding, sequence and state, then the use —
with the moment binding of each event verified against the grant right
after its sequence is confirmed.
"""

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from .db import (
    AuthorizationGrant,
    AuthorizationGrantLifecycleEvent,
    AuthorizationGrantUse,
)

TIMESTAMP_UNPARSEABLE = "timestamp_unparseable"
REFERENCE_MISMATCH = "reference_mismatch"
SEQUENCE_OR_STATE_MISMATCH = "sequence_or_state_mismatch"
USE_MISMATCH = "use_mismatch"

# Check precedence: a moment that cannot be parsed is reported before a
# binding break, which is reported before an impossible sequence or state,
# which is reported before a use inconsistency.
_ANOMALY_PRIORITY = {
    TIMESTAMP_UNPARSEABLE: 0,
    REFERENCE_MISMATCH: 1,
    SEQUENCE_OR_STATE_MISMATCH: 2,
    USE_MISMATCH: 3,
}

# A stored moment only means something under the RFC 3339 ``Z`` contract
# every writer of these tables commits to; fractional seconds are optional
# and offset forms or a missing suffix never parse.
_UTC_Z_STAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)

# Sentinels keep damaged values orderable without ever raising: an
# unparseable event stamp sorts after every parseable one inside its grant,
# and candidates without an orderable instant sort by tier and id alone.
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


def _grant_anomaly(
    grant: AuthorizationGrant,
    events: list[AuthorizationGrantLifecycleEvent],
    uses: list[AuthorizationGrantUse],
) -> str | None:
    """First anomaly of one grant that has at least one lifecycle event.

    The checks run in a fixed order — moments parse, decision-event
    reference binding, event sequence and stored state, the moments of the
    verified sequence binding to the grant, then the use record — and the
    first failure decides the grant's anomaly; ``None`` means the grant,
    its events, and its use record are fully consistent.
    """
    # Moments: every stamp the reconciliation relies on must still parse to
    # a UTC instant — the grant's issue moment, its terminal moment when the
    # stored status is terminal, and every event's occurrence moment.
    if _parse_moment(grant.issued_at) is None:
        return TIMESTAMP_UNPARSEABLE
    if grant.status == "consumed" and _parse_moment(grant.consumed_at) is None:
        return TIMESTAMP_UNPARSEABLE
    if grant.status == "revoked" and _parse_moment(grant.revoked_at) is None:
        return TIMESTAMP_UNPARSEABLE
    for event in events:
        if _parse_moment(event.occurred_at) is None:
            return TIMESTAMP_UNPARSEABLE

    # Reference binding: every event must name the grant's own decision
    # event. (The event's ``grant_id`` binds by construction: events are
    # grouped under the grant they name, and a ``grant_id`` naming no grant
    # of the machine is reported as an orphan.)
    for event in events:
        if event.authorization_event_id != grant.event_id:
            return REFERENCE_MISMATCH

    # Sequence and state: ordered by (occurred-at instant, id), the events
    # must be exactly the unique first ``issued`` plus the one terminal
    # event the stored status demands — an unknown status, a missing or
    # duplicated issued, an unknown type, a terminal event on an active
    # grant, or a missing terminal event on a terminal grant all break it.
    required = _REQUIRED_SEQUENCE.get(grant.status)
    actual = [event.type for event in sorted(events, key=_event_order_key)]
    if required is None or actual != required:
        return SEQUENCE_OR_STATE_MISMATCH

    # Moment binding of the verified sequence: each event's occurrence
    # moment must be the exact stored moment the grant carries for that
    # action — the issue moment for ``issued``, the terminal moment for the
    # one terminal event.
    for event in events:
        if event.type == "issued" and event.occurred_at != grant.issued_at:
            return REFERENCE_MISMATCH
        if event.type == "consumed" and event.occurred_at != grant.consumed_at:
            return REFERENCE_MISMATCH
        if event.type == "revoked" and event.occurred_at != grant.revoked_at:
            return REFERENCE_MISMATCH

    # Use record: exactly one use of the same ownership, decision event,
    # and consumption moment for a consumed grant; none otherwise.
    if grant.status == "consumed":
        if len(uses) != 1:
            return USE_MISMATCH
        use = uses[0]
        if (
            use.machine_id != grant.machine_id
            or use.event_id != grant.event_id
            or use.consumed_at != grant.consumed_at
        ):
            return USE_MISMATCH
    elif uses:
        return USE_MISMATCH

    return None


def reconcile(session, machine_id: str) -> dict[str, Any]:
    """Read-only reconciliation conclusion for one machine's grants.

    Reads only rows owned by the path machine — its grants, its use
    records, and its lifecycle events — and never writes, repairs,
    recomputes, or fabricates anything. Grants with no lifecycle event are
    historical (old databases predate the feature): they are counted, never
    judged, and never given events. Returns the fixed-shape conclusion
    ``{valid, checked_grant_count, historical_grant_count,
    broken_grant_id, anomaly}``; when every checked grant is consistent,
    ``valid`` is ``True`` and both ``broken_grant_id`` and ``anomaly`` are
    ``None``.
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

    events_by_grant: dict[Any, list[AuthorizationGrantLifecycleEvent]] = {}
    for event in events:
        events_by_grant.setdefault(event.grant_id, []).append(event)
    uses_by_grant: dict[Any, list[AuthorizationGrantUse]] = {}
    for use in uses:
        uses_by_grant.setdefault(use.grant_id, []).append(use)

    grant_ids = {grant.id for grant in grants}
    historical_grant_count = 0
    # Each candidate: (sort key, broken grant id, anomaly). The sort key is
    # tiered: a grant whose ``issued_at`` no longer parses cannot be ordered
    # and is reported first (tier 0); other problem grants order by their
    # actual ``issued_at`` UTC instant and then grant id (tier 1); orphan
    # events and uses have no grant to order by and follow, located by
    # their stored ``grant_id`` (tier 2).
    candidates: list[tuple[tuple, Any, str]] = []
    for grant in grants:
        grant_events = events_by_grant.get(grant.id, [])
        if not grant_events:
            # Historical grant from before the lifecycle feature: counted,
            # never judged, and never given fabricated events.
            historical_grant_count += 1
            continue
        anomaly = _grant_anomaly(
            grant, grant_events, uses_by_grant.get(grant.id, [])
        )
        if anomaly is None:
            continue
        instant = _parse_moment(grant.issued_at)
        if instant is None:
            key = (0, _FAR_PAST, _id_key(grant.id), _ANOMALY_PRIORITY[anomaly])
        else:
            key = (1, instant, _id_key(grant.id), _ANOMALY_PRIORITY[anomaly])
        candidates.append((key, grant.id, anomaly))

    # Orphan records: a lifecycle event or use whose stored ``grant_id``
    # names no grant of this machine. The dangling reference is located by
    # that ``grant_id``; an orphan event is a reference break, an orphan
    # use a use inconsistency.
    for orphan_grant_id in events_by_grant:
        if orphan_grant_id in grant_ids:
            continue
        candidates.append(
            (
                (
                    2,
                    _FAR_PAST,
                    _id_key(orphan_grant_id),
                    _ANOMALY_PRIORITY[REFERENCE_MISMATCH],
                ),
                orphan_grant_id,
                REFERENCE_MISMATCH,
            )
        )
    for orphan_grant_id in uses_by_grant:
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
                USE_MISMATCH,
            )
        )

    broken_grant_id: Any = None
    anomaly: str | None = None
    if candidates:
        candidates.sort(key=lambda candidate: candidate[0])
        _, broken_grant_id, anomaly = candidates[0]

    return {
        "valid": not candidates,
        "checked_grant_count": len(grants),
        "historical_grant_count": historical_grant_count,
        "broken_grant_id": broken_grant_id,
        "anomaly": anomaly,
    }
