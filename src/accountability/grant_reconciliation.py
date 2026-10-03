"""Read-only reconciliation of grants, their uses, and lifecycle events.

:func:`reconcile_machine_grants` is the independent, strictly read-only
cross-check of one machine's authorization grants against the single
consumption record and the immutable grant-lifecycle audit events. For every
grant of the path machine that carries lifecycle events it verifies, without
ever writing, repairing, recomputing, or rewriting a record:

* the event sequence starts with a unique ``issued`` event whose
  ``occurred_at`` is the grant's own ``issued_at``;
* an ``active`` grant carries no terminal (``consumed``/``revoked``) event
  and no use record;
* a ``consumed`` grant carries exactly one ``consumed`` event whose
  ``occurred_at`` is the grant's ``consumed_at``, plus exactly one use
  record of the same machine ownership and the same consumption moment;
* a ``revoked`` grant carries exactly one ``revoked`` event whose
  ``occurred_at`` is the grant's ``revoked_at``, and no use record;
* every event's ``authorization_event_id`` and ``grant_id`` bind to the
  grant they are filed under.

Grants with no lifecycle event at all are historical rows from databases
that predate the lifecycle feature: they are counted in
``historical_grant_count`` for old-database compatibility, never judged and
never backfilled. Damaged stored values are reported, never crashed on,
repaired, rewritten, or recomputed for storage, and only the path machine's
rows are ever read, so another machine's damaged records never change the
conclusion and repeated reads of unchanged data stay byte-identical across
restarts.
"""

from datetime import datetime, timezone
from typing import Any

from .db import (
    AuthorizationGrant,
    AuthorizationGrantLifecycleEvent,
    AuthorizationGrantUse,
)

_GRANT_TABLE = AuthorizationGrant.__table__
_USE_TABLE = AuthorizationGrantUse.__table__
_EVENT_TABLE = AuthorizationGrantLifecycleEvent.__table__

_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)

_KNOWN_TYPES = ("issued", "consumed", "revoked")
_TERMINAL_TYPES = ("consumed", "revoked")


def _parse_instant(value: object) -> datetime | None:
    """Parse a stored UTC stamp to its actual instant, or ``None``.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract;
    a missing, non-text, or malformed value returns ``None`` so the
    read-only reconciliation reports the record instead of crashing on,
    repairing, or rewriting it.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            pass
    return None


def _instant_or_far_future(value: object) -> datetime:
    """Ordering instant for a stored stamp; damaged values sort last."""
    instant = _parse_instant(value)
    return instant if instant is not None else _FAR_FUTURE


def _id_key(value: object) -> tuple[int, str]:
    """Tie-break key for a stored identifier, tolerant of a damaged value."""
    if isinstance(value, str):
        return (0, value)
    return (1, "")


def _event_order(events: list[Any]) -> list[Any]:
    """Order one grant's events by (occurred-at instant, id)."""
    return sorted(
        events,
        key=lambda event: (
            _instant_or_far_future(event["occurred_at"]),
            _id_key(event["id"]),
        ),
    )


def _grant_anomaly(grant: Any, events: list[Any], uses: list[Any]) -> str | None:
    """First anomaly category for one grant, or ``None`` when consistent.

    The categories are checked in the fixed order ``reference_mismatch``
    (an event's decision-event or grant binding does not name this grant),
    ``sequence_or_state_mismatch`` (the event sequence or the grant state
    does not match the lifecycle contract, including a terminal or issue
    moment that disagrees with the grant's own stamp), and
    ``use_mismatch`` (the consumption record does not match the state).
    Timestamp parseability is judged by the caller before this scan.
    """
    # Reference binding: every event filed under this grant must name the
    # grant's own decision event (the grant id already matches by grouping).
    for event in events:
        if event["authorization_event_id"] != grant["event_id"]:
            return "reference_mismatch"

    ordered = _event_order(events)
    issued = [event for event in ordered if event["type"] == "issued"]
    # The sequence must open with the grant's unique issued event, stamped
    # with the grant's own issue moment.
    if len(issued) != 1 or ordered[0]["type"] != "issued":
        return "sequence_or_state_mismatch"
    if issued[0]["occurred_at"] != grant["issued_at"]:
        return "sequence_or_state_mismatch"
    if any(event["type"] not in _KNOWN_TYPES for event in ordered):
        return "sequence_or_state_mismatch"
    terminal = [event for event in ordered if event["type"] in _TERMINAL_TYPES]

    status = grant["status"]
    if status == "active":
        # An active grant carries no terminal event and no use.
        if terminal:
            return "sequence_or_state_mismatch"
    elif status == "consumed":
        # Exactly one terminal event: the single consumption, stamped with
        # the grant's own consumed_at.
        if len(terminal) != 1 or terminal[0]["type"] != "consumed":
            return "sequence_or_state_mismatch"
        if terminal[0]["occurred_at"] != grant["consumed_at"]:
            return "sequence_or_state_mismatch"
    elif status == "revoked":
        # Exactly one terminal event: the one revocation, stamped with the
        # grant's own revoked_at.
        if len(terminal) != 1 or terminal[0]["type"] != "revoked":
            return "sequence_or_state_mismatch"
        if terminal[0]["occurred_at"] != grant["revoked_at"]:
            return "sequence_or_state_mismatch"
    else:
        return "sequence_or_state_mismatch"

    # The use record must match the terminal state: exactly one use of the
    # same ownership and consumption moment for a consumed grant, and no
    # use at all otherwise.
    if status == "consumed":
        if len(uses) != 1:
            return "use_mismatch"
        use = uses[0]
        if (
            use["machine_id"] != grant["machine_id"]
            or use["consumed_at"] != grant["consumed_at"]
        ):
            return "use_mismatch"
    elif uses:
        return "use_mismatch"
    return None


def reconcile_machine_grants(session, machine_id: str) -> dict[str, Any]:
    """Read-only reconciliation of one machine's grants, uses, and events.

    Returns ``{valid, checked_grant_count, historical_grant_count,
    broken_grant_id, anomaly}``. ``checked_grant_count`` is the machine's
    total grant count and ``historical_grant_count`` the number of grants
    carrying no lifecycle event (historical rows are counted for
    old-database compatibility, never judged and never backfilled). Every
    other grant is checked against its events and its use record; the first
    problem grant — ordered by the actual UTC instant of ``issued_at`` and
    then by grant id — sets ``broken_grant_id`` and ``anomaly``. A grant
    whose own ``issued_at`` or required terminal stamp no longer parses
    cannot be ordered and is reported first of all as
    ``timestamp_unparseable``; an orphan lifecycle event whose ``grant_id``
    names no grant of the path machine is located by that ``grant_id`` as a
    ``reference_mismatch``. When everything is consistent ``valid`` is
    ``true`` and both ``broken_grant_id`` and ``anomaly`` are ``None``.

    Only rows owned by the path machine are read, and the query never
    inserts, updates, deletes, backfills, repairs, recomputes, or
    normalizes a grant, a use, or a lifecycle event, so repeated reads of
    unchanged data return identical conclusions and the data survives
    restarts untouched.
    """
    grant_rows = list(
        session.execute(
            _GRANT_TABLE.select().where(_GRANT_TABLE.c.machine_id == machine_id)
        )
    )
    use_rows = list(
        session.execute(
            _USE_TABLE.select().where(_USE_TABLE.c.machine_id == machine_id)
        )
    )
    event_rows = list(
        session.execute(
            _EVENT_TABLE.select().where(_EVENT_TABLE.c.machine_id == machine_id)
        )
    )

    checked_grant_count = len(grant_rows)

    uses_by_grant: dict[Any, list[Any]] = {}
    for row in use_rows:
        uses_by_grant.setdefault(row._mapping["grant_id"], []).append(
            row._mapping
        )
    events_by_grant: dict[Any, list[Any]] = {}
    for row in event_rows:
        events_by_grant.setdefault(row._mapping["grant_id"], []).append(
            row._mapping
        )

    grant_ids: set[Any] = set()
    historical_grant_count = 0
    # Grants whose own stamps no longer parse cannot be ordered by the
    # issued_at instant; they are reported before every other anomaly.
    timestamp_broken: list[Any] = []
    # (issued-at instant, id tie-break, grant id, anomaly) per problem.
    problems: list[tuple[datetime, tuple[int, str], Any, str]] = []

    for row in grant_rows:
        grant = row._mapping
        grant_id = grant["id"]
        grant_ids.add(grant_id)
        events = events_by_grant.get(grant_id, [])
        if not events:
            # A grant with no lifecycle event is a historical row from a
            # database that predates the feature: counted, never judged,
            # and no events are fabricated for it.
            historical_grant_count += 1
            continue

        issued_instant = _parse_instant(grant["issued_at"])
        terminal_stamp = None
        if grant["status"] == "consumed":
            terminal_stamp = grant["consumed_at"]
        elif grant["status"] == "revoked":
            terminal_stamp = grant["revoked_at"]
        if issued_instant is None or (
            grant["status"] in _TERMINAL_TYPES
            and _parse_instant(terminal_stamp) is None
        ):
            timestamp_broken.append(grant_id)
            continue

        anomaly = _grant_anomaly(
            grant, events, uses_by_grant.get(grant_id, [])
        )
        if anomaly is not None:
            problems.append(
                (issued_instant, _id_key(grant_id), grant_id, anomaly)
            )

    def _conclusion(
        valid: bool, broken_grant_id: Any, anomaly: str | None
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "checked_grant_count": checked_grant_count,
            "historical_grant_count": historical_grant_count,
            "broken_grant_id": broken_grant_id,
            "anomaly": anomaly,
        }

    # An unparseable grant stamp is reported before every other anomaly:
    # the problem ordering itself rests on the issued_at instant.
    if timestamp_broken:
        return _conclusion(
            False,
            sorted(timestamp_broken, key=_id_key)[0],
            "timestamp_unparseable",
        )

    # Orphan events name a grant the path machine does not own; they are
    # located by their stored grant_id and sort after every grant problem,
    # since no issued_at instant exists to order them by.
    for grant_id in events_by_grant:
        if grant_id not in grant_ids:
            problems.append(
                (
                    _FAR_FUTURE,
                    _id_key(grant_id),
                    grant_id,
                    "reference_mismatch",
                )
            )

    if problems:
        problems.sort(key=lambda problem: (problem[0], problem[1]))
        _, _, broken_grant_id, anomaly = problems[0]
        return _conclusion(False, broken_grant_id, anomaly)

    return _conclusion(True, None, None)
