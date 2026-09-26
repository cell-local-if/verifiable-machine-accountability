"""Authorization decision computation.

The decision rule is the single source of truth for the non-persisting
evaluation endpoints (single and batch) and the decision-event write path:

* a ``suspended`` machine is denied with ``machine_suspended`` before any
  behavior declaration or policy rule is consulted;
* an active machine needs at least one enabled behavior declaration whose
  resource pattern matches, otherwise ``no_enabled_declaration``;
* at least one matching policy rule is required, otherwise
  ``no_matching_policy``;
* among the matching rules the lowest priority decides; a deny at that
  priority wins over same-priority allows (``denied_by_policy``), otherwise
  the request is ``allowed_by_policy``.

The functions deliberately take a SQLAlchemy execution (``Session`` or
``Connection``) so the same logic can run on a plain read connection or
inside the locked write transaction that appends a decision event. That
shared execution is what lets the event path observe machine status,
declarations, and rules together with the chain append in one indivisible
write: the status read and the decision are never based on a state older
than the transaction the event commits in.
"""

import re

from sqlalchemy import Connection
from sqlalchemy.orm import Session

from .db import BehaviorDeclaration, Machine, PolicyRule

Execution = Session | Connection


def pattern_matches(pattern: str, value: str) -> bool:
    regex = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(regex, value, re.DOTALL) is not None


def _split_glob(pattern: str) -> tuple[str, tuple[str, ...], str]:
    """Split a ``*``-glob into its prefix literal, middle literals, suffix."""
    parts = pattern.split("*")
    return parts[0], tuple(parts[1:-1]), parts[-1]


def _common_supersequence_lengths(
    first_middles: tuple[str, ...], second_middles: tuple[str, ...]
) -> list[list[int]]:
    """Suffix DP table of shortest common supersequence lengths.

    ``length[i][j]`` is the length of a shortest sequence embedding both
    ``first_middles[i:]`` and ``second_middles[j:]`` as subsequences. Every
    fragment occurrence advances the embedding on its own: occurrences with
    the same text are never deduplicated or substituted for one another, and
    two equal fragments may share one position only when both runs reach them
    in the same relative order.
    """
    first_len, second_len = len(first_middles), len(second_middles)
    length = [[0] * (second_len + 1) for _ in range(first_len + 1)]
    for i in range(first_len - 1, -1, -1):
        length[i][second_len] = first_len - i
    for j in range(second_len - 1, -1, -1):
        length[first_len][j] = second_len - j
    for i in range(first_len - 1, -1, -1):
        for j in range(second_len - 1, -1, -1):
            if first_middles[i] == second_middles[j]:
                length[i][j] = 1 + length[i + 1][j + 1]
            else:
                length[i][j] = 1 + min(length[i + 1][j], length[i][j + 1])
    return length


def _middle_runs_compatible(
    first_middles: tuple[str, ...], second_middles: tuple[str, ...]
) -> bool:
    """Whether two middle literal runs can hold inside a single resource.

    Each run is a sequence of fragment occurrences: a repeated literal keeps
    every occurrence, in order, and each occurrence must be laid out
    separately. A common resource lays both runs out over one shared sequence
    in which equal fragments coincide, so the runs are compatible exactly when
    a common supersequence exists that needs no more occurrences of any
    fragment than the richer of the two runs already carries — that is, when
    the shortest common supersequence holds exactly ``max`` of each
    fragment's two occurrence counts. Runs demanding opposite orders for the
    same fragments (directly, as in ``*a*b*`` against ``*b*a*``, or through a
    longer contradiction) force an extra, unshareable occurrence and can
    never be satisfied by one resource, no matter how the remaining text is
    chosen; a run that merely repeats a fragment (``*a*b*a*`` against
    ``*b*a*``) still shares every occurrence and stays compatible.
    """
    first_counts: dict[str, int] = {}
    for fragment in first_middles:
        first_counts[fragment] = first_counts.get(fragment, 0) + 1
    second_counts: dict[str, int] = {}
    for fragment in second_middles:
        second_counts[fragment] = second_counts.get(fragment, 0) + 1
    shared = sum(
        max(first_counts.get(fragment, 0), second_counts.get(fragment, 0))
        for fragment in first_counts.keys() | second_counts.keys()
    )
    length = _common_supersequence_lengths(first_middles, second_middles)
    return length[0][0] == shared


def _merge_middle_runs(
    first_middles: tuple[str, ...], second_middles: tuple[str, ...]
) -> tuple[str, ...]:
    """Canonical literal run covering both middle runs in order.

    The merge is the lexicographically smallest of the shortest common
    supersequences of the two runs: it keeps every literal fragment
    occurrence of both patterns — repeats included, never deduplicated — in
    an order both runs can satisfy, shares fragments the runs have in common
    instead of duplicating them, and is independent of which pattern was
    passed first, so the reported intersection never varies with the order a
    pair is examined in.
    """
    first_len, second_len = len(first_middles), len(second_middles)
    length = _common_supersequence_lengths(first_middles, second_middles)
    # best[i][j]: the lexicographically smallest shortest common
    # supersequence of the two suffixes, built bottom-up.
    best: list[list[tuple[str, ...]]] = [
        [()] * (second_len + 1) for _ in range(first_len + 1)
    ]
    for j in range(second_len, -1, -1):
        best[first_len][j] = tuple(second_middles[j:])
    for i in range(first_len, -1, -1):
        best[i][second_len] = tuple(first_middles[i:])
    for i in range(first_len - 1, -1, -1):
        for j in range(second_len - 1, -1, -1):
            candidates = []
            if first_middles[i] == second_middles[j]:
                candidates.append((first_middles[i],) + best[i + 1][j + 1])
            if length[i][j] == 1 + length[i + 1][j]:
                candidates.append((first_middles[i],) + best[i + 1][j])
            if length[i][j] == 1 + length[i][j + 1]:
                candidates.append((second_middles[j],) + best[i][j + 1])
            best[i][j] = min(candidates)
    return best[0][0]


def patterns_intersect(first: str, second: str) -> bool:
    """Whether two resource patterns can both match some common resource.

    The existing ``*`` semantics are kept: a star matches any (possibly empty)
    text and every other segment matches literally. Two patterns intersect
    exactly when a single resource string can satisfy both. A pattern without a
    star matches exactly one string, so a literal pair intersects only when
    equal and a literal intersects a glob only when the glob matches the
    literal. When both patterns carry a star, a common string exists exactly
    when their prefix literals are prefix-comparable (one starts with the
    other), their suffix literals are suffix-comparable, and their middle
    literal runs can hold inside one resource: each run is a sequence of
    fragment occurrences (a repeated literal keeps every occurrence, in
    order) and a common resource lays both runs out over one shared sequence,
    so runs whose occurrences cannot all line up — such as ``*a*b*`` against
    ``*b*a*`` — share no common match.
    """
    return intersection_pattern(first, second) is not None


def intersection_pattern(first: str, second: str) -> str | None:
    """A glob describing the scope matched by both resource patterns.

    Returns ``None`` when the patterns cannot match a common resource (the
    same test as ``patterns_intersect``). Otherwise the result is a glob under
    the existing ``*`` semantics whose every match is matched by both
    patterns: the longer of the two prefix literals, the merged middle
    literal runs, and the longer of the two suffix literals, joined by stars.
    The middle runs merge into the canonical common supersequence that keeps
    every literal fragment occurrence of both patterns — repeats included —
    in an order both can satisfy, so the result never depends on which
    pattern is examined first. A literal pattern's intersection with a glob
    it matches is the literal itself, so when one rule uses an exact resource
    its intersection with any overlapping pattern is that exact resource.
    """
    if "*" not in first:
        if "*" not in second:
            return first if first == second else None
        return first if pattern_matches(second, first) else None
    if "*" not in second:
        return second if pattern_matches(first, second) else None
    first_prefix, first_middles, first_suffix = _split_glob(first)
    second_prefix, second_middles, second_suffix = _split_glob(second)
    if not (
        first_prefix.startswith(second_prefix)
        or second_prefix.startswith(first_prefix)
    ):
        return None
    if not (
        first_suffix.endswith(second_suffix)
        or second_suffix.endswith(first_suffix)
    ):
        return None
    prefix = (
        first_prefix if len(first_prefix) >= len(second_prefix) else second_prefix
    )
    suffix = (
        first_suffix if len(first_suffix) >= len(second_suffix) else second_suffix
    )
    # Empty fragments (from adjacent or edge stars) pin nothing: ``a**b``
    # matches exactly the same resources as ``a*b``.
    first_middles = tuple(part for part in first_middles if part)
    second_middles = tuple(part for part in second_middles if part)
    if not _middle_runs_compatible(first_middles, second_middles):
        return None
    middles = _merge_middle_runs(first_middles, second_middles)
    return "*".join([prefix, *middles, suffix])


def machine_status(executor: Execution, machine_id: str) -> str | None:
    """Return one machine's stored status, or ``None`` if it does not exist."""
    return executor.execute(
        Machine.__table__.select()
        .where(Machine.__table__.c.id == machine_id)
        .with_only_columns(Machine.__table__.c.status)
    ).scalar()


def decide(
    executor: Execution,
    machine_id: str,
    status: str,
    action_type: str,
    resource: str,
) -> tuple[bool, str]:
    """Compute ``(allowed, reason)`` for a machine whose status was just read.

    A suspended machine short-circuits before declarations and rules. The
    reads happen on ``executor``, so inside a locked write transaction they
    see the same machine/declaration/rule snapshot the event commits against.
    """
    if status == "suspended":
        return False, "machine_suspended"

    declarations = _enabled_declaration_patterns(executor, machine_id, action_type)
    if not any(pattern_matches(pattern, resource) for pattern in declarations):
        return False, "no_enabled_declaration"

    rules = _policy_rules(executor, action_type)
    return _decide_against_rules(rules, resource)


def decide_from_snapshot(
    status: str,
    declarations_by_action: dict[str, list[str]] | None,
    rules_by_action: dict[str, list[tuple[str, str, int]]] | None,
    action_type: str,
    resource: str,
) -> tuple[bool, str]:
    """Compute ``(allowed, reason)`` from an already-loaded snapshot.

    Pure counterpart of :func:`decide` over data read at one earlier instant:
    a suspended machine is denied with ``machine_suspended`` without touching
    declarations or rules; otherwise the enabled-declaration check and the
    priority-first policy judgement use exactly the single-evaluation
    semantics. The snapshot maps group declaration resource patterns and
    ``(resource_pattern, effect, priority)`` rule rows by action type.
    """
    if status == "suspended":
        return False, "machine_suspended"

    patterns = (declarations_by_action or {}).get(action_type, [])
    if not any(pattern_matches(pattern, resource) for pattern in patterns):
        return False, "no_enabled_declaration"

    return _decide_against_rules(
        (rules_by_action or {}).get(action_type, []), resource
    )


def evaluate_batch(
    executor: Execution, machine_id: str, items: list[tuple[str, str]]
) -> list[tuple[bool, str]] | None:
    """Evaluate many ``(action_type, resource)`` pairs on one read snapshot.

    The machine's status is read once; when it exists every enabled
    declaration and policy rule is read at that same instant and shared by
    every item, so no input can observe or influence another input's result.
    A suspended machine reads neither declarations nor rules: every item is
    ``machine_suspended``. Returns ``None`` when the machine does not exist
    (the caller then answers ``404`` without emitting any item result). The
    per-item judgement is identical to :func:`decide`.
    """
    status = machine_status(executor, machine_id)
    if status is None:
        return None

    if status == "suspended":
        # No declaration or policy read happens for a suspended machine.
        return [(False, "machine_suspended") for _ in items]

    declarations_rows = executor.execute(
        BehaviorDeclaration.__table__.select()
        .where(
            BehaviorDeclaration.__table__.c.machine_id == machine_id,
            BehaviorDeclaration.__table__.c.enabled.is_(True),
        )
        .with_only_columns(
            BehaviorDeclaration.__table__.c.action_type,
            BehaviorDeclaration.__table__.c.resource_pattern,
        )
    ).all()
    declarations_by_action: dict[str, list[str]] = {}
    for row in declarations_rows:
        declarations_by_action.setdefault(row[0], []).append(row[1])

    rules_rows = executor.execute(
        PolicyRule.__table__.select().with_only_columns(
            PolicyRule.__table__.c.action_type,
            PolicyRule.__table__.c.resource_pattern,
            PolicyRule.__table__.c.effect,
            PolicyRule.__table__.c.priority,
        )
    ).all()
    rules_by_action: dict[str, list[tuple[str, str, int]]] = {}
    for row in rules_rows:
        rules_by_action.setdefault(row[0], []).append((row[1], row[2], row[3]))

    return [
        decide_from_snapshot(
            status, declarations_by_action, rules_by_action, action_type, resource
        )
        for action_type, resource in items
    ]


def _enabled_declaration_patterns(
    executor: Execution, machine_id: str, action_type: str
) -> list[str]:
    """The resource patterns of a machine's enabled declarations for an action."""
    rows = executor.execute(
        BehaviorDeclaration.__table__.select()
        .where(
            BehaviorDeclaration.__table__.c.machine_id == machine_id,
            BehaviorDeclaration.__table__.c.action_type == action_type,
            BehaviorDeclaration.__table__.c.enabled.is_(True),
        )
        .with_only_columns(BehaviorDeclaration.__table__.c.resource_pattern)
    ).all()
    return [row[0] for row in rows]


def _policy_rules(
    executor: Execution, action_type: str
) -> list[tuple[str, str, int]]:
    """All global rules for an action as ``(pattern, effect, priority)`` rows."""
    rows = executor.execute(
        PolicyRule.__table__.select()
        .where(PolicyRule.__table__.c.action_type == action_type)
        .with_only_columns(
            PolicyRule.__table__.c.resource_pattern,
            PolicyRule.__table__.c.effect,
            PolicyRule.__table__.c.priority,
        )
    ).all()
    return [(row[0], row[1], row[2]) for row in rows]


def _decide_against_rules(
    rules: list[tuple[str, str, int]], resource: str
) -> tuple[bool, str]:
    """The priority-first policy judgement over already-loaded rule rows."""
    matching = [row for row in rules if pattern_matches(row[0], resource)]
    if not matching:
        return False, "no_matching_policy"

    lowest = min(row[2] for row in matching)
    decisive = [row for row in matching if row[2] == lowest]
    if any(row[1] == "deny" for row in decisive):
        return False, "denied_by_policy"
    return True, "allowed_by_policy"
