"""Authorization decision computation.

The decision rule is the single source of truth for both the non-persisting
evaluation endpoint and the decision-event write path:

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

    declarations = executor.execute(
        BehaviorDeclaration.__table__.select()
        .where(
            BehaviorDeclaration.__table__.c.machine_id == machine_id,
            BehaviorDeclaration.__table__.c.action_type == action_type,
            BehaviorDeclaration.__table__.c.enabled.is_(True),
        )
        .with_only_columns(BehaviorDeclaration.__table__.c.resource_pattern)
    ).all()
    if not any(pattern_matches(row[0], resource) for row in declarations):
        return False, "no_enabled_declaration"

    rules = executor.execute(
        PolicyRule.__table__.select()
        .where(PolicyRule.__table__.c.action_type == action_type)
        .with_only_columns(
            PolicyRule.__table__.c.resource_pattern,
            PolicyRule.__table__.c.effect,
            PolicyRule.__table__.c.priority,
        )
    ).all()
    matching = [row for row in rules if pattern_matches(row[0], resource)]
    if not matching:
        return False, "no_matching_policy"

    lowest = min(row[2] for row in matching)
    decisive = [row for row in matching if row[2] == lowest]
    if any(row[1] == "deny" for row in decisive):
        return False, "denied_by_policy"
    return True, "allowed_by_policy"
