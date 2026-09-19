import os
import re
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, StrictInt, StringConstraints
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import chain
from .db import (
    AuthorizationDecisionCausalLink,
    AuthorizationDecisionEvent,
    Base,
    BehaviorDeclaration,
    Machine,
    PolicyRule,
)

DEFAULT_DATABASE_URL = "sqlite:///./accountability.db"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_engine(database_url: str):
    connect_args = (
        {"check_same_thread": False, "timeout": 15}
        if database_url.startswith("sqlite")
        else {}
    )
    engine = create_engine(database_url, connect_args=connect_args)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.close()

    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    database_url = os.environ.get("ACCOUNTABILITY_DATABASE_URL", DEFAULT_DATABASE_URL)
    engine = make_engine(database_url)
    Base.metadata.create_all(engine)
    chain.migrate_schema(engine)
    chain.backfill_chains(engine)
    app.state.engine = engine
    yield
    engine.dispose()


app = FastAPI(
    title="Verifiable Machine Accountability",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def error_response(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code}})


class MachineCreate(BaseModel):
    external_id: NonEmptyStr
    display_name: NonEmptyStr
    public_key: NonEmptyStr


class RotateKeyRequest(BaseModel):
    public_key: NonEmptyStr
    expected_version: int


class MachineOut(BaseModel):
    id: str
    external_id: str
    display_name: str
    public_key: str
    status: str
    version: int
    created_at: str
    updated_at: str


def to_out(machine: Machine) -> MachineOut:
    return MachineOut(
        id=machine.id,
        external_id=machine.external_id,
        display_name=machine.display_name,
        public_key=machine.public_key,
        status=machine.status,
        version=machine.version,
        created_at=machine.created_at,
        updated_at=machine.updated_at,
    )


class BehaviorDeclarationCreate(BaseModel):
    action_type: NonEmptyStr
    resource_pattern: NonEmptyStr
    enabled: StrictBool


class BehaviorDeclarationOut(BaseModel):
    id: str
    machine_id: str
    action_type: str
    resource_pattern: str
    enabled: bool
    created_at: str
    updated_at: str


def declaration_to_out(declaration: BehaviorDeclaration) -> BehaviorDeclarationOut:
    return BehaviorDeclarationOut(
        id=declaration.id,
        machine_id=declaration.machine_id,
        action_type=declaration.action_type,
        resource_pattern=declaration.resource_pattern,
        enabled=declaration.enabled,
        created_at=declaration.created_at,
        updated_at=declaration.updated_at,
    )


@app.post("/machines", status_code=201, response_model=MachineOut)
def create_machine(body: MachineCreate, session: SessionDep):
    now = utc_now_iso()
    machine = Machine(
        id=str(uuid.uuid4()),
        external_id=body.external_id,
        display_name=body.display_name,
        public_key=body.public_key,
        status="active",
        version=1,
        created_at=now,
        updated_at=now,
    )
    session.add(machine)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return error_response(409, "duplicate_external_id")
    return to_out(machine)


@app.get("/machines/{machine_id}", response_model=MachineOut)
def get_machine(machine_id: str, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")
    return to_out(machine)


@app.post("/machines/{machine_id}/rotate-key", response_model=MachineOut)
def rotate_key(machine_id: str, body: RotateKeyRequest, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")
    if body.public_key == machine.public_key:
        return error_response(422, "same_public_key")
    if body.expected_version != machine.version:
        return error_response(409, "version_conflict")

    now = utc_now_iso()
    result = session.execute(
        update(Machine)
        .where(Machine.id == machine_id, Machine.version == body.expected_version)
        .values(public_key=body.public_key, version=Machine.version + 1, updated_at=now)
    )
    if result.rowcount == 0:
        session.rollback()
        current = session.get(Machine, machine_id)
        if current is None:
            return error_response(404, "not_found")
        if current.public_key == body.public_key:
            return error_response(422, "same_public_key")
        return error_response(409, "version_conflict")

    session.commit()
    session.refresh(machine)
    return to_out(machine)


@app.post(
    "/machines/{machine_id}/behavior-declarations",
    status_code=201,
    response_model=BehaviorDeclarationOut,
)
def create_behavior_declaration(
    machine_id: str, body: BehaviorDeclarationCreate, session: SessionDep
):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    now = utc_now_iso()
    declaration = BehaviorDeclaration(
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        action_type=body.action_type,
        resource_pattern=body.resource_pattern,
        enabled=body.enabled,
        created_at=now,
        updated_at=now,
    )
    session.add(declaration)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        if session.get(Machine, machine_id) is None:
            return error_response(404, "not_found")
        return error_response(409, "duplicate_behavior_declaration")
    return declaration_to_out(declaration)


@app.get(
    "/machines/{machine_id}/behavior-declarations",
    response_model=list[BehaviorDeclarationOut],
)
def list_behavior_declarations(machine_id: str, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    declarations = session.scalars(
        select(BehaviorDeclaration)
        .where(BehaviorDeclaration.machine_id == machine_id)
        .order_by(BehaviorDeclaration.created_at, BehaviorDeclaration.id)
    ).all()
    return [declaration_to_out(d) for d in declarations]


class PolicyRuleCreate(BaseModel):
    action_type: NonEmptyStr
    resource_pattern: NonEmptyStr
    effect: Literal["allow", "deny"]
    priority: Annotated[StrictInt, Field(ge=0)]


class PolicyRuleOut(BaseModel):
    id: str
    action_type: str
    resource_pattern: str
    effect: str
    priority: int
    created_at: str
    updated_at: str


def policy_rule_to_out(rule: PolicyRule) -> PolicyRuleOut:
    return PolicyRuleOut(
        id=rule.id,
        action_type=rule.action_type,
        resource_pattern=rule.resource_pattern,
        effect=rule.effect,
        priority=rule.priority,
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


@app.post("/policy-rules", status_code=201, response_model=PolicyRuleOut)
def create_policy_rule(body: PolicyRuleCreate, session: SessionDep):
    now = utc_now_iso()
    rule = PolicyRule(
        id=str(uuid.uuid4()),
        action_type=body.action_type,
        resource_pattern=body.resource_pattern,
        effect=body.effect,
        priority=body.priority,
        created_at=now,
        updated_at=now,
    )
    session.add(rule)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return error_response(409, "duplicate_policy_rule")
    return policy_rule_to_out(rule)


class AuthorizationEvaluationCreate(BaseModel):
    action_type: NonEmptyStr
    resource: NonEmptyStr


class AuthorizationEvaluationOut(BaseModel):
    allowed: bool
    reason: str


def pattern_matches(pattern: str, value: str) -> bool:
    regex = ".*".join(re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(regex, value, re.DOTALL) is not None


def compute_authorization_decision(
    session: Session, machine_id: str, action_type: str, resource: str
) -> AuthorizationEvaluationOut:
    declarations = session.scalars(
        select(BehaviorDeclaration).where(
            BehaviorDeclaration.machine_id == machine_id,
            BehaviorDeclaration.action_type == action_type,
            BehaviorDeclaration.enabled.is_(True),
        )
    ).all()
    if not any(pattern_matches(d.resource_pattern, resource) for d in declarations):
        return AuthorizationEvaluationOut(
            allowed=False, reason="no_enabled_declaration"
        )

    rules = session.scalars(
        select(PolicyRule).where(PolicyRule.action_type == action_type)
    ).all()
    matching = [r for r in rules if pattern_matches(r.resource_pattern, resource)]
    if not matching:
        return AuthorizationEvaluationOut(allowed=False, reason="no_matching_policy")

    lowest = min(r.priority for r in matching)
    decisive = [r for r in matching if r.priority == lowest]
    if any(r.effect == "deny" for r in decisive):
        return AuthorizationEvaluationOut(allowed=False, reason="denied_by_policy")
    return AuthorizationEvaluationOut(allowed=True, reason="allowed_by_policy")


@app.post(
    "/machines/{machine_id}/authorization-evaluations",
    response_model=AuthorizationEvaluationOut,
)
def evaluate_authorization(
    machine_id: str, body: AuthorizationEvaluationCreate, session: SessionDep
):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    return compute_authorization_decision(
        session, machine_id, body.action_type, body.resource
    )


class AuthorizationDecisionEventOut(BaseModel):
    id: str
    machine_id: str
    action_type: str
    resource: str
    allowed: bool
    reason: str
    created_at: str
    previous_event_id: str | None
    content_hash: str
    chain_hash: str


def decision_event_to_out(event: AuthorizationDecisionEvent) -> AuthorizationDecisionEventOut:
    return AuthorizationDecisionEventOut(
        id=event.id,
        machine_id=event.machine_id,
        action_type=event.action_type,
        resource=event.resource,
        allowed=event.allowed,
        reason=event.reason,
        created_at=event.created_at,
        previous_event_id=event.previous_event_id,
        content_hash=event.content_hash,
        chain_hash=event.chain_hash,
    )


@app.post(
    "/machines/{machine_id}/authorization-decision-events",
    status_code=201,
    response_model=AuthorizationDecisionEventOut,
)
def create_authorization_decision_event(
    machine_id: str, body: AuthorizationEvaluationCreate, session: SessionDep
):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    decision = compute_authorization_decision(
        session, machine_id, body.action_type, body.resource
    )
    engine = session.get_bind()
    # Commit and return the read connection before opening the locked write
    # transaction, so concurrent writers never hold two pool connections at
    # once. The decision is fully materialized above.
    session.commit()
    session.close()
    # Read tail, mint the new link, and insert in one write transaction so
    # concurrent appenders cannot lose events or fork the per-machine chain.
    result = chain.append_event(
        engine,
        machine_id=machine_id,
        action_type=body.action_type,
        resource=body.resource,
        allowed=decision.allowed,
        reason=decision.reason,
    )
    return AuthorizationDecisionEventOut(**result)


@app.get(
    "/machines/{machine_id}/authorization-decision-events",
    response_model=list[AuthorizationDecisionEventOut],
)
def list_authorization_decision_events(machine_id: str, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    events = session.scalars(
        select(AuthorizationDecisionEvent)
        .where(AuthorizationDecisionEvent.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionEvent.created_at, AuthorizationDecisionEvent.id
        )
    ).all()
    return [decision_event_to_out(e) for e in events]


class IntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_event_id: str | None


@app.get(
    "/machines/{machine_id}/authorization-decision-events/integrity",
    response_model=IntegrityOut,
)
def check_authorization_decision_event_integrity(machine_id: str, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    valid, checked_count, broken_event_id = chain.verify_chain(session, machine_id)
    return IntegrityOut(
        valid=valid,
        checked_count=checked_count,
        broken_event_id=broken_event_id,
    )


class CausalLinkCreate(BaseModel):
    effect_event_id: NonEmptyStr


class CausalLinkOut(BaseModel):
    id: str
    machine_id: str
    cause_event_id: str
    effect_event_id: str
    created_at: str


def causal_link_to_out(link: AuthorizationDecisionCausalLink) -> CausalLinkOut:
    return CausalLinkOut(
        id=link.id,
        machine_id=link.machine_id,
        cause_event_id=link.cause_event_id,
        effect_event_id=link.effect_event_id,
        created_at=link.created_at,
    )


def get_machine_event(
    session: Session, machine_id: str, event_id: str
) -> AuthorizationDecisionEvent | None:
    return session.scalar(
        select(AuthorizationDecisionEvent).where(
            AuthorizationDecisionEvent.id == event_id,
            AuthorizationDecisionEvent.machine_id == machine_id,
        )
    )


def introduces_cycle(
    session: Session, machine_id: str, cause_event_id: str, effect_event_id: str
) -> bool:
    """Whether a new cause -> effect edge would close a directed cycle.

    Follows existing cause -> effect edges from the proposed effect; reaching
    the proposed cause means the new edge closes a loop.
    """
    links = session.scalars(
        select(AuthorizationDecisionCausalLink).where(
            AuthorizationDecisionCausalLink.machine_id == machine_id
        )
    ).all()
    adjacency: dict[str, list[str]] = {}
    for link in links:
        adjacency.setdefault(link.cause_event_id, []).append(link.effect_event_id)

    visited = {effect_event_id}
    stack = [effect_event_id]
    while stack:
        current = stack.pop()
        if current == cause_event_id:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False


@app.post(
    "/machines/{machine_id}/authorization-decision-events/{cause_event_id}/causal-links",
    status_code=201,
    response_model=CausalLinkOut,
)
def create_causal_link(
    machine_id: str,
    cause_event_id: str,
    body: CausalLinkCreate,
    session: SessionDep,
):
    cause_event = get_machine_event(session, machine_id, cause_event_id)
    effect_event = get_machine_event(session, machine_id, body.effect_event_id)
    if cause_event is None or effect_event is None:
        return error_response(404, "not_found")
    if cause_event_id == body.effect_event_id:
        return error_response(422, "self_causal_link")

    existing = session.scalar(
        select(AuthorizationDecisionCausalLink).where(
            AuthorizationDecisionCausalLink.machine_id == machine_id,
            AuthorizationDecisionCausalLink.cause_event_id == cause_event_id,
            AuthorizationDecisionCausalLink.effect_event_id == body.effect_event_id,
        )
    )
    if existing is not None:
        return error_response(409, "duplicate_causal_link")
    if introduces_cycle(session, machine_id, cause_event_id, body.effect_event_id):
        return error_response(409, "causal_cycle")

    link = AuthorizationDecisionCausalLink(
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        cause_event_id=cause_event_id,
        effect_event_id=body.effect_event_id,
        created_at=utc_now_iso(),
    )
    session.add(link)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return error_response(409, "duplicate_causal_link")
    return causal_link_to_out(link)


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{cause_event_id}/causal-links",
    response_model=list[CausalLinkOut],
)
def list_causal_links(machine_id: str, cause_event_id: str, session: SessionDep):
    cause_event = get_machine_event(session, machine_id, cause_event_id)
    if cause_event is None:
        return error_response(404, "not_found")

    links = session.scalars(
        select(AuthorizationDecisionCausalLink)
        .where(
            AuthorizationDecisionCausalLink.machine_id == machine_id,
            AuthorizationDecisionCausalLink.cause_event_id == cause_event_id,
        )
        .order_by(
            AuthorizationDecisionCausalLink.created_at,
            AuthorizationDecisionCausalLink.id,
        )
    ).all()
    return [causal_link_to_out(link) for link in links]


class CausalLinkIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_link_id: str | None


def find_broken_causal_link(
    session: Session, machine_id: str
) -> tuple[int, str | None]:
    """Scan one machine's causal links in (created_at, id) order.

    Returns ``(total_count, broken_link_id)``. Each link is broken when its
    cause or effect does not resolve to an event of the path machine (missing
    or owned by another machine) or when both endpoints are the same event.
    Links that pass are added to a running edge set, and a link whose
    cause -> effect edge would close a directed cycle among the edges scanned
    so far is reported as broken. Read-only: issues no writes.
    """
    links = session.scalars(
        select(AuthorizationDecisionCausalLink)
        .where(AuthorizationDecisionCausalLink.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionCausalLink.created_at,
            AuthorizationDecisionCausalLink.id,
        )
    ).all()

    machine_event_ids = set(
        session.scalars(
            select(AuthorizationDecisionEvent.id).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    )

    adjacency: dict[str, list[str]] = {}
    for link in links:
        if (
            link.cause_event_id == link.effect_event_id
            or link.cause_event_id not in machine_event_ids
            or link.effect_event_id not in machine_event_ids
        ):
            return len(links), link.id
        # Follow already-scanned cause -> effect edges from the new effect;
        # reaching the new cause means this edge closes a directed cycle.
        visited = {link.effect_event_id}
        stack = [link.effect_event_id]
        while stack:
            current = stack.pop()
            if current == link.cause_event_id:
                return len(links), link.id
            for nxt in adjacency.get(current, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        adjacency.setdefault(link.cause_event_id, []).append(link.effect_event_id)

    return len(links), None


@app.get(
    "/machines/{machine_id}/authorization-decision-events/causal-links/integrity",
    response_model=CausalLinkIntegrityOut,
)
def check_causal_link_integrity(machine_id: str, session: SessionDep):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    checked_count, broken_link_id = find_broken_causal_link(session, machine_id)
    return CausalLinkIntegrityOut(
        valid=broken_link_id is None,
        checked_count=checked_count,
        broken_link_id=broken_link_id,
    )


_INTEGER_QUERY_RE = re.compile(r"-?\d+")


class CausalTraceParams(BaseModel):
    direction: Literal["downstream", "upstream"]
    max_depth: int


def validate_causal_trace_params(
    direction: Annotated[str | None, Query()] = None,
    max_depth: Annotated[str | None, Query()] = None,
) -> CausalTraceParams:
    """Validate the causal-trace query string before any event lookup.

    Query parameters arrive as strings, so integer-ness is checked explicitly
    against the raw value: decimal forms such as ``3.0`` and the literals
    ``true``/``false`` are rejected instead of being coerced, and the value
    must lie in 1..20.
    """
    errors: list[dict[str, object]] = []

    if direction is None:
        errors.append(
            {"type": "missing", "loc": ["query", "direction"],
             "msg": "Field required", "input": None}
        )
    elif direction not in ("downstream", "upstream"):
        errors.append(
            {
                "type": "literal_error",
                "loc": ["query", "direction"],
                "msg": "Input should be 'downstream' or 'upstream'",
                "input": direction,
                "ctx": {"expected": "'downstream' or 'upstream'"},
            }
        )

    depth_value: int | None = None
    if max_depth is None:
        errors.append(
            {"type": "missing", "loc": ["query", "max_depth"],
             "msg": "Field required", "input": None}
        )
    elif not _INTEGER_QUERY_RE.fullmatch(max_depth):
        errors.append(
            {"type": "int_parsing", "loc": ["query", "max_depth"],
             "msg": "Input should be a valid integer", "input": max_depth}
        )
    else:
        depth_value = int(max_depth)
        if depth_value < 1:
            errors.append(
                {
                    "type": "greater_than_equal",
                    "loc": ["query", "max_depth"],
                    "msg": "Input should be greater than or equal to 1",
                    "input": max_depth,
                    "ctx": {"ge": 1},
                }
            )
        elif depth_value > 20:
            errors.append(
                {
                    "type": "less_than_equal",
                    "loc": ["query", "max_depth"],
                    "msg": "Input should be less than or equal to 20",
                    "input": max_depth,
                    "ctx": {"le": 20},
                }
            )

    if errors:
        raise RequestValidationError(errors)
    return CausalTraceParams(direction=direction, max_depth=depth_value)  # type: ignore[arg-type]


class TracedEventOut(BaseModel):
    event_id: str
    depth: int


class CausalTraceOut(BaseModel):
    event_id: str
    direction: str
    max_depth: int
    events: list[TracedEventOut]


def trace_causal_events(
    session: Session,
    machine_id: str,
    start_event_id: str,
    direction: str,
    max_depth: int,
) -> list[tuple[AuthorizationDecisionEvent, int]]:
    """Bounded breadth-first walk over one machine's causal links.

    ``downstream`` follows existing ``cause_event_id -> effect_event_id``
    edges; ``upstream`` reverses them. Only links belonging to the machine are
    loaded and only targets that still exist as events of the machine are
    followed, so the walk never crosses machines or enters dangling targets.
    BFS plus a visited set seeded with the start makes the first encounter the
    shortest distance and guarantees termination when links form a ring.
    """
    links = session.scalars(
        select(AuthorizationDecisionCausalLink).where(
            AuthorizationDecisionCausalLink.machine_id == machine_id
        )
    ).all()

    existing_event_ids = set(
        session.scalars(
            select(AuthorizationDecisionEvent.id).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    )

    adjacency: dict[str, list[str]] = {}
    for link in links:
        if direction == "downstream":
            adjacency.setdefault(link.cause_event_id, []).append(link.effect_event_id)
        else:
            adjacency.setdefault(link.effect_event_id, []).append(link.cause_event_id)

    distances = {start_event_id: 0}
    queue: deque[tuple[str, int]] = deque([(start_event_id, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor in distances:
                continue
            # Skip link endpoints that do not resolve to an event of this
            # machine (missing target or one outside the machine boundary).
            if neighbor not in existing_event_ids:
                continue
            distances[neighbor] = depth + 1
            queue.append((neighbor, depth + 1))

    distances.pop(start_event_id, None)
    if not distances:
        return []

    events = session.scalars(
        select(AuthorizationDecisionEvent).where(
            AuthorizationDecisionEvent.machine_id == machine_id,
            AuthorizationDecisionEvent.id.in_(distances),
        )
    ).all()
    ordered = sorted(
        events, key=lambda event: (distances[event.id], event.created_at, event.id)
    )
    return [(event, distances[event.id]) for event in ordered]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/causal-trace",
    response_model=CausalTraceOut,
)
def get_authorization_decision_event_causal_trace(
    machine_id: str,
    event_id: str,
    params: Annotated[CausalTraceParams, Depends(validate_causal_trace_params)],
    session: SessionDep,
):
    start_event = get_machine_event(session, machine_id, event_id)
    if start_event is None:
        return error_response(404, "not_found")

    traced = trace_causal_events(
        session,
        machine_id=machine_id,
        start_event_id=event_id,
        direction=params.direction,
        max_depth=params.max_depth,
    )
    return CausalTraceOut(
        event_id=event_id,
        direction=params.direction,
        max_depth=params.max_depth,
        events=[
            TracedEventOut(event_id=event.id, depth=depth) for event, depth in traced
        ],
    )


# Strict RFC3339 UTC shape matching the stored created_at format: full
# date-time with a literal "T" and a "Z" designator (optional fraction).
_RFC3339_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z")


def parse_utc_rfc3339(value: str) -> datetime | None:
    """Parse a strict UTC RFC3339 timestamp ending in ``Z``; None if invalid."""
    if not _RFC3339_UTC_RE.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class ComplianceExportParams(BaseModel):
    from_created_at: str
    to_created_at: str
    from_dt: datetime
    to_dt: datetime


def validate_compliance_export_params(
    from_created_at: Annotated[str | None, Query()] = None,
    to_created_at: Annotated[str | None, Query()] = None,
) -> ComplianceExportParams:
    """Validate the compliance-export time range before any machine lookup.

    Both bounds are required, must be UTC RFC3339 timestamps ending in ``Z``,
    and ``from_created_at`` must not be later than ``to_created_at``. Any
    violation raises 422 before the handler touches the database.
    """
    errors: list[dict[str, object]] = []

    from_dt: datetime | None = None
    if from_created_at is None:
        errors.append(
            {"type": "missing", "loc": ["query", "from_created_at"],
             "msg": "Field required", "input": None}
        )
    else:
        from_dt = parse_utc_rfc3339(from_created_at)
        if from_dt is None:
            errors.append(
                {
                    "type": "datetime_parsing",
                    "loc": ["query", "from_created_at"],
                    "msg": "Input should be a valid UTC RFC3339 datetime ending in 'Z'",
                    "input": from_created_at,
                }
            )

    to_dt: datetime | None = None
    if to_created_at is None:
        errors.append(
            {"type": "missing", "loc": ["query", "to_created_at"],
             "msg": "Field required", "input": None}
        )
    else:
        to_dt = parse_utc_rfc3339(to_created_at)
        if to_dt is None:
            errors.append(
                {
                    "type": "datetime_parsing",
                    "loc": ["query", "to_created_at"],
                    "msg": "Input should be a valid UTC RFC3339 datetime ending in 'Z'",
                    "input": to_created_at,
                }
            )

    if not errors and from_dt > to_dt:  # type: ignore[operator]
        errors.append(
            {
                "type": "value_error",
                "loc": ["query", "from_created_at"],
                "msg": "from_created_at must not be later than to_created_at",
                "input": from_created_at,
            }
        )

    if errors:
        raise RequestValidationError(errors)
    return ComplianceExportParams(
        from_created_at=from_created_at,  # type: ignore[arg-type]
        to_created_at=to_created_at,  # type: ignore[arg-type]
        from_dt=from_dt,  # type: ignore[arg-type]
        to_dt=to_dt,  # type: ignore[arg-type]
    )


class ComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    events: list[AuthorizationDecisionEventOut]
    causal_links: list[CausalLinkOut]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/compliance-export",
    response_model=ComplianceExportOut,
)
def export_authorization_decision_events_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export of one machine's decision events and links.

    Events are those whose created_at falls in the closed
    [from_created_at, to_created_at] range; causal links are the machine's
    links whose cause and effect events are both in that event set. Issues no
    writes, so repeated exports over the same data are identical.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    all_events = session.scalars(
        select(AuthorizationDecisionEvent).where(
            AuthorizationDecisionEvent.machine_id == machine_id
        )
    ).all()
    # Compare parsed instants rather than raw strings: stored timestamps may
    # omit the fractional part when it is zero, which would misorder a plain
    # lexicographic comparison.
    events = sorted(
        (
            event
            for event in all_events
            if (created := parse_utc_rfc3339(event.created_at)) is not None
            and params.from_dt <= created <= params.to_dt
        ),
        key=lambda event: (event.created_at, event.id),
    )
    event_ids = {event.id for event in events}

    links = session.scalars(
        select(AuthorizationDecisionCausalLink)
        .where(
            AuthorizationDecisionCausalLink.machine_id == machine_id,
            AuthorizationDecisionCausalLink.cause_event_id.in_(event_ids),
            AuthorizationDecisionCausalLink.effect_event_id.in_(event_ids),
        )
        .order_by(
            AuthorizationDecisionCausalLink.created_at,
            AuthorizationDecisionCausalLink.id,
        )
    ).all()

    return ComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        events=[decision_event_to_out(event) for event in events],
        causal_links=[causal_link_to_out(link) for link in links],
    )
