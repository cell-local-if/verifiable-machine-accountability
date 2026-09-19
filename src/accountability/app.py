import os
import re
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
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
