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
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import assignment_chain, chain, incidents, machines, rotation_chain
from .db import (
    AuthorizationDecisionCausalLink,
    AuthorizationDecisionEvent,
    AuthorizationDecisionEvidence,
    AuthorizationDecisionIncident,
    Base,
    BehaviorDeclaration,
    IncidentResponsibilityAssignment,
    IncidentStatusEvent,
    KeyRotationEvent,
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
    rotation_chain.migrate_schema(engine)
    rotation_chain.backfill_chains(engine)
    assignment_chain.migrate_schema(engine)
    assignment_chain.backfill_chains(engine)
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


class MachineStatusUpdate(BaseModel):
    # Literal validation runs before the handler, so a missing body field, a
    # non-object body, a non-string value, or any string other than
    # "active"/"suspended" is a 422 before the machine is ever looked up.
    status: Literal["active", "suspended"]


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
    engine = session.get_bind()
    # Return the read connection before opening the locked write transaction,
    # so concurrent rotations never hold two pool connections at once.
    session.close()
    # Update the machine row and append the rotation record to the machine's
    # chain tail in one write transaction, so a successful rotation always
    # leaves exactly one audit record and concurrent rotations cannot lose
    # records, fork, or break the chain.
    result = rotation_chain.rotate_key(
        engine,
        machine_id=machine_id,
        new_public_key=body.public_key,
        expected_version=body.expected_version,
    )
    status = result["status"]
    if status == "not_found":
        return error_response(404, "not_found")
    if status == "same_public_key":
        return error_response(422, "same_public_key")
    if status == "version_conflict":
        return error_response(409, "version_conflict")
    return MachineOut(**result["machine"])


@app.post("/machines/{machine_id}/status", response_model=MachineOut)
def update_machine_status(
    machine_id: str, body: MachineStatusUpdate, session: SessionDep
):
    """Persistently suspend or reactivate one machine.

    Body validation runs before any machine lookup, so a missing body, a
    non-object body, a missing or non-string ``status``, or any value other
    than ``active``/``suspended`` is a 422 even when the machine does not
    exist. A missing machine is a 404 ``not_found``. Requesting the machine's
    current status returns 409 ``invalid_status_transition`` and writes
    nothing, so concurrent requests for the same target status have at most
    one success. On success only ``status`` and ``updated_at`` are atomically
    updated; ``version``, ``public_key``, ``created_at``, and every other
    record are unchanged, and the full updated machine is returned with 200.
    The status is stored in the machines table and survives restarts.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the chain appenders: concurrent updates never hold two pool
    # connections at once.
    session.close()
    result = machines.change_machine_status(
        engine,
        machine_id=machine_id,
        to_status=body.status,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    if result["status"] == "invalid_status_transition":
        return error_response(409, "invalid_status_transition")
    return MachineOut(**result["machine"])


class KeyRotationEventOut(BaseModel):
    id: str
    machine_id: str
    old_public_key: str
    new_public_key: str
    version: int
    created_at: str
    previous_rotation_id: str | None
    content_hash: str
    chain_hash: str


def key_rotation_event_to_out(event: KeyRotationEvent) -> KeyRotationEventOut:
    return KeyRotationEventOut(
        id=event.id,
        machine_id=event.machine_id,
        old_public_key=event.old_public_key,
        new_public_key=event.new_public_key,
        version=event.version,
        created_at=event.created_at,
        previous_rotation_id=event.previous_rotation_id,
        content_hash=event.content_hash,
        chain_hash=event.chain_hash,
    )


@app.get(
    "/machines/{machine_id}/key-rotation-events",
    response_model=list[KeyRotationEventOut],
)
def list_key_rotation_events(machine_id: str, session: SessionDep):
    """Read-only audit history of one machine's key rotations.

    Returns every rotation record owned by the path machine in (created_at,
    id) order; a machine with no rotations yields ``[]``. The query only
    reads: it never writes or modifies machines, events, evidence, chains, or
    causal links.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    events = session.scalars(
        select(KeyRotationEvent)
        .where(KeyRotationEvent.machine_id == machine_id)
        .order_by(KeyRotationEvent.created_at, KeyRotationEvent.id)
    ).all()
    return [key_rotation_event_to_out(event) for event in events]


class KeyRotationIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_rotation_id: str | None


@app.get(
    "/machines/{machine_id}/key-rotation-events/integrity",
    response_model=KeyRotationIntegrityOut,
)
def check_key_rotation_event_integrity(machine_id: str, session: SessionDep):
    """Read-only verification of one machine's key rotation hash chain.

    Returns ``{valid, checked_count, broken_rotation_id}``: an empty or fully
    sound chain reports ``true``, the machine's total rotation count, and
    ``null``; otherwise the first record whose content hash, previous-rotation
    link, or chain hash does not verify is reported. Only the path machine's
    records are examined, and the query never writes, repairs, or deletes.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    valid, checked_count, broken_rotation_id = rotation_chain.verify_chain(
        session, machine_id
    )
    return KeyRotationIntegrityOut(
        valid=valid,
        checked_count=checked_count,
        broken_rotation_id=broken_rotation_id,
    )


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


@app.get("/policy-rules", response_model=list[PolicyRuleOut])
def list_policy_rules(session: SessionDep):
    """Read-only listing of every global policy rule.

    Returns all rules ordered by the actual UTC instant of ``created_at``,
    then by ``id``; a table with no rules yields ``[]``. Each item carries
    exactly the persisted ``{id, action_type, resource_pattern, effect,
    priority, created_at, updated_at}`` values, with no normalization or
    repair. ISO-8601 text ordering is not chronological once fractional
    seconds are present (``...:00.5Z`` sorts before ``...:00Z`` because ``.``
    precedes ``Z``), so stamps are parsed to UTC instants before the id
    tie-break. The query only reads: it never writes, updates, deletes,
    normalizes, or repairs a rule, and it never participates in authorization
    evaluation, so repeated calls are stable and rules remain queryable across
    restarts.
    """
    rules = session.scalars(select(PolicyRule)).all()
    return [policy_rule_to_out(rule) for rule in order_by_created_at_instant(rules)]


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

    # A suspended machine is denied before any declaration or policy lookup,
    # so its stored behavior declarations and policy rules can never produce
    # an allow (or any other reason) while suspended.
    if machine.status == "suspended":
        return AuthorizationEvaluationOut(
            allowed=False, reason="machine_suspended"
        )

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

    # A suspended machine is denied without reading its declarations or any
    # policy rule. The decision is still recorded below under the same
    # persistence and hash-chain rules as every other decision, so the
    # suspension leaves a complete audit trail.
    if machine.status == "suspended":
        decision = AuthorizationEvaluationOut(
            allowed=False, reason="machine_suspended"
        )
    else:
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


# RFC 3339 date-time expressed in UTC with a literal ``Z`` suffix. Fractional
# seconds are optional; offset forms (``+00:00``) and a missing suffix are
# rejected. The calendar/time fields are range-checked by ``datetime``.
_RFC3339_Z_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)


class ComplianceExportParams(BaseModel):
    from_created_at: str
    to_created_at: str


def parse_utc_z_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def order_by_created_at_instant(records: list) -> list:
    """Order persisted rows by the actual UTC instant of ``created_at``.

    Text ordering of ISO-8601 stamps is not chronological across the
    fractional-second boundary: within one second ``...:00.5Z`` precedes
    ``...:00Z`` lexicographically (``.`` < ``Z``) even though the instant is
    later. Parsing first makes an exact-second stamp sort before any
    fractional stamp of the same second; rows that share an instant break
    ties by ``id`` ascending.
    """
    return sorted(
        records,
        key=lambda record: (parse_utc_z_datetime(record.created_at), record.id),
    )


def _datetime_error(field_name: str, raw: str) -> dict[str, object]:
    return {
        "type": "value_error",
        "loc": ["query", field_name],
        "msg": "Input should be an RFC 3339 date-time in UTC ending with 'Z'",
        "input": raw,
    }


def validate_compliance_export_params(
    from_created_at: Annotated[str | None, Query()] = None,
    to_created_at: Annotated[str | None, Query()] = None,
) -> ComplianceExportParams:
    """Validate the compliance-export query string before any machine lookup.

    Both bounds are required UTC RFC 3339 date-times ending in ``Z``, and the
    lower bound must not be after the upper bound. Any missing, malformed, or
    inverted value raises a 422 before the database is consulted, so an
    invalid export against a non-existent machine still reports 422 rather
    than 404.
    """
    errors: list[dict[str, object]] = []
    parsed: dict[str, datetime] = {}

    for field_name, raw in (
        ("from_created_at", from_created_at),
        ("to_created_at", to_created_at),
    ):
        if raw is None:
            errors.append(
                {"type": "missing", "loc": ["query", field_name],
                 "msg": "Field required", "input": None}
            )
        elif not _RFC3339_Z_DATETIME_RE.fullmatch(raw):
            errors.append(_datetime_error(field_name, raw))
        else:
            try:
                parsed[field_name] = parse_utc_z_datetime(raw)
            except ValueError:
                # Regex passed but the calendar/time values are out of range
                # (e.g. month 13, day 30 in February, hour 24).
                errors.append(_datetime_error(field_name, raw))

    if not errors and parsed["from_created_at"] > parsed["to_created_at"]:
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
    )


class KeyRotationComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    rotations: list[KeyRotationEventOut]


@app.get(
    "/machines/{machine_id}/key-rotation-events/compliance-export",
    response_model=KeyRotationComplianceExportOut,
)
def export_key_rotation_events_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export of one machine's key rotations over a window.

    Includes every rotation record owned by the path machine whose
    ``created_at`` falls within the inclusive bounds, in ``(created_at, id)``
    order, with exactly the fields of the rotation list endpoint. Records are
    exported exactly as stored: a damaged machine public key or a corrupt
    previous-rotation link, content hash, or chain hash never causes a record
    to be rewritten, filtered out, or repaired. The endpoint only issues reads
    and never returns another machine's rotations.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load the machine's rotations, order them by the actual UTC instant of
    # ``created_at`` (then id), and apply the closed window to parsed instants.
    # Neither step is safe lexicographically: within one second an exact-second
    # ISO stamp (no fractional part) sorts *before* a fractional stamp as text
    # (``.`` precedes ``Z``), so ordering by the stored string would put
    # ``...:00.5Z`` ahead of ``...:00Z``; parsing both the bounds and each
    # stamp compares true instants.
    machine_rotations = session.scalars(
        select(KeyRotationEvent).where(KeyRotationEvent.machine_id == machine_id)
    ).all()
    in_window = [
        record
        for record in machine_rotations
        if window_start <= parse_utc_z_datetime(record.created_at) <= window_end
    ]
    records = order_by_created_at_instant(in_window)

    return KeyRotationComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        rotations=[key_rotation_event_to_out(record) for record in records],
    )


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


# A SHA-256-style fingerprint: exactly 64 lowercase hexadecimal characters.
# The pattern anchors the full string and admits no uppercase, so the stored
# value is compared and persisted exactly as supplied, never case-folded.
LowerHexHash = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{64}$")
]


class EvidenceCreate(BaseModel):
    evidence_type: NonEmptyStr
    content_hash: LowerHexHash


class EvidenceOut(BaseModel):
    id: str
    machine_id: str
    event_id: str
    evidence_type: str
    content_hash: str
    created_at: str


def evidence_to_out(record: AuthorizationDecisionEvidence) -> EvidenceOut:
    return EvidenceOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        evidence_type=record.evidence_type,
        content_hash=record.content_hash,
        created_at=record.created_at,
    )


@app.post(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/evidence",
    status_code=201,
    response_model=EvidenceOut,
)
def create_evidence(
    machine_id: str,
    event_id: str,
    body: EvidenceCreate,
    session: SessionDep,
):
    """Attach one immutable evidence fingerprint to a decision event.

    Body validation runs before any path lookup, so a malformed payload is a
    422 even when the machine or event does not exist. The write touches only
    the evidence table: the event, its hash chain, and causal links are never
    modified.
    """
    event = get_machine_event(session, machine_id, event_id)
    if event is None:
        return error_response(404, "not_found")

    existing = session.scalar(
        select(AuthorizationDecisionEvidence).where(
            AuthorizationDecisionEvidence.event_id == event_id,
            AuthorizationDecisionEvidence.content_hash == body.content_hash,
        )
    )
    if existing is not None:
        return error_response(409, "duplicate_evidence")

    record = AuthorizationDecisionEvidence(
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        event_id=event_id,
        evidence_type=body.evidence_type,
        content_hash=body.content_hash,
        created_at=utc_now_iso(),
    )
    session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return error_response(409, "duplicate_evidence")
    return evidence_to_out(record)


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/evidence",
    response_model=list[EvidenceOut],
)
def list_evidence(machine_id: str, event_id: str, session: SessionDep):
    event = get_machine_event(session, machine_id, event_id)
    if event is None:
        return error_response(404, "not_found")

    records = session.scalars(
        select(AuthorizationDecisionEvidence)
        .where(
            AuthorizationDecisionEvidence.machine_id == machine_id,
            AuthorizationDecisionEvidence.event_id == event_id,
        )
        .order_by(
            AuthorizationDecisionEvidence.created_at,
            AuthorizationDecisionEvidence.id,
        )
    ).all()
    return [evidence_to_out(record) for record in records]


class IncidentCreate(BaseModel):
    incident_type: NonEmptyStr
    summary: NonEmptyStr


class IncidentOut(BaseModel):
    id: str
    machine_id: str
    event_id: str
    incident_type: str
    summary: str
    status: str
    created_at: str


def incident_to_out(record: AuthorizationDecisionIncident) -> IncidentOut:
    return IncidentOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        incident_type=record.incident_type,
        summary=record.summary,
        status=record.status,
        created_at=record.created_at,
    )


@app.post(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
    status_code=201,
    response_model=IncidentOut,
)
def create_incident(
    machine_id: str,
    event_id: str,
    body: IncidentCreate,
    session: SessionDep,
):
    """Register one persistent exception-handling incident on a decision event.

    Body validation runs before any path lookup, so a malformed payload is a
    422 even when the machine or event does not exist. The write touches only
    the incidents table: the event, its evidence, hash chain, and causal links
    are never modified.
    """
    event = get_machine_event(session, machine_id, event_id)
    if event is None:
        return error_response(404, "not_found")

    existing = session.scalar(
        select(AuthorizationDecisionIncident).where(
            AuthorizationDecisionIncident.event_id == event_id,
            AuthorizationDecisionIncident.incident_type == body.incident_type,
            AuthorizationDecisionIncident.summary == body.summary,
        )
    )
    if existing is not None:
        return error_response(409, "duplicate_incident")

    record = AuthorizationDecisionIncident(
        id=str(uuid.uuid4()),
        machine_id=machine_id,
        event_id=event_id,
        incident_type=body.incident_type,
        summary=body.summary,
        status="open",
        created_at=utc_now_iso(),
    )
    session.add(record)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        return error_response(409, "duplicate_incident")
    return incident_to_out(record)


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents",
    response_model=list[IncidentOut],
)
def list_incidents(machine_id: str, event_id: str, session: SessionDep):
    event = get_machine_event(session, machine_id, event_id)
    if event is None:
        return error_response(404, "not_found")

    records = session.scalars(
        select(AuthorizationDecisionIncident)
        .where(
            AuthorizationDecisionIncident.machine_id == machine_id,
            AuthorizationDecisionIncident.event_id == event_id,
        )
        .order_by(
            AuthorizationDecisionIncident.created_at,
            AuthorizationDecisionIncident.id,
        )
    ).all()
    return [incident_to_out(record) for record in records]


class IncidentStatusUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]


class IncidentStatusEventOut(BaseModel):
    id: str
    machine_id: str
    event_id: str
    incident_id: str
    from_status: str
    to_status: str
    created_at: str


def incident_status_event_to_out(
    record: IncidentStatusEvent,
) -> IncidentStatusEventOut:
    return IncidentStatusEventOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        incident_id=record.incident_id,
        from_status=record.from_status,
        to_status=record.to_status,
        created_at=record.created_at,
    )


@app.post(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
    "{incident_id}/status",
    response_model=IncidentOut,
)
def transition_incident_status(
    machine_id: str,
    event_id: str,
    incident_id: str,
    body: IncidentStatusUpdate,
    session: SessionDep,
):
    """Advance one incident along ``open -> acknowledged -> resolved``.

    Body validation runs before any path lookup, so a missing, non-string, or
    otherwise invalid ``status`` is a 422 even when the machine, event, or
    incident does not exist. A missing machine, event, or incident, or an
    ownership mismatch, is a 404. Any transition other than ``open ->
    acknowledged`` and ``acknowledged -> resolved`` returns 409
    ``invalid_status_transition`` and writes nothing. On success the incident
    status update and one append-only history record are committed in a single
    transaction, and the updated incident is returned with status 200.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the chain appenders: concurrent transitions never hold two pool
    # connections at once.
    session.close()
    result = incidents.change_incident_status(
        engine,
        machine_id=machine_id,
        event_id=event_id,
        incident_id=incident_id,
        to_status=body.status,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    if result["status"] == "invalid_status_transition":
        return error_response(409, "invalid_status_transition")
    return IncidentOut(**result["incident"])


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
    "{incident_id}/status-history",
    response_model=list[IncidentStatusEventOut],
)
def list_incident_status_history(
    machine_id: str,
    event_id: str,
    incident_id: str,
    session: SessionDep,
):
    """Read-only, immutable history of one incident's status transitions.

    The machine, event, and incident must all exist and belong together; a
    missing one or an ownership mismatch returns 404 ``not_found``. Entries are
    returned in ``created_at``, then ``id`` order (``[]`` for an incident that
    has never moved). The query only reads: history records are never updated
    or deleted, and no other table is touched.
    """
    incident = incidents.get_machine_event_incident(
        session, machine_id, event_id, incident_id
    )
    if incident is None:
        return error_response(404, "not_found")

    records = incidents.list_status_history(session, machine_id, event_id, incident_id)
    return [incident_status_event_to_out(record) for record in records]


class ResponsibilityAssignmentCreate(BaseModel):
    party: NonEmptyStr
    role: NonEmptyStr


class ResponsibilityAssignmentOut(BaseModel):
    id: str
    machine_id: str
    event_id: str
    incident_id: str
    party: str
    role: str
    created_at: str
    previous_assignment_id: str | None
    content_hash: str
    chain_hash: str


def responsibility_assignment_to_out(
    record: IncidentResponsibilityAssignment,
) -> ResponsibilityAssignmentOut:
    return ResponsibilityAssignmentOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        incident_id=record.incident_id,
        party=record.party,
        role=record.role,
        created_at=record.created_at,
        previous_assignment_id=record.previous_assignment_id,
        content_hash=record.content_hash,
        chain_hash=record.chain_hash,
    )


@app.post(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
    "{incident_id}/responsibility-assignments",
    status_code=201,
    response_model=ResponsibilityAssignmentOut,
)
def create_responsibility_assignment(
    machine_id: str,
    event_id: str,
    incident_id: str,
    body: ResponsibilityAssignmentCreate,
    session: SessionDep,
):
    """Assign a responsible ``(party, role)`` pair to one registered incident.

    Body validation runs before any path lookup, so a missing, non-string, or
    blank ``party``/``role`` is a 422 even when the machine, event, or
    incident does not exist. A missing machine, event, or incident, or an
    ownership mismatch, is a 404. Assigning the same ``party`` and ``role`` to
    the same incident twice returns 409 ``duplicate_assignment`` and writes
    nothing. The new record is appended to the machine's assignment hash
    chain: the duplicate check, tail read, and insert commit in a single
    locked write transaction, so concurrent creators cannot lose records,
    fork the chain, or break a link. The incident, event, evidence, decision
    chain, and causal links are never modified.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the other chain appenders: concurrent creators never hold two
    # pool connections at once.
    session.close()
    result = assignment_chain.append_assignment(
        engine,
        machine_id=machine_id,
        event_id=event_id,
        incident_id=incident_id,
        party=body.party,
        role=body.role,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    if result["status"] == "duplicate_assignment":
        return error_response(409, "duplicate_assignment")
    return ResponsibilityAssignmentOut(**result["assignment"])


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/incidents/"
    "{incident_id}/responsibility-assignments",
    response_model=list[ResponsibilityAssignmentOut],
)
def list_responsibility_assignments(
    machine_id: str,
    event_id: str,
    incident_id: str,
    session: SessionDep,
):
    """Read-only list of one incident's responsibility assignments.

    The machine, event, and incident must all exist and belong together; a
    missing one or an ownership mismatch returns 404 ``not_found``. Entries are
    returned in ``created_at``, then ``id`` order (``[]`` for an incident with
    no assignments). The query only reads: it never writes, updates, or deletes
    any record, and assignments of another incident or machine are never
    returned.
    """
    incident = incidents.get_machine_event_incident(
        session, machine_id, event_id, incident_id
    )
    if incident is None:
        return error_response(404, "not_found")

    records = session.scalars(
        select(IncidentResponsibilityAssignment)
        .where(
            IncidentResponsibilityAssignment.machine_id == machine_id,
            IncidentResponsibilityAssignment.event_id == event_id,
            IncidentResponsibilityAssignment.incident_id == incident_id,
        )
        .order_by(
            IncidentResponsibilityAssignment.created_at,
            IncidentResponsibilityAssignment.id,
        )
    ).all()
    return [responsibility_assignment_to_out(record) for record in records]


class ResponsibilityAssignmentIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_assignment_id: str | None


@app.get(
    "/machines/{machine_id}/responsibility-assignments/integrity",
    response_model=ResponsibilityAssignmentIntegrityOut,
)
def check_responsibility_assignment_integrity(machine_id: str, session: SessionDep):
    """Read-only verification of one machine's responsibility-assignment chain.

    Returns ``{valid, checked_count, broken_assignment_id}``: an empty or
    fully sound chain reports ``true``, the machine's total assignment count,
    and ``null``; otherwise the first record whose content hash,
    previous-assignment link, or chain hash does not verify is reported. Only
    the path machine's records are examined, and the query never writes,
    repairs, or deletes, so repeated calls and restarts return stable results.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    valid, checked_count, broken_assignment_id = assignment_chain.verify_chain(
        session, machine_id
    )
    return ResponsibilityAssignmentIntegrityOut(
        valid=valid,
        checked_count=checked_count,
        broken_assignment_id=broken_assignment_id,
    )


class ResponsibilityAssignmentComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    assignments: list[ResponsibilityAssignmentOut]


@app.get(
    "/machines/{machine_id}/responsibility-assignments/compliance-export",
    response_model=ResponsibilityAssignmentComplianceExportOut,
)
def export_responsibility_assignments_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export of one machine's responsibility assignments
    over a time window.

    Includes every assignment owned by the path machine whose ``created_at``
    falls within the inclusive bounds, in ``(created_at, id)`` order, with
    exactly the fields of the assignment list endpoint. Records are exported
    exactly as stored: a missing, foreign, or damaged event or incident
    reference, field value, or chain link never causes a record to be
    rewritten, filtered out, or repaired. The endpoint only issues reads and
    never returns another machine's assignments.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load in the same (created_at, id) order as the list endpoint, then apply
    # the closed window to parsed instants: a stored exact-second ISO stamp
    # (no fractional part) would not compare correctly lexicographically
    # against a bound carrying a fractional part.
    machine_assignments = session.scalars(
        select(IncidentResponsibilityAssignment)
        .where(IncidentResponsibilityAssignment.machine_id == machine_id)
        .order_by(
            IncidentResponsibilityAssignment.created_at,
            IncidentResponsibilityAssignment.id,
        )
    ).all()
    records = [
        record
        for record in machine_assignments
        if window_start <= parse_utc_z_datetime(record.created_at) <= window_end
    ]

    return ResponsibilityAssignmentComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        assignments=[responsibility_assignment_to_out(record) for record in records],
    )


class IncidentIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_incident_id: str | None


# The exact status-history sequences that match an incident's current status.
# History is ordered by (created_at, id); each pair is (from_status, to_status).
_EXPECTED_INCIDENT_HISTORY = {
    "open": [],
    "acknowledged": [("open", "acknowledged")],
    "resolved": [("open", "acknowledged"), ("acknowledged", "resolved")],
}


def find_broken_incident(
    session: Session, machine_id: str
) -> tuple[int, str | None]:
    """Audit one machine's incidents for lifecycle and responsibility closure.

    Returns ``(total_count, broken_incident_id)``. Incidents are scanned in
    ``(created_at, id)`` order. An incident is broken when:

    * its ``event_id`` does not resolve to an existing decision event owned by
      the path machine;
    * its status history does not exactly match the status: ``open`` has no
      history, ``acknowledged`` has exactly ``open -> acknowledged``, and
      ``resolved`` has that edge followed by ``acknowledged -> resolved``;
    * a history record's ``machine_id``, ``event_id``, or ``incident_id`` does
      not match the incident;
    * a responsibility record's ownership triple does not match, its ``party``
      or ``role`` is empty after trimming surrounding whitespace, or its
      trimmed ``(party, role)`` pair duplicates another record;
    * a ``resolved`` incident has no valid responsibility record (the
      responsibility closure is open).

    Read-only: it issues no writes and never normalizes, repairs, or deletes a
    bad value.
    """
    incidents = session.scalars(
        select(AuthorizationDecisionIncident)
        .where(AuthorizationDecisionIncident.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionIncident.created_at,
            AuthorizationDecisionIncident.id,
        )
    ).all()

    machine_event_ids = set(
        session.scalars(
            select(AuthorizationDecisionEvent.id).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    )

    for incident in incidents:
        if incident.event_id not in machine_event_ids:
            return len(incidents), incident.id

        history = session.scalars(
            select(IncidentStatusEvent)
            .where(IncidentStatusEvent.incident_id == incident.id)
            .order_by(IncidentStatusEvent.created_at, IncidentStatusEvent.id)
        ).all()
        expected_edges = _EXPECTED_INCIDENT_HISTORY.get(incident.status)
        edges = [(entry.from_status, entry.to_status) for entry in history]
        if expected_edges is None or edges != expected_edges or any(
            entry.machine_id != machine_id
            or entry.event_id != incident.event_id
            or entry.incident_id != incident.id
            for entry in history
        ):
            return len(incidents), incident.id

        assignments = session.scalars(
            select(IncidentResponsibilityAssignment)
            .where(IncidentResponsibilityAssignment.incident_id == incident.id)
            .order_by(
                IncidentResponsibilityAssignment.created_at,
                IncidentResponsibilityAssignment.id,
            )
        ).all()
        valid_pairs: set[tuple[str, str]] = set()
        for assignment in assignments:
            party = assignment.party.strip() if isinstance(assignment.party, str) else ""
            role = assignment.role.strip() if isinstance(assignment.role, str) else ""
            if (
                assignment.machine_id != machine_id
                or assignment.event_id != incident.event_id
                or assignment.incident_id != incident.id
                or not party
                or not role
            ):
                return len(incidents), incident.id
            if (party, role) in valid_pairs:
                return len(incidents), incident.id
            valid_pairs.add((party, role))

        # The responsibility closure is complete only once a resolved incident
        # has at least one sound (party, role) attribution.
        if incident.status == "resolved" and not valid_pairs:
            return len(incidents), incident.id

    return len(incidents), None


@app.get(
    "/machines/{machine_id}/authorization-decision-events/incidents/integrity",
    response_model=IncidentIntegrityOut,
)
def check_incident_integrity(machine_id: str, session: SessionDep):
    """Read-only lifecycle and responsibility-closure audit of one machine's
    registered incidents.

    Returns ``{valid, checked_count, broken_incident_id}``: no incidents or all
    sound incidents report ``true``, the machine's total incident count, and
    ``null``; otherwise the first incident failing the event-reference,
    status-history, ownership, non-blank/unique party-role, or
    resolved-has-responsibility check is reported. Only incidents owned by the
    path machine are examined, and the query never writes, repairs, or deletes.
    A missing machine returns ``404 not_found``.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    checked_count, broken_incident_id = find_broken_incident(session, machine_id)
    return IncidentIntegrityOut(
        valid=broken_incident_id is None,
        checked_count=checked_count,
        broken_incident_id=broken_incident_id,
    )


class IncidentComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    incidents: list[IncidentOut]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/incidents/compliance-export",
    response_model=IncidentComplianceExportOut,
)
def export_incidents_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export of one machine's incidents over a time window.

    Includes every incident owned by the path machine whose ``created_at``
    falls within the inclusive bounds, in ``(created_at, id)`` order, with
    exactly the fields of the incident list endpoint. Incidents are exported
    exactly as stored: a damaged, missing, or foreign event reference, status
    history, or responsibility record never causes an incident to be rewritten,
    filtered out, or repaired. The endpoint only issues reads and never returns
    another machine's incidents.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load in the same (created_at, id) order as the list endpoint, then apply
    # the closed window to parsed instants: a stored exact-second ISO stamp
    # (no fractional part) would not compare correctly lexicographically
    # against a bound carrying a fractional part.
    machine_incidents = session.scalars(
        select(AuthorizationDecisionIncident)
        .where(AuthorizationDecisionIncident.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionIncident.created_at,
            AuthorizationDecisionIncident.id,
        )
    ).all()
    records = [
        record
        for record in machine_incidents
        if window_start <= parse_utc_z_datetime(record.created_at) <= window_end
    ]

    return IncidentComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        incidents=[incident_to_out(record) for record in records],
    )


class IncidentStatusHistoryComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    status_history: list[IncidentStatusEventOut]


@app.get(
    "/machines/{machine_id}/incident-status-events/compliance-export",
    response_model=IncidentStatusHistoryComplianceExportOut,
)
def export_incident_status_history_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only machine-level compliance export of incident status history.

    Includes every incident status event owned by the path machine whose
    ``created_at`` falls within the inclusive bounds, in ``(created_at, id)``
    order, with exactly the fields of the per-incident status-history endpoint.
    Records are exported exactly as stored: a missing, foreign, or damaged
    incident or event reference or a corrupt status edge never causes a record
    to be rewritten, filtered out, or repaired. The endpoint only issues reads
    and never returns another machine's status history.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load in the same (created_at, id) order as the status-history endpoint,
    # then apply the closed window to parsed instants: a stored exact-second ISO
    # stamp (no fractional part) would not compare correctly lexicographically
    # against a bound carrying a fractional part.
    machine_records = session.scalars(
        select(IncidentStatusEvent)
        .where(IncidentStatusEvent.machine_id == machine_id)
        .order_by(IncidentStatusEvent.created_at, IncidentStatusEvent.id)
    ).all()
    records = [
        record
        for record in machine_records
        if window_start <= parse_utc_z_datetime(record.created_at) <= window_end
    ]

    return IncidentStatusHistoryComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        status_history=[incident_status_event_to_out(record) for record in records],
    )


# Evidence fingerprints are checked exactly as stored: 64 characters drawn
# only from lowercase hexadecimal. The pattern never case-folds, so an
# uppercase fingerprint fails verification instead of being normalized away.
_EVIDENCE_LOWER_HEX_HASH_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_evidence_id: str | None


def find_broken_evidence(
    session: Session, machine_id: str
) -> tuple[int, str | None]:
    """Scan one machine's evidence records in (created_at, id) order.

    Returns ``(total_count, broken_evidence_id)``. A record is broken when its
    ``event_id`` does not resolve to an existing authorization decision event
    owned by the path machine (missing or owned by another machine), when
    ``evidence_type`` is not a string that stays non-empty after trimming
    surrounding whitespace, or when ``content_hash`` is not exactly 64
    lowercase hexadecimal characters compared as stored. Read-only: it issues
    no writes and never normalizes, repairs, or deletes a bad value.
    """
    records = session.scalars(
        select(AuthorizationDecisionEvidence)
        .where(AuthorizationDecisionEvidence.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionEvidence.created_at,
            AuthorizationDecisionEvidence.id,
        )
    ).all()

    machine_event_ids = set(
        session.scalars(
            select(AuthorizationDecisionEvent.id).where(
                AuthorizationDecisionEvent.machine_id == machine_id
            )
        ).all()
    )

    for record in records:
        if (
            record.event_id not in machine_event_ids
            or not isinstance(record.evidence_type, str)
            or not record.evidence_type.strip()
            or not isinstance(record.content_hash, str)
            or _EVIDENCE_LOWER_HEX_HASH_RE.fullmatch(record.content_hash) is None
        ):
            return len(records), record.id

    return len(records), None


@app.get(
    "/machines/{machine_id}/authorization-decision-events/evidence/integrity",
    response_model=EvidenceIntegrityOut,
)
def check_evidence_integrity(machine_id: str, session: SessionDep):
    """Read-only integrity audit of one machine's evidence records.

    Returns ``{valid, checked_count, broken_evidence_id}``: no records or all
    sound records report ``true``, the machine's total evidence count, and
    ``null``; otherwise the first record failing the event reference,
    evidence-type, or exact lowercase-hex-content-hash check is reported. Only
    evidence owned by the path machine is examined, so another machine's
    damaged records can never fail this machine's audit. The query never
    writes, repairs, or deletes evidence, events, chains, or links.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    checked_count, broken_evidence_id = find_broken_evidence(session, machine_id)
    return EvidenceIntegrityOut(
        valid=broken_evidence_id is None,
        checked_count=checked_count,
        broken_evidence_id=broken_evidence_id,
    )


class EvidenceComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    evidence: list[EvidenceOut]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/evidence/compliance-export",
    response_model=EvidenceComplianceExportOut,
)
def export_evidence_compliance(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export of one machine's evidence over a time window.

    Includes every evidence record owned by the path machine whose
    ``created_at`` falls within the inclusive bounds, in ``(created_at, id)``
    order, with exactly the fields of the evidence list endpoint. Records are
    exported exactly as stored: a damaged or foreign ``event_id`` never causes
    a record to be rewritten, filtered out, or repaired. The endpoint only
    issues reads and never returns another machine's evidence.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load in the same (created_at, id) order as the list endpoint, then apply
    # the closed window to parsed instants: a stored exact-second ISO stamp
    # (no fractional part) would not compare correctly lexicographically
    # against a bound carrying a fractional part.
    machine_evidence = session.scalars(
        select(AuthorizationDecisionEvidence)
        .where(AuthorizationDecisionEvidence.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionEvidence.created_at,
            AuthorizationDecisionEvidence.id,
        )
    ).all()
    records = [
        record
        for record in machine_evidence
        if window_start <= parse_utc_z_datetime(record.created_at) <= window_end
    ]

    return EvidenceComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        evidence=[evidence_to_out(record) for record in records],
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
def export_authorization_decision_events(
    machine_id: str,
    params: Annotated[ComplianceExportParams, Depends(validate_compliance_export_params)],
    session: SessionDep,
):
    """Read-only compliance export for one machine over a closed time window.

    Includes the machine's authorization decision events whose ``created_at``
    falls within the inclusive bounds and the machine's causal links whose two
    endpoints are both among the exported events. The endpoint only issues
    reads: it never writes, repairs, or deletes records, and never returns data
    owned by another machine.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Load in the same (created_at, id) order as the list endpoint, then apply
    # the closed window to parsed instants: a stored exact-second ISO stamp
    # (no fractional part) would not compare correctly lexicographically
    # against a bound carrying a fractional part.
    machine_events = session.scalars(
        select(AuthorizationDecisionEvent)
        .where(AuthorizationDecisionEvent.machine_id == machine_id)
        .order_by(
            AuthorizationDecisionEvent.created_at, AuthorizationDecisionEvent.id
        )
    ).all()
    events = [
        event
        for event in machine_events
        if window_start <= parse_utc_z_datetime(event.created_at) <= window_end
    ]
    event_id_set = {event.id for event in events}

    if event_id_set:
        links = session.scalars(
            select(AuthorizationDecisionCausalLink)
            .where(
                AuthorizationDecisionCausalLink.machine_id == machine_id,
                AuthorizationDecisionCausalLink.cause_event_id.in_(event_id_set),
                AuthorizationDecisionCausalLink.effect_event_id.in_(event_id_set),
            )
            .order_by(
                AuthorizationDecisionCausalLink.created_at,
                AuthorizationDecisionCausalLink.id,
            )
        ).all()
    else:
        links = []

    return ComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        events=[decision_event_to_out(event) for event in events],
        causal_links=[causal_link_to_out(link) for link in links],
    )


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
