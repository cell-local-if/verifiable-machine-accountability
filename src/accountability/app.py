import os
import bisect
import hashlib
import json
import re
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import (
    assignment_chain,
    authorization,
    chain,
    diagnostics,
    evidence_chain,
    incidents,
    machines,
    policy_conflicts,
    policy_preview,
    policy_rule_chain,
    privacy_chain,
    rotation_chain,
)
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
    PrivacyAccess,
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
    privacy_chain.migrate_schema(engine)
    privacy_chain.backfill_chains(engine)
    evidence_chain.migrate_schema(engine)
    evidence_chain.backfill_chains(engine)
    policy_rule_chain.migrate_schema(engine)
    policy_rule_chain.backfill_chains(engine)
    diagnostics.migrate_schema(engine)
    # Finalize joint-write markers left by a crashed previous process from
    # the committed evidence, before serving any request.
    diagnostics.recover_pending(engine)
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


class QueryError(Exception):
    """A query-string failure reported as ``422 {"error":{"code": ...}}``."""

    def __init__(self, code: str):
        self.code = code


@app.exception_handler(QueryError)
def _query_error_handler(request: Request, exc: QueryError) -> JSONResponse:
    return error_response(422, exc.code)


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


def validate_machine_status_history_params(request: Request) -> None:
    """Validate the machine status-history query string.

    The history query is keyed on the path machine alone and accepts no
    business filter parameters; any parameter name is a 422 ``invalid_query``.
    The check runs before the machine is looked up and issues no database
    access, so an extra parameter against a non-existent machine still
    reports 422 rather than 404.
    """
    if request.query_params:
        raise QueryError("invalid_query")


@app.get("/machines/{machine_id}/status-history")
def list_machine_status_history(
    machine_id: str,
    _: Annotated[None, Depends(validate_machine_status_history_params)],
    session: SessionDep,
):
    """Read-only, immutable history of one machine's status transitions.

    The caller submits only the path machine id — no time range, business
    filter, or request body; any query parameter is a 422 ``invalid_query``
    raised before the machine is ever looked up. A missing machine is a 404
    ``not_found`` carrying no history data. Only ``GET`` is routed; other
    methods return 405 without reading records, computing a result, or
    writing anything.

    On success the response is an array — empty when the machine has never
    changed status — of the machine's own transition records, ordered by the
    actual UTC instant of ``created_at`` and then by id, so an exact-second
    record sorts before any fractional-second record of the same second. Each
    item carries exactly ``{id, machine_id, from_status, to_status,
    created_at}`` in this fixed field order, with ``created_at`` the UTC
    commit-moment stamp ending in ``Z``. Records are returned exactly as
    stored: they are never rewritten, recomputed, filtered out, or repaired,
    and another machine's records can never enter the result. The query only
    issues reads — it never creates, updates, deletes, repairs, or normalizes
    status, history, or any other accountability record — and the body is
    compact UTF-8 JSON terminated by a single newline, free of any
    floating-point or non-finite value, byte-identical on repeat calls
    against unchanged data, including data persisted across application
    restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    # Ordering parses stamps to UTC instants because an exact-second stamp
    # sorts before a fractional stamp of the same second only after parsing
    # (lexicographically '.' precedes 'Z').
    records = order_by_created_at_instant(
        machines.list_status_history(session, machine_id)
    )
    payload = [
        {
            "id": record.id,
            "machine_id": record.machine_id,
            "from_status": record.from_status,
            "to_status": record.to_status,
            "created_at": record.created_at,
        }
        for record in records
    ]
    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a
    # fixed field order, terminated by a single newline, and free of any
    # floating-point or non-finite value (allow_nan=False).
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


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


class PolicyRuleChainOut(PolicyRuleOut):
    # The creation response and the chain view add the tamper-evident chain
    # fields; the plain listing and compliance export keep the 7-field shape.
    previous_rule_id: str | None
    content_hash: str
    chain_hash: str


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


@app.post("/policy-rules", status_code=201, response_model=PolicyRuleChainOut)
def create_policy_rule(body: PolicyRuleCreate, session: SessionDep):
    """Create one global policy rule and append it to the rule chain tail.

    Body validation runs before any write, so a malformed payload (missing or
    blank action/resource, an effect other than ``allow``/``deny``, a negative
    or non-integer priority) is a 422 and writes neither a rule nor a chain
    link. The duplicate ``(action_type, resource_pattern, priority)`` check,
    the chain-tail read, and the rule insert run in one locked write
    transaction: a duplicate business triple is a 409
    ``duplicate_policy_rule`` and writes nothing, while concurrent successful
    creations cannot lose rules, skip a link, fork the chain, or point two
    rules at the same predecessor. On success the rule is returned with 201
    carrying its visible fields together with ``previous_rule_id`` (``null``
    for the first rule), its 64-lowercase-hex ``content_hash`` over those
    fields, and its ``chain_hash``; these are exactly the values later shown
    by ``GET /policy-rules/chain``.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the other chain appenders: concurrent creations never hold two
    # pool connections at once.
    session.close()
    result = policy_rule_chain.append_policy_rule(
        engine,
        action_type=body.action_type,
        resource_pattern=body.resource_pattern,
        effect=body.effect,
        priority=body.priority,
    )
    if result["status"] == "duplicate_policy_rule":
        return error_response(409, "duplicate_policy_rule")
    return PolicyRuleChainOut(**result["rule"])


@app.get("/policy-rules", response_model=list[PolicyRuleOut])
def list_policy_rules(session: SessionDep):
    """Read-only listing of every global policy rule.

    Returns all rules ordered by the actual UTC instant of ``created_at``,
    then by ``id``; a table with no rules yields ``[]``. Each item carries
    exactly the persisted ``{id, action_type, resource_pattern, effect,
    priority, created_at, updated_at}`` values, with no normalization or
    repair and no chain fields (those live on the creation response and
    ``GET /policy-rules/chain``). ISO-8601 text ordering is not chronological
    once fractional seconds are present (``...:00.5Z`` sorts before
    ``...:00Z`` because ``.`` precedes ``Z``), so stamps are parsed to UTC
    instants before the id tie-break. A stored ``created_at`` that no longer
    parses never crashes the listing: it deterministically sorts after every
    parseable instant while its stored text is emitted untouched. The query
    only reads: it never writes, updates, deletes, normalizes, or repairs a
    rule, and it never participates in authorization evaluation, so repeated
    calls are stable and rules remain queryable across restarts.
    """
    rules = session.scalars(select(PolicyRule)).all()
    records = sorted(
        rules,
        key=lambda rule: (_policy_rule_created_instant(rule.created_at), rule.id),
    )
    return [policy_rule_to_out(rule) for rule in records]


def validate_policy_rule_chain_query_params(request: Request) -> None:
    """Validate the policy-rule chain/integrity/conflicts query string.

    The read-only chain, integrity, and conflict-analysis endpoints are keyed
    on the global rule table alone and accept no filter parameters; any
    parameter name is a 422 ``invalid_query``. The check runs during
    validation, before any rule is read and without any database access, so an
    extra parameter is rejected identically against an empty database and
    never changes into another error type.
    """
    if request.query_params:
        raise QueryError("invalid_query")


@app.get("/policy-rules/chain")
def list_policy_rule_chain(
    _: Annotated[None, Depends(validate_policy_rule_chain_query_params)],
    session: SessionDep,
):
    """Read-only view of the complete global policy-rule hash chain.

    The caller submits no machine path and no filter parameters; any query
    parameter is a 422 ``invalid_query`` raised before any rule is read, and
    non-GET methods return 405 without reading rule content. The response is
    an array — empty when no rules exist — of every global rule in the chain
    order used to build and verify it: the actual UTC instant of
    ``created_at`` and then ``id``, so an exact-second rule sorts before any
    fractional-second rule of the same second. Each item carries the rule's
    visible fields ``{id, action_type, resource_pattern, effect, priority,
    created_at, updated_at}`` exactly as stored plus ``previous_rule_id``
    (``null`` only on the first rule), ``content_hash`` (the SHA-256 of the
    compact key-sorted JSON of the visible fields), and ``chain_hash``
    (``sha256("<previous chain_hash>:<content_hash>")`` from the empty-prefix
    root), all 64 lowercase hexadecimal characters. The values are identical
    to those returned at creation. A stored ``created_at`` that no longer
    parses never crashes the query: it sorts after every parseable instant, is
    still returned, and keeps its stored text. The query is strictly
    read-only — it never creates, updates, deletes, repairs, recomputes, or
    normalizes a rule — and the body is compact UTF-8 JSON terminated by a
    single newline, byte-identical on repeat calls against unchanged data,
    including data persisted across restarts.
    """
    rows = policy_rule_chain.ordered_rules(session)
    payload = [
        {
            "id": row._mapping["id"],
            "action_type": row._mapping["action_type"],
            "resource_pattern": row._mapping["resource_pattern"],
            "effect": row._mapping["effect"],
            "priority": row._mapping["priority"],
            "created_at": row._mapping["created_at"],
            "updated_at": row._mapping["updated_at"],
            "previous_rule_id": row._mapping["previous_rule_id"],
            "content_hash": row._mapping["content_hash"],
            "chain_hash": row._mapping["chain_hash"],
        }
        for row in rows
    ]
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


# A stored ``created_at`` that no longer parses is ordered after every
# parseable record (the same tolerant convention the evidence chain uses), so
# a damaged stamp sorts deterministically last instead of crashing a read-only
# audit; such a degenerate instant can never fall inside a finite window.
_POLICY_RULE_FAR_FUTURE = datetime.max.replace(tzinfo=timezone.utc)


def _policy_rule_created_instant(value: object) -> datetime:
    """Parse a stored policy-rule ``created_at`` to its UTC instant.

    Legitimately written values always satisfy the RFC 3339 ``Z`` contract; a
    damaged value sorts after every parseable record instead of raising, so the
    read-only listing and window export neither crash nor repair the stored
    text.
    """
    if isinstance(value, str) and _RFC3339_Z_DATETIME_RE.fullmatch(value):
        try:
            return parse_utc_z_datetime(value)
        except ValueError:
            pass
    return _POLICY_RULE_FAR_FUTURE


class AuthorizationEvaluationCreate(BaseModel):
    action_type: NonEmptyStr
    resource: NonEmptyStr


class AuthorizationEvaluationOut(BaseModel):
    allowed: bool
    reason: str


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
    allowed, reason = authorization.decide(
        session,
        machine_id,
        machine.status,
        body.action_type,
        body.resource,
    )
    return AuthorizationEvaluationOut(allowed=allowed, reason=reason)


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
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the other writers: concurrent requests never hold two pool
    # connections at once. Body validation (422) has already run before this
    # handler, so the only lookup failure below is a missing machine.
    session.close()
    # The machine/status lookup, the suspended/declaration/policy decision,
    # and the chain-tail append run in one locked write transaction, the same
    # lock the status-change endpoint takes. A status change and this append
    # therefore commit in one definite serial order: when the status change
    # commits first the event is decided against the new status (a suspension
    # yields machine_suspended, never a stale active-era result), and when the
    # append commits first the later status change never rewrites the stored
    # event. No event can be lost, forked, or half-written.
    result = chain.append_decision_event(
        engine,
        machine_id=machine_id,
        action_type=body.action_type,
        resource=body.resource,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    return AuthorizationDecisionEventOut(**result["event"])


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


def validate_accountability_export_params(
    request: Request,
) -> ComplianceExportParams:
    """Validate the machine-level accountability export query string.

    Exactly two parameters are accepted: ``from_created_at`` and
    ``to_created_at``, both required UTC RFC 3339 date-times ending in ``Z``
    (fractional seconds optional; offset forms, surrounding whitespace, and
    non-``Z`` suffixes are rejected), with the lower bound not later than the
    upper bound (equal bounds allowed). Any other parameter name is a 422
    ``invalid_query``; a missing, blank, malformed, or inverted bound is a 422
    ``bad_time``. Validation runs entirely before the machine is looked up and
    issues no database access, so an invalid query against a non-existent
    machine still reports 422 rather than 404 and never reads machine data.
    """
    allowed = {"from_created_at", "to_created_at"}
    unknown = [name for name in request.query_params if name not in allowed]
    if unknown:
        raise QueryError("invalid_query")

    raw_from = request.query_params.get("from_created_at")
    raw_to = request.query_params.get("to_created_at")

    def _valid(value: str | None) -> bool:
        if not value or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
            return False
        try:
            parse_utc_z_datetime(value)
        except ValueError:
            return False
        return True

    if not _valid(raw_from) or not _valid(raw_to):
        raise QueryError("bad_time")

    if parse_utc_z_datetime(raw_from) > parse_utc_z_datetime(raw_to):
        raise QueryError("bad_time")

    return ComplianceExportParams(
        from_created_at=raw_from,  # type: ignore[arg-type]
        to_created_at=raw_to,  # type: ignore[arg-type]
    )


@app.get("/policy-rules/compliance-export")
def export_policy_rules_compliance(
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only compliance export of global policy rules over a time window.

    The query accepts exactly the two required bounds ``from_created_at``/
    ``to_created_at`` — UTC RFC 3339 date-times ending in ``Z`` (fractional
    seconds optional, equal bounds allowed); any other query parameter is a
    422 ``invalid_query`` and a missing, blank, offset, whitespace-padded,
    malformed, out-of-range, or inverted bound is a 422 ``bad_time``. Both
    checks run during validation, before any policy rule is read and without
    any database access. Only ``GET`` is routed; other methods return 405
    without filtering, ordering, reading rule content, or writing anything.

    On success the response carries exactly ``{from_created_at,
    to_created_at, policy_rules}`` in this fixed field order; the bounds are
    echoed verbatim and ``policy_rules`` is always present, an empty array when
    the window contains nothing (including an empty database). The array holds
    only global rules whose own ``created_at`` falls in the closed UTC
    interval, ordered by the actual UTC instant of ``created_at`` and then by
    id, so an exact-second rule sorts before any fractional-second rule of the
    same second. Each item exposes exactly the policy-rule list fields ``{id,
    action_type, resource_pattern, effect, priority, created_at,
    updated_at}`` with stored values verbatim — never filtered, repaired, or
    normalized for missing, illegal, or duplicated data — and no key, secret,
    or extra policy text is added. A stored ``created_at`` that no longer
    parses never crashes the export: it deterministically sorts after every
    parseable instant (and so never falls inside a finite window) while its
    stored text is left untouched. The query is strictly read-only — it never
    creates, updates, deletes, repairs, recomputes, or normalizes a rule and
    never changes an authorization-evaluation result; the body is compact
    UTF-8 JSON with a single trailing newline, contains no floating-point,
    ``-0.0``, or non-finite value, is byte-identical on repeat calls against
    unchanged data, and reads rules persisted across application restarts.
    """
    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    rules = session.scalars(select(PolicyRule)).all()
    in_window = [
        rule
        for rule in rules
        if window_start <= _policy_rule_created_instant(rule.created_at) <= window_end
    ]
    records = sorted(
        in_window,
        key=lambda rule: (_policy_rule_created_instant(rule.created_at), rule.id),
    )

    payload = {
        "from_created_at": params.from_created_at,
        "to_created_at": params.to_created_at,
        "policy_rules": [
            {
                "id": rule.id,
                "action_type": rule.action_type,
                "resource_pattern": rule.resource_pattern,
                "effect": rule.effect,
                "priority": rule.priority,
                "created_at": rule.created_at,
                "updated_at": rule.updated_at,
            }
            for rule in records
        ],
    }
    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a fixed
    # field order, terminated by a single newline, and free of any
    # floating-point or non-finite value (allow_nan=False).
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


class PolicyRuleIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_policy_rule_id: str | None


@app.get("/policy-rules/integrity", response_model=PolicyRuleIntegrityOut)
def check_policy_rule_integrity(
    _: Annotated[None, Depends(validate_policy_rule_chain_query_params)],
    session: SessionDep,
):
    """Read-only verification of the global policy-rule hash chain.

    The caller submits no machine path and no filter parameters; any query
    parameter is a 422 ``invalid_query`` raised during validation before any
    rule is read, and non-GET methods return 405 without reading rule content.
    Returns ``{valid, checked_count, broken_policy_rule_id}``: an empty rule
    table reports ``true``, zero, and ``null``; ``checked_count`` always
    counts every global rule, even one whose stored ``created_at`` no longer
    parses (it is taken in last chain order rather than crashing the scan).
    Rules are examined in the chain order — the actual UTC instant of
    ``created_at`` and then ``id``, so an exact-second rule precedes any
    fractional-second rule of the same second — and the first rule whose
    stored ``content_hash``, ``previous_rule_id`` link, or ``chain_hash`` does
    not verify against the recomputed chain makes the conclusion ``false`` and
    is reported by its stored id, emitted verbatim; when every rule verifies,
    the broken id is ``null``. The query is strictly read-only: it never
    creates, updates, deletes, repairs, recomputes, or normalizes a rule and
    never changes an authorization-evaluation result, and repeated calls
    against unchanged data return byte-identical results.
    """
    valid, checked_count, broken_policy_rule_id = policy_rule_chain.verify_chain(
        session
    )
    return PolicyRuleIntegrityOut(
        valid=valid,
        checked_count=checked_count,
        broken_policy_rule_id=broken_policy_rule_id,
    )


@app.get("/policy-rules/conflicts")
def analyze_policy_rule_conflicts(
    _: Annotated[None, Depends(validate_policy_rule_chain_query_params)],
    session: SessionDep,
):
    """Read-only conflict and override analysis over the global policy rules.

    The caller submits no machine path and no filter parameters; any query
    parameter is a 422 ``invalid_query`` raised during validation before any
    rule is read and without any database access, and only ``GET`` is routed,
    so non-GET methods return 405 without reading rules, computing matches, or
    writing anything. An empty rule table returns three empty arrays.

    On success the response always carries exactly ``{rules, conflicts,
    overrides}`` in this fixed field order. ``rules`` keeps every stored
    global rule with the seven visible fields ``{id, action_type,
    resource_pattern, effect, priority, created_at, updated_at}`` emitted
    exactly as stored — illegal or damaged values are never repaired, deleted,
    or normalized — plus a ``relation`` annotation:

    - ``invalid`` — the stored action type or resource pattern is not a
      string, the effect is not exactly ``allow``/``deny``, or the priority is
      a boolean, non-integer, or negative; such a rule never takes part in any
      matching judgement. Damaged ``id`` or timestamp values neither
      invalidate a rule nor are repaired (matching never depends on them);
    - ``unmatched`` — a valid rule with no same-action candidate whose
      resource pattern has a common matching scope;
    - ``conflict`` — a valid rule in a same-action pair whose patterns
      intersect, at the same priority, with opposite effects;
    - ``override`` — a valid rule in a same-action intersecting pair whose
      priorities differ; ``overrides`` states which side covers which.

    Details are ordered by the actual UTC instant of ``created_at`` and then
    by id, so an exact-second record sorts before any fractional-second
    record of the same second and a damaged stamp sorts after every parseable
    one. Two valid rules are candidates only when their actions are equal and
    their resource patterns can match a common resource (the existing ``*``
    matches-any-text, everything-else-literal semantics); ``intersection`` is
    the glob describing exactly that shared scope. Each conflict entry is
    ``{rule_ids, intersection, reason}`` with both ids ascending and reason
    ``"same_priority_opposite_effect"``; each override entry is
    ``{overriding_rule_id, overridden_rule_id, intersection, reason}`` with
    the smaller numeric priority covering the larger and reason
    ``"lower_priority_overrides"``. Both lists sort by their pair of rule ids
    ascending, so repeat queries over unchanged data are byte-identical.

    A failure that prevents reading the rules returns 500
    ``internal_error`` with no partial analysis. The query is strictly
    read-only — it never creates, updates, deletes, repairs, recomputes, or
    normalizes a rule and never changes an authorization result — and the body
    is compact UTF-8 JSON terminated by a single newline, free of
    floating-point, ``-0.0``, or non-finite values, including data persisted
    across application restarts; old and empty databases work unchanged.
    """
    # A failure while *reading* the rules is an internal read-layer fault:
    # answer 500 internal_error with no partial analysis. Damaged business
    # fields are not a read failure — the rows were read — so the pure, total
    # analysis over them runs outside this guard and a content anomaly is
    # reported as a normal 200 (every damaged value surfaces verbatim), never
    # misclassified as an internal read error.
    try:
        stored_rules = policy_conflicts.load_rules(session)
    except Exception:
        return error_response(500, "internal_error")
    payload = policy_conflicts.build_analysis(stored_rules)

    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a fixed
    # field order, terminated by a single newline, and free of any
    # floating-point or non-finite computed value (allow_nan=False).
    body = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    return Response(content=body, media_type="application/json")


@app.post("/policy-rules/decision-preview")
async def preview_policy_rule_decision(request: Request):
    """Read-only preview of the global policy decision for one request.

    The body is a JSON object carrying exactly two string fields, ``action``
    and ``resource``; both stay non-empty after surrounding whitespace is
    stripped. Validation runs entirely before any rule is read:

    - any query parameter is ``422 invalid_query`` first of all;
    - a body that is not a JSON object, that lacks or adds a field, or that
      types either field wrongly is ``422 invalid_request``;
    - an action or resource empty after trimming is ``422 invalid_value``.

    Every malformed request is rejected identically against an empty rule
    table, and non-POST methods return ``405`` without reading rules. The
    preview reads global policy rules only — never machine declarations,
    machine status, or authorization events — and issues no writes.

    Every stored rule is retained in ``rules`` and annotated ``invalid``
    (an illegal stored field), ``unmatched``, ``winner``, ``overridden`` (a
    larger-priority matching candidate), or ``conflict`` (a same-minimum
    mixed allow/deny tier, which is decided as a denial). Rules sort by
    priority, then the actual UTC instant of ``created_at`` (damaged stamps
    last), then id. ``conflicts`` reports the mixed minimum tier as id-sorted
    pairs; ``winners`` lists the decisive rules ordered by
    ``(created_at instant, id)`` and is empty on a conflict. With no legal
    candidate the decision is ``{"allowed": false,
    "reason": "no_matching_policy"}`` and both groups are empty. A failure
    that prevents reading the rules returns ``500 internal_error`` with no
    partial preview. The body is compact UTF-8 JSON with one trailing newline.
    """
    # Query validation precedes body parsing and every database read.
    if request.query_params:
        return error_response(422, "invalid_query")

    try:
        payload = await request.json()
    except Exception:
        return error_response(422, "invalid_request")
    if not isinstance(payload, dict) or set(payload) != {"action", "resource"}:
        return error_response(422, "invalid_request")
    action_raw, resource_raw = payload["action"], payload["resource"]
    if isinstance(action_raw, bool) or not isinstance(action_raw, str):
        return error_response(422, "invalid_request")
    if isinstance(resource_raw, bool) or not isinstance(resource_raw, str):
        return error_response(422, "invalid_request")

    action = action_raw.strip()
    resource = resource_raw.strip()
    if not action or not resource:
        # Empty-after-trim is a value-domain failure, reported without ever
        # reading the rule table.
        return error_response(422, "invalid_value")

    try:
        with Session(request.app.state.engine) as session:
            stored_rules = policy_preview.load_rules(session)
            result = policy_preview.build_preview(action, resource, stored_rules)
    except Exception:
        # Never emit a partial preview when the rules cannot be read.
        return error_response(500, "internal_error")

    # Serialize by hand so the body is compact UTF-8 JSON in a fixed field
    # order, terminated by a single newline, and free of floating-point or
    # non-finite values (allow_nan=False).
    body = (
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    )
    return Response(content=body, media_type="application/json")


@app.post("/policy-rules/decision-preview/batch")
async def preview_policy_rule_decision_batch(request: Request):
    """Read-only batch preview of global policy decisions for many requests.

    The body is a JSON object carrying exactly one field, ``requests``, an
    array whose items each carry exactly the string fields ``action`` and
    ``resource``; the empty array is legal and yields an empty result. Every
    check runs entirely before any rule is read, and any single illegal item
    rejects the whole batch with no partial analysis:

    - any query parameter is ``422 invalid_query`` first of all;
    - a body that is not a JSON object, that lacks or adds a top-level field,
      whose ``requests`` is not an array, or whose items are not objects with
      exactly ``action`` and ``resource`` is ``422 invalid_batch``;
    - an item whose action or resource is not a string, or is empty after
      trimming, is ``422 invalid_value``.

    Non-POST methods return ``405`` without reading rules. All items are
    analyzed against one same-instant snapshot of the global rule table, read
    once; no input can change another item's result, and a failure that
    prevents reading the rules returns ``500 internal_error`` with no partial
    analysis. Each item applies the single-preview matching, conflict, winner,
    override, and final-decision semantics unchanged, including the relation
    annotations.

    The response is ``{batch_count, analyses, summary, decisions}`` in this
    fixed field order. ``batch_count`` is the number of submitted requests.
    ``analyses`` has one ``{input, result}`` entry per request in input
    order, where ``input`` echoes the trimmed ``{action, resource}`` and
    ``result`` is exactly the single-preview payload for that pair.
    ``summary`` counts, in the fixed key order ``no_match``, ``allow``,
    ``deny``, ``conflict``, ``override``: inputs with no matching candidate,
    inputs decided allow, inputs decided deny, inputs reporting a conflict
    group, and inputs with at least one overridden candidate. ``decisions``
    counts final decisions under the three existing reason keys
    ``no_matching_policy``, ``allowed_by_policy``, ``denied_by_policy``; empty
    categories stay present with the JSON integer zero. The query is strictly
    read-only — it never creates, updates, deletes, repairs, recomputes, or
    normalizes anything — and the body is compact UTF-8 JSON terminated by a
    single newline, free of floating-point or non-finite values,
    byte-identical on repeat calls against unchanged data, including data
    persisted across restarts.
    """
    # Query validation precedes body parsing and every database read.
    if request.query_params:
        return error_response(422, "invalid_query")

    try:
        payload = await request.json()
    except Exception:
        return error_response(422, "invalid_batch")
    if not isinstance(payload, dict) or set(payload) != {"requests"}:
        return error_response(422, "invalid_batch")
    requests_raw = payload["requests"]
    if isinstance(requests_raw, bool) or not isinstance(requests_raw, list):
        return error_response(422, "invalid_batch")

    # Structural item failures are invalid_batch; a well-shaped item whose
    # action/resource fails the value domain is invalid_value. Structure is
    # checked for every item before any value is (the single preview's
    # structural-before-value precedence, applied batch-wide), and either
    # failure rejects the whole batch before any rule is read.
    for item in requests_raw:
        if not isinstance(item, dict) or set(item) != {"action", "resource"}:
            return error_response(422, "invalid_batch")

    items: list[tuple[str, str]] = []
    for item in requests_raw:
        action_raw, resource_raw = item["action"], item["resource"]
        if isinstance(action_raw, bool) or not isinstance(action_raw, str):
            return error_response(422, "invalid_value")
        if isinstance(resource_raw, bool) or not isinstance(resource_raw, str):
            return error_response(422, "invalid_value")
        action = action_raw.strip()
        resource = resource_raw.strip()
        if not action or not resource:
            return error_response(422, "invalid_value")
        items.append((action, resource))

    try:
        with Session(request.app.state.engine) as session:
            # One snapshot of the rule table decides the whole batch, so no
            # item can observe or cause a change in another item's analysis.
            stored_rules = policy_preview.load_rules(session)
            results = [
                policy_preview.build_preview(action, resource, stored_rules)
                for action, resource in items
            ]
    except Exception:
        # Never emit a partial analysis when the rules cannot be read.
        return error_response(500, "internal_error")

    analyses = [
        {
            "input": {"action": action, "resource": resource},
            "result": result,
        }
        for (action, resource), result in zip(items, results)
    ]

    summary = {"no_match": 0, "allow": 0, "deny": 0, "conflict": 0, "override": 0}
    decisions = {
        "no_matching_policy": 0,
        "allowed_by_policy": 0,
        "denied_by_policy": 0,
    }
    for result in results:
        decision = result["decision"]
        decisions[decision["reason"]] += 1
        if decision["reason"] == "no_matching_policy":
            summary["no_match"] += 1
        if decision["allowed"]:
            summary["allow"] += 1
        else:
            summary["deny"] += 1
        if result["conflicts"]:
            summary["conflict"] += 1
        if any(rule["relation"] == "overridden" for rule in result["rules"]):
            summary["override"] += 1

    batch_payload = {
        "batch_count": len(items),
        "analyses": analyses,
        "summary": summary,
        "decisions": decisions,
    }
    # Serialize by hand so the body is compact UTF-8 JSON in a fixed field
    # order, terminated by a single newline, and free of floating-point or
    # non-finite values (allow_nan=False).
    body = (
        json.dumps(
            batch_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


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
    previous_evidence_id: str | None
    # Real registrations always carry a 64-hex chain digest; nullable only so
    # read-only exports can emit a pre-chain/external row verbatim as stored.
    chain_hash: str | None


def evidence_to_out(record: AuthorizationDecisionEvidence) -> EvidenceOut:
    return EvidenceOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        evidence_type=record.evidence_type,
        content_hash=record.content_hash,
        created_at=record.created_at,
        previous_evidence_id=record.previous_evidence_id,
        chain_hash=record.chain_hash,
    )


class AccountabilityEvidenceOut(BaseModel):
    # The aggregate accountability export keeps the pre-chain evidence shape;
    # the chain fields appear on the evidence create/list/export surfaces.
    id: str
    machine_id: str
    event_id: str
    evidence_type: str
    content_hash: str
    created_at: str


def accountability_evidence_to_out(
    record: AuthorizationDecisionEvidence,
) -> AccountabilityEvidenceOut:
    return AccountabilityEvidenceOut(
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
    422 even when the machine or event does not exist. The machine/event
    ownership lookup, the duplicate-fingerprint check, the insert, and the
    append to the machine's evidence hash chain
    (``previous_evidence_id``/``chain_hash``) all commit in a single locked
    write transaction, so concurrent registrations cannot lose records, fork
    the chain, skip a link, or point two records at the same predecessor. A
    missing machine or event, or an event owned by another machine, is a 404
    ``not_found`` and writes nothing; a repeated fingerprint on the same event
    is a 409 ``duplicate_evidence`` and writes nothing. The event, its hash
    chain, and causal links are never modified.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the other chain appenders: concurrent registrations never hold
    # two pool connections at once.
    session.close()
    result = evidence_chain.append_evidence(
        engine,
        machine_id=machine_id,
        event_id=event_id,
        evidence_type=body.evidence_type,
        content_hash=body.content_hash,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    if result["status"] == "duplicate_evidence":
        return error_response(409, "duplicate_evidence")
    return EvidenceOut(**result["evidence"])


@app.get(
    "/machines/{machine_id}/authorization-decision-events/{event_id}/evidence",
)
def list_evidence(machine_id: str, event_id: str, session: SessionDep):
    """Read-only list of one decision event's evidence records.

    The event must exist and belong to the path machine; a missing event or an
    event owned by another machine is a 404 ``not_found``. Returns only the
    path machine's records for the event, an empty array when there are none,
    ordered stably by ``created_at`` then ``id``. Each record carries the same
    chain fields as the registration response (``previous_evidence_id`` and
    ``chain_hash``) in the same positions. The query only reads, and the body
    is compact UTF-8 JSON terminated by a single newline, with no
    floating-point or non-finite values.
    """
    event = get_machine_event(session, machine_id, event_id)
    if event is None:
        return error_response(404, "not_found")

    rows = session.scalars(
        select(AuthorizationDecisionEvidence).where(
            AuthorizationDecisionEvidence.machine_id == machine_id,
            AuthorizationDecisionEvidence.event_id == event_id,
        )
    ).all()
    # Order by the actual UTC instant of created_at, then id — the same chain
    # order used by the export and chain verification (an exact-second stamp
    # sorts before a fractional stamp of the same second only after parsing).
    # The tolerant instant key keeps a tampered stamp sorting last instead of
    # raising, matching the audit's "count it and flag it" behavior.
    records = evidence_chain.order_evidence_rows(rows)
    payload = [
        {
            "id": record.id,
            "machine_id": record.machine_id,
            "event_id": record.event_id,
            "evidence_type": record.evidence_type,
            "content_hash": record.content_hash,
            "created_at": record.created_at,
            "previous_evidence_id": record.previous_evidence_id,
            "chain_hash": record.chain_hash,
        }
        for record in records
    ]
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


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


# --- read-only desensitized privacy responsibility compliance export --------


def privacy_reference_digest(kind: str, machine_id: str, raw: object) -> str | None:
    """Desensitizing digest for one responsibility chain field.

    The three segments ``privacy:v1|<kind>``, the path machine id, and the
    stored value with surrounding whitespace removed are joined directly (no
    separator beyond the one inside the prefix) and hashed as UTF-8 with
    SHA-256, yielding 64 lowercase hexadecimal characters. ``kind`` is
    ``party`` or ``role``. When the stored value is not a string or is empty
    after stripping surrounding whitespace, the digest is ``null`` so the raw
    value is never emitted.
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    message = f"privacy:v1|{kind}{machine_id}{value}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


class PrivacyResponsibilityAssignmentOut(BaseModel):
    id: str
    machine_id: str
    event_id: str
    incident_id: str
    created_at: str
    previous_assignment_id: str | None
    content_hash: str
    chain_hash: str
    party_ref: str | None
    role_ref: str | None


def privacy_responsibility_assignment_to_out(
    record: IncidentResponsibilityAssignment,
) -> PrivacyResponsibilityAssignmentOut:
    # The raw party/role never leave the service: only their desensitizing
    # digests are emitted. Every other field is emitted exactly as stored.
    return PrivacyResponsibilityAssignmentOut(
        id=record.id,
        machine_id=record.machine_id,
        event_id=record.event_id,
        incident_id=record.incident_id,
        created_at=record.created_at,
        previous_assignment_id=record.previous_assignment_id,
        content_hash=record.content_hash,
        chain_hash=record.chain_hash,
        party_ref=privacy_reference_digest("party", record.machine_id, record.party),
        role_ref=privacy_reference_digest("role", record.machine_id, record.role),
    )


class PrivacyResponsibilityComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    responsibility_assignments: list[PrivacyResponsibilityAssignmentOut]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/privacy-responsibility/"
    "compliance-export",
    response_model=PrivacyResponsibilityComplianceExportOut,
)
def export_privacy_responsibility_compliance(
    machine_id: str,
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only desensitized privacy view of one machine's responsibility
    assignments over a closed time window.

    Includes every responsibility assignment owned by the path machine whose
    own ``created_at`` falls within the inclusive bounds, ordered by the actual
    UTC instant of ``created_at`` and then by id, so an exact-second record
    sorts before any fractional-second record of the same second. The raw
    ``party`` and ``role`` values are never returned; their positions carry
    ``party_ref`` and ``role_ref`` instead: ``SHA-256(UTF-8("privacy:v1|party"
    + machine_id + trimmed party))`` and the same with the ``role`` prefix. A
    value that is not a string or is blank after trimming yields ``null``
    while the record is still included. Every other field — id, machine/event/
    incident ids, created_at, and the responsibility chain fields — is emitted
    exactly as stored: a missing, misowned, duplicated, or chain-damaged
    related object never causes a record to be rewritten, filtered out, or
    repaired, and another machine's assignments can never enter the result.
    The query only issues reads, produces byte-identical output for identical
    data and parameters on repeat calls, and reads assignments persisted
    across application restarts; an empty or old database needs no migration.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Membership is decided by the record's own machine_id and created_at only;
    # the referenced event/incident are never looked up, so dangling or
    # misowned references export verbatim. Ordering parses stamps to UTC
    # instants because an exact-second stamp sorts before a fractional stamp
    # of the same second only after parsing (lexicographically '.' < 'Z').
    records = _machine_rows_in_window(
        session,
        IncidentResponsibilityAssignment,
        machine_id,
        window_start,
        window_end,
    )

    return PrivacyResponsibilityComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        responsibility_assignments=[
            privacy_responsibility_assignment_to_out(record) for record in records
        ],
    )


# --- read-only desensitized privacy key-rotation export ----------------------


def privacy_key_rotation_to_dict(record: KeyRotationEvent) -> dict[str, object]:
    """Desensitized privacy view of one stored key-rotation record.

    The raw ``old_public_key``/``new_public_key`` never leave the service:
    their positions carry ``old_public_key_ref``/``new_public_key_ref``, the
    SHA-256 digests of ``privacy:v1|old_public_key`` /
    ``privacy:v1|new_public_key`` concatenated with the machine id and the
    stored key with surrounding whitespace removed (``null`` when the stored
    value is not a string or is blank after trimming). Every other field —
    id, machine id, version, created_at, previous-rotation link, and chain
    hash — is emitted exactly as stored, in this fixed field order.
    """
    return {
        "id": record.id,
        "machine_id": record.machine_id,
        "version": record.version,
        "created_at": record.created_at,
        "previous_rotation_id": record.previous_rotation_id,
        "chain_hash": record.chain_hash,
        "old_public_key_ref": privacy_reference_digest(
            "old_public_key", record.machine_id, record.old_public_key
        ),
        "new_public_key_ref": privacy_reference_digest(
            "new_public_key", record.machine_id, record.new_public_key
        ),
    }


@app.get("/machines/{machine_id}/key-rotation-events/privacy-export")
def export_key_rotation_events_privacy(
    machine_id: str,
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only desensitized privacy export of one machine's key rotations
    over a closed time window.

    The caller submits only the path machine id and the two required bounds
    ``from_created_at``/``to_created_at`` — UTC RFC 3339 date-times ending in
    ``Z`` (fractional seconds optional, equal bounds allowed); any other
    query parameter is a 422 ``invalid_query`` and a missing, blank, offset,
    malformed, or inverted bound is a 422 ``bad_time``, both raised before
    any machine or rotation data is read. A missing machine is a 404
    ``not_found`` carrying no rotation data. Only ``GET`` is routed; other
    methods return 405 without filtering, digesting, or writing anything.

    On success the response carries the machine id, the bounds echoed
    verbatim, and ``rotations`` — always present, an empty array when the
    window contains nothing. The array holds only records owned by the path
    machine whose own ``created_at`` falls in the inclusive interval, ordered
    by the actual UTC instant of ``created_at`` and then by id, so an
    exact-second record sorts before any fractional-second record of the same
    second. Each item exposes exactly ``{id, machine_id, version, created_at,
    previous_rotation_id, chain_hash, old_public_key_ref,
    new_public_key_ref}``: the raw public keys are never returned, only their
    desensitizing digests. Records are exported exactly as stored — a
    missing, misowned, duplicated, or chain-damaged record is never
    rewritten, filtered out, or repaired, and another machine's rotations can
    never enter the result. The query only issues reads; the body is compact
    UTF-8 JSON with a fixed field order ending in a newline, contains no
    floating-point values, and is byte-identical on repeat calls against
    unchanged data, including data persisted across application restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Membership is decided by the record's own machine_id and created_at
    # only; ordering parses stamps to UTC instants because an exact-second
    # stamp sorts before a fractional stamp of the same second only after
    # parsing (lexicographically '.' precedes 'Z').
    records = _machine_rows_in_window(
        session, KeyRotationEvent, machine_id, window_start, window_end
    )

    payload = {
        "machine_id": machine_id,
        "from_created_at": params.from_created_at,
        "to_created_at": params.to_created_at,
        "rotations": [privacy_key_rotation_to_dict(record) for record in records],
    }
    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a
    # fixed field order, terminated by a single newline, and free of any
    # floating-point or non-finite value (allow_nan=False).
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


# --- read-only desensitized privacy authorization-decision-event export ------


def privacy_decision_event_to_dict(
    record: AuthorizationDecisionEvent,
) -> dict[str, object]:
    """Desensitized privacy view of one stored authorization decision event.

    The raw ``action_type``/``resource`` never leave the service: their
    positions carry ``action_ref``/``resource_ref``, the SHA-256 digests of
    ``privacy:v1|action`` / ``privacy:v1|resource`` concatenated with the
    machine id and the stored value with surrounding whitespace removed
    (``null`` when the stored value is not a string or is blank after
    trimming). Every other field — id, machine id, allowed, reason,
    created_at, and the event chain fields — is emitted exactly as stored, in
    this fixed field order.
    """
    return {
        "id": record.id,
        "machine_id": record.machine_id,
        "action_ref": privacy_reference_digest(
            "action", record.machine_id, record.action_type
        ),
        "resource_ref": privacy_reference_digest(
            "resource", record.machine_id, record.resource
        ),
        "allowed": record.allowed,
        "reason": record.reason,
        "created_at": record.created_at,
        "previous_event_id": record.previous_event_id,
        "content_hash": record.content_hash,
        "chain_hash": record.chain_hash,
    }


@app.get("/machines/{machine_id}/authorization-decision-events/privacy-export")
def export_authorization_decision_events_privacy(
    machine_id: str,
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only desensitized privacy export of one machine's authorization
    decision events over a closed time window.

    The caller submits only the path machine id and the two required bounds
    ``from_created_at``/``to_created_at`` — UTC RFC 3339 date-times ending in
    ``Z`` (fractional seconds optional, equal bounds allowed); any other
    query parameter is a 422 ``invalid_query`` and a missing, blank, offset,
    malformed, or inverted bound is a 422 ``bad_time``, both raised before
    any machine or event data is read. A missing machine is a 404
    ``not_found`` carrying no event data. Only ``GET`` is routed; other
    methods return 405 without filtering, digesting, or writing anything.

    On success the response carries the machine id, the bounds echoed
    verbatim, and ``events`` — always present, an empty array when the window
    contains nothing. The array holds only records owned by the path machine
    whose own ``created_at`` falls in the inclusive interval, ordered by the
    actual UTC instant of ``created_at`` and then by id, so an exact-second
    record sorts before any fractional-second record of the same second. Each
    item exposes exactly ``{id, machine_id, action_ref, resource_ref,
    allowed, reason, created_at, previous_event_id, content_hash,
    chain_hash}``: the raw action and resource are never returned, only their
    desensitizing digests. Records are exported exactly as stored — a
    missing, misowned, duplicated, or chain-damaged record is never
    rewritten, filtered out, or repaired, and another machine's events can
    never enter the result. The query only issues reads; the body is compact
    UTF-8 JSON with a fixed field order ending in a newline, contains no
    floating-point values, and is byte-identical on repeat calls against
    unchanged data, including data persisted across application restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Membership is decided by the record's own machine_id and created_at
    # only; ordering parses stamps to UTC instants because an exact-second
    # stamp sorts before a fractional stamp of the same second only after
    # parsing (lexicographically '.' precedes 'Z').
    records = _machine_rows_in_window(
        session, AuthorizationDecisionEvent, machine_id, window_start, window_end
    )

    payload = {
        "machine_id": machine_id,
        "from_created_at": params.from_created_at,
        "to_created_at": params.to_created_at,
        "events": [privacy_decision_event_to_dict(record) for record in records],
    }
    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a
    # fixed field order, terminated by a single newline, and free of any
    # floating-point or non-finite value (allow_nan=False).
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


# --- read-only desensitized machine identity privacy export ------------------


def validate_machine_privacy_export_params(request: Request) -> None:
    """Validate the machine identity privacy-export query string.

    The export is keyed on the path machine alone and accepts no query
    parameters; any parameter name is a 422 ``invalid_query``. The check runs
    before the machine is looked up and issues no database access, so an extra
    parameter against a non-existent machine still reports 422 rather than
    404.
    """
    if request.query_params:
        raise QueryError("invalid_query")


@app.get("/machines/{machine_id}/privacy-export")
def export_machine_identity_privacy(
    machine_id: str,
    _: Annotated[None, Depends(validate_machine_privacy_export_params)],
    session: SessionDep,
):
    """Read-only desensitized privacy export of one machine's identity.

    The caller submits only the path machine id — no time range, business
    filter, or request body; any query parameter is a 422 ``invalid_query``
    raised before the machine is ever looked up. A missing machine is a 404
    ``not_found`` carrying no identity data. Only ``GET`` is routed; other
    methods return 405 without reading the identity, computing a digest, or
    writing anything.

    On success the response carries exactly ``{id, ext_ref, name_ref,
    key_ref, version, status, created_at, updated_at}`` in this fixed field
    order. The raw external id, display name, and public key never leave the
    service: their positions carry the desensitizing digests
    ``SHA-256(UTF-8("privacy:v1|external" + machine_id + trimmed
    external_id))``, and the same with the ``display`` and ``public``
    prefixes — 64 lowercase hexadecimal characters, or ``null`` when the
    stored value is not a string or is blank after trimming, while every
    other field is still returned. ``version``, ``status``, ``created_at``,
    and ``updated_at`` are emitted exactly as stored, with ``version`` as a
    JSON integer. The query only issues reads — it never creates, updates,
    deletes, repairs, or normalizes the machine identity — and the body is
    compact UTF-8 JSON terminated by a single newline, free of any
    floating-point or non-finite value, byte-identical on repeat calls
    against unchanged data, including data persisted across application
    restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    payload = {
        "id": machine.id,
        "ext_ref": privacy_reference_digest(
            "external", machine.id, machine.external_id
        ),
        "name_ref": privacy_reference_digest(
            "display", machine.id, machine.display_name
        ),
        "key_ref": privacy_reference_digest(
            "public", machine.id, machine.public_key
        ),
        "version": machine.version,
        "status": machine.status,
        "created_at": machine.created_at,
        "updated_at": machine.updated_at,
    }
    # Serialize by hand so the body is guaranteed compact UTF-8 JSON in a
    # fixed field order, terminated by a single newline, and free of any
    # floating-point or non-finite value (allow_nan=False).
    body = (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    )
    return Response(content=body, media_type="application/json")


# --- machine-level privacy access registration and read-only query ---------


# An RFC 3339 date-time in UTC ending in ``Z`` (the same shape the compliance
# exports use), enforced as a string pattern so an offset, a missing suffix, or
# surrounding whitespace can never be accepted; out-of-range calendar/time
# values are rejected by the field validator that parses each value.
ZuluDatetime = Annotated[
    str, StringConstraints(pattern=_RFC3339_Z_DATETIME_RE.pattern)
]


class PrivacyAccessCreate(BaseModel):
    """Registration payload for one machine privacy-data access.

    The body must be an object carrying exactly the access metadata: the
    access time, the desensitized export window actually used, the access
    result, and the number of records hit. No responsible-party rawtext or key
    material is a field, so such material sent by a client is never stored or
    echoed. Every timestamp must be a UTC RFC 3339 date-time ending in ``Z``,
    ``window_start`` must not be later than ``window_end`` (equal bounds are
    allowed), ``result`` is exactly ``success``/``failed``, and
    ``matches_count`` is a non-negative integer; a failed access reports zero
    matches. Pydantic rejects a missing field, a non-object body, or any value
    of the wrong shape with 422 before the handler — and therefore before the
    machine is ever looked up.
    """

    accessed_at: ZuluDatetime
    window_start: ZuluDatetime
    window_end: ZuluDatetime
    result: Literal["success", "failed"]
    matches_count: Annotated[StrictInt, Field(ge=0)]

    @field_validator("accessed_at", "window_start", "window_end")
    @classmethod
    def _range_check_zulu(cls, value: str) -> str:
        # The pattern already ruled out offsets, whitespace, and malformed
        # text; parsing rejects out-of-range calendar/time values (month 13,
        # day 30 in February, hour 24, ...).
        parse_utc_z_datetime(value)
        return value

    @model_validator(mode="after")
    def _check_window_and_failed_count(self) -> "PrivacyAccessCreate":
        if parse_utc_z_datetime(self.window_start) > parse_utc_z_datetime(
            self.window_end
        ):
            raise ValueError("window_start must not be later than window_end")
        # A failed access returns no data, so its hit count is always zero.
        if self.result == "failed" and self.matches_count != 0:
            raise ValueError("a failed access must report zero matches")
        return self


class PrivacyAccessOut(BaseModel):
    id: str
    machine_id: str
    accessed_at: str
    window_start: str
    window_end: str
    result: str
    matches_count: int


def privacy_access_to_out(record: PrivacyAccess) -> PrivacyAccessOut:
    return PrivacyAccessOut(
        id=record.id,
        machine_id=record.machine_id,
        accessed_at=record.accessed_at,
        window_start=record.window_start,
        window_end=record.window_end,
        result=record.result,
        matches_count=record.matches_count,
    )


@app.post(
    "/machines/{machine_id}/privacy-accesses",
    status_code=201,
    response_model=PrivacyAccessOut,
)
def create_privacy_access(
    machine_id: str, body: PrivacyAccessCreate, session: SessionDep
):
    """Register one machine-level privacy data access.

    Body validation runs before any path lookup, so a missing, non-object, or
    otherwise malformed payload is a 422 even when the machine does not exist
    and leaves no record. After validation, a missing machine is a 404
    ``not_found`` and writes nothing. A repeat registration with the same
    ``(accessed_at, window_start, window_end, result)`` for the same machine
    returns 409 ``duplicate_access`` and writes nothing (the hit count is not
    part of the identity). On success the record is persisted in its own
    append-only table — it never modifies machines, events, chains, or the
    desensitized responsibility export — and is returned with 201 carrying a
    fresh UUID, the machine id, the access time and window exactly as
    submitted, the result, and the hit count. The machine lookup, duplicate
    check, insert, and append to the machine's privacy-access hash chain
    (``previous_access_id``/``content_hash``/``chain_hash``) commit in a
    single locked write transaction, so concurrent registrations cannot lose
    records, fork the chain, or break a link. No responsible-party rawtext or
    key material is ever stored or echoed.
    """
    engine = session.get_bind()
    # Release the read connection before opening the locked write transaction,
    # matching the other chain appenders: concurrent registrations never hold
    # two pool connections at once.
    session.close()
    result = privacy_chain.append_access(
        engine,
        machine_id=machine_id,
        accessed_at=body.accessed_at,
        window_start=body.window_start,
        window_end=body.window_end,
        result=body.result,
        matches_count=body.matches_count,
    )
    if result["status"] == "not_found":
        return error_response(404, "not_found")
    if result["status"] == "duplicate_access":
        return error_response(409, "duplicate_access")
    return PrivacyAccessOut(**result["access"])


# Batch request field names, in the order per-item errors are reported.
_PRIVACY_ACCESS_FIELDS = (
    "accessed_at",
    "window_start",
    "window_end",
    "result",
    "matches_count",
)


class BatchValidationError(Exception):
    """A batch payload failure reported as ``422 {"error":{"code": ...}}``.

    Codes distinguish the failure family: ``invalid_batch`` (body/item shape
    or field type), ``bad_time`` (a timestamp or the export window), or
    ``invalid_value`` (the result string or a negative hit count).
    """

    def __init__(self, code: str):
        self.code = code


def _validate_privacy_access_batch_item(item: object) -> dict[str, object]:
    """Validate one privacy-access batch item into a plain dict.

    Checks run in failure-family order: object shape, presence, and field
    types (``invalid_batch``); the three ``Z`` timestamps and the export
    window (``bad_time``); then ``result`` and ``matches_count`` value-domain
    rules (``invalid_value``), including the failed-access-must-hit-zero
    rule. Extra submitted keys are ignored exactly as on the single-registration
    path, never stored or echoed.
    """
    if not isinstance(item, dict):
        raise BatchValidationError("invalid_batch")

    for field_name in _PRIVACY_ACCESS_FIELDS:
        if field_name not in item:
            raise BatchValidationError("invalid_batch")

    timestamps = {
        field_name: item[field_name]
        for field_name in ("accessed_at", "window_start", "window_end")
    }
    result_value = item["result"]
    matches_count = item["matches_count"]

    # Business field types: the three times and the result must be strings,
    # and the hit count must be an integer (booleans are not integers here,
    # matching the single path's StrictInt).
    if not all(isinstance(value, str) for value in timestamps.values()):
        raise BatchValidationError("invalid_batch")
    if not isinstance(result_value, str):
        raise BatchValidationError("invalid_batch")
    if isinstance(matches_count, bool) or not isinstance(matches_count, int):
        raise BatchValidationError("invalid_batch")

    # Timestamp shape (Z suffix only; no offsets or surrounding whitespace)
    # first, then calendar/time range, then the non-inverted window.
    for value in timestamps.values():
        if not _RFC3339_Z_DATETIME_RE.fullmatch(value):
            raise BatchValidationError("bad_time")
    parsed: dict[str, datetime] = {}
    try:
        for field_name, value in timestamps.items():
            parsed[field_name] = parse_utc_z_datetime(value)
    except ValueError:
        raise BatchValidationError("bad_time") from None
    if parsed["window_start"] > parsed["window_end"]:
        raise BatchValidationError("bad_time")

    if result_value not in ("success", "failed"):
        raise BatchValidationError("invalid_value")
    if matches_count < 0:
        raise BatchValidationError("invalid_value")
    # A failed access returns no data, so its hit count is always zero.
    if result_value == "failed" and matches_count != 0:
        raise BatchValidationError("invalid_value")

    return {
        "accessed_at": timestamps["accessed_at"],
        "window_start": timestamps["window_start"],
        "window_end": timestamps["window_end"],
        "result": result_value,
        "matches_count": matches_count,
    }


@app.post(
    "/machines/{machine_id}/privacy-accesses/batch",
)
async def create_privacy_accesses_batch(machine_id: str, request: Request):
    """Register a batch of machine-level privacy data accesses at once.

    The body must be a JSON object carrying a ``privacy_accesses`` array;
    each item carries exactly the same five business fields as one single
    registration. Every structural and business check completes before the
    machine is looked up or any row is written:

    - a non-object body, a missing/non-array ``privacy_accesses``, a
      non-object item, a missing item field, or a wrong-typed business field
      is ``422 invalid_batch``;
    - a malformed/offset/whitespace/out-of-range timestamp or an inverted
      window is ``422 bad_time``;
    - a ``result`` other than ``success``/``failed`` or a negative integer
      hit count (also a failed access with non-zero hits) is
      ``422 invalid_value``;
    - any query parameter is ``422 invalid_query``.

    An empty array is legal and returns ``200 {"results": []}`` without
    touching the database. After validation, a missing machine is
    ``404 not_found`` and writes nothing. Otherwise the whole batch enters
    one locked write transaction — the same lock the single registration
    takes, so the two serialize against each other — and items are processed
    in request-array order with the existing duplicate check and insert. The
    first item of a given access identity
    ``(accessed_at, window_start, window_end, result)`` for the machine
    registers; already-registered or earlier-in-batch repeats come back as
    ``duplicate_access`` without a new row, never aborting the other items.
    On success the status is 200 with a ``results`` array aligned to the
    request: each success item is ``{"outcome": "success", ...record}`` and
    each duplicate is ``{"outcome": "duplicate_access", ...submitted fields}``.
    Any persistence failure returns ``500 internal_error`` and the single
    transaction rolls back, leaving no partial records. The path accepts
    ``POST`` only; other methods return ``405``.
    """
    # Unknown query parameters are rejected first, then the batch itself;
    # both precede the machine lookup.
    if request.query_params:
        return error_response(422, "invalid_query")

    try:
        payload = await request.json()
    except Exception:
        return error_response(422, "invalid_batch")
    if not isinstance(payload, dict) or not isinstance(
        payload.get("privacy_accesses"), list
    ):
        return error_response(422, "invalid_batch")

    try:
        items = [
            _validate_privacy_access_batch_item(item)
            for item in payload["privacy_accesses"]
        ]
    except BatchValidationError as error:
        return error_response(422, error.code)

    # An empty batch is legal: nothing to look up or register.
    if not items:
        return {"results": []}

    engine = request.app.state.engine
    # The batch runs on its own locked connection; the request session is not
    # needed and must not hold a pooled read connection meanwhile.
    try:
        result = privacy_chain.append_accesses_batch(
            engine, machine_id=machine_id, items=items
        )
    except Exception:
        # The locked write transaction rolls back on any failure, so a batch
        # can never leave part of its rows behind.
        return error_response(500, "internal_error")

    if result["status"] == "not_found":
        return error_response(404, "not_found")

    results = []
    for item, outcome in zip(items, result["outcomes"], strict=True):
        if outcome["status"] == "duplicate_access":
            # A duplicate only echoes the fields as submitted — no id, no
            # machine id, and no record is created.
            results.append({"outcome": "duplicate_access", **item})
        else:
            results.append({"outcome": "success", **outcome["access"]})
    return {"results": results}


class PrivacyAccessExportParams(BaseModel):
    from_accessed_at: str
    to_accessed_at: str


def validate_privacy_access_export_params(
    request: Request,
) -> PrivacyAccessExportParams:
    """Validate the privacy-access query string before any data access.

    Exactly two parameters are accepted: ``from_accessed_at`` and
    ``to_accessed_at``, both required UTC RFC 3339 date-times ending in ``Z``
    (fractional seconds optional; offset forms, surrounding whitespace, and
    non-``Z`` suffixes are rejected), with the lower bound not later than the
    upper bound (equal bounds allowed). Any other parameter name is a 422
    ``invalid_query``; a missing, blank, malformed, or inverted bound is a 422
    ``bad_time``. Both checks run before the machine or any access record is
    read, so an invalid query against a non-existent machine still reports 422
    rather than 404.
    """
    allowed = {"from_accessed_at", "to_accessed_at"}
    unknown = [name for name in request.query_params if name not in allowed]
    if unknown:
        raise QueryError("invalid_query")

    raw_from = request.query_params.get("from_accessed_at")
    raw_to = request.query_params.get("to_accessed_at")

    def _valid(value: str | None) -> bool:
        if not value or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
            return False
        try:
            parse_utc_z_datetime(value)
        except ValueError:
            return False
        return True

    if not _valid(raw_from) or not _valid(raw_to):
        raise QueryError("bad_time")

    if parse_utc_z_datetime(raw_from) > parse_utc_z_datetime(raw_to):
        raise QueryError("bad_time")

    return PrivacyAccessExportParams(
        from_accessed_at=raw_from,  # type: ignore[arg-type]
        to_accessed_at=raw_to,  # type: ignore[arg-type]
    )


class PrivacyAccessComplianceExportOut(BaseModel):
    machine_id: str
    from_accessed_at: str
    to_accessed_at: str
    privacy_accesses: list[PrivacyAccessOut]


def _machine_accesses_in_window(
    session: Session,
    machine_id: str,
    window_start: datetime,
    window_end: datetime,
) -> list[PrivacyAccess]:
    """Load one machine's privacy accesses and apply the closed UTC window.

    Membership is decided by the row's own ``machine_id`` and the actual UTC
    instant of ``accessed_at`` only. Rows are ordered by that instant and then
    by ``id`` ascending, so an exact-second record sorts before any
    fractional-second record of the same second (ISO text alone is not
    chronological across that boundary). Issues reads only.
    """
    rows = session.scalars(
        select(PrivacyAccess).where(PrivacyAccess.machine_id == machine_id)
    ).all()
    in_window = [
        row
        for row in rows
        if window_start <= parse_utc_z_datetime(row.accessed_at) <= window_end
    ]
    return sorted(
        in_window,
        key=lambda row: (parse_utc_z_datetime(row.accessed_at), row.id),
    )


@app.get(
    "/machines/{machine_id}/privacy-accesses/compliance-export",
    response_model=PrivacyAccessComplianceExportOut,
)
def export_privacy_accesses(
    machine_id: str,
    params: Annotated[
        PrivacyAccessExportParams, Depends(validate_privacy_access_export_params)
    ],
    session: SessionDep,
):
    """Read-only query of one machine's registered privacy accesses.

    The caller submits only the path machine id and a closed UTC window over
    access time; query validation (``invalid_query`` for unknown parameters,
    ``bad_time`` for missing/blank/offset/malformed/inverted bounds) completes
    before the machine or any access record is read. A missing machine is a
    404 ``not_found`` with no access data. On success the response carries the
    machine id, the bounds echoed verbatim, and ``privacy_accesses`` — always
    present, an empty array when the window contains nothing. The array holds
    only records owned by the path machine whose own ``accessed_at`` falls in
    the inclusive interval, ordered by the actual UTC instant of
    ``accessed_at`` and then by id; each record exposes exactly
    ``{id, machine_id, accessed_at, window_start, window_end, result,
    matches_count}`` — never a responsible party or key rawtext. The query only
    issues reads, never returns another machine's records, gives identical
    results on repeat calls, and reads records persisted across restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_accessed_at)
    window_end = parse_utc_z_datetime(params.to_accessed_at)
    records = _machine_accesses_in_window(
        session, machine_id, window_start, window_end
    )

    return PrivacyAccessComplianceExportOut(
        machine_id=machine_id,
        from_accessed_at=params.from_accessed_at,
        to_accessed_at=params.to_accessed_at,
        privacy_accesses=[privacy_access_to_out(record) for record in records],
    )


class PrivacyAccessSummaryOut(BaseModel):
    machine_id: str
    from_accessed_at: str
    to_accessed_at: str
    success_count: int
    failed_count: int
    matches_count: int


@app.get(
    "/machines/{machine_id}/privacy-accesses/summary",
    response_model=PrivacyAccessSummaryOut,
)
def summarize_privacy_accesses(
    machine_id: str,
    params: Annotated[
        PrivacyAccessExportParams, Depends(validate_privacy_access_export_params)
    ],
    session: SessionDep,
):
    """Read-only aggregate summary of one machine's privacy accesses.

    A summary view alongside — never part of — the per-machine privacy-access
    audit chain. The caller submits only the path machine id and a closed UTC
    window over access time; query validation (``invalid_query`` for unknown
    parameters, ``bad_time`` for missing/blank/offset/missing-``Z``/malformed/
    inverted bounds) completes before the machine or any access record is read.
    A missing machine is a ``404 not_found`` with no summary data.

    On success the response carries the machine id, the bounds echoed verbatim,
    and three totals computed from stored values over exactly the records owned
    by the path machine whose own ``accessed_at`` falls in the inclusive
    interval: ``success_count`` (records whose ``result`` is ``success``),
    ``failed_count`` (records whose ``result`` is ``failed``), and
    ``matches_count`` (the sum of every in-window record's stored
    ``matches_count``, counted independently of the result). An empty window
    still returns the full envelope with all three totals at zero. The query
    only issues reads — it never creates, updates, deletes, repairs, or
    normalizes an access record — never totals another machine's records, gives
    byte-identical results on repeat calls against unchanged data, and reads
    records persisted across application restarts. It exposes no responsible-
    party rawtext or key material.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_accessed_at)
    window_end = parse_utc_z_datetime(params.to_accessed_at)
    records = _machine_accesses_in_window(
        session, machine_id, window_start, window_end
    )

    success_count = sum(1 for record in records if record.result == "success")
    failed_count = sum(1 for record in records if record.result == "failed")
    # The hit total is an independent sum over every in-window record, including
    # failed records whose stored count is zero; it is never derived from the
    # success/failure tallies.
    matches_count = sum(record.matches_count for record in records)

    return PrivacyAccessSummaryOut(
        machine_id=machine_id,
        from_accessed_at=params.from_accessed_at,
        to_accessed_at=params.to_accessed_at,
        success_count=success_count,
        failed_count=failed_count,
        matches_count=matches_count,
    )


# Fixed summary bucket width: 900 seconds = one UTC quarter hour. Bucket edges
# are the absolute UTC quarter-hour boundaries (``...:00``/``:15``/``:30``/
# ``:45``), independent of the request window, and each bucket is the
# half-open interval ``[bucket_start, bucket_end)``.
PRIVACY_ACCESS_BUCKET_WIDTH_SECONDS = 900


def _floor_to_access_bucket(instant: datetime) -> datetime:
    """Floor a UTC instant to its fixed 900-second quarter-hour bucket edge."""
    # 900 seconds divides the day evenly, so flooring to UTC quarter-hour edges
    # is plain calendar arithmetic on hour/minute; seconds and fractional
    # seconds truncate toward the earlier edge.
    minute_of_day = instant.hour * 60 + instant.minute
    floored_minute = (minute_of_day // 15) * 15
    return instant.replace(
        hour=floored_minute // 60,
        minute=floored_minute % 60,
        second=0,
        microsecond=0,
    )


def _format_bucket_edge(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


class PrivacyAccessSummaryBucketOut(BaseModel):
    bucket_start: str
    bucket_end: str
    success_count: int
    failed_count: int
    matches_count: int


class PrivacyAccessBucketSummaryOut(BaseModel):
    machine_id: str
    from_accessed_at: str
    to_accessed_at: str
    bucket_width_seconds: int
    summary_buckets: list[PrivacyAccessSummaryBucketOut]


@app.get(
    "/machines/{machine_id}/privacy-accesses/summary/buckets",
    response_model=PrivacyAccessBucketSummaryOut,
)
def summarize_privacy_access_buckets(
    machine_id: str,
    params: Annotated[
        PrivacyAccessExportParams, Depends(validate_privacy_access_export_params)
    ],
    session: SessionDep,
):
    """Read-only fixed-width bucketed summary of one machine's privacy accesses.

    A bucketed view alongside — never part of — the per-machine privacy-access
    audit chain. The caller submits only the path machine id and the same
    closed UTC window over access time as the summary; query validation
    (``invalid_query`` for unknown parameters, ``bad_time`` for missing/blank/
    offset/missing-``Z``/malformed/inverted bounds) completes before the
    machine or any access record is read. A missing machine is a
    ``404 not_found`` with no bucket data.

    On success the response carries the machine id, the bounds echoed verbatim,
    the fixed ``bucket_width_seconds`` of 900, and ``summary_buckets`` — always
    present, an empty array when the window contains nothing. Only records
    owned by the path machine whose own ``accessed_at`` falls in the inclusive
    request interval are examined; each record is then assigned to the fixed
    UTC quarter-hour bucket (edges at ``:00``/``:15``/``:30``/``:45``) that
    contains its actual access instant, with buckets half-open
    (``[bucket_start, bucket_end)``): an access exactly on an edge belongs to
    the bucket starting there, never the preceding one. Bucket edges are
    independent of the request window. Only buckets containing at least one
    in-window record appear, ordered ascending by ``bucket_start``; each bucket
    carries ``bucket_start``, ``bucket_end``, ``success_count`` (records whose
    ``result`` is ``success``), ``failed_count`` (records whose ``result`` is
    ``failed``), and ``matches_count`` (the sum of every bucketed record's
    stored ``matches_count``, counted independently of the result). The query
    only issues reads — it never creates, updates, deletes, repairs, or
    normalizes an access record — never buckets another machine's records,
    gives byte-identical results on repeat calls against unchanged data, and
    reads records persisted across application restarts. It exposes no
    responsible-party rawtext or key material. Non-GET methods return ``405``
    without reading access records.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_accessed_at)
    window_end = parse_utc_z_datetime(params.to_accessed_at)
    records = _machine_accesses_in_window(
        session, machine_id, window_start, window_end
    )

    # ``records`` is already ordered by (accessed-at instant, id), so buckets
    # are first encountered in ascending start order; the dict preserves that
    # order, making the output stable without a second sort.
    totals_by_bucket: dict[datetime, dict[str, int]] = {}
    for record in records:
        accessed_instant = parse_utc_z_datetime(record.accessed_at)
        bucket_start = _floor_to_access_bucket(accessed_instant)
        totals = totals_by_bucket.setdefault(
            bucket_start,
            {"success_count": 0, "failed_count": 0, "matches_count": 0},
        )
        if record.result == "success":
            totals["success_count"] += 1
        elif record.result == "failed":
            totals["failed_count"] += 1
        # The hit total is an independent sum over every bucketed record,
        # including failed records whose stored count is zero; it is never
        # derived from the success/failure tallies.
        totals["matches_count"] += record.matches_count

    summary_buckets = [
        PrivacyAccessSummaryBucketOut(
            bucket_start=_format_bucket_edge(bucket_start),
            bucket_end=_format_bucket_edge(
                bucket_start
                + timedelta(seconds=PRIVACY_ACCESS_BUCKET_WIDTH_SECONDS)
            ),
            success_count=totals["success_count"],
            failed_count=totals["failed_count"],
            matches_count=totals["matches_count"],
        )
        for bucket_start, totals in totals_by_bucket.items()
    ]

    return PrivacyAccessBucketSummaryOut(
        machine_id=machine_id,
        from_accessed_at=params.from_accessed_at,
        to_accessed_at=params.to_accessed_at,
        bucket_width_seconds=PRIVACY_ACCESS_BUCKET_WIDTH_SECONDS,
        summary_buckets=summary_buckets,
    )


class PrivacyAccessChangesParams(BaseModel):
    limit: int
    cursor: str | None = None


# An exclusive pagination cursor is ``<accessed_at original text>|<record id>``.
# The split is on the first separator, so the timestamp segment may itself
# contain ``|``; the record-id segment must be exactly a UUID.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def validate_privacy_access_changes_params(
    request: Request,
) -> PrivacyAccessChangesParams:
    """Validate the incremental ``changes`` query string before any lookup.

    Exactly two parameters are accepted: the required ``limit`` and the
    optional ``cursor``. ``limit`` must be a non-boolean integer in 1..100;
    a missing, non-integer (``3.0``, ``true``, blank, non-decimal text),
    out-of-range, or boolean value is a 422 ``bad_limit``. ``cursor``, when
    present, must be a string shaped ``<accessed_at original text>|<uuid>``;
    a malformed, non-string, or shape-mismatching cursor is a 422
    ``invalid_cursor`` and the machine is never queried. Any other parameter
    name is a 422 ``invalid_query``. All three checks run before the machine
    is looked up and issue no database access, so an invalid query against a
    non-existent machine still reports 422 rather than 404.
    """
    allowed = {"limit", "cursor"}
    unknown = [name for name in request.query_params if name not in allowed]
    if unknown:
        raise QueryError("invalid_query")

    # A repeated parameter name is itself an unknown-shape query: reject
    # ``limit=1&limit=2`` rather than silently taking one occurrence.
    if len(request.query_params.getlist("limit")) != 1:
        raise QueryError("bad_limit")
    if len(request.query_params.getlist("cursor")) > 1:
        raise QueryError("invalid_cursor")

    raw_limit = request.query_params.get("limit")
    if (
        raw_limit is None
        or not _INTEGER_QUERY_RE.fullmatch(raw_limit)
    ):
        raise QueryError("bad_limit")
    limit = int(raw_limit)
    if not 1 <= limit <= 100:
        raise QueryError("bad_limit")

    cursor = request.query_params.get("cursor")
    if cursor is not None:
        # Query parameters arrive as strings; an empty value carries no
        # separator and fails the shape check below.
        parts = cursor.split("|", 1)
        if len(parts) != 2 or not parts[0] or not _UUID_RE.fullmatch(parts[1]):
            raise QueryError("invalid_cursor")
        # The position segment is the original ``accessed_at`` text, always a
        # UTC RFC 3339 date-time ending in ``Z``; a damaged value is rejected
        # here rather than failing later while parsing the position.
        cursor_accessed_at = parts[0]
        if not _RFC3339_Z_DATETIME_RE.fullmatch(cursor_accessed_at):
            raise QueryError("invalid_cursor")
        try:
            parse_utc_z_datetime(cursor_accessed_at)
        except ValueError:
            raise QueryError("invalid_cursor") from None

    return PrivacyAccessChangesParams(limit=limit, cursor=cursor)


class PrivacyAccessChangesOut(BaseModel):
    machine_id: str
    limit: int
    records: list[PrivacyAccessOut]
    next_cursor: str | None
    has_more: bool


def _encode_access_cursor(accessed_at: str, access_id: str) -> str:
    return f"{accessed_at}|{access_id}"


@app.get(
    "/machines/{machine_id}/privacy-accesses/changes",
    response_model=PrivacyAccessChangesOut,
)
def get_privacy_access_changes(
    machine_id: str,
    params: Annotated[
        PrivacyAccessChangesParams,
        Depends(validate_privacy_access_changes_params),
    ],
    session: SessionDep,
):
    """Read-only incremental, keyset-paginated query of one machine's accesses.

    The caller submits the path machine id, a required ``limit`` (an integer
    from 1 to 100), and an optional opaque ``cursor`` returned by a previous
    page. Query validation (``invalid_query`` for unknown parameters,
    ``bad_limit`` for a missing/non-integer/boolean/out-of-range limit,
    ``invalid_cursor`` for a malformed, non-string, or shape-mismatching
    cursor) completes before the machine or any access record is read. A
    missing machine is a ``404 not_found`` carrying no access records.

    On success the response is ``{machine_id, limit, records, next_cursor,
    has_more}``. ``records`` contains only records owned by the path machine,
    ordered by the actual UTC instant of ``accessed_at`` and then by record id
    ascending (so an exact-second record sorts before any fractional-second
    record of the same second), and is an empty array on an empty page. Each
    record exposes exactly the seven registered visible fields — never a
    responsible-party rawtext, key, policy text, or identity material. The
    cursor is an exclusive position ``<accessed_at original text>|<record
    id>`` pointing just after a page's last record, so repeating the same
    cursor against unchanged data returns the byte-identical next page, a
    record inserted before an old cursor position never resurfaces on later
    pages, and already-returned records are never read back. ``next_cursor``
    is that position when at least one record follows the current page and
    ``null`` on the last page; ``has_more`` is true exactly when a record
    exists after the current position (false on an empty page). The endpoint
    adds no schema (cursors are stateless), issues only reads — it never
    writes, updates, deletes, repairs, or normalizes a record and never
    returns another machine's records — and keeps reading data persisted
    across application restarts. Non-GET methods return ``405`` without
    reading records, computing a page, or writing anything.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    rows = session.scalars(
        select(PrivacyAccess).where(PrivacyAccess.machine_id == machine_id)
    ).all()
    ordered = sorted(
        rows,
        key=lambda row: (parse_utc_z_datetime(row.accessed_at), row.id),
    )

    start = 0
    if params.cursor is not None:
        cursor_accessed_at, cursor_id = params.cursor.split("|", 1)
        # Exclusive keyset position: the first record strictly after
        # (cursor-instant, cursor-id). Registered rows always parse, so a
        # cursor built by this endpoint lands exactly at its record.
        start = bisect.bisect_right(
            ordered,
            (parse_utc_z_datetime(cursor_accessed_at), cursor_id),
            key=lambda row: (parse_utc_z_datetime(row.accessed_at), row.id),
        )

    remaining = ordered[start:]
    page = remaining[: params.limit]
    has_more = len(remaining) > len(page)
    next_cursor = (
        _encode_access_cursor(page[-1].accessed_at, page[-1].id) if page and has_more
        else None
    )

    return PrivacyAccessChangesOut(
        machine_id=machine_id,
        limit=params.limit,
        records=[privacy_access_to_out(record) for record in page],
        next_cursor=next_cursor,
        has_more=has_more,
    )


def validate_privacy_access_integrity_params(request: Request) -> None:
    """Validate the privacy-access integrity query string before any lookup.

    The integrity check is keyed on the path machine alone and accepts no
    query parameters; any parameter name is a 422 ``invalid_query``. The
    check runs before the machine is looked up and issues no database access,
    so an extra parameter against a non-existent machine still reports 422
    rather than 404.
    """
    if request.query_params:
        raise QueryError("invalid_query")


class PrivacyAccessIntegrityOut(BaseModel):
    valid: bool
    checked_count: int
    broken_access_id: str | None


@app.get(
    "/machines/{machine_id}/privacy-accesses/integrity",
    response_model=PrivacyAccessIntegrityOut,
)
def check_privacy_access_integrity(
    machine_id: str,
    _: Annotated[None, Depends(validate_privacy_access_integrity_params)],
    session: SessionDep,
):
    """Read-only verification of one machine's privacy access hash chain.

    Returns ``{valid, checked_count, broken_access_id}``: an empty or fully
    sound chain reports ``true``, the machine's total access count, and
    ``null``; otherwise the first record — in (accessed-at instant, id)
    order, an exact-second record before any fractional-second record of the
    same second — whose content hash, previous-access link, or chain hash
    does not verify is reported. Only the path machine's records are
    examined, and the query never writes, repairs, or deletes, so repeated
    calls and restarts return stable results.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    valid, checked_count, broken_access_id = privacy_chain.verify_chain(
        session, machine_id
    )
    return PrivacyAccessIntegrityOut(
        valid=valid,
        checked_count=checked_count,
        broken_access_id=broken_access_id,
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


def validate_evidence_chain_integrity_params(request: Request) -> None:
    """Validate the evidence-chain integrity query string before any lookup.

    The check is keyed on the path machine alone and accepts no query
    parameters; any parameter name is a 422 ``invalid_query``. The check runs
    before the machine is looked up and issues no database access, so an extra
    parameter against a non-existent machine still reports 422 rather than 404.
    """
    if request.query_params:
        raise QueryError("invalid_query")


@app.get(
    "/machines/{machine_id}/authorization-decision-events/"
    "evidence-chain/integrity",
    response_model=EvidenceIntegrityOut,
)
def check_evidence_chain_integrity(
    machine_id: str,
    _: Annotated[None, Depends(validate_evidence_chain_integrity_params)],
    session: SessionDep,
):
    """Read-only verification of one machine's tamper-evident evidence chain.

    Returns the same three conclusions as the existing evidence integrity
    audit — ``{valid, checked_count, broken_evidence_id}`` — while additionally
    verifying the per-machine hash chain. Records are scanned in
    ``(created_at, id)`` chain order. A complete or empty chain reports
    ``true``, the machine's total evidence count, and ``null``; otherwise the
    first record is reported that fails an existing evidence-audit check (its
    event reference is missing or foreign-owned, its ``evidence_type`` is
    blank, or its stored fingerprint is not exactly 64 lowercase hexadecimal
    characters) or whose recomputed content digest, previous-evidence link, or
    chain hash does not verify. A record with a corrupted ``created_at`` still
    enters the total and is reported as broken; a missing associated event
    never removes the record. Only the path machine's records are examined, so
    another machine's damage cannot change this result, and only the first
    broken record is reported.

    Any query parameter is a 422 ``invalid_query`` raised before the machine
    is looked up; a missing machine is a 404 ``not_found`` carrying no chain
    conclusion. Only ``GET`` is routed; other methods return 405 without
    reading records. The query is strictly read-only — it never writes,
    repairs, deletes, or normalizes evidence, events, chains, or links — and
    repeated calls against unchanged data return byte-identical results.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    valid, checked_count, broken_evidence_id = evidence_chain.verify_full_chain(
        session, machine_id
    )
    return EvidenceIntegrityOut(
        valid=valid,
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


class CausalLinkComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    causal_links: list[CausalLinkOut]


@app.get(
    "/machines/{machine_id}/authorization-decision-events/causal-links/"
    "compliance-export",
    response_model=CausalLinkComplianceExportOut,
)
def export_causal_links_compliance(
    machine_id: str,
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only machine-level compliance slice of causal links over a window.

    Unlike the event-window compliance export — whose ``causal_links`` require
    both endpoint events to fall inside the exported event set — this slice is
    keyed on each link's own ``created_at``: it includes every causal link
    whose stored ``machine_id`` is the path machine and whose own
    ``created_at`` falls within the closed interval
    ``[from_created_at, to_created_at]``, independently of when either
    endpoint event was created. Each item has exactly the fields of the
    causal-link list endpoint (``id``, ``machine_id``, ``cause_event_id``,
    ``effect_event_id``, ``created_at``), emitted exactly as stored: a cause or
    effect event that is missing, owned by another machine, duplicated, or
    otherwise damaged never causes a link to be rewritten, filtered out, or
    repaired, and the events table is not consulted at all. Links are ordered
    by the actual UTC instant of ``created_at`` and then by id, so an
    exact-second link sorts before any fractional-second link of the same
    second, and the array is present (and empty) even when the window contains
    nothing. The query only issues reads — it never creates, updates, deletes,
    repairs, recomputes, or normalizes a link or any other record — never
    returns another machine's links, produces identical output for identical
    data and parameters on repeat calls, and reads links persisted across
    application restarts. An empty or old database needs no migration: the
    endpoint adds no schema.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    # Membership is decided by the link's own machine_id and created_at only;
    # the endpoint events are never looked up, so dangling or misowned
    # endpoints export verbatim. Ordering parses stamps to UTC instants
    # because an exact-second ISO stamp sorts before a fractional stamp of the
    # same second only after parsing (lexicographically '.' precedes 'Z').
    records = _machine_rows_in_window(
        session,
        AuthorizationDecisionCausalLink,
        machine_id,
        window_start,
        window_end,
    )

    return CausalLinkComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        causal_links=[causal_link_to_out(record) for record in records],
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


# --- read-only joint-write transaction diagnostics -------------------------


class DiagParams(BaseModel):
    from_: str
    to: str


def validate_diag_params(request: Request) -> DiagParams:
    """Validate the diagnostics query string before any machine lookup.

    Exactly two parameters are accepted: ``from`` and ``to``, both required
    UTC RFC 3339 date-times ending in ``Z`` (fractional seconds optional;
    offset forms, surrounding whitespace, and non-``Z`` suffixes are
    rejected), with ``from`` not later than ``to`` (equal bounds allowed).
    Any other parameter name is a 422 ``invalid_query``; a missing, blank,
    malformed, or inverted bound is a 422 ``bad_time``. Both checks run
    before the machine is looked up, so invalid parameters against a
    non-existent machine still report 422 rather than 404.
    """
    allowed = {"from", "to"}
    unknown = [name for name in request.query_params if name not in allowed]
    if unknown:
        raise QueryError("invalid_query")

    raw_from = request.query_params.get("from")
    raw_to = request.query_params.get("to")

    def _valid(value: str | None) -> bool:
        if not value or not _RFC3339_Z_DATETIME_RE.fullmatch(value):
            return False
        try:
            parse_utc_z_datetime(value)
        except ValueError:
            return False
        return True

    if not _valid(raw_from) or not _valid(raw_to):
        raise QueryError("bad_time")

    if parse_utc_z_datetime(raw_from) > parse_utc_z_datetime(raw_to):
        raise QueryError("bad_time")

    return DiagParams(from_=raw_from, to=raw_to)  # type: ignore[arg-type]


class DiagCheckOut(BaseModel):
    valid: bool
    checked_count: int
    broken_event_id: str | None


class DiagRecordOut(BaseModel):
    tid: str
    at: str
    phase: str
    op: str
    fail: str
    flags: list[str]
    status: Literal["committed", "rolled_back"]
    machine_status: str
    event: str | None
    count: int
    check: DiagCheckOut


class DiagOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    from_: str = Field(serialization_alias="from")
    to: str
    records: list[DiagRecordOut]
    check: DiagCheckOut


@app.get("/machines/{machine_id}/diag", response_model=DiagOut)
def get_machine_diagnostics(
    machine_id: str,
    params: Annotated[DiagParams, Depends(validate_diag_params)],
    session: SessionDep,
):
    """Read-only diagnostics for one machine's joint-write transaction attempts.

    Returns every finalized diagnostic in the closed UTC window, ordered by
    the actual UTC instant of ``at`` then ``tid``, plus a top-level ``check``
    of the machine's current event hash chain. Each record carries its own
    ``check`` snapshot in the event-integrity-audit shape. The query only
    reads: it never writes, repairs, recomputes, or deletes a diagnostic or
    any business record, and only the path machine's records are returned.
    A missing machine returns ``404 not_found``.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_)
    window_end = parse_utc_z_datetime(params.to)

    rows = diagnostics.fetch_records(session, machine_id, window_start, window_end)
    records = []
    for row in rows:
        mapping = row._mapping
        records.append(
            DiagRecordOut(
                tid=mapping["id"],
                at=mapping["at"],
                phase=mapping["phase"],
                op=mapping["op"],
                fail=mapping["fail"],
                flags=json.loads(mapping["flags"]),
                status=mapping["status"],
                machine_status=mapping["machine_status"] or "",
                event=mapping["event"],
                count=mapping["count"],
                check=DiagCheckOut(
                    valid=bool(mapping["check_valid"]),
                    checked_count=mapping["check_checked_count"],
                    broken_event_id=mapping["check_broken_event_id"],
                ),
            )
        )

    valid, checked_count, broken_event_id = chain.verify_chain(session, machine_id)
    return DiagOut(
        id=machine_id,
        from_=params.from_,
        to=params.to,
        records=records,
        check=DiagCheckOut(
            valid=valid,
            checked_count=checked_count,
            broken_event_id=broken_event_id,
        ),
    )


# --- read-only machine-level accountability compliance export --------------


class AccountabilityComplianceExportOut(BaseModel):
    machine_id: str
    from_created_at: str
    to_created_at: str
    events: list[AuthorizationDecisionEventOut]
    evidence: list[AccountabilityEvidenceOut]
    incidents: list[IncidentOut]
    status_history: list[IncidentStatusEventOut]
    responsibility_assignments: list[ResponsibilityAssignmentOut]


def _machine_rows_in_window(session, model, machine_id: str, window_start, window_end):
    """Load one machine's rows of a table and apply the closed UTC window.

    Membership is decided by the row's own ``created_at`` parsed to a UTC
    instant and its ``machine_id`` only; a missing, foreign, or otherwise
    damaged related-object reference never filters a row out (exports are
    raw, never repaired). Rows are ordered by the actual UTC instant of
    ``created_at`` and then by ``id``, so an exact-second record sorts before
    any fractional-second record of the same second. Issues reads only.
    """
    rows = session.scalars(
        select(model).where(model.machine_id == machine_id)
    ).all()
    in_window = [
        row
        for row in rows
        if window_start <= parse_utc_z_datetime(row.created_at) <= window_end
    ]
    return order_by_created_at_instant(in_window)


@app.get(
    "/machines/{machine_id}/accountability/compliance-export",
    response_model=AccountabilityComplianceExportOut,
)
def export_machine_accountability(
    machine_id: str,
    params: Annotated[
        ComplianceExportParams, Depends(validate_accountability_export_params)
    ],
    session: SessionDep,
):
    """Read-only machine-level closed-loop accountability slice over a window.

    The response carries five accountability record groups, each containing
    only path-machine records whose own ``created_at`` falls inside the closed
    interval ``[from_created_at, to_created_at]``, each ordered by the actual
    UTC instant of ``created_at`` and then by id (an exact-second record sorts
    before a fractional-second record of the same second), and each emitted as
    an empty array when the window contains nothing:

    - ``events`` — authorization decision events with their decision result
      (``allowed``/``reason``) and integrity-chain fields
      (``previous_event_id``/``content_hash``/``chain_hash``);
    - ``evidence`` — evidence records with their raw ``content_hash``
      fingerprint, exported even when the referenced event is missing or owned
      by another machine;
    - ``incidents`` — registered incidents with their registration content and
      current lifecycle status;
    - ``status_history`` — incident status transitions with their
      before/after statuses, following the existing machine/incident
      ownership;
    - ``responsibility_assignments`` — responsibility attributions with their
      party, role, and chain fields, following the existing machine/incident
      ownership.

    Records are exported exactly as stored: a missing or misowned related
    object never causes filtering, rewriting, or repair, and only the path
    machine's records are returned. Event associations (causal links) require
    both endpoints to be events of this export; the other three dependent
    groups follow their existing machine/entity ownership alone. The query
    issues no writes, repairs, deletions, recomputations, or normalizations,
    produces byte-identical output for identical data and parameters on repeat
    calls, and reads records persisted across application restarts.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    window_start = parse_utc_z_datetime(params.from_created_at)
    window_end = parse_utc_z_datetime(params.to_created_at)

    events = _machine_rows_in_window(
        session, AuthorizationDecisionEvent, machine_id, window_start, window_end
    )
    evidence = _machine_rows_in_window(
        session,
        AuthorizationDecisionEvidence,
        machine_id,
        window_start,
        window_end,
    )
    incidents = _machine_rows_in_window(
        session,
        AuthorizationDecisionIncident,
        machine_id,
        window_start,
        window_end,
    )
    status_history = _machine_rows_in_window(
        session, IncidentStatusEvent, machine_id, window_start, window_end
    )
    assignments = _machine_rows_in_window(
        session,
        IncidentResponsibilityAssignment,
        machine_id,
        window_start,
        window_end,
    )

    return AccountabilityComplianceExportOut(
        machine_id=machine_id,
        from_created_at=params.from_created_at,
        to_created_at=params.to_created_at,
        events=[decision_event_to_out(event) for event in events],
        evidence=[accountability_evidence_to_out(record) for record in evidence],
        incidents=[incident_to_out(record) for record in incidents],
        status_history=[
            incident_status_event_to_out(record) for record in status_history
        ],
        responsibility_assignments=[
            responsibility_assignment_to_out(record) for record in assignments
        ],
    )


# --- read-only machine-level integrity summary ------------------------------


def validate_integrity_summary_params(request: Request) -> None:
    """Validate the integrity-summary query string before any machine lookup.

    The summary is keyed on the path machine alone and accepts no query
    parameters; any parameter name is a 422 ``invalid_query``. The check runs
    before the machine is looked up and issues no database access, so an extra
    parameter against a non-existent machine still reports 422 rather than 404.
    """
    if request.query_params:
        raise QueryError("invalid_query")


class IntegritySummaryOut(BaseModel):
    machine_id: str
    valid: bool
    events: IntegrityOut
    rotations: KeyRotationIntegrityOut
    evidence: EvidenceIntegrityOut
    incidents: IncidentIntegrityOut


@app.get(
    "/machines/{machine_id}/integrity-summary",
    response_model=IntegritySummaryOut,
)
def get_machine_integrity_summary(
    machine_id: str,
    _: Annotated[None, Depends(validate_integrity_summary_params)],
    session: SessionDep,
):
    """Read-only aggregate integrity summary for one machine.

    Runs the four existing machine-level audits back to back, each in its
    audit's stable order, and returns their single-audit result shapes:

    - ``events`` — the authorization decision event hash chain
      (``{valid, checked_count, broken_event_id}``);
    - ``rotations`` — the key rotation hash chain
      (``{valid, checked_count, broken_rotation_id}``);
    - ``evidence`` — evidence ownership/type/fingerprint audit
      (``{valid, checked_count, broken_evidence_id}``);
    - ``incidents`` — incident lifecycle and responsibility-closure audit
      (``{valid, checked_count, broken_incident_id}``).

    The top-level ``valid`` is true only when all four blocks pass; any block's
    first anomaly makes it false. A machine with no records still returns the
    complete response, every block reporting zero records, valid, and no broken
    id. Only records owned by the path machine are examined, so damage under
    another machine never changes the result. The query is strictly read-only
    — it never creates, updates, deletes, repairs, recomputes, or normalizes
    a machine or any accountability record — gives identical results on repeat
    calls against unchanged data, and reads data persisted across application
    restarts. A missing machine returns ``404 not_found`` with no summary
    data; any query parameter returns ``422 invalid_query`` before the machine
    is looked up; non-GET methods return ``405``.
    """
    machine = session.get(Machine, machine_id)
    if machine is None:
        return error_response(404, "not_found")

    events_valid, events_count, broken_event_id = chain.verify_chain(
        session, machine_id
    )
    rotations_valid, rotations_count, broken_rotation_id = (
        rotation_chain.verify_chain(session, machine_id)
    )
    evidence_count, broken_evidence_id = find_broken_evidence(session, machine_id)
    incidents_count, broken_incident_id = find_broken_incident(session, machine_id)

    events_block = IntegrityOut(
        valid=events_valid,
        checked_count=events_count,
        broken_event_id=broken_event_id,
    )
    rotations_block = KeyRotationIntegrityOut(
        valid=rotations_valid,
        checked_count=rotations_count,
        broken_rotation_id=broken_rotation_id,
    )
    evidence_block = EvidenceIntegrityOut(
        valid=broken_evidence_id is None,
        checked_count=evidence_count,
        broken_evidence_id=broken_evidence_id,
    )
    incidents_block = IncidentIntegrityOut(
        valid=broken_incident_id is None,
        checked_count=incidents_count,
        broken_incident_id=broken_incident_id,
    )

    return IntegritySummaryOut(
        machine_id=machine_id,
        valid=all(
            (
                events_block.valid,
                rotations_block.valid,
                evidence_block.valid,
                incidents_block.valid,
            )
        ),
        events=events_block,
        rotations=rotations_block,
        evidence=evidence_block,
        incidents=incidents_block,
    )
