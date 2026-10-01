from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Machine(Base):
    __tablename__ = "machines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    public_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class MachineStatusEvent(Base):
    """One immutable machine status transition (``active`` <-> ``suspended``).

    Rows are append-only: each accepted status change inserts exactly one row
    in the same locked write transaction that updates the machine, and rows
    are never updated or deleted afterwards, so the table is a complete
    per-machine history of every committed status change. Each row also
    carries a per-machine tamper-evident hash chain
    (``previous_status_event_id``/``content_hash``/``chain_hash``) computed
    from the record identifier, ownership, status edge, and creation moment.
    """

    __tablename__ = "machine_status_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    from_status: Mapped[str] = mapped_column(String, nullable=False)
    to_status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_status_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class KeyRotationEvent(Base):
    __tablename__ = "key_rotation_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    old_public_key: Mapped[str] = mapped_column(String, nullable=False)
    new_public_key: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_rotation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class BehaviorDeclaration(Base):
    __tablename__ = "behavior_declarations"
    __table_args__ = (
        UniqueConstraint(
            "machine_id",
            "action_type",
            "resource_pattern",
            name="uq_behavior_declaration_machine_action_resource",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String, nullable=False)
    resource_pattern: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class PolicyRule(Base):
    __tablename__ = "policy_rules"
    __table_args__ = (
        UniqueConstraint(
            "action_type",
            "resource_pattern",
            "priority",
            name="uq_policy_rule_action_resource_priority",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action_type: Mapped[str] = mapped_column(String, nullable=False)
    resource_pattern: Mapped[str] = mapped_column(String, nullable=False)
    effect: Mapped[str] = mapped_column(String, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    # Global tamper-evident hash chain across all policy rules. Nullable so
    # databases created before the chain feature keep working; the startup
    # migration backfills any missing values.
    previous_rule_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuthorizationDecisionEvent(Base):
    __tablename__ = "authorization_decision_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String, nullable=False)
    resource: Mapped[str] = mapped_column(String, nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuthorizationDecisionBasis(Base):
    """Immutable decision-basis snapshot captured for one decision event.

    A row is written in the same locked write transaction that appends its
    event, so the snapshot and the event either commit together or leave no
    trace. Rows are append-only and never updated, deleted, or recomputed
    afterwards: the read-only ``decision-basis`` query only re-emits the
    stored document. Events written before the feature existed have no row;
    querying their basis answers ``404 snapshot_not_found`` rather than
    fabricating a basis from current data. The table is created automatically
    at startup on databases that predate the feature.

    Only visible business fields are recorded (machine status, enabled
    declaration matching scope, policy candidate relations, the event result
    and reason, and the capture moment) — never machine keys or policy text
    beyond the rule's existing visible columns.
    """

    __tablename__ = "authorization_decision_basis"

    # One snapshot per decision event; the event id is the natural key and
    # the machine column enforces strict path-machine isolation.
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        primary_key=True,
    )
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # The complete five-group basis document, stored as the exact compact
    # key-sorted JSON text later re-emitted byte-for-byte.
    document: Mapped[str] = mapped_column(Text, nullable=False)


class AuthorizationDecisionEvidence(Base):
    __tablename__ = "authorization_decision_evidence"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "content_hash",
            name="uq_evidence_event_content_hash",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    evidence_type: Mapped[str] = mapped_column(String, nullable=False)
    # Client-supplied evidence fingerprint: exactly 64 lowercase hexadecimal
    # characters, compared as stored with no case folding. Distinct from the
    # per-record chain digest below, which covers this fingerprint together
    # with the record's other content fields.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine evidence hash chain. Nullable so databases created before
    # the evidence-chain feature keep working; the startup migration
    # backfills any missing values.
    previous_evidence_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    # SHA-256 content digest of the record's six content fields; the name
    # stays distinct from ``content_hash`` (the evidence fingerprint).
    content_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuthorizationDecisionIncident(Base):
    __tablename__ = "authorization_decision_incidents"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "incident_type",
            "summary",
            name="uq_incident_event_type_summary",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    incident_type: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="open")
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class IncidentStatusEvent(Base):
    """One immutable incident status transition (``open -> acknowledged`` or
    ``acknowledged -> resolved``). Rows are append-only: they are never updated
    or deleted, so the table is a complete, tamper-evident-by-construction
    history of every status change. Each row also carries a per-machine
    tamper-evident hash chain
    (``previous_status_event_id``/``content_hash``/``chain_hash``) computed
    from the record, ownership, status-edge, and creation-moment columns.
    """

    __tablename__ = "incident_status_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_incidents.id"),
        nullable=False,
        index=True,
    )
    from_status: Mapped[str] = mapped_column(String, nullable=False)
    to_status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_status_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class IncidentResponsibilityAssignment(Base):
    """Responsibility attribution for one registered incident.

    A ``(party, role)`` pair can be assigned to a given incident at most once;
    the same pair on another incident is a distinct assignment. Rows live in
    their own table and never modify incidents, events, evidence, chains, or
    links.
    """

    __tablename__ = "incident_responsibility_assignments"
    __table_args__ = (
        UniqueConstraint(
            "incident_id",
            "party",
            "role",
            name="uq_responsibility_assignment_incident_party_role",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_incidents.id"),
        nullable=False,
        index=True,
    )
    party: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_assignment_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuthorizationDecisionCausalLink(Base):
    __tablename__ = "authorization_decision_causal_links"
    __table_args__ = (
        UniqueConstraint(
            "machine_id",
            "cause_event_id",
            "effect_event_id",
            name="uq_causal_link_machine_cause_effect",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    cause_event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    effect_event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class PrivacyAccess(Base):
    """One registered machine-level privacy data access.

    Rows are append-only audit of when a machine accessed desensitized data
    and over which export window, with the access result and the number of
    records hit. No responsible-party rawtext or key material is ever stored:
    the columns carry only operational access metadata. A repeat registration
    with the same ``(machine_id, accessed_at, window_start, window_end,
    result)`` tuple is a duplicate and is never written; the same access at a
    different time, window, or result is a distinct record. Each row also
    carries a per-machine tamper-evident hash chain
    (``previous_access_id``/``content_hash``/``chain_hash``) computed from the
    seven audit columns only.
    """

    __tablename__ = "privacy_accesses"
    __table_args__ = (
        UniqueConstraint(
            "machine_id",
            "accessed_at",
            "window_start",
            "window_end",
            "result",
            name="uq_privacy_access_machine_time_window_result",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    accessed_at: Mapped[str] = mapped_column(String, nullable=False)
    window_start: Mapped[str] = mapped_column(String, nullable=False)
    window_end: Mapped[str] = mapped_column(String, nullable=False)
    result: Mapped[str] = mapped_column(String, nullable=False)
    matches_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Per-machine hash chain. Nullable so databases created before the chain
    # feature keep working; the startup migration backfills any missing values.
    previous_access_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chain_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuthorizationGrant(Base):
    """One one-time, short-lived authorization credential for one event.

    A grant is minted only for an existing decision event of the path machine
    that committed ``allowed = true`` with ``reason = "allowed_by_policy"`` and
    whose immutable decision-basis snapshot passes the read-only consistency
    audit. Rows are never updated after issue except the one atomic terminal
    transition performed by the consume or revoke operation, and
    ``event_id`` is unique: at most one grant can ever be signed per event,
    enforced by the database itself so a concurrent burst of issue requests
    has exactly one winner. The table is created automatically at startup on
    databases that predate the feature, and the ``revoked_at`` column is
    added safely to databases created before emergency revocation.
    """

    __tablename__ = "authorization_grants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        unique=True,
    )
    issued_at: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[str] = mapped_column(String, nullable=False)
    # ``active`` after issue; exactly one terminal transition follows:
    # ``consumed`` exactly once (with its single use record) or ``revoked``
    # once by an emergency revocation. Grants are never renewed or
    # transferred; expiry is a derived state checked against ``expires_at``
    # and needs no column update.
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    consumed_at: Mapped[str | None] = mapped_column(String, nullable=True)
    # Set only by the one-way ``active`` -> ``revoked`` transition. Nullable
    # so databases created before the revocation feature keep working; the
    # startup migration adds the column and leaves every old grant's value
    # null. Stays null on consumed and expired grants.
    revoked_at: Mapped[str | None] = mapped_column(String, nullable=True)


class AuthorizationGrantUse(Base):
    """The single consumption record of one authorization grant.

    Written in the same locked transaction that flips the grant to
    ``consumed``: the grant state change and the use record commit together or
    leave no trace, so exactly one use row can ever exist per grant (enforced
    by the unique ``grant_id``), even under concurrent consumption.
    """

    __tablename__ = "authorization_grant_uses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    grant_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_grants.id"),
        nullable=False,
        unique=True,
    )
    machine_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("machines.id"), nullable=False, index=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("authorization_decision_events.id"),
        nullable=False,
        index=True,
    )
    consumed_at: Mapped[str] = mapped_column(String, nullable=False)


class WriteTransactionDiagnostic(Base):
    """One read-only diagnostic record per joint-write transaction attempt.

    A joint write is one of the two public mutating operations serialized by
    the same machine write lock: a machine status change (``op = "change"``)
    and an authorization decision-event creation (``op = "event"``). The
    table is append-only observability: the records are written in their own
    transaction(s), never updated or deleted, and never participate in
    authorization evaluation. Columns store only operational metadata — no
    keys, secrets, policy text, or identity material.

    * ``tid`` — stable id of the diagnostic record (a fresh UUID per attempt);
    * ``at`` — UTC RFC 3339 timestamp (ending in ``Z``) of the attempt's
      terminal outcome, also the ordering key;
    * ``phase`` — ``started-commit`` (the joint write committed) or
      ``started-rollback`` (it rolled back, with a stable failure category in
      ``fail``);
    * ``fail`` — ``none`` for committed attempts, otherwise ``race``, ``io``,
      or ``other``;
    * ``flags`` — ``[]`` or a JSON array listing ``lock_wait`` and ``retry``
      in the order actually experienced (a lock wait and its retries collapse
      into this one record);
    * ``status`` — the attempt's transaction terminal state: ``committed``
      when the joint write landed or ``rolled_back`` when it did not;
    * ``machine_status`` — the machine's ``active``/``suspended`` state at
      the terminal state, kept separate from the transaction outcome;
    * ``event`` — the created event id for ``op = "event"`` commits, else
      ``null``;
    * ``count`` — the machine's decision-event count after the attempt.
    """

    __tablename__ = "write_transaction_diagnostics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    op: Mapped[str] = mapped_column(String(16), nullable=False)
    at: Mapped[str] = mapped_column(String, nullable=False, index=True)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    fail: Mapped[str] = mapped_column(String(16), nullable=False)
    flags: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    status: Mapped[str] = mapped_column(String, nullable=False)
    # The machine's own active/suspended state at the terminal state, distinct
    # from the transaction outcome stored in ``status``. Nullable so databases
    # created before the split keep working; the startup migration backfills
    # it from the pre-split ``status`` value.
    machine_status: Mapped[str | None] = mapped_column(String, nullable=True)
    event: Mapped[str | None] = mapped_column(String(36), nullable=True)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Snapshot of the machine's event hash-chain audit at the attempt's
    # terminal state, with the same shape as the event integrity endpoint:
    # {valid, checked_count, broken_event_id}.
    check_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    check_checked_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    check_broken_event_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    # Transient start timestamp of the attempt; copied into ``at`` when the
    # marker is finalized and used to classify crash residuals at startup.
    started_at: Mapped[str] = mapped_column(String, nullable=False)
