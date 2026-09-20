from sqlalchemy import Boolean, ForeignKey, Integer, String, UniqueConstraint
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
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


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
    history of every status change.
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
