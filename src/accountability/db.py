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
    # The machine version after the rotation was applied.
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


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
