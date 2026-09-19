"""Database layer for machine identity persistence.

Uses SQLAlchemy with SQLite. The database URL can be overridden with the
``ACCOUNTABILITY_DATABASE_URL`` environment variable; by default a file-based
SQLite database is used so data survives application restarts.
"""

from __future__ import annotations

import os

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./accountability.db"

DATABASE_URL = os.environ.get("ACCOUNTABILITY_DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    pass


class Machine(Base):
    __tablename__ = "machines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    external_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    public_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


engine = make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db(eng=None) -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(eng if eng is not None else engine)


def get_session():
    """FastAPI dependency yielding a database session."""
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# Ensure the schema exists as soon as the module is imported, so the
# application is usable even without running lifespan startup hooks.
init_db()
