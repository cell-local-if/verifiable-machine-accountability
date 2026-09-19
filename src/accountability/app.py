from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, BeforeValidator
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import Machine, get_session, init_db


def _stripped_non_empty(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("must be a string")
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must be a non-empty string")
    return trimmed


TrimmedNonEmptyStr = Annotated[str, BeforeValidator(_stripped_non_empty)]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MachineCreate(BaseModel):
    external_id: TrimmedNonEmptyStr
    display_name: TrimmedNonEmptyStr
    public_key: TrimmedNonEmptyStr


class RotateKeyRequest(BaseModel):
    public_key: TrimmedNonEmptyStr
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


def _error(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code}})


def _to_out(machine: Machine) -> MachineOut:
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Verifiable Machine Accountability", version="0.1.0", lifespan=lifespan)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/machines", status_code=201, response_model=MachineOut)
def create_machine(payload: MachineCreate, session: Session = Depends(get_session)):
    now = _utcnow()
    machine = Machine(
        id=str(uuid.uuid4()),
        external_id=payload.external_id,
        display_name=payload.display_name,
        public_key=payload.public_key,
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
        return _error(409, "duplicate_external_id")
    return _to_out(machine)


@app.get("/machines/{machine_id}", response_model=MachineOut)
def get_machine(machine_id: str, session: Session = Depends(get_session)):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return _error(404, "not_found")
    return _to_out(machine)


@app.post("/machines/{machine_id}/rotate-key", response_model=MachineOut)
def rotate_key(
    machine_id: str, payload: RotateKeyRequest, session: Session = Depends(get_session)
):
    machine = session.get(Machine, machine_id)
    if machine is None:
        return _error(404, "not_found")
    if machine.version != payload.expected_version:
        return _error(409, "version_conflict")
    if machine.public_key == payload.public_key:
        return _error(422, "public_key_unchanged")

    # Atomic conditional update: only one concurrent rotation for a given
    # version can win; the loser sees rowcount == 0 and gets a conflict.
    now = _utcnow()
    result = session.execute(
        update(Machine)
        .where(Machine.id == machine_id, Machine.version == payload.expected_version)
        .values(public_key=payload.public_key, version=Machine.version + 1, updated_at=now)
    )
    if result.rowcount == 0:
        session.rollback()
        return _error(409, "version_conflict")
    session.commit()

    updated = session.get(Machine, machine_id)
    return _to_out(updated)
