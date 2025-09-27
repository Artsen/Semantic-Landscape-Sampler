"""Sampling run ORM model.

Classes:
    RunStatus: Simple enumeration of valid run lifecycle states.
    Run: Records the prompt, sampling parameters, and status for a sampling run.

Functions:
    set_updated_at(_, __, target): SQLAlchemy event hook that maintains the `updated_at` timestamp.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, Text, event
from sqlmodel import Field, SQLModel


class RunStatus(str):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    prompt: str
    n: int
    model: str
    temperature: float
    top_p: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    status: str = Field(default=RunStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    notes: Optional[str] = Field(default=None, sa_column=Column(Text))


@event.listens_for(Run, "before_update", propagate=True)
def set_updated_at(_, __, target):
    target.updated_at = datetime.utcnow()



