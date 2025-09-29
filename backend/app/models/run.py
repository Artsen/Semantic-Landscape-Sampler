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

from sqlalchemy import Column, Float, Integer, Text, event
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
    chunk_size: Optional[int] = Field(default=None, sa_column=Column(Integer))
    chunk_overlap: Optional[int] = Field(default=None, sa_column=Column(Integer))
    system_prompt: Optional[str] = Field(default=None, sa_column=Column(Text))
    embedding_model: str = Field(default="text-embedding-3-large")
    status: str = Field(default=RunStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    notes: Optional[str] = Field(default=None, sa_column=Column(Text))
    progress_stage: Optional[str] = Field(default=None, sa_column=Column(Text))
    progress_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    progress_percent: Optional[float] = Field(default=None, sa_column=Column(Float))
    progress_metadata: Optional[str] = Field(default=None, sa_column=Column(Text))


@event.listens_for(Run, "before_update", propagate=True)
def set_updated_at(_, __, target):
    target.updated_at = datetime.utcnow()
