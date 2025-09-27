"""Response ORM model.

Classes:
    Response: Captures raw LLM output text plus token usage metadata for a sampled run.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


class Response(SQLModel, table=True):
    __tablename__ = "responses"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id")
    index: int
    raw_text: str
    tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    usage_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))


