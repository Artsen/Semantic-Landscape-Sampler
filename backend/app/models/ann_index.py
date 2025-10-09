"""ANN index metadata model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class AnnIndex(SQLModel, table=True):
    __tablename__ = "ann_index"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id", unique=True, index=True)
    method: str = Field(default="annoy")
    params_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    vector_count: int = Field(default=0)
    built_at: datetime = Field(default_factory=datetime.utcnow)
    index_path: Optional[str] = Field(default=None)

