"""Response embedding persistence model.

Classes:
    Embedding: Persists the embedding vector produced for a response along with dimensionality metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel


class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"

    response_id: UUID = Field(foreign_key="responses.id", primary_key=True)
    dim: int
    vector: bytes = Field(sa_column=Column(LargeBinary))
    created_at: datetime = Field(default_factory=datetime.utcnow)


