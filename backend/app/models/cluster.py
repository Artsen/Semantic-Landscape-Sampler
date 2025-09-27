"""Cluster ORM model definition.

Classes:
    Cluster: Stores clustering metadata for a response including method, label, probability, similarity, and outlier score.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class Cluster(SQLModel, table=True):
    __tablename__ = "clusters"
    __table_args__ = (
        UniqueConstraint("response_id", "method", name="uq_cluster_response_method"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    response_id: UUID = Field(foreign_key="responses.id")
    method: str
    label: int
    probability: Optional[float] = None
    similarity: Optional[float] = None
    outlier_score: Optional[float] = None

