"""Response segment ORM model.

Classes:
    ResponseSegment: Persists discourse segments along with embeddings, roles, and cluster metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, LargeBinary, BigInteger
from sqlmodel import Field, SQLModel


class ResponseSegment(SQLModel, table=True):
    __tablename__ = "response_segments"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    response_id: UUID = Field(foreign_key="responses.id")
    position: int
    text: str
    role: Optional[str] = None
    tokens: Optional[int] = None
    prompt_similarity: Optional[float] = None
    silhouette_score: Optional[float] = None
    cluster_label: Optional[int] = None
    cluster_probability: Optional[float] = None
    cluster_similarity: Optional[float] = None
    outlier_score: Optional[float] = None
    embedding_dim: Optional[int] = None
    embedding_vector: Optional[bytes] = Field(
        default=None, sa_column=Column(LargeBinary)
    )
    coord_x: float = Field(default=0.0)
    coord_y: float = Field(default=0.0)
    coord_z: float = Field(default=0.0)
    coord2_x: float = Field(default=0.0)
    coord2_y: float = Field(default=0.0)
    text_hash: Optional[str] = Field(default=None, index=True)
    is_cached: bool = Field(default=False)
    is_duplicate: bool = Field(default=False)
    simhash64: Optional[int] = Field(default=None, sa_column=Column(BigInteger))
    created_at: datetime = Field(default_factory=datetime.utcnow)
