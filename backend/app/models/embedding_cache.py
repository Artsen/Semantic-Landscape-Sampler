"""Embedding cache model for deduplicating vectors across runs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Column, Index, LargeBinary, UniqueConstraint
from sqlmodel import Field, SQLModel


class EmbeddingCache(SQLModel, table=True):
    __tablename__ = "embedding_cache"
    __table_args__ = (
        UniqueConstraint(
            "text_hash",
            "model_id",
            "preproc_version",
            name="uq_embedding_cache_hash_model_preproc",
        ),
        Index(
            "ix_embedding_cache_hash_model_preproc",
            "text_hash",
            "model_id",
            "preproc_version",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    text_hash: str = Field(index=True)
    model_id: str = Field(index=True)
    preproc_version: str = Field(default="norm-nfkc-v1", index=True)
    provider: str | None = Field(default=None, index=True)
    model_revision: str | None = Field(default=None)
    vector: bytes = Field(sa_column=Column(LargeBinary))
    vector_dtype: str = Field(default="float16")
    vector_norm: float | None = Field(default=None)
    dim: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
