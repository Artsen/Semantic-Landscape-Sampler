"""Projection cache model for reusable dimensionality reductions."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Column, Index, LargeBinary, UniqueConstraint
from sqlmodel import Field, SQLModel


class ProjectionCache(SQLModel, table=True):
    """Persist cached projection layouts to avoid recomputation."""

    __tablename__ = "projection_cache"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "method",
            "params_hash",
            "feature_version",
            name="uq_projection_cache_run_method_params",
        ),
        Index("ix_projection_cache_run_method", "run_id", "method"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id", index=True)
    method: str = Field(index=True)
    params_hash: str = Field(index=True)
    params_json: str
    feature_version: str = Field(index=True)
    coord_dtype: str = Field(default="float32")
    point_count: int
    total_count: int
    is_subsample: bool = Field(default=False)
    subsample_strategy: str | None = Field(default=None)
    coords_2d: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    coords_3d: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    response_ids_json: str
    trustworthiness_2d: float | None = Field(default=None)
    trustworthiness_3d: float | None = Field(default=None)
    continuity_2d: float | None = Field(default=None)
    continuity_3d: float | None = Field(default=None)
    warnings_json: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
