"""Run comparison metadata models.

Classes:
    RunPair: Stores alignment metadata for a pair of runs compared together.
    PointLink: Persists segment-level linkages between two runs within a comparison.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON, UniqueConstraint, Index
from sqlmodel import Field, SQLModel


class RunPair(SQLModel, table=True):
    """Persisted comparison metadata between two runs.

    Attributes:
        id: Primary key for the comparison pair.
        left_run_id: Identifier of the baseline/left run.
        right_run_id: Identifier of the run being aligned to the left run.
        alignment_method: Strategy used to align the right run to the left run.
        transform_json: Serialized alignment transform (rotation, scale, translation).
        created_at: Timestamp when the comparison metadata was recorded.
    """

    __tablename__ = "run_pairs"
    __table_args__ = (
        UniqueConstraint("left_run_id", "right_run_id", name="uq_run_pairs_runs"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    left_run_id: UUID = Field(foreign_key="runs.id")
    right_run_id: UUID = Field(foreign_key="runs.id")
    alignment_method: str = Field(default="procrustes", max_length=64)
    transform_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class PointLink(SQLModel, table=True):
    """Link between segments across two runs within a comparison pair.

    Attributes:
        id: Primary key for the link record.
        run_pair_id: Identifier of the owning run pair.
        left_segment_id: Segment identifier from the left run.
        right_segment_id: Segment identifier from the right run.
        link_type: Nature of the link (exact hash, nearest neighbour, etc.).
        distance: Optional similarity distance in feature space.
        created_at: Timestamp when the linkage was stored.
    """

    __tablename__ = "point_links"
    __table_args__ = (
        Index("ix_point_links_pair_left", "run_pair_id", "left_segment_id"),
        Index("ix_point_links_pair_right", "run_pair_id", "right_segment_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_pair_id: UUID = Field(foreign_key="run_pairs.id")
    left_segment_id: UUID = Field(foreign_key="response_segments.id")
    right_segment_id: UUID = Field(foreign_key="response_segments.id")
    link_type: str = Field(default="exact_hash", max_length=16)
    distance: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
