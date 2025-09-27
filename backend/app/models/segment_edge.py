"""Segment similarity edge model.

Classes:
    SegmentEdge: Stores similarity scores connecting semantically related response segments.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class SegmentEdge(SQLModel, table=True):
    __tablename__ = "segment_edges"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", name="uq_segment_edge_pair"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id")
    source_id: UUID = Field(foreign_key="response_segments.id")
    target_id: UUID = Field(foreign_key="response_segments.id")
    score: float

