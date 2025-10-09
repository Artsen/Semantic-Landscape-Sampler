"""Segment insight metadata for tooltips and neighbor context."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class SegmentInsight(SQLModel, table=True):
    __tablename__ = "segment_insights"

    segment_id: UUID = Field(foreign_key="response_segments.id", primary_key=True)
    top_terms_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    neighbors_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    cluster_exemplar_id: Optional[UUID] = Field(default=None)
    metrics_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)

