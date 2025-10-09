"""Cluster metrics persistence models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, Float, Integer, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


class ClusterMetrics(SQLModel, table=True):
    __tablename__ = "cluster_metrics"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_cluster_metrics_run_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id", index=True)
    algo: str = Field(default="hdbscan")
    params_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    silhouette_embed: Optional[float] = Field(default=None, sa_column=Column(Float))
    silhouette_feature: Optional[float] = Field(default=None, sa_column=Column(Float))
    davies_bouldin: Optional[float] = Field(default=None, sa_column=Column(Float))
    calinski_harabasz: Optional[float] = Field(default=None, sa_column=Column(Float))
    n_clusters: Optional[int] = Field(default=None, sa_column=Column(Integer))
    n_noise: Optional[int] = Field(default=None, sa_column=Column(Integer))
    stability_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    sweep_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow)
