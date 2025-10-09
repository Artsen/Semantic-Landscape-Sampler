"""Run provenance metadata model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class RunProvenance(SQLModel, table=True):
    __tablename__ = "run_provenance"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="runs.id", unique=True, index=True)
    python_version: Optional[str] = None
    node_version: Optional[str] = None
    blas_impl: Optional[str] = None
    openmp_threads: Optional[int] = None
    numba_version: Optional[str] = None
    numba_target: Optional[str] = None
    lib_versions_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    feature_weights_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    input_space_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    umap_params_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    cluster_params_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    commit_sha: Optional[str] = None
    env_label: Optional[str] = None
    random_state_seed_source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

