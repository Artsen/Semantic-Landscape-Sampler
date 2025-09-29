"""Pydantic schemas for run lifecycle and analytics payloads.

Classes:
    RunCreateRequest, RunCreatedResponse, RunResource, SampleRequest: Manage run configuration workflows.
    ResponsePoint, SegmentPoint, ResponseHull, SegmentEdge: Visualisation payload primitives.
    ClusterSummary, SegmentClusterSummary: Aggregate metrics for clustered responses/segments.
    RunResultsResponse, ExportRow, UsageInfo: Response formats returned by result and export endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator


class RunCreateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    n: int = Field(ge=1, le=500)
    model: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=16, le=4096)
    chunk_size: Optional[int] = Field(default=None, ge=1, le=30)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=30)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("chunk_overlap")
    @classmethod
    def clamp_chunk_overlap(
        cls,
        value: Optional[int],
        info: ValidationInfo,
    ) -> Optional[int]:
        if value is None:
            return None
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None:
            max_overlap = max(chunk_size - 1, 0)
            return min(value, max_overlap)
        return value

    embedding_model: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("prompt")
    @classmethod
    def trim_prompt(cls, value: str) -> str:
        return value.strip()

    @field_validator("notes")
    @classmethod
    def normalise_notes(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("system_prompt")
    @classmethod
    def normalise_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None


class RunCreatedResponse(BaseModel):
    run_id: UUID



class RunUpdateRequest(BaseModel):
    notes: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("notes")
    @classmethod
    def normalise_notes(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None


class RunResource(BaseModel):
    id: UUID
    prompt: str
    n: int
    model: str
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    system_prompt: Optional[str]
    embedding_model: str
    temperature: float
    top_p: Optional[float]
    seed: Optional[int]
    max_tokens: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    notes: Optional[str] = None
    progress_stage: Optional[str] = None
    progress_message: Optional[str] = None
    progress_percent: Optional[float] = None
    progress_metadata: Optional[dict[str, Any]] = None


class RunSummary(BaseModel):
    id: UUID
    prompt: str
    n: int
    model: str
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    system_prompt: Optional[str]
    embedding_model: str
    temperature: float
    top_p: Optional[float]
    seed: Optional[int]
    max_tokens: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime
    response_count: int
    segment_count: int
    notes: Optional[str] = None
    progress_stage: Optional[str] = None
    progress_message: Optional[str] = None
    progress_percent: Optional[float] = None
    progress_metadata: Optional[dict[str, Any]] = None


class SampleRequest(BaseModel):
    jitter_prompt_token: Optional[str] = None
    force_refresh: bool = False
    overwrite_previous: bool = False
    callback_url: Optional[HttpUrl] = None
    include_segments: bool = True
    include_discourse_tags: bool = True


class UsageInfo(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ResponsePoint(BaseModel):
    id: UUID
    index: int
    text_preview: str
    full_text: str
    tokens: Optional[int]
    finish_reason: Optional[str]
    usage: Optional[UsageInfo]
    cluster: Optional[int]
    probability: Optional[float]
    similarity_to_centroid: Optional[float]
    outlier_score: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    embedding_tokens: Optional[int] = None
    completion_cost: Optional[float] = None
    embedding_cost: Optional[float] = None
    total_cost: Optional[float] = None
    coords_3d: tuple[float, float, float]
    coords_2d: tuple[float, float]


class SegmentPoint(BaseModel):
    id: UUID
    response_id: UUID
    response_index: int
    position: int
    text: str
    role: Optional[str]
    tokens: Optional[int]
    embedding_tokens: Optional[int] = None
    embedding_cost: Optional[float] = None
    prompt_similarity: Optional[float]
    silhouette_score: Optional[float]
    cluster: Optional[int]
    probability: Optional[float]
    similarity_to_centroid: Optional[float]
    outlier_score: Optional[float]
    coords_3d: tuple[float, float, float]
    coords_2d: tuple[float, float]


class SegmentEdge(BaseModel):
    source_id: UUID
    target_id: UUID
    score: float


class ClusterSummary(BaseModel):
    label: int
    size: int
    centroid_xyz: tuple[float, float, float]
    exemplar_ids: list[UUID]
    average_similarity: Optional[float]
    method: str
    keywords: list[str] = Field(default_factory=list)
    noise: bool = False


class SegmentClusterSummary(BaseModel):
    label: int
    size: int
    exemplar_ids: list[UUID]
    average_similarity: Optional[float]
    method: str
    keywords: list[str] = Field(default_factory=list)
    theme: Optional[str] = None
    noise: bool = False


class ResponseHull(BaseModel):
    response_id: UUID
    coords_2d: list[tuple[float, float]]
    coords_3d: list[tuple[float, float, float]]


class RunCostSummary(BaseModel):
    model: str
    embedding_model: str
    completion_input_tokens: int
    completion_output_tokens: int
    completion_cost: float
    embedding_tokens: int
    embedding_cost: float
    total_cost: float


class RunResultsResponse(BaseModel):
    run: RunResource
    points: list[ResponsePoint]
    clusters: list[ClusterSummary]
    segments: list[SegmentPoint]
    segment_clusters: list[SegmentClusterSummary]
    segment_edges: list[SegmentEdge]
    response_hulls: list[ResponseHull]
    prompt: str
    model: str
    system_prompt: Optional[str]
    embedding_model: str
    n: int
    costs: RunCostSummary
    chunk_size: Optional[int] | None = None
    chunk_overlap: Optional[int] | None = None


class ExportRow(BaseModel):
    response_id: UUID
    index: int
    text: str
    tokens: Optional[int]
    cluster: Optional[int]
    probability: Optional[float]
    x: float
    y: float
    z: Optional[float]

    def to_csv_row(self) -> list[Any]:
        return [
            str(self.response_id),
            self.index,
            self.text,
            self.tokens if self.tokens is not None else "",
            self.cluster if self.cluster is not None else "",
            f"{self.probability:.4f}" if self.probability is not None else "",
            f"{self.x:.6f}",
            f"{self.y:.6f}",
            f"{self.z:.6f}" if self.z is not None else "",
        ]
