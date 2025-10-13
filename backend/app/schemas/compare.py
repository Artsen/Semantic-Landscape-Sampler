"""Schemas for run comparison endpoints.

Classes:
    CompareRunsRequest: Validate payload for triggering a run comparison.
    ComparisonTransformComponent: Describes an alignment transform for a specific dimensionality.
    ComparisonTransformSummary: Aggregated transform metadata reported back to the client.
    MovementHistogramBin: Simple bucketed histogram for movement magnitudes.
    TopTermShift: Captures keyword overlap deltas between matching clusters.
    ComparisonMetrics: Diff metrics summarising alignment quality and cluster deltas.
    ComparisonLink: Segment linkage metadata between the two runs.
    ComparisonPoint: Visualisation payload representing aligned points for each run.
    CompareRunsResponse: Full payload returned by the comparison endpoint.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.run import RunResource


class ComparisonMode(str, Enum):
    SEGMENTS = "segments"
    RESPONSES = "responses"


class CompareRunsRequest(BaseModel):
    left_run_id: UUID
    right_run_id: UUID
    mode: ComparisonMode = ComparisonMode.SEGMENTS
    min_shared: int = Field(default=5, ge=0, le=1000)
    save: bool = Field(default=True)
    histogram_bins: int = Field(default=10, ge=4, le=60)
    max_links: Optional[int] = Field(default=None, ge=1, le=5000)


class ComparisonTransformComponent(BaseModel):
    dimension: Literal["2d", "3d"]
    rotation: list[list[float]]
    scale: float
    translation: list[float]
    rmsd: Optional[float] = None
    anchor_count: int


class ComparisonTransformSummary(BaseModel):
    method: str
    anchor_kind: str
    anchor_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, ComparisonTransformComponent] = Field(default_factory=dict)


class MovementHistogramBin(BaseModel):
    start: float
    end: float
    count: int


class TopTermShift(BaseModel):
    cluster_label: int
    left_terms: list[str] = Field(default_factory=list)
    right_terms: list[str] = Field(default_factory=list)
    shared_terms: list[str] = Field(default_factory=list)
    jaccard: Optional[float] = None


class ComparisonMetrics(BaseModel):
    shared_segment_count: int
    linked_segment_count: int
    alignment_method: str
    ari: Optional[float] = None
    nmi: Optional[float] = None
    mean_movement: Optional[float] = None
    median_movement: Optional[float] = None
    max_movement: Optional[float] = None
    movement_histogram: list[MovementHistogramBin] = Field(default_factory=list)
    cluster_count_left: int
    cluster_count_right: int
    delta_cluster_count: int
    mean_cluster_size_left: Optional[float] = None
    mean_cluster_size_right: Optional[float] = None
    median_cluster_size_left: Optional[float] = None
    median_cluster_size_right: Optional[float] = None
    noise_left: int
    noise_right: int
    top_term_shifts: list[TopTermShift] = Field(default_factory=list)


class ComparisonLink(BaseModel):
    left_segment_id: UUID
    right_segment_id: UUID
    link_type: Literal["exact_hash", "nn"]
    distance: Optional[float] = None
    text_hash: Optional[str] = None
    movement_vector: Optional[tuple[float, float, float]] = None
    movement_distance: Optional[float] = None


class ComparisonPoint(BaseModel):
    id: UUID
    run_id: UUID
    response_id: UUID
    response_index: Optional[int]
    segment_position: Optional[int]
    source: Literal["left", "right"]
    text_hash: Optional[str]
    cluster_label: Optional[int]
    cluster_probability: Optional[float]
    cluster_similarity: Optional[float]
    role: Optional[str]
    prompt_similarity: Optional[float]
    text_preview: Optional[str] = None
    coords_3d: tuple[float, float, float]
    coords_2d: tuple[float, float]
    aligned_coords_3d: tuple[float, float, float]
    aligned_coords_2d: tuple[float, float]
    is_anchor: bool = False


class CompareRunsResponse(BaseModel):
    run_pair_id: Optional[UUID]
    left_run: RunResource
    right_run: RunResource
    transforms: ComparisonTransformSummary
    points: list[ComparisonPoint]
    links: list[ComparisonLink]
    metrics: ComparisonMetrics
    anchor_kind: str

