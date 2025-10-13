"""Pydantic schemas for run lifecycle and analytics payloads.
Classes:
    RunCreateRequest, RunCreatedResponse, RunResource, SampleRequest: Manage run configuration workflows.
    ResponsePoint, SegmentPoint, ResponseHull, SegmentEdge: Visualisation payload primitives.
    ClusterSummary, SegmentClusterSummary: Aggregate metrics for clustered responses/segments.
    RunResultsResponse, ExportRow, UsageInfo: Response formats returned by result and export endpoints.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator

class UMAPParamsRequest(BaseModel):
    n_neighbors: Optional[int] = None
    min_dist: Optional[float] = None
    metric: Optional[str] = None
    seed: Optional[int] = None

class UMAPParams(BaseModel):
    n_neighbors: int
    min_dist: float
    metric: str
    seed: Optional[int]
    seed_source: str

class QualityGauge(BaseModel):
    trustworthiness_2d: Optional[float]
    trustworthiness_3d: Optional[float]
    continuity_2d: Optional[float]
    continuity_3d: Optional[float]

class ProjectionMode(str, Enum):
    BOTH = "both"
    D2 = "2d"
    D3 = "3d"


class ProjectionVariantResponse(BaseModel):
    run_id: UUID
    method: str
    requested_params: dict[str, Any]
    resolved_params: dict[str, Any]
    feature_version: str
    point_count: int
    total_count: int
    is_subsample: bool
    subsample_strategy: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    response_ids: list[UUID]
    coords_2d: Optional[list[list[float]]] = None
    coords_3d: Optional[list[list[float]]] = None
    trustworthiness_2d: Optional[float] = None
    trustworthiness_3d: Optional[float] = None
    continuity_2d: Optional[float] = None
    continuity_3d: Optional[float] = None
    from_cache: bool
    cached_at: Optional[datetime] = None


class StageTiming(BaseModel):
    name: str
    duration_ms: float
    offset_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

class RunCreateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    n: int = Field(ge=1, le=500)
    model: str
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=16, le=4096)
    chunk_size: Optional[int] = Field(default=None, ge=2, le=200)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=199)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    embedding_model: Optional[str] = None
    preproc_version: Optional[str] = None
    use_cache: bool = True
    notes: Optional[str] = Field(default=None, max_length=2000)
    cluster_algo: Optional[str] = None
    hdbscan_min_cluster_size: Optional[int] = Field(default=None, ge=2)
    hdbscan_min_samples: Optional[int] = Field(default=None, ge=1)
    umap: Optional[UMAPParamsRequest] = None

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
        return 
    
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
    
    @field_validator("preproc_version")
    @classmethod
    def normalise_preproc_version(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        return text or None
    
    @field_validator("cluster_algo")
    @classmethod
    def normalise_cluster_algo(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        algo = value.strip().lower()
        if algo not in {"hdbscan", "kmeans"}:
            raise ValueError("cluster_algo must be 'hdbscan' or 'kmeans'")
        return algo
    
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
    preproc_version: str
    use_cache: bool
    cluster_algo: str
    hdbscan_min_cluster_size: Optional[int]
    hdbscan_min_samples: Optional[int]
    umap: UMAPParams
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
    processing_time_ms: Optional[float] = None
    stage_timings: list[StageTiming] = Field(default_factory=list)

class RunSummary(BaseModel):
    id: UUID
    prompt: str
    n: int
    model: str
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    system_prompt: Optional[str]
    embedding_model: str
    preproc_version: str
    use_cache: bool
    cluster_algo: str
    hdbscan_min_cluster_size: Optional[int]
    hdbscan_min_samples: Optional[int]
    umap: UMAPParams
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
    processing_time_ms: Optional[float] = None

class RunMetrics(BaseModel):
    run_id: UUID
    total_segments: int
    cached_segments: int
    duplicate_segments: int
    cache_hit_rate: float
    processing_time_ms: Optional[float] = None
    stage_timings: list[StageTiming] = Field(default_factory=list)
    silhouette_embed: Optional[float] = None
    silhouette_feature: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    n_clusters: Optional[int] = None
    n_noise: Optional[int] = None

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
    text_hash: Optional[str]
    is_cached: bool
    is_duplicate: bool
    simhash64: Optional[int] = None
    coords_3d: tuple[float, float, float]
    coords_2d: tuple[float, float]

class SegmentEdge(BaseModel):
    source_id: UUID
    target_id: UUID
    score: float

class SegmentGraphEdge(BaseModel):
    source: UUID
    target: UUID
    similarity: float

class SegmentTopTerm(BaseModel):
    term: str
    weight: float

class SegmentNeighborPreview(BaseModel):
    id: UUID
    similarity: float
    text: str
    cluster: Optional[int] = None

class SegmentContextMetrics(BaseModel):
    sim_to_exemplar: Optional[float] = None
    sim_to_nn: Optional[float] = None

class SegmentContextResponse(BaseModel):
    segment_id: UUID
    cluster_id: Optional[int] = None
    cluster_exemplar_id: Optional[UUID] = None
    exemplar_preview: Optional[str] = None
    top_terms: list[SegmentTopTerm]
    neighbors: list[SegmentNeighborPreview]
    why_here: SegmentContextMetrics
    preview: str
    
class SegmentGraphResponse(BaseModel):
    mode: Literal["full", "simplified"]
    edges: list[SegmentGraphEdge]
    auto_simplified: bool
    k: int
    threshold: float
    node_count: int


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

class ClusterMetricsSummary(BaseModel):
    algo: str
    params: dict[str, Any] = Field(default_factory=dict)
    silhouette_embed: Optional[float] = None
    silhouette_feature: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    n_clusters: Optional[int] = None
    n_noise: Optional[int] = None
    stability: Optional[dict[str, Any]] = None
    sweep: Optional[dict[str, Any]] = None
    created_at: datetime

class ClusterMetricsResponse(ClusterMetricsSummary):
    run_id: UUID

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
    preproc_version: str
    n: int
    costs: RunCostSummary
    chunk_size: Optional[int] | None = None
    chunk_overlap: Optional[int] | None = None
    umap: UMAPParams
    quality: QualityGauge
    projection_quality: Optional[dict[str, QualityGauge]] = None
    cluster_metrics: Optional[ClusterMetricsSummary] = None
    provenance: Optional[dict[str, Any]] = None

class ExportMode(str, Enum):
    RESPONSES = "responses"
    SEGMENTS = "segments"

class ExportScope(str, Enum):
    RUN = "run"
    CLUSTER = "cluster"
    SELECTION = "selection"
    VIEWPORT = "viewport"

class ExportFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"

class ExportInclude(str, Enum):
    PROVENANCE = "provenance"
    VECTORS = "vectors"
    METADATA = "metadata"

class ExportViewport(BaseModel):
    dimension: Literal["2d", "3d"] = "2d"
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None

    @field_validator("max_x")
    @classmethod
    def validate_x_bounds(cls, value: Optional[float], info: ValidationInfo) -> Optional[float]:
        min_value = info.data.get("min_x")
        if value is not None and min_value is not None and value < min_value:
            raise ValueError("max_x must be greater than or equal to min_x")
        return value
    
    @field_validator("max_y")
    @classmethod
    def validate_y_bounds(cls, value: Optional[float], info: ValidationInfo) -> Optional[float]:
        min_value = info.data.get("min_y")
        if value is not None and min_value is not None and value < min_value:
            raise ValueError("max_y must be greater than or equal to min_y")
        return 
    
    @field_validator("max_z")
    @classmethod
    def validate_z_bounds(cls, value: Optional[float], info: ValidationInfo) -> Optional[float]:
        min_value = info.data.get("min_z")
        if value is not None and min_value is not None and value < min_value:
            raise ValueError("max_z must be greater than or equal to min_z")
        return value
    
class RunExportRequest(BaseModel):
    selection_ids: list[UUID] = Field(default_factory=list)

class ExportRow(BaseModel):
    schema_version: str
    run_id: UUID
    kind: Literal["segment", "response"]
    segment_id: Optional[UUID]
    response_id: UUID
    response_index: int
    position: Optional[int]
    text: str
    text_hash: str
    cluster_id: Optional[int]
    cluster_probability: Optional[float]
    cluster_similarity: Optional[float]
    coords_2d: tuple[float, float]
    coords_3d: tuple[float, float, float]
    top_terms: list[Any] = Field(default_factory=list)
    neighbors: list[Any] = Field(default_factory=list)
    metadata: Optional[dict[str, Any]] = None
    embedding: Optional[dict[str, Any]] = None
    
class RunProvenanceResponse(BaseModel):
    run_id: UUID
    created_at: datetime
    python_version: Optional[str]
    node_version: Optional[str]
    blas_impl: Optional[str]
    openmp_threads: Optional[int]
    numba_version: Optional[str]
    numba_target: Optional[str]
    lib_versions: dict[str, Any]
    embedding_model: Optional[str]
    embedding_dim: Optional[int]
    llm_model: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    feature_weights: dict[str, Any]
    input_space: dict[str, Any]
    umap_params: dict[str, Any]
    cluster_params: dict[str, Any]
    commit_sha: Optional[str]
    env_label: Optional[str]
    random_state_seed_source: Optional[str]
