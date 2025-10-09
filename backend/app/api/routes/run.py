"""Run management endpoints for sampling and retrieving semantic landscapes.

Endpoints:
    create_run(payload, session): Persist a sampling configuration and return its identifier.
    sample_run(run_id, sample_request, session): Fan out chat completions, embeddings, clustering, and persistence.
    get_results(run_id, session): Materialise all derived artefacts for a run in a single payload.
    export_run_payload(...): Stream fine-grained exports (run, cluster, selection, viewport).

Helpers:
    _to_run_resource(run): Convert a Run ORM instance into its response schema.
    _build_results(...): Stitch together responses, projections, clusters, segments, edges, and hulls.
    _extract_cluster_keywords(...): Derive TF-IDF keywords that describe response clusters.
    _extract_segment_keywords(...): Derive TF-IDF keywords for segment clusters.
"""

from __future__ import annotations
from datetime import datetime
import csv
import hashlib
import io
import json
from collections import defaultdict
import math
from typing import Any, Optional, Sequence
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import select

from app.db.session import get_session
from app.core.config import get_settings
from app.models import (
    Cluster,
    ClusterMetrics,
    Embedding,
    Projection,
    Response as ResponseModel,
    ResponseHull as ResponseHullModel,
    ResponseSegment,
    Run,
    RunProvenance,
    SegmentEdge as SegmentEdgeModel,
    SegmentInsight,
)
from app.schemas import (
    ClusterSummary,
    ClusterMetricsSummary,
    ClusterMetricsResponse,
    ExportFormat,
    ExportInclude,
    ExportMode,
    ExportRow,
    ExportScope,
    ExportViewport,
    ResponseHull as ResponseHullSchema,
    ResponsePoint,
    RunCreateRequest,
    RunCreatedResponse,
    RunExportRequest,
    RunCostSummary,
    RunResource,
    RunResultsResponse,
    RunMetrics,
    RunProvenanceResponse,
    RunUpdateRequest,
    SegmentGraphResponse,
    SegmentGraphEdge,
    SampleRequest,
    SegmentClusterSummary,
    SegmentEdge as SegmentEdgeSchema,
    SegmentPoint,
    SegmentTopTerm,
    SegmentNeighborPreview,
    SegmentContextMetrics,
    SegmentContextResponse,
    UsageInfo,
    RunSummary,
    UMAPParams,
    QualityGauge,
)
from app.services.pricing import get_completion_pricing, get_embedding_pricing
from app.services.runs import RunService, load_run_with_details, list_recent_runs
from app.utils.tokenization import count_tokens as estimate_tokens
from app.utils.text import normalise_for_embedding

router = APIRouter(prefix="/run", tags=["runs"])

_EXPORT_SCHEMA_VERSION = "1.0.0"
_SETTINGS = get_settings()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None




@router.get("", response_model=list[RunSummary])
async def list_runs(limit: int = 20, session: AsyncSession = Depends(get_session)) -> list[RunSummary]:
    summaries = await list_recent_runs(session, limit=limit)
    return summaries

@router.post("", response_model=RunCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    payload: RunCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> RunCreatedResponse:
    service = RunService()
    run = await service.create_run(session, payload)
    return RunCreatedResponse(run_id=run.id)


@router.get("/{run_id}", response_model=RunResource)
async def get_run(run_id: UUID, session: AsyncSession = Depends(get_session)) -> RunResource:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return _to_run_resource(run)


@router.patch("/{run_id}", response_model=RunResource)
async def update_run(
    run_id: UUID,
    payload: RunUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> RunResource:
    service = RunService()
    try:
        run = await service.update_run(session, run_id=run_id, payload=payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _to_run_resource(run)




@router.get("/{run_id}/provenance", response_model=RunProvenanceResponse)
async def get_run_provenance(
    run_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> RunProvenanceResponse:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    result = await session.exec(select(RunProvenance).where(RunProvenance.run_id == run_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provenance not found")

    return _provenance_record_to_response(record)


@router.get("/{run_id}/metrics", response_model=RunMetrics)
async def get_run_metrics(
    run_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> RunMetrics:
    service = RunService()
    try:
        metrics = await service.compute_run_metrics(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return metrics


@router.get("/{run_id}/graph", response_model=SegmentGraphResponse)
async def get_run_graph(
    run_id: UUID,
    mode: str = Query(default="full"),
    k: int = Query(default=15, ge=1, le=200),
    sim: float = Query(default=0.35, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
) -> SegmentGraphResponse:
    run = await session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    resolved_mode = mode.lower().strip() if mode else "full"
    if resolved_mode not in {"full", "simplified"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown graph mode '{mode}'")

    service = RunService()
    try:
        graph_data = await service.build_segment_graph(
            session,
            run,
            mode=resolved_mode,
            neighbor_k=k,
            threshold=sim,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    edges = [
        SegmentGraphEdge(source=edge[0], target=edge[1], similarity=edge[2])
        for edge in graph_data["edges"]
    ]

    return SegmentGraphResponse(
        mode=graph_data["mode"],
        edges=edges,
        auto_simplified=graph_data["auto_simplified"],
        k=graph_data["k"],
        threshold=graph_data["threshold"],
        node_count=graph_data["node_count"],
    )


@router.post("/{run_id}/sample", response_model=RunResource)
async def sample_run(
    run_id: UUID,
    sample_request: SampleRequest | None = Body(default=None),
    session: AsyncSession = Depends(get_session),
) -> RunResource:
    service = RunService()
    try:
        run = await service.sample_run(session, run_id=run_id, sample_request=sample_request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _to_run_resource(run)


@router.get("/{run_id}/results", response_model=RunResultsResponse)
async def get_results(run_id: UUID, session: AsyncSession = Depends(get_session)) -> RunResultsResponse:
    details = await load_run_with_details(session, run_id)
    if details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    (
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics,
        provenance,
    ) = details

    return _build_results(
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics=cluster_metrics,
        provenance=provenance,
    )





@router.get("/{run_id}/cluster-metrics", response_model=ClusterMetricsResponse)
async def get_cluster_metrics(
    run_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ClusterMetricsResponse:
    result = await session.exec(
        select(ClusterMetrics).where(ClusterMetrics.run_id == run_id)
    )
    metrics = result.scalar_one_or_none()
    if metrics is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cluster metrics not found")

    def _parse_json(payload: str | None) -> dict[str, Any] | None:
        if not payload:
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    params = _parse_json(getattr(metrics, "params_json", None)) or {}
    stability = _parse_json(getattr(metrics, "stability_json", None))
    sweep = _parse_json(getattr(metrics, "sweep_json", None))

    return ClusterMetricsResponse(
        run_id=run_id,
        algo=metrics.algo,
        params=params,
        silhouette_embed=_finite(metrics.silhouette_embed),
        silhouette_feature=_finite(metrics.silhouette_feature),
        davies_bouldin=_finite(metrics.davies_bouldin),
        calinski_harabasz=_finite(metrics.calinski_harabasz),
        n_clusters=metrics.n_clusters,
        n_noise=metrics.n_noise,
        stability=stability,
        sweep=sweep,
        created_at=metrics.created_at,
    )


@router.get("/{run_id}/clusters", response_model=RunResultsResponse)
async def recompute_clusters_endpoint(
    run_id: UUID,
    min_cluster_size: Optional[int] = Query(default=None, ge=2),
    min_samples: Optional[int] = Query(default=None, ge=1),
    algo: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> RunResultsResponse:
    service = RunService()
    try:
        await service.recompute_clusters(
            session,
            run_id=run_id,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algo=algo,
        )
    except ValueError as exc:
        message = str(exc)
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in message.lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=message) from exc

    details = await load_run_with_details(session, run_id)
    if details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    (
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics,
        provenance,
    ) = details

    return _build_results(
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics=cluster_metrics,
        provenance=provenance,
    )
@router.api_route("/{run_id}/export", methods=["GET", "POST"])
async def export_run_payload(
    run_id: UUID,
    scope: ExportScope = Query(default=ExportScope.RUN),
    mode: ExportMode = Query(default=ExportMode.SEGMENTS),
    export_format: ExportFormat = Query(default=ExportFormat.JSON, alias="format"),
    cluster_id: Optional[int] = Query(default=None),
    selection_ids_query: Optional[str] = Query(default=None, alias="selection_ids"),
    include: Optional[list[str]] = Query(default=None),
    viewport_min_x: Optional[float] = Query(default=None),
    viewport_max_x: Optional[float] = Query(default=None),
    viewport_min_y: Optional[float] = Query(default=None),
    viewport_max_y: Optional[float] = Query(default=None),
    viewport_min_z: Optional[float] = Query(default=None),
    viewport_max_z: Optional[float] = Query(default=None),
    viewport_dimension: str = Query(default="2d", alias="viewport_dim"),
    payload: RunExportRequest | None = Body(default=None),
    session: AsyncSession = Depends(get_session),
) -> Response:
    include_tokens: list[str] = []
    if include:
        for token in include:
            include_tokens.extend(part.strip() for part in token.split(","))
    include_set: set[ExportInclude] = set()
    for token in include_tokens:
        if not token:
            continue
        try:
            include_set.add(ExportInclude(token.lower()))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown include option '{token}'",
            ) from exc

    selection_ids: set[UUID] = set(payload.selection_ids if payload else [])
    if selection_ids_query:
        for part in selection_ids_query.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                selection_ids.add(UUID(token))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid selection id '{token}'",
                ) from exc

    if scope is ExportScope.CLUSTER and cluster_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cluster scope requires cluster_id",
        )
    if scope is ExportScope.SELECTION and not selection_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="selection scope requires selection ids",
        )

    viewport: ExportViewport | None = None
    if scope is ExportScope.VIEWPORT:
        dimension = viewport_dimension.lower()
        if dimension not in {"2d", "3d"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="viewport_dim must be '2d' or '3d'",
            )
        if (
            viewport_min_x is None
            or viewport_max_x is None
            or viewport_min_y is None
            or viewport_max_y is None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="viewport scope requires min_x, max_x, min_y, max_y values",
            )
        viewport = ExportViewport(
            dimension=dimension,  # type: ignore[arg-type]
            min_x=viewport_min_x,
            max_x=viewport_max_x,
            min_y=viewport_min_y,
            max_y=viewport_max_y,
            min_z=viewport_min_z,
            max_z=viewport_max_z,
        )

    details = await load_run_with_details(session, run_id)
    if details is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    (
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics,
        provenance,
    ) = details

    result = _build_results(
        run,
        responses,
        projections,
        clusters,
        embeddings,
        segment_rows,
        edge_rows,
        hull_rows,
        cluster_metrics=cluster_metrics,
        provenance=provenance,
    )

    rows, provenance_payload = await _assemble_export_rows(
        session=session,
        run=run,
        result=result,
        responses=responses,
        segments=segment_rows,
        embeddings=embeddings,
        scope=scope,
        mode=mode,
        cluster_id=cluster_id,
        selection_ids=selection_ids,
        viewport=viewport,
        include_set=include_set,
        provenance=provenance,
    )

    filename = _derive_export_filename(
        run.id,
        scope,
        mode,
        export_format,
        cluster_id,
        len(rows),
        viewport,
    )

    content, media_type = _render_export_content(
        rows=rows,
        export_format=export_format,
        include_set=include_set,
        run_id=run.id,
        scope=scope,
        mode=mode,
        cluster_id=cluster_id,
        selection_ids=selection_ids,
        viewport=viewport,
        provenance_payload=provenance_payload,
    )

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _to_run_resource(run: Run) -> RunResource:
    umap_params = UMAPParams(
        n_neighbors=getattr(run, "umap_n_neighbors", _SETTINGS.umap_default_n_neighbors),
        min_dist=float(getattr(run, "umap_min_dist", _SETTINGS.umap_default_min_dist)),
        metric=getattr(run, "umap_metric", _SETTINGS.umap_default_metric),
        seed=getattr(run, "umap_seed", None),
        seed_source=getattr(run, "random_state_seed_source", "default"),
    )

    progress_metadata = None
    raw_metadata = getattr(run, "progress_metadata", None)
    if raw_metadata:
        try:
            progress_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            progress_metadata = None

    processing_time_ms = None
    stage_timings: list[dict[str, Any]] = []
    raw_timings = getattr(run, "timings_json", None)
    if raw_timings:
        try:
            parsed_timings = json.loads(raw_timings)
        except json.JSONDecodeError:
            parsed_timings = None
        if isinstance(parsed_timings, dict):
            stage_values = parsed_timings.get("stages") or []
            total_value = parsed_timings.get("total_duration_ms")
            if isinstance(total_value, (int, float)):
                processing_time_ms = float(total_value)
        elif isinstance(parsed_timings, list):
            stage_values = parsed_timings
        else:
            stage_values = []
        stage_timings = [value for value in stage_values if isinstance(value, dict)]
    attr_duration = getattr(run, "processing_time_ms", None)
    if processing_time_ms is None and isinstance(attr_duration, (int, float)):
        processing_time_ms = float(attr_duration)

    return RunResource(
        id=run.id,
        prompt=run.prompt,
        n=run.n,
        model=run.model,
        chunk_size=getattr(run, "chunk_size", None),
        chunk_overlap=getattr(run, "chunk_overlap", None),
        system_prompt=getattr(run, "system_prompt", None),
        embedding_model=getattr(run, "embedding_model", None)
        or _SETTINGS.openai_embedding_model,
        preproc_version=getattr(run, "preproc_version", None)
        or _SETTINGS.embedding_preproc_version,
        use_cache=bool(getattr(run, "use_cache", True)),
        cluster_algo=getattr(run, "cluster_algo", _SETTINGS.cluster_default_algo),
        hdbscan_min_cluster_size=getattr(run, "hdbscan_min_cluster_size", None),
        hdbscan_min_samples=getattr(run, "hdbscan_min_samples", None),
        umap=umap_params,
        temperature=run.temperature,
        top_p=run.top_p,
        seed=run.seed,
        max_tokens=run.max_tokens,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        error_message=run.error_message,
        notes=getattr(run, "notes", None),
        progress_stage=getattr(run, "progress_stage", None),
        progress_message=getattr(run, "progress_message", None),
        progress_percent=getattr(run, "progress_percent", None),
        progress_metadata=progress_metadata,
        processing_time_ms=processing_time_ms,
        stage_timings=stage_timings,
    )


def _build_results(
    run: Run,
    responses: list[ResponseModel],
    projections: list[Projection],
    clusters: list[Cluster],
    embeddings: list[Embedding],
    segments: list[ResponseSegment],
    segment_edges: list[SegmentEdgeModel],
    hulls: list[ResponseHullModel],
    *,
    cluster_metrics: ClusterMetrics | None = None,
    provenance: Any | None = None,
) -> RunResultsResponse:
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    proj_map: dict[tuple[UUID, int], Projection] = {}
    for projection in projections:
        proj_map[(projection.response_id, projection.dim)] = projection

    cluster_map: dict[UUID, Cluster] = {cluster.response_id: cluster for cluster in clusters}
    method_by_label: dict[int, str] = {}
    label_to_ids: dict[int, list[UUID]] = defaultdict(list)
    centroid_xyz: dict[int, tuple[float, float, float]] = {}

    cluster_metrics_summary: ClusterMetricsSummary | None = None
    if cluster_metrics is not None:
        params: dict[str, Any] = {}
        if getattr(cluster_metrics, "params_json", None):
            try:
                params = json.loads(cluster_metrics.params_json)
            except json.JSONDecodeError:
                params = {}
        stability = None
        if getattr(cluster_metrics, "stability_json", None):
            try:
                stability = json.loads(cluster_metrics.stability_json)
            except json.JSONDecodeError:
                stability = None
        sweep = None
        if getattr(cluster_metrics, "sweep_json", None):
            try:
                sweep = json.loads(cluster_metrics.sweep_json)
            except json.JSONDecodeError:
                sweep = None
        cluster_metrics_summary = ClusterMetricsSummary(
            algo=cluster_metrics.algo,
            params=params,
            silhouette_embed=_finite(cluster_metrics.silhouette_embed),
            silhouette_feature=_finite(cluster_metrics.silhouette_feature),
            davies_bouldin=_finite(cluster_metrics.davies_bouldin),
            calinski_harabasz=_finite(cluster_metrics.calinski_harabasz),
            n_clusters=cluster_metrics.n_clusters,
            n_noise=cluster_metrics.n_noise,
            stability=stability,
            sweep=sweep,
            created_at=cluster_metrics.created_at,
        )

    if clusters and projections:
        coords_by_label: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for cluster in clusters:
            projection3d = proj_map.get((cluster.response_id, 3))
            if projection3d:
                coords_by_label[cluster.label].append(
                    (float(projection3d.x), float(projection3d.y), float(projection3d.z or 0.0))
                )
        for label, coords in coords_by_label.items():
            if coords:
                arr = np.asarray(coords, dtype=float)
                centroid_xyz[label] = (
                    float(arr[:, 0].mean()),
                    float(arr[:, 1].mean()),
                    float(arr[:, 2].mean()),
                )
            else:
                centroid_xyz[label] = (0.0, 0.0, 0.0)

    keywords_by_label = _extract_cluster_keywords(responses, label_to_ids)

    cluster_summaries: list[ClusterSummary] = []
    ordered_labels = sorted(label_to_ids.keys(), key=lambda lbl: (lbl < 0, lbl))
    for label in ordered_labels:
        response_ids = label_to_ids[label]
        members = [cluster_map[response_id] for response_id in response_ids if response_id in cluster_map]
        similarities = [member.similarity for member in members if member.similarity is not None]
        avg_similarity = float(np.mean(similarities)) if similarities else None
        exemplars = sorted(
            members,
            key=lambda item: (item.probability or 0.0, item.similarity or 0.0),
            reverse=True,
        )[:2]

        cluster_summaries.append(
            ClusterSummary(
                label=label,
                size=len(response_ids),
                centroid_xyz=centroid_xyz.get(label, (0.0, 0.0, 0.0)),
                exemplar_ids=[ex.response_id for ex in exemplars],
                average_similarity=avg_similarity,
                method=method_by_label.get(label, "unknown"),
                keywords=keywords_by_label.get(label, []),
                noise=label < 0,
            )
        )

    response_index_map = {response.id: response.index for response in responses}

    segment_points: list[SegmentPoint] = []
    segments_by_label: dict[int, list[ResponseSegment]] = defaultdict(list)
    for segment in segments:
        coords3d = (float(segment.coord_x), float(segment.coord_y), float(segment.coord_z))
        coords2d = (float(segment.coord2_x), float(segment.coord2_y))
        cluster_label = int(segment.cluster_label) if segment.cluster_label is not None else None
        if cluster_label is not None:
            segments_by_label[cluster_label].append(segment)

        segment_points.append(
            SegmentPoint(
                id=segment.id,
                response_id=segment.response_id,
                response_index=response_index_map.get(segment.response_id, -1),
                position=segment.position,
                text=segment.text,
                role=segment.role,
                tokens=segment.tokens,
                prompt_similarity=_finite(segment.prompt_similarity),
                silhouette_score=_finite(segment.silhouette_score),
                cluster=cluster_label,
                probability=_finite(segment.cluster_probability),
                similarity_to_centroid=_finite(segment.cluster_similarity),
                outlier_score=_finite(segment.outlier_score),
                text_hash=segment.text_hash,
                is_cached=bool(getattr(segment, "is_cached", False)),
                is_duplicate=bool(getattr(segment, "is_duplicate", False)),
                simhash64=getattr(segment, "simhash64", None),
                coords_3d=coords3d,
                coords_2d=coords2d,
            )
        )

    segment_keywords = _extract_segment_keywords(segments, segments_by_label)

    segment_cluster_summaries: list[SegmentClusterSummary] = []
    ordered_segment_labels = sorted(segments_by_label.keys(), key=lambda lbl: (lbl < 0, lbl))
    for label in ordered_segment_labels:
        members = segments_by_label[label]
        similarities = [seg.cluster_similarity for seg in members if seg.cluster_similarity is not None]
        avg_similarity = float(np.mean(similarities)) if similarities else None
        exemplars = sorted(
            members,
            key=lambda seg: ((seg.cluster_probability or 0.0), (seg.cluster_similarity or 0.0)),
            reverse=True,
        )[:3]
        keywords = segment_keywords.get(label, [])
        theme = " ".join(keywords[:3]) if keywords else None
        segment_cluster_summaries.append(
            SegmentClusterSummary(
                label=label,
                size=len(members),
                exemplar_ids=[seg.id for seg in exemplars],
                average_similarity=avg_similarity,
                method="hdbscan",
                keywords=keywords,
                theme=theme,
                noise=label < 0,
            )
        )

    edge_items = [
        SegmentEdgeSchema(source_id=edge.source_id, target_id=edge.target_id, score=float(edge.score))
        for edge in segment_edges
    ]

    hull_map: dict[UUID, dict[int, list[list[float]]]] = defaultdict(dict)
    for hull in hulls:
        try:
            coords = json.loads(hull.points_json) if hull.points_json else []
        except json.JSONDecodeError:
            coords = []
        hull_map[hull.response_id][hull.dim] = coords

    response_hulls: list[ResponseHullSchema] = []
    for response_id, dims in hull_map.items():
        coords2d = [tuple(float(v) for v in point[:2]) for point in dims.get(2, [])]
        coords3d = [tuple(float(v) for v in point[:3]) for point in dims.get(3, [])]
        response_hulls.append(
            ResponseHullSchema(
                response_id=response_id,
                coords_2d=coords2d,
                coords_3d=coords3d,
            )
        )
    response_hulls.sort(key=lambda item: response_index_map.get(item.response_id, 0))

    points: list[ResponsePoint] = []
    for response in responses:
        cluster_info = cluster_map.get(response.id)
        projection3d = proj_map.get((response.id, 3))
        projection2d = proj_map.get((response.id, 2))
        coords3d = (
            float(projection3d.x) if projection3d else 0.0,
            float(projection3d.y) if projection3d else 0.0,
            float(projection3d.z or 0.0) if projection3d else 0.0,
        )
        coords2d = (
            float(projection2d.x) if projection2d else 0.0,
            float(projection2d.y) if projection2d else 0.0,
        )
        usage = response.usage_json or {}
        usage_info = (
            UsageInfo(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
            if usage
            else None
        )

        full_text = response.raw_text or ""
        preview = (full_text[:197] + "...") if len(full_text) > 200 else full_text

        outlier_score = usage.get("outlier_score")
        if outlier_score is None and cluster_info and cluster_info.label == -1:
            outlier_score = 1.0

        points.append(
            ResponsePoint(
                id=response.id,
                index=response.index,
                text_preview=preview,
                full_text=full_text,
                tokens=response.tokens,
                finish_reason=response.finish_reason,
                usage=usage_info,
                cluster=cluster_info.label if cluster_info else None,
                probability=_finite(cluster_info.probability) if cluster_info else None,
                similarity_to_centroid=_finite(cluster_info.similarity) if cluster_info else None,
                outlier_score=_finite(outlier_score),
                coords_3d=coords3d,
                coords_2d=coords2d,
            )
        )

    run_resource = _to_run_resource(run)

    completion_input_tokens = 0
    completion_output_tokens = 0
    completion_cached_tokens = 0
    for response in responses:
        usage = response.usage_json or {}
        completion_input_tokens += _safe_int(usage.get("prompt_tokens", 0))
        completion_output_tokens += _safe_int(usage.get("completion_tokens", 0))
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            completion_cached_tokens += _safe_int(prompt_details.get("cached_tokens"))

    embedding_model_id = run_resource.embedding_model or _SETTINGS.openai_embedding_model

    def _embedding_token_count(text: str | None) -> int:
        if not text:
            return 0
        try:
            return estimate_tokens(text, embedding_model_id)
        except Exception:
            return estimate_tokens(text, None)

    embedding_tokens = 0

    if run.prompt:
        embedding_tokens += _embedding_token_count(run.prompt)

    response_hashes_seen: set[str] = set()
    for response in responses:
        text = response.raw_text or ""
        if not text.strip():
            continue
        collapsed, normalised = normalise_for_embedding(text)
        base = normalised or collapsed
        text_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
        if text_hash in response_hashes_seen:
            continue
        response_hashes_seen.add(text_hash)
        embedding_tokens += _embedding_token_count(text)

    segment_hashes_seen: set[str] = set()
    for segment in segments:
        if getattr(segment, "is_cached", False):
            continue
        text_hash = getattr(segment, "text_hash", None)
        key = text_hash or f"segment:{segment.id}"
        if key in segment_hashes_seen:
            continue
        segment_hashes_seen.add(key)
        token_count = segment.tokens if isinstance(segment.tokens, int) and segment.tokens > 0 else None
        if token_count is None:
            token_count = _embedding_token_count(segment.text)
        embedding_tokens += token_count

    completion_pricing = get_completion_pricing(run.model)
    embedding_pricing = get_embedding_pricing(embedding_model_id)

    billable_input_tokens = max(completion_input_tokens - completion_cached_tokens, 0)
    cached_input_tokens = min(completion_cached_tokens, completion_input_tokens)

    completion_cost = 0.0
    if completion_pricing:
        completion_cost = (
            billable_input_tokens * completion_pricing.input_cost
            + cached_input_tokens * completion_pricing.cached_input_cost
            + completion_output_tokens * completion_pricing.output_cost
        )

    embedding_cost = 0.0
    if embedding_pricing:
        embedding_cost = float(embedding_tokens) * embedding_pricing.input_cost

    total_cost = completion_cost + embedding_cost

    cost_summary = RunCostSummary(
        model=run.model,
        embedding_model=run_resource.embedding_model,
        completion_input_tokens=completion_input_tokens,
        completion_output_tokens=completion_output_tokens,
        completion_cost=completion_cost,
        embedding_tokens=int(embedding_tokens),
        embedding_cost=embedding_cost,
        total_cost=total_cost,
    )

    umap_params = UMAPParams(
        n_neighbors=getattr(run, "umap_n_neighbors", _SETTINGS.umap_default_n_neighbors),
        min_dist=float(getattr(run, "umap_min_dist", _SETTINGS.umap_default_min_dist)),
        metric=getattr(run, "umap_metric", _SETTINGS.umap_default_metric),
        seed=getattr(run, "umap_seed", None),
        seed_source=getattr(run, "random_state_seed_source", "default"),
    )

    quality = QualityGauge(
        trustworthiness_2d=_finite(getattr(run, "trustworthiness_2d", None)),
        trustworthiness_3d=_finite(getattr(run, "trustworthiness_3d", None)),
        continuity_2d=_finite(getattr(run, "continuity_2d", None)),
        continuity_3d=_finite(getattr(run, "continuity_3d", None)),
    )

    return RunResultsResponse(
        run=run_resource,
        points=points,
        clusters=cluster_summaries,
        segments=segment_points,
        segment_clusters=segment_cluster_summaries,
        segment_edges=edge_items,
        response_hulls=response_hulls,
        cluster_metrics=cluster_metrics_summary,
        prompt=run.prompt,
        model=run.model,
        system_prompt=getattr(run, "system_prompt", None),
        embedding_model=run_resource.embedding_model,
        preproc_version=run_resource.preproc_version,
        n=run.n,
        chunk_size=getattr(run, "chunk_size", None),
        chunk_overlap=getattr(run, "chunk_overlap", None),
        costs=cost_summary,
        umap=umap_params,
        quality=quality,
        provenance=_provenance_to_dict(provenance),
    )



def _extract_cluster_keywords(
    responses: Sequence[ResponseModel],
    label_to_ids: dict[int, list[UUID]],
    *,
    top_k: int = 5,
) -> dict[int, list[str]]:
    if not responses:
        return {}

    texts = [response.raw_text or "" for response in responses]
    if not any(texts):
        return {}

    vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    id_to_index = {response.id: idx for idx, response in enumerate(responses)}

    keywords_by_label: dict[int, list[str]] = {}
    for label, response_ids in label_to_ids.items():
        if label < 0:
            continue
        indices = [id_to_index[rid] for rid in response_ids if rid in id_to_index]
        if not indices:
            continue
        weights = tfidf_matrix[indices].mean(axis=0)
        weights = np.asarray(weights).ravel()
        if not weights.size:
            continue
        sorted_idx = weights.argsort()[::-1]
        keywords = [feature_names[i] for i in sorted_idx if weights[i] > 0][:top_k]
        keywords_by_label[label] = keywords

    return keywords_by_label


def _extract_segment_keywords(
    segments: Sequence[ResponseSegment],
    segments_by_label: dict[int, list[ResponseSegment]],
    *,
    top_k: int = 5,
) -> dict[int, list[str]]:
    if not segments_by_label:
        return {}

    texts = [segment.text or "" for segment in segments]
    if not any(texts):
        return {}

    vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    id_to_index = {segment.id: idx for idx, segment in enumerate(segments)}

    keywords_by_label: dict[int, list[str]] = {}
    for label, members in segments_by_label.items():
        if label < 0:
            continue
        indices = [id_to_index.get(member.id) for member in members if member.id in id_to_index]
        indices = [idx for idx in indices if idx is not None]
        if not indices:
            continue
        weights = tfidf_matrix[indices].mean(axis=0)
        weights = np.asarray(weights).ravel()
        if not weights.size:
            continue
        sorted_idx = weights.argsort()[::-1]
        keywords = [feature_names[i] for i in sorted_idx if weights[i] > 0][:top_k]
        keywords_by_label[label] = keywords

    return keywords_by_label

async def _assemble_export_rows(
    *,
    session: AsyncSession,
    run: Run,
    result: RunResultsResponse,
    responses: Sequence[ResponseModel],
    segments: Sequence[ResponseSegment],
    embeddings: Sequence[Embedding],
    scope: ExportScope,
    mode: ExportMode,
    cluster_id: Optional[int],
    selection_ids: set[UUID],
    viewport: ExportViewport | None,
    include_set: set[ExportInclude],
    provenance: Any | None,
) -> tuple[list[ExportRow], dict[str, Any] | None]:
    response_map = {response.id: response for response in responses}
    segment_map = {segment.id: segment for segment in segments}
    embedding_map = {embedding.response_id: embedding for embedding in embeddings}
    keywords_by_label = {cluster.label: cluster.keywords for cluster in result.clusters}

    items: Sequence[ResponsePoint | SegmentPoint]
    if mode is ExportMode.RESPONSES:
        items = result.points
    else:
        items = result.segments

    filtered: list[ResponsePoint | SegmentPoint] = []
    for item in items:
        if scope is ExportScope.CLUSTER and cluster_id is not None:
            item_cluster = getattr(item, "cluster", None)
            if item_cluster != cluster_id:
                continue
        if scope is ExportScope.SELECTION and selection_ids and item.id not in selection_ids:
            continue
        if viewport and not _within_viewport(item.coords_2d, item.coords_3d, viewport):
            continue
        filtered.append(item)

    segment_insights: dict[UUID, SegmentInsight] = {}
    if mode is ExportMode.SEGMENTS and filtered:
        segment_ids = [item.id for item in filtered]  # type: ignore[attr-defined]
        segment_insights = await _load_segment_insights(session, segment_ids)

    include_metadata = ExportInclude.METADATA in include_set
    include_vectors = ExportInclude.VECTORS in include_set

    rows: list[ExportRow] = []

    for item in filtered:
        if mode is ExportMode.SEGMENTS:
            segment_point: SegmentPoint = item  # type: ignore[assignment]
            segment_model = segment_map.get(segment_point.id)
            if not segment_model:
                continue
            insight = segment_insights.get(segment_point.id)
            text_hash = segment_model.text_hash or hashlib.sha256(
                (segment_point.text or "").encode("utf-8")
            ).hexdigest()
            top_terms = _top_terms_from_insight(insight)
            neighbors = _neighbors_from_insight(insight)
            metadata = None
            if include_metadata:
                metadata = {
                    "role": segment_point.role,
                    "tokens": segment_point.tokens,
                    "prompt_similarity": segment_point.prompt_similarity,
                    "silhouette_score": segment_point.silhouette_score,
                    "probability": segment_point.probability,
                    "similarity_to_centroid": segment_point.similarity_to_centroid,
                    "outlier_score": segment_point.outlier_score,
                    "is_cached": bool(getattr(segment_model, "is_cached", False)),
                    "is_duplicate": bool(getattr(segment_model, "is_duplicate", False)),
                    "text_hash": segment_model.text_hash,
                    "simhash64": getattr(segment_model, "simhash64", None),
                }
                metrics = _metrics_from_insight(insight)
                if metrics:
                    metadata["insight_metrics"] = metrics
            embedding_payload = None
            if include_vectors:
                vector = _decode_vector(segment_model.embedding_vector, segment_model.embedding_dim)
                if vector:
                    preview = vector[: min(16, len(vector))]
                    embedding_payload = {
                        "dim": len(vector),
                        "preview": preview,
                        "vector": vector,
                    }
            rows.append(
                ExportRow(
                    schema_version=_EXPORT_SCHEMA_VERSION,
                    run_id=run.id,
                    kind="segment",
                    segment_id=segment_point.id,
                    response_id=segment_point.response_id,
                    response_index=segment_point.response_index,
                    position=segment_point.position,
                    text=segment_point.text,
                    text_hash=text_hash,
                    cluster_id=segment_point.cluster,
                    cluster_probability=segment_point.probability,
                    cluster_similarity=segment_point.similarity_to_centroid,
                    coords_2d=(
                        float(segment_point.coords_2d[0]),
                        float(segment_point.coords_2d[1]),
                    ),
                    coords_3d=(
                        float(segment_point.coords_3d[0]),
                        float(segment_point.coords_3d[1]),
                        float(segment_point.coords_3d[2]),
                    ),
                    top_terms=top_terms,
                    neighbors=neighbors,
                    metadata=metadata,
                    embedding=embedding_payload,
                )
            )
        else:
            response_point: ResponsePoint = item  # type: ignore[assignment]
            response_model = response_map.get(response_point.id)
            if not response_model:
                continue
            text = response_point.full_text or ""
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
            top_terms = (
                keywords_by_label.get(response_point.cluster, [])
                if response_point.cluster is not None
                else []
            )
            metadata = None
            if include_metadata:
                metadata = {
                    "tokens": response_point.tokens,
                    "finish_reason": response_point.finish_reason,
                    "usage": response_point.usage.model_dump() if response_point.usage else None,
                    "probability": response_point.probability,
                    "similarity_to_centroid": response_point.similarity_to_centroid,
                    "outlier_score": response_point.outlier_score,
                }
            embedding_payload = None
            if include_vectors:
                embedding_model = embedding_map.get(response_point.id)
                vector = _decode_vector(
                    embedding_model.vector if embedding_model else None,
                    embedding_model.dim if embedding_model else None,
                )
                if vector:
                    preview = vector[: min(16, len(vector))]
                    embedding_payload = {
                        "dim": len(vector),
                        "preview": preview,
                        "vector": vector,
                    }
            rows.append(
                ExportRow(
                    schema_version=_EXPORT_SCHEMA_VERSION,
                    run_id=run.id,
                    kind="response",
                    segment_id=None,
                    response_id=response_point.id,
                    response_index=response_point.index,
                    position=None,
                    text=text,
                    text_hash=text_hash,
                    cluster_id=response_point.cluster,
                    cluster_probability=response_point.probability,
                    cluster_similarity=response_point.similarity_to_centroid,
                    coords_2d=(
                        float(response_point.coords_2d[0]),
                        float(response_point.coords_2d[1]),
                    ),
                    coords_3d=(
                        float(response_point.coords_3d[0]),
                        float(response_point.coords_3d[1]),
                        float(response_point.coords_3d[2]),
                    ),
                    top_terms=top_terms,
                    neighbors=[],
                    metadata=metadata,
                    embedding=embedding_payload,
                )
            )

    provenance_payload = _provenance_to_dict(provenance) if ExportInclude.PROVENANCE in include_set else None
    return rows, provenance_payload


def _render_export_content(
    *,
    rows: list[ExportRow],
    export_format: ExportFormat,
    include_set: set[ExportInclude],
    run_id: UUID,
    scope: ExportScope,
    mode: ExportMode,
    cluster_id: Optional[int],
    selection_ids: set[UUID],
    viewport: ExportViewport | None,
    provenance_payload: dict[str, Any] | None,
) -> tuple[bytes, str]:
    row_dicts = [row.model_dump(mode="json", exclude_none=True) for row in rows]
    include_provenance = ExportInclude.PROVENANCE in include_set and provenance_payload

    if export_format is ExportFormat.JSON:
        payload: dict[str, Any] = {
            "schema_version": _EXPORT_SCHEMA_VERSION,
            "run_id": str(run_id),
            "scope": scope.value,
            "mode": mode.value,
            "count": len(rows),
            "rows": row_dicts,
        }
        if scope is ExportScope.CLUSTER and cluster_id is not None:
            payload["cluster_id"] = cluster_id
        if scope is ExportScope.SELECTION:
            payload["selection_ids"] = [str(item) for item in selection_ids]
        if scope is ExportScope.VIEWPORT and viewport is not None:
            payload["viewport"] = viewport.model_dump(exclude_none=True)
        if include_provenance:
            payload["provenance"] = provenance_payload
        data = json.dumps(payload, ensure_ascii=False)
        return data.encode("utf-8"), "application/json"

    if export_format is ExportFormat.JSONL:
        lines: list[str] = []
        for row_dict in row_dicts:
            row_dict.setdefault("scope", scope.value)
            row_dict.setdefault("mode", mode.value)
            if include_provenance:
                row_dict.setdefault("provenance", provenance_payload)
            lines.append(json.dumps(row_dict, ensure_ascii=False))
        content = "\n".join(lines)
        if lines:
            content += "\n"
        return content.encode("utf-8"), "application/x-ndjson"

    tabular_rows, fieldnames = _build_tabular_rows(
        row_dicts=row_dicts,
        include_set=include_set,
        provenance_payload=provenance_payload if include_provenance else None,
    )

    if export_format is ExportFormat.CSV:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in tabular_rows:
            writer.writerow(row)
        return buffer.getvalue().encode("utf-8"), "text/csv"

    if export_format is ExportFormat.PARQUET:
        buffer = io.BytesIO()
        df = pd.DataFrame(tabular_rows, columns=fieldnames)
        df.to_parquet(buffer, index=False)
        return buffer.getvalue(), "application/x-parquet"

    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported export format")


def _build_tabular_rows(
    *,
    row_dicts: list[dict[str, Any]],
    include_set: set[ExportInclude],
    provenance_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    base_fields: list[str] = [
        "schema_version",
        "run_id",
        "kind",
        "segment_id",
        "response_id",
        "response_index",
        "position",
        "text",
        "text_hash",
        "cluster_id",
        "cluster_probability",
        "cluster_similarity",
        "coords_2d",
        "coords_3d",
        "top_terms",
        "neighbors",
    ]
    if ExportInclude.METADATA in include_set:
        base_fields.append("metadata")
    if ExportInclude.VECTORS in include_set:
        base_fields.append("embedding")
    if provenance_payload is not None:
        base_fields.append("provenance")

    rows: list[dict[str, Any]] = []
    provenance_json = (
        json.dumps(provenance_payload, ensure_ascii=False) if provenance_payload is not None else ""
    )

    for data in row_dicts:
        flat = {field: "" for field in base_fields}
        for key in base_fields:
            if key not in data:
                continue
            value = data[key]
            if key in {"coords_2d", "coords_3d", "top_terms", "neighbors", "metadata", "embedding"}:
                flat[key] = json.dumps(value, ensure_ascii=False) if value not in (None, "") else ""
            else:
                flat[key] = value if value is not None else ""
        if provenance_payload is not None:
            flat["provenance"] = provenance_json
        rows.append(flat)
    return rows, base_fields


def _derive_export_filename(
    run_id: UUID,
    scope: ExportScope,
    mode: ExportMode,
    export_format: ExportFormat,
    cluster_id: Optional[int],
    item_count: int,
    viewport: ExportViewport | None,
) -> str:
    base = f"run_{run_id}__{mode.value}"
    if scope is ExportScope.CLUSTER and cluster_id is not None:
        base += f"__cluster_{cluster_id}"
    elif scope is ExportScope.SELECTION:
        base += f"__selection_{item_count}"
    elif scope is ExportScope.VIEWPORT:
        suffix = "viewport3d" if viewport and viewport.dimension == "3d" else "viewport"
        base += f"__{suffix}"
    else:
        base += "__full"
    extension = {
        ExportFormat.JSON: "json",
        ExportFormat.JSONL: "jsonl",
        ExportFormat.CSV: "csv",
        ExportFormat.PARQUET: "parquet",
    }[export_format]
    return f"{base}.{extension}"


def _within_viewport(
    coords_2d: tuple[float, float],
    coords_3d: tuple[float, float, float],
    viewport: ExportViewport,
) -> bool:
    def _check(value: float, minimum: Optional[float], maximum: Optional[float]) -> bool:
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
        return True

    x2, y2 = coords_2d
    x3, y3, z3 = coords_3d

    if viewport.dimension == "3d":
        return (
            _check(x3, viewport.min_x, viewport.max_x)
            and _check(y3, viewport.min_y, viewport.max_y)
            and _check(z3, viewport.min_z, viewport.max_z)
        )

    if not _check(x2, viewport.min_x, viewport.max_x):
        return False
    if not _check(y2, viewport.min_y, viewport.max_y):
        return False
    if viewport.min_z is not None or viewport.max_z is not None:
        if not _check(z3, viewport.min_z, viewport.max_z):
            return False
    return True


async def _load_segment_insights(
    session: AsyncSession,
    segment_ids: Sequence[UUID],
) -> dict[UUID, SegmentInsight]:
    if not segment_ids:
        return {}
    result = await session.exec(
        select(SegmentInsight).where(SegmentInsight.segment_id.in_(segment_ids))
    )
    return {row.segment_id: row for row in result.scalars()}


def _top_terms_from_insight(insight: SegmentInsight | None) -> list[Any]:
    if not insight or not insight.top_terms_json:
        return []
    try:
        data = json.loads(insight.top_terms_json)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _neighbors_from_insight(insight: SegmentInsight | None) -> list[dict[str, Any]]:
    if not insight or not insight.neighbors_json:
        return []
    try:
        data = json.loads(insight.neighbors_json)
    except json.JSONDecodeError:
        return []
    payload: list[dict[str, Any]] = []
    if isinstance(data, list):
        for entry in data:
            try:
                segment_id = UUID(str(entry.get("segment_id")))
                similarity = float(entry.get("similarity", 0.0))
            except (TypeError, ValueError):
                continue
            payload.append({
                "segment_id": str(segment_id),
                "similarity": similarity,
            })
    return payload


def _metrics_from_insight(insight: SegmentInsight | None) -> dict[str, Any]:
    if not insight or not insight.metrics_json:
        return {}
    try:
        data = json.loads(insight.metrics_json)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _decode_vector(vector_bytes: bytes | None, dim: Optional[int]) -> list[float] | None:
    if not vector_bytes or not dim:
        return None
    array = np.frombuffer(vector_bytes, dtype=np.float32)
    if dim and len(array) != dim:
        array = array[:dim]
    return array.astype(float).tolist()


def _provenance_record_to_response(record: RunProvenance) -> RunProvenanceResponse:
    return RunProvenanceResponse(
        run_id=record.run_id,
        created_at=record.created_at,
        python_version=record.python_version,
        node_version=record.node_version,
        blas_impl=record.blas_impl,
        openmp_threads=record.openmp_threads,
        numba_version=record.numba_version,
        numba_target=record.numba_target,
        lib_versions=json.loads(record.lib_versions_json) if record.lib_versions_json else {},
        embedding_model=record.embedding_model,
        embedding_dim=record.embedding_dim,
        llm_model=record.llm_model,
        temperature=record.temperature,
        top_p=record.top_p,
        max_tokens=record.max_tokens,
        feature_weights=json.loads(record.feature_weights_json) if record.feature_weights_json else {},
        input_space=json.loads(record.input_space_json) if record.input_space_json else {},
        umap_params=json.loads(record.umap_params_json) if record.umap_params_json else {},
        cluster_params=json.loads(record.cluster_params_json) if record.cluster_params_json else {},
        commit_sha=record.commit_sha,
        env_label=record.env_label,
        random_state_seed_source=record.random_state_seed_source,
    )


def _provenance_to_dict(provenance: Any | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    if hasattr(provenance, "model_dump"):
        data = provenance.model_dump()
    else:
        data = {
            key: value
            for key, value in vars(provenance).items()
            if not key.startswith("_")
        }
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


@router.get("/segments/{segment_id}/context", response_model=SegmentContextResponse)
async def get_segment_context(
    segment_id: UUID,
    k: int = Query(default=8, ge=1, le=50),
    session: AsyncSession = Depends(get_session),
) -> SegmentContextResponse:
    segment = await session.get(ResponseSegment, segment_id)
    if segment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")

    insight = await session.get(SegmentInsight, segment_id)

    top_terms_raw = _top_terms_from_insight(insight)
    top_terms: list[SegmentTopTerm] = []
    for item in top_terms_raw:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        weight = item.get("weight")
        weight_value = _finite(weight)
        if not term or weight_value is None:
            continue
        top_terms.append(SegmentTopTerm(term=str(term), weight=weight_value))

    neighbors_raw = _neighbors_from_insight(insight)
    neighbor_ids: list[UUID] = []
    for entry in neighbors_raw:
        segment_token = entry.get("segment_id") if isinstance(entry, dict) else None
        try:
            neighbor_uuid = UUID(str(segment_token))
        except (TypeError, ValueError):
            continue
        neighbor_ids.append(neighbor_uuid)
    neighbor_ids = neighbor_ids[:k]

    neighbor_segments: dict[UUID, ResponseSegment] = {}
    if neighbor_ids:
        result = await session.exec(
            select(ResponseSegment).where(ResponseSegment.id.in_(neighbor_ids))
        )
        neighbor_segments = {row.id: row for row in result.scalars().all()}

    neighbors: list[SegmentNeighborPreview] = []
    for entry in neighbors_raw:
        if len(neighbors) >= k:
            break
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("segment_id")
        try:
            neighbor_uuid = UUID(str(raw_id))
        except (TypeError, ValueError):
            continue
        neighbor_segment = neighbor_segments.get(neighbor_uuid)
        if neighbor_segment is None:
            continue
        similarity = _finite(entry.get("similarity"))
        similarity = similarity if similarity is not None else 0.0
        preview_text = neighbor_segment.text or ""
        preview = preview_text[:240].strip()
        neighbor_cluster = (
            int(neighbor_segment.cluster_label)
            if neighbor_segment.cluster_label is not None
            else None
        )
        neighbors.append(
            SegmentNeighborPreview(
                id=neighbor_uuid,
                similarity=similarity,
                text=preview,
                cluster=neighbor_cluster,
            )
        )

    cluster_id = int(segment.cluster_label) if segment.cluster_label is not None else None

    exemplar_preview = None
    exemplar_id = None
    if insight and insight.cluster_exemplar_id:
        exemplar_id = insight.cluster_exemplar_id
        exemplar_segment = neighbor_segments.get(exemplar_id)
        if exemplar_segment is None:
            exemplar_segment = await session.get(ResponseSegment, exemplar_id)
        if exemplar_segment is not None:
            exemplar_preview = (exemplar_segment.text or "")[:240].strip()

    metrics = _metrics_from_insight(insight)
    why_here = SegmentContextMetrics(
        sim_to_exemplar=_finite(metrics.get("sim_to_exemplar")) if metrics else None,
        sim_to_nn=_finite(metrics.get("sim_to_nn")) if metrics else None,
    )

    preview = (segment.text or "")[:240].strip()

    return SegmentContextResponse(
        segment_id=segment_id,
        cluster_id=cluster_id,
        cluster_exemplar_id=exemplar_id,
        exemplar_preview=exemplar_preview,
        top_terms=top_terms,
        neighbors=neighbors,
        why_here=why_here,
        preview=preview,
    )
