"""Run management endpoints for sampling and retrieving semantic landscapes.


Endpoints:
    create_run(payload, session): Persist a sampling configuration and return its identifier.
    sample_run(run_id, sample_request, session): Fan out chat completions, embeddings, clustering, and persistence.
    get_results(run_id, session): Materialise all derived artefacts for a run in a single payload.
    export_json(run_id, session): Convenience wrapper mirroring `get_results`.
    export_csv(run_id, session): Stream a flattened CSV export of response-level metrics.

Helpers:
    _to_run_resource(run): Convert a Run ORM instance into its response schema.
    _build_results(...): Stitch together responses, projections, clusters, segments, edges, and hulls.
    _extract_cluster_keywords(...): Derive TF-IDF keywords that describe response clusters.
    _extract_segment_keywords(...): Derive TF-IDF keywords for segment clusters.
"""

from __future__ import annotations
import csv
import io
import json
from collections import defaultdict
from typing import Sequence
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Response, status
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.db.session import get_session
from app.models import (
    Cluster,
    Embedding,
    Projection,
    Response as ResponseModel,
    ResponseHull as ResponseHullModel,
    ResponseSegment,
    Run,
    SegmentEdge as SegmentEdgeModel,
)
from app.schemas import (
    ClusterSummary,
    ResponseHull as ResponseHullSchema,
    ResponsePoint,
    RunCreateRequest,
    RunCreatedResponse,
    RunResource,
    RunResultsResponse,
    RunUpdateRequest,
    SampleRequest,
    SegmentClusterSummary,
    SegmentEdge as SegmentEdgeSchema,
    SegmentPoint, UsageInfo, RunSummary,
    RunCostSummary,
)
from app.services.runs import RunService, load_run_with_details, list_recent_runs
from app.services.pricing import get_completion_pricing, get_embedding_pricing
from app.utils.tokenization import count_tokens

_SETTINGS = get_settings()

router = APIRouter(prefix="/run", tags=["runs"])




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

@router.get("/{run_id}", response_model=RunResource)
async def get_run(run_id: UUID, session: AsyncSession = Depends(get_session)) -> RunResource:
    run = await session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return _to_run_resource(run)





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

    run, responses, projections, clusters, embeddings, segment_rows, edge_rows, hull_rows = details
    return _build_results(run, responses, projections, clusters, embeddings, segment_rows, edge_rows, hull_rows)


@router.get("/{run_id}/export.json")
async def export_json(run_id: UUID, session: AsyncSession = Depends(get_session)) -> RunResultsResponse:
    return await get_results(run_id, session)


@router.get("/{run_id}/export.csv")
async def export_csv(run_id: UUID, session: AsyncSession = Depends(get_session)) -> Response:
    details = await load_run_with_details(session, run_id)
    if details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    run, responses, projections, clusters, embeddings, segment_rows, edge_rows, hull_rows = details
    result = _build_results(run, responses, projections, clusters, embeddings, segment_rows, edge_rows, hull_rows)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["response_id", "index", "cluster", "prob", "similarity", "x", "y", "z", "preview"])
    for point in result.points:
        writer.writerow(
            [
                str(point.id),
                point.index,
                point.cluster if point.cluster is not None else "",
                f"{point.probability:.4f}" if point.probability is not None else "",
                f"{point.similarity_to_centroid:.4f}" if point.similarity_to_centroid is not None else "",
                f"{point.coords_3d[0]:.6f}",
                f"{point.coords_3d[1]:.6f}",
                f"{point.coords_3d[2]:.6f}",
                point.text_preview,
            ]
        )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=run_{run_id}.csv",
        },
    )

def _to_run_resource(run: Run) -> RunResource:
    metadata = None
    if run.progress_metadata:
        try:
            metadata = json.loads(run.progress_metadata)
        except json.JSONDecodeError:
            metadata = None

    return RunResource(
        id=run.id,
        prompt=run.prompt,
        n=run.n,
        model=run.model,
        chunk_size=run.chunk_size,
        chunk_overlap=run.chunk_overlap,
        system_prompt=run.system_prompt,
        embedding_model=run.embedding_model or _SETTINGS.openai_embedding_model,
        temperature=run.temperature,
        top_p=run.top_p,
        seed=run.seed,
        max_tokens=run.max_tokens,
        status=run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        error_message=run.error_message,
        notes=run.notes,
        progress_stage=run.progress_stage,
        progress_message=run.progress_message,
        progress_percent=run.progress_percent,
        progress_metadata=metadata,
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
) -> RunResultsResponse:
    embedding_model = run.embedding_model or _SETTINGS.openai_embedding_model
    completion_pricing = get_completion_pricing(run.model)
    embedding_pricing = get_embedding_pricing(embedding_model)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_completion_cost = 0.0
    total_embedding_tokens = 0
    total_embedding_cost = 0.0

    prompt_embedding_tokens = count_tokens(run.prompt, embedding_model)
    total_embedding_tokens += prompt_embedding_tokens
    if embedding_pricing:
        total_embedding_cost += prompt_embedding_tokens * embedding_pricing.input_cost

    proj_map: dict[tuple[UUID, int], Projection] = {}
    for projection in projections:
        proj_map[(projection.response_id, projection.dim)] = projection

    cluster_map: dict[UUID, Cluster] = {cluster.response_id: cluster for cluster in clusters}
    method_by_label: dict[int, str] = {}
    label_to_ids: dict[int, list[UUID]] = defaultdict(list)
    centroid_xyz: dict[int, tuple[float, float, float]] = {}

    for cluster in clusters:
        method_by_label[cluster.label] = cluster.method
        label_to_ids[cluster.label].append(cluster.response_id)

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
        segment_tokens = segment.tokens if segment.tokens is not None else count_tokens(segment.text, embedding_model)
        embedding_tokens = segment_tokens or 0
        total_embedding_tokens += embedding_tokens
        embedding_cost = None
        if embedding_pricing:
            embedding_cost = embedding_tokens * embedding_pricing.input_cost
            total_embedding_cost += embedding_cost
        segment_points.append(
            SegmentPoint(
                id=segment.id,
                response_id=segment.response_id,
                response_index=response_index_map.get(segment.response_id, -1),
                position=segment.position,
                text=segment.text,
                role=segment.role,
                tokens=segment_tokens,
                embedding_tokens=embedding_tokens,
                embedding_cost=embedding_cost,
                prompt_similarity=float(segment.prompt_similarity) if segment.prompt_similarity is not None else None,
                silhouette_score=float(segment.silhouette_score) if segment.silhouette_score is not None else None,
                cluster=cluster_label,
                probability=float(segment.cluster_probability) if segment.cluster_probability is not None else None,
                similarity_to_centroid=float(segment.cluster_similarity) if segment.cluster_similarity is not None else None,
                outlier_score=float(segment.outlier_score) if segment.outlier_score is not None else None,
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

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        completion_cost = None
        if completion_pricing:
            completion_cost = (
                prompt_tokens * completion_pricing.input_cost
                + completion_tokens * completion_pricing.output_cost
            )
            total_completion_cost += completion_cost

        embedding_tokens = count_tokens(full_text, embedding_model)
        total_embedding_tokens += embedding_tokens
        embedding_cost = None
        if embedding_pricing:
            embedding_cost = embedding_tokens * embedding_pricing.input_cost
            total_embedding_cost += embedding_cost

        total_cost = (completion_cost or 0.0) + (embedding_cost or 0.0)

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
                probability=cluster_info.probability if cluster_info else None,
                similarity_to_centroid=cluster_info.similarity if cluster_info else None,
                outlier_score=float(outlier_score) if outlier_score is not None else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                embedding_tokens=embedding_tokens,
                completion_cost=completion_cost,
                embedding_cost=embedding_cost,
                total_cost=total_cost,
                coords_3d=coords3d,
                coords_2d=coords2d,
            )
        )

    total_embedding_tokens = int(total_embedding_tokens)
    total_prompt_tokens = int(total_prompt_tokens)
    total_completion_tokens = int(total_completion_tokens)
    cost_summary = RunCostSummary(
        model=run.model,
        embedding_model=embedding_model,
        completion_input_tokens=total_prompt_tokens,
        completion_output_tokens=total_completion_tokens,
        completion_cost=float(total_completion_cost),
        embedding_tokens=total_embedding_tokens,
        embedding_cost=float(total_embedding_cost),
        total_cost=float(total_completion_cost + total_embedding_cost),
    )

    return RunResultsResponse(
        run=_to_run_resource(run),
        points=points,
        clusters=cluster_summaries,
        segments=segment_points,
        segment_clusters=segment_cluster_summaries,
        segment_edges=edge_items,
        response_hulls=response_hulls,
        prompt=run.prompt,
        model=run.model,
        system_prompt=run.system_prompt,
        embedding_model=run.embedding_model or _SETTINGS.openai_embedding_model,
        n=run.n,
        costs=cost_summary,
        chunk_size=run.chunk_size,
        chunk_overlap=run.chunk_overlap,
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
