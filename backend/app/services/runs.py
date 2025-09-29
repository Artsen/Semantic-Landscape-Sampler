"""High level orchestration for sampling runs.

Classes:

    RunService: Coordinates OpenAI calls, segmentation, embeddings, projections, clustering, and persistence for runs.

Functions:

    load_run_with_details(session, run_id): Fetch a run and all derived ORM records needed to build API responses.

"""

from __future__ import annotations

import json

from collections import defaultdict

from datetime import datetime

from typing import Any, Optional, Sequence

from uuid import UUID

import numpy as np

from scipy.spatial import ConvexHull

from sqlalchemy import delete, func, select

from app.core.config import get_settings

from app.models import (
    Cluster,
    Embedding,
    Projection,
    Response,
    ResponseHull,
    ResponseSegment,
    Run,
    RunStatus,
    SegmentEdge,
)

from app.schemas import RunCreateRequest, RunSummary, RunUpdateRequest, SampleRequest

from app.services.openai_client import EmbeddingBatch, OpenAIService

from app.services.projection import (
    build_feature_matrix,
    cluster_with_fallback,
    compute_umap,
)

from app.services.segmentation import SegmentDraft, flatten_drafts, make_segment_drafts

from app.utils.tokenization import count_tokens

_SETTINGS = get_settings()


class RunService:

    def __init__(self, openai_service: OpenAIService | None = None) -> None:

        self._openai = openai_service or OpenAIService()

    async def _update_progress(
        self,
        session,
        run: Run,
        *,
        stage: str,
        message: str,
        percent: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        run.progress_stage = stage

        run.progress_message = message

        run.progress_percent = percent

        run.progress_metadata = json.dumps(metadata) if metadata else None

        run.updated_at = datetime.utcnow()

        session.add(run)

        await session.commit()

        await session.refresh(run)

    async def create_run(self, session, payload: RunCreateRequest) -> Run:

        chunk_size = (
            payload.chunk_size
            if payload.chunk_size is not None
            else _SETTINGS.segment_word_window
        )

        chunk_overlap = (
            payload.chunk_overlap
            if payload.chunk_overlap is not None
            else _SETTINGS.segment_word_overlap
        )

        if chunk_overlap is not None:

            max_overlap = max(chunk_size - 1, 0)

            chunk_overlap = min(chunk_overlap, max_overlap)

        system_prompt = payload.system_prompt if payload.system_prompt else None

        embedding_model = payload.embedding_model or _SETTINGS.openai_embedding_model

        run = Run(
            prompt=payload.prompt,
            n=payload.n,
            model=payload.model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            system_prompt=system_prompt,
            embedding_model=embedding_model,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
            max_tokens=payload.max_tokens,
            notes=payload.notes,
            status=RunStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            progress_stage="queued",
            progress_message=f"Queued run for {payload.n} completions",
            progress_percent=0.0,
            progress_metadata=json.dumps({"n": payload.n, "model": payload.model}),
        )

        session.add(run)

        await session.commit()

        await session.refresh(run)

        return run

    async def update_run(
        self, session, *, run_id: UUID, payload: RunUpdateRequest
    ) -> Run:

        run = await session.get(Run, run_id)

        if not run:

            raise ValueError(f"Run {run_id} not found")

        run.notes = payload.notes

        run.updated_at = datetime.utcnow()

        await session.commit()

        await session.refresh(run)

        return run

    async def sample_run(
        self,
        session,
        *,
        run_id: UUID,
        sample_request: SampleRequest | None = None,
    ) -> Run:

        run = await session.get(Run, run_id)

        if not run:

            raise ValueError(f"Run {run_id} not found")

        if not run.embedding_model:

            run.embedding_model = _SETTINGS.openai_embedding_model

        if not run.chunk_size:
            run.chunk_size = _SETTINGS.segment_word_window

        if run.chunk_overlap is None:
            run.chunk_overlap = _SETTINGS.segment_word_overlap

        if run.chunk_overlap is not None and run.chunk_size:
            max_overlap = max(run.chunk_size - 1, 0)
            run.chunk_overlap = min(run.chunk_overlap, max_overlap)

        options = sample_request or SampleRequest()

        if options.overwrite_previous or options.force_refresh:

            await self._clear_existing(session, run_id)

        run.status = RunStatus.PENDING

        run.error_message = None

        current_stage = "initializing"

        try:

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Preparing sampling pipeline",
                percent=0.02,
                metadata={"n": run.n, "model": run.model},
            )

            current_stage = "requesting-completions"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message=f"Requesting {run.n} completions from {run.model}",
                percent=0.1,
                metadata={"n": run.n, "model": run.model},
            )

            chat_results = await self._openai.sample_chat(
                prompt=run.prompt,
                n=run.n,
                model=run.model,
                system_prompt=run.system_prompt,
                temperature=run.temperature,
                top_p=run.top_p,
                seed=run.seed,
                jitter_token=options.jitter_prompt_token,
                max_tokens=run.max_tokens,
            )

            current_stage = "responses-received"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message=f"Received {len(chat_results)} responses from model",
                percent=0.15,
                metadata={"responses": len(chat_results)},
            )

            responses = [
                Response(
                    run_id=run.id,
                    index=sample.index,
                    raw_text=sample.text,
                    tokens=sample.tokens,
                    finish_reason=sample.finish_reason,
                    usage_json=sample.usage,
                )
                for sample in chat_results
            ]

            current_stage = "embedding-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message=f"Embedding {len(responses)} responses",
                percent=0.2,
                metadata={
                    "responses": len(responses),
                    "embedding_model": run.embedding_model,
                },
            )

            response_embeddings = await self._embed_responses(
                responses, run.embedding_model
            )

            session.add_all(responses)

            await session.commit()

            for response in responses:

                await session.refresh(response)

            current_stage = "persisting-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message=f"Persisted {len(responses)} responses",
                percent=0.24,
                metadata={"responses": len(responses)},
            )

            prompt_embedding = await self._embed_prompt(run.prompt, run.embedding_model)

            current_stage = "embedding-prompt"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Prompt embedding prepared for similarity features",
                percent=0.28,
                metadata={"has_prompt_embedding": bool(prompt_embedding)},
            )

            response_ids = [response.id for response in responses]

            segment_drafts: list[SegmentDraft] = []

            segment_embeddings: EmbeddingBatch | None = None

            total_segments = 0

            segment_edge_count = 0

            if options.include_segments:

                current_stage = "segmenting-responses"

                await self._update_progress(
                    session,
                    run,
                    stage=current_stage,
                    message="Segmenting responses into discourse windows",
                    percent=0.34,
                    metadata={
                        "responses": len(responses),
                        "chunk_size": run.chunk_size,
                    },
                )

                segment_drafts = flatten_drafts(
                    make_segment_drafts(
                        response.id,
                        response.index,
                        response.raw_text or "",
                        word_window=run.chunk_size or _SETTINGS.segment_word_window,
                        word_overlap=(
                            run.chunk_overlap
                            if run.chunk_overlap is not None
                            else _SETTINGS.segment_word_overlap
                        ),
                    )
                    for response in responses
                )

                segment_drafts = self._ensure_minimum_segments(
                    responses, segment_drafts
                )

                total_segments = len(segment_drafts)

                for seg in segment_drafts:

                    seg.tokens = count_tokens(seg.text, run.embedding_model)

                if options.include_discourse_tags and self._openai.is_configured:

                    roles = await self._openai.discourse_tag_segments(
                        [seg.text for seg in segment_drafts]
                    )

                    for seg, role in zip(segment_drafts, roles, strict=False):

                        if role:

                            seg.role = role

                if segment_drafts:

                    current_stage = "embedding-segments"

                    await self._update_progress(
                        session,
                        run,
                        stage=current_stage,
                        message=f"Embedding {total_segments} segments",
                        percent=0.42,
                        metadata={
                            "segments": total_segments,
                            "embedding_model": run.embedding_model,
                        },
                    )

                    segment_embeddings = await self._openai.embed_texts(
                        [seg.text for seg in segment_drafts],
                        model=run.embedding_model,
                    )

                    for seg, vector in zip(
                        segment_drafts, segment_embeddings.vectors, strict=False
                    ):

                        seg.embedding = list(vector)

                    self._apply_prompt_similarity(segment_drafts, prompt_embedding)

                    current_stage = "analysing-segments"

                    await self._update_progress(
                        session,
                        run,
                        stage=current_stage,
                        message="Projecting segment manifold",
                        percent=0.5,
                        metadata={
                            "segments": total_segments,
                            "min_dist": _SETTINGS.projection_min_dist,
                        },
                    )

                    segment_feature_matrix = build_feature_matrix(
                        [seg.text for seg in segment_drafts],
                        segment_embeddings.vectors,
                        prompt_embedding=prompt_embedding,
                        keyword_axes=_SETTINGS.segment_keyword_axes,
                    )

                    segment_projection = compute_umap(
                        segment_feature_matrix, min_dist=_SETTINGS.projection_min_dist
                    )

                    min_cluster, min_samples = self._suggest_cluster_params(
                        len(segment_drafts)
                    )

                    segment_clustering = cluster_with_fallback(
                        segment_feature_matrix,
                        coords_3d=segment_projection.coords_3d,
                        similarity_basis=segment_embeddings.vectors,
                        min_cluster_size=min_cluster,
                        min_samples=min_samples,
                    )

                    self._apply_segment_analysis(
                        segment_drafts, segment_projection, segment_clustering
                    )

                    segment_edges = self._build_segment_edges(run.id, segment_drafts)

                    segment_edge_count = len(segment_edges)

                    response_hulls = self._build_response_hulls(segment_drafts)

                    await self._persist_segment_edges(session, run.id, segment_edges)

                    await self._persist_response_hulls(
                        session, response_ids, response_hulls
                    )

                    await self._persist_segments(session, response_ids, segment_drafts)

                    current_stage = "persisting-segment-artifacts"

                    await self._update_progress(
                        session,
                        run,
                        stage=current_stage,
                        message=f"Stored {total_segments} segments and overlays",
                        percent=0.58,
                        metadata={
                            "segments": total_segments,
                            "edges": segment_edge_count,
                        },
                    )

                else:

                    current_stage = "segments-minimum"

                    await self._update_progress(
                        session,
                        run,
                        stage=current_stage,
                        message="Segmentation produced no additional slices; using full responses",
                        percent=0.4,
                        metadata={"segments": 0},
                    )

                    await self._persist_segment_edges(session, run.id, [])

                    await self._persist_response_hulls(session, response_ids, [])

                    await self._persist_segments(session, response_ids, [])

            else:

                current_stage = "segments-disabled"

                await self._update_progress(
                    session,
                    run,
                    stage=current_stage,
                    message="Segment overlays disabled for this run",
                    percent=0.38,
                    metadata={"responses": len(responses)},
                )

                await self._persist_segment_edges(session, run.id, [])

                await self._persist_response_hulls(session, response_ids, [])

                await self._persist_segments(session, response_ids, [])

            if segment_drafts and segment_embeddings is not None:

                aggregated = self._aggregate_response_embeddings(
                    responses, segment_drafts, response_embeddings
                )

                if aggregated:

                    vectors = [vec.tolist() for vec in aggregated]

                    dim = len(vectors[0]) if vectors else segment_embeddings.dim

                    response_embeddings = EmbeddingBatch(
                        vectors=vectors, model=segment_embeddings.model, dim=dim
                    )

            texts = [response.raw_text or "" for response in responses]

            current_stage = "projecting-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Projecting responses with UMAP",
                percent=0.72,
                metadata={"responses": len(responses)},
            )

            feature_matrix = build_feature_matrix(
                texts,
                response_embeddings.vectors,
                prompt_embedding=prompt_embedding,
                keyword_axes=_SETTINGS.segment_keyword_axes,
            )

            projection_result = compute_umap(
                feature_matrix, min_dist=_SETTINGS.projection_min_dist
            )

            min_cluster, min_samples = self._suggest_cluster_params(run.n)

            clustering_result = cluster_with_fallback(
                feature_matrix,
                coords_3d=projection_result.coords_3d,
                similarity_basis=response_embeddings.vectors,
                min_cluster_size=min_cluster,
                min_samples=min_samples,
            )

            current_stage = "clustering-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Clustering responses and computing outlier scores",
                percent=0.82,
                metadata={
                    "clusters": len(
                        set(int(label) for label in clustering_result.labels)
                    )
                },
            )

            await self._persist_embeddings(session, responses, response_embeddings)

            await self._persist_projection(session, responses, projection_result)

            await self._persist_clusters(session, responses, clustering_result)

            current_stage = "finalizing"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Finalising run artifacts",
                percent=0.92,
                metadata={"responses": len(responses), "segments": total_segments},
            )

            run.status = RunStatus.COMPLETED

            run.error_message = None

            current_stage = "completed"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Run completed successfully",
                percent=1.0,
                metadata={
                    "responses": len(responses),
                    "segments": total_segments,
                    "clusters": len(
                        set(int(label) for label in clustering_result.labels)
                    ),
                    "segment_edges": segment_edge_count,
                },
            )

            return run

        except Exception as exc:

            run.status = RunStatus.FAILED

            run.error_message = str(exc)

            await self._update_progress(
                session,
                run,
                stage="failed",
                message=f"Run failed during {current_stage}: {exc}",
                percent=None,
                metadata={"phase": current_stage},
            )

            raise

    async def _embed_prompt(
        self, prompt: str | None, embedding_model: str
    ) -> list[float] | None:

        if not prompt:

            return None

        try:

            batch = await self._openai.embed_texts([prompt], model=embedding_model)

        except Exception:

            return None

        return batch.vectors[0] if batch.vectors else None

    async def _embed_responses(
        self, responses: Sequence[Response], embedding_model: str
    ):

        texts = [response.raw_text for response in responses]

        return await self._openai.embed_texts(texts, model=embedding_model)

    def _apply_prompt_similarity(
        self, segments: Sequence[SegmentDraft], prompt_embedding: list[float] | None
    ) -> None:

        if not prompt_embedding or not segments:

            return

        prompt_vec = np.asarray(prompt_embedding, dtype=float)

        norm = np.linalg.norm(prompt_vec)

        if norm == 0:

            return

        prompt_vec /= norm

        vectors = [
            np.asarray(seg.embedding, dtype=float) for seg in segments if seg.embedding
        ]

        if not vectors:

            return

        matrix = np.vstack(vectors)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        norms[norms == 0] = 1.0

        normalised = matrix / norms

        sims = normalised @ prompt_vec

        cursor = 0

        for seg in segments:

            if not seg.embedding:

                continue

            seg.prompt_similarity = float(np.clip(sims[cursor], -1.0, 1.0))

            cursor += 1

    def _apply_segment_analysis(
        self, segments: Sequence[SegmentDraft], projection, clustering
    ) -> None:

        coords3d = projection.coords_3d

        coords2d = projection.coords_2d

        labels = clustering.labels

        probabilities = clustering.probabilities

        similarities = clustering.per_point_similarity

        outliers = getattr(clustering, "outlier_scores", np.zeros_like(probabilities))

        silhouettes = getattr(
            clustering, "silhouette_scores", np.zeros_like(probabilities)
        )

        outliers = np.nan_to_num(outliers, nan=0.0, posinf=0.0, neginf=0.0)

        silhouettes = np.nan_to_num(silhouettes, nan=0.0, posinf=0.0, neginf=0.0)

        for idx, seg in enumerate(segments):

            if idx < len(coords3d):

                seg.coords3d = tuple(float(value) for value in coords3d[idx])

            if idx < len(coords2d):

                seg.coords2d = tuple(float(value) for value in coords2d[idx])

            label = int(labels[idx]) if idx < len(labels) else -1

            seg.cluster_label = label if label >= 0 else None

            if idx < len(probabilities):

                seg.cluster_probability = float(probabilities[idx])

            if label >= 0 and idx < len(similarities):

                seg.cluster_similarity = float(similarities[idx])

            if idx < len(outliers):

                seg.outlier_score = float(outliers[idx])

            if idx < len(silhouettes):

                seg.silhouette_score = float(silhouettes[idx])

    def _aggregate_response_embeddings(
        self,
        responses: Sequence[Response],
        segments: Sequence[SegmentDraft],
        fallback: EmbeddingBatch,
    ) -> list[np.ndarray]:

        grouped: dict[UUID, list[np.ndarray]] = defaultdict(list)

        for seg in segments:

            if seg.embedding:

                grouped[seg.response_id].append(np.asarray(seg.embedding, dtype=float))

        fallback_map = {
            response.id: np.asarray(vector, dtype=float)
            for response, vector in zip(responses, fallback.vectors, strict=False)
        }

        default_dim = (
            fallback.dim
            if fallback.dim
            else (
                fallback_map[responses[0].id].shape[0]
                if responses and responses[0].id in fallback_map
                else 1536
            )
        )

        aggregated: list[np.ndarray] = []

        for response in responses:

            vectors = grouped.get(response.id)

            if vectors:

                aggregated.append(np.vstack(vectors).mean(axis=0))

            else:

                fallback_vec = fallback_map.get(response.id)

                if fallback_vec is not None:

                    aggregated.append(fallback_vec)

                else:

                    aggregated.append(np.zeros(default_dim, dtype=float))

        return aggregated

    def _ensure_minimum_segments(
        self,
        responses: Sequence[Response],
        segments: list[SegmentDraft],
    ) -> list[SegmentDraft]:

        existing = defaultdict(list)

        for seg in segments:

            existing[seg.response_id].append(seg)

        enriched: list[SegmentDraft] = []

        for response in responses:

            segs = existing.get(response.id)

            if not segs:

                enriched.append(
                    SegmentDraft(
                        response_id=response.id,
                        response_index=response.index,
                        position=0,
                        text=response.raw_text or "",
                        role="background",
                        tokens=len((response.raw_text or "").split()),
                    )
                )

            else:

                enriched.extend(sorted(segs, key=lambda item: item.position))

        return enriched

    async def _persist_embeddings(
        self, session, responses: Sequence[Response], embeddings: EmbeddingBatch
    ) -> None:

        if not responses:

            return

        await session.execute(
            delete(Embedding).where(
                Embedding.response_id.in_([r.id for r in responses])
            )
        )

        entries = []

        for response, vector in zip(responses, embeddings.vectors, strict=False):

            data = np.asarray(vector, dtype=np.float32).tobytes()

            entries.append(
                Embedding(
                    response_id=response.id,
                    dim=len(vector),
                    vector=data,
                )
            )

        session.add_all(entries)

        await session.commit()

    async def _persist_segments(
        self,
        session,
        response_ids: Sequence[UUID],
        segments: Sequence[SegmentDraft],
    ) -> None:

        if response_ids:

            await session.execute(
                delete(ResponseSegment).where(
                    ResponseSegment.response_id.in_(response_ids)
                )
            )

        if not segments:

            await session.commit()

            return

        entries = []

        for seg in segments:

            vector_bytes = (
                np.asarray(seg.embedding, dtype=np.float32).tobytes()
                if seg.embedding
                else None
            )

            entries.append(
                ResponseSegment(
                    id=seg.id,
                    response_id=seg.response_id,
                    position=seg.position,
                    text=seg.text,
                    role=seg.role,
                    tokens=seg.tokens,
                    prompt_similarity=seg.prompt_similarity,
                    silhouette_score=seg.silhouette_score,
                    cluster_label=seg.cluster_label,
                    cluster_probability=seg.cluster_probability,
                    cluster_similarity=seg.cluster_similarity,
                    outlier_score=seg.outlier_score,
                    embedding_dim=len(seg.embedding) if seg.embedding else None,
                    embedding_vector=vector_bytes,
                    coord_x=seg.coords3d[0],
                    coord_y=seg.coords3d[1],
                    coord_z=seg.coords3d[2],
                    coord2_x=seg.coords2d[0],
                    coord2_y=seg.coords2d[1],
                )
            )

        session.add_all(entries)

        await session.commit()

    async def _persist_projection(
        self,
        session,
        responses: Sequence[Response],
        projection,
    ) -> None:

        if not responses:

            return

        await session.execute(
            delete(Projection).where(
                Projection.response_id.in_([r.id for r in responses])
            )
        )

        for response, coords3d, coords2d in zip(
            responses, projection.coords_3d, projection.coords_2d, strict=False
        ):

            session.add(
                Projection(
                    response_id=response.id,
                    method=projection.method,
                    dim=3,
                    x=float(coords3d[0]),
                    y=float(coords3d[1]),
                    z=float(coords3d[2]),
                )
            )

            session.add(
                Projection(
                    response_id=response.id,
                    method=projection.method,
                    dim=2,
                    x=float(coords2d[0]),
                    y=float(coords2d[1]),
                    z=None,
                )
            )

        await session.commit()

    async def _persist_clusters(
        self,
        session,
        responses: Sequence[Response],
        clustering,
    ) -> None:

        if not responses:

            return

        await session.execute(
            delete(Cluster).where(Cluster.response_id.in_([r.id for r in responses]))
        )

        labels = clustering.labels

        probabilities = clustering.probabilities

        similarities = clustering.per_point_similarity

        outliers = getattr(clustering, "outlier_scores", None)

        for idx, response in enumerate(responses):

            label = int(labels[idx])

            probability = float(probabilities[idx])

            similarity = float(similarities[idx]) if label >= 0 else None

            cluster_entry = Cluster(
                response_id=response.id,
                method=getattr(clustering, "method", "unknown"),
                label=label,
                probability=probability,
                similarity=similarity,
            )

            session.add(cluster_entry)

            usage = dict(response.usage_json or {})

            if outliers is not None and len(outliers) > idx:

                usage["outlier_score"] = float(outliers[idx])

            elif label < 0:

                usage["outlier_score"] = 1.0

            else:

                usage.pop("outlier_score", None)

            response.usage_json = usage

        await session.commit()

    async def _persist_segment_edges(
        self, session, run_id: UUID, edges: Sequence[SegmentEdge]
    ) -> None:

        await session.execute(delete(SegmentEdge).where(SegmentEdge.run_id == run_id))

        if edges:

            session.add_all(edges)

        await session.commit()

    async def _persist_response_hulls(
        self,
        session,
        response_ids: Sequence[UUID],
        hulls: Sequence[ResponseHull],
    ) -> None:

        if response_ids:

            await session.execute(
                delete(ResponseHull).where(ResponseHull.response_id.in_(response_ids))
            )

        if hulls:

            session.add_all(hulls)

        await session.commit()

    def _build_segment_edges(
        self, run_id: UUID, segments: Sequence[SegmentDraft]
    ) -> list[SegmentEdge]:

        vectors = [
            np.asarray(seg.embedding, dtype=float) for seg in segments if seg.embedding
        ]

        filtered = [seg for seg in segments if seg.embedding]

        if len(filtered) < 2:

            return []

        matrix = np.vstack(vectors)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        norms[norms == 0] = 1.0

        normalised = matrix / norms

        similarities = normalised @ normalised.T

        threshold = _SETTINGS.segment_similarity_threshold

        edges: list[SegmentEdge] = []

        count = similarities.shape[0]

        for i in range(count):

            for j in range(i + 1, count):

                score = float(similarities[i, j])

                if score >= threshold:

                    edges.append(
                        SegmentEdge(
                            run_id=run_id,
                            source_id=filtered[i].id,
                            target_id=filtered[j].id,
                            score=score,
                        )
                    )

        return edges

    def _build_response_hulls(
        self, segments: Sequence[SegmentDraft]
    ) -> list[ResponseHull]:

        if not segments:

            return []

        grouped: dict[UUID, list[SegmentDraft]] = defaultdict(list)

        for seg in segments:

            grouped[seg.response_id].append(seg)

        hulls: list[ResponseHull] = []

        for response_id, segs in grouped.items():

            coords2d = np.array([seg.coords2d for seg in segs], dtype=float)

            coords3d = np.array([seg.coords3d for seg in segs], dtype=float)

            hulls.append(
                ResponseHull(
                    response_id=response_id,
                    dim=2,
                    points_json=json.dumps(self._convex_hull(coords2d)),
                )
            )

            hulls.append(
                ResponseHull(
                    response_id=response_id,
                    dim=3,
                    points_json=json.dumps(self._convex_hull(coords3d)),
                )
            )

        return hulls

    def _convex_hull(self, points: np.ndarray) -> list[list[float]]:

        if points.size == 0:

            return []

        unique = np.unique(points, axis=0)

        if unique.shape[0] <= 2:

            return unique.tolist()

        try:

            hull = ConvexHull(unique)

        except Exception:

            return unique.tolist()

        vertices = unique[hull.vertices]

        return vertices.tolist()

    def _suggest_cluster_params(self, count: int) -> tuple[int, int]:

        if count <= 15:

            return 3, 1

        if count <= 50:

            return 5, 2

        return 8, 3

    async def _clear_existing(self, session, run_id: UUID) -> None:

        response_ids = await session.exec(
            select(Response.id).where(Response.run_id == run_id)
        )

        ids = response_ids.scalars().all()

        if ids:

            await session.execute(delete(Cluster).where(Cluster.response_id.in_(ids)))

            await session.execute(
                delete(Projection).where(Projection.response_id.in_(ids))
            )

            await session.execute(
                delete(Embedding).where(Embedding.response_id.in_(ids))
            )

            await session.execute(
                delete(ResponseSegment).where(ResponseSegment.response_id.in_(ids))
            )

            await session.execute(
                delete(ResponseHull).where(ResponseHull.response_id.in_(ids))
            )

            await session.execute(delete(Response).where(Response.id.in_(ids)))

            await session.execute(
                delete(SegmentEdge).where(SegmentEdge.run_id == run_id)
            )

            await session.commit()


async def load_run_with_details(session, run_id: UUID):

    run = await session.get(Run, run_id)

    if not run:

        return None

    results = await session.exec(
        select(Response).where(Response.run_id == run_id).order_by(Response.index)
    )

    responses = results.scalars().all()

    if not responses:

        return run, [], [], [], [], [], [], []

    response_ids = [response.id for response in responses]

    projection_rows = await session.exec(
        select(Projection)
        .where(Projection.response_id.in_(response_ids))
        .order_by(Projection.response_id)
    )

    cluster_rows = await session.exec(
        select(Cluster)
        .where(Cluster.response_id.in_(response_ids))
        .order_by(Cluster.response_id)
    )

    embedding_rows = await session.exec(
        select(Embedding)
        .where(Embedding.response_id.in_(response_ids))
        .order_by(Embedding.response_id)
    )

    segment_rows = await session.exec(
        select(ResponseSegment)
        .where(ResponseSegment.response_id.in_(response_ids))
        .order_by(ResponseSegment.response_id, ResponseSegment.position)
    )

    edge_rows = await session.exec(
        select(SegmentEdge).where(SegmentEdge.run_id == run_id)
    )

    hull_rows = await session.exec(
        select(ResponseHull)
        .where(ResponseHull.response_id.in_(response_ids))
        .order_by(ResponseHull.response_id)
    )

    return (
        run,
        responses,
        projection_rows.scalars().all(),
        cluster_rows.scalars().all(),
        embedding_rows.scalars().all(),
        segment_rows.scalars().all(),
        edge_rows.scalars().all(),
        hull_rows.scalars().all(),
    )


async def list_recent_runs(session, limit: int = 20) -> list[RunSummary]:

    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)

    results = await session.exec(stmt)

    runs = results.scalars().all()

    if not runs:

        return []

    run_ids = [run.id for run in runs]

    response_counts_query = (
        select(Response.run_id, func.count())
        .where(Response.run_id.in_(run_ids))
        .group_by(Response.run_id)
    )

    response_counts = await session.exec(response_counts_query)

    response_map = {row[0]: int(row[1]) for row in response_counts.all()}

    segment_counts_query = (
        select(Response.run_id, func.count())
        .join(ResponseSegment, ResponseSegment.response_id == Response.id)
        .where(Response.run_id.in_(run_ids))
        .group_by(Response.run_id)
    )

    segment_counts = await session.exec(segment_counts_query)

    segment_map = {row[0]: int(row[1]) for row in segment_counts.all()}

    summaries: list[RunSummary] = []

    for run in runs:

        metadata = None

        if run.progress_metadata:

            try:

                metadata = json.loads(run.progress_metadata)

            except json.JSONDecodeError:

                metadata = None

        summaries.append(
            RunSummary(
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
                response_count=response_map.get(run.id, 0),
                segment_count=segment_map.get(run.id, 0),
                notes=run.notes,
                progress_stage=run.progress_stage,
                progress_message=run.progress_message,
                progress_percent=run.progress_percent,
                progress_metadata=metadata,
            )
        )

    return summaries
