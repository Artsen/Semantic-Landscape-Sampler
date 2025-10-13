"""High level orchestration for sampling runs.

Classes:
    RunService: Coordinates OpenAI calls, segmentation, embeddings, projections, clustering, and persistence for runs.

Functions:
    load_run_with_details(session, run_id): Fetch a run and all derived ORM records needed to build API responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import UUID

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from sqlalchemy import delete, func, select
from sqlalchemy.exc import OperationalError
from threadpoolctl import threadpool_info

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

try:  # optional dependency
    from annoy import AnnoyIndex  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    AnnoyIndex = None  # type: ignore

try:  # optional dependency
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    hnswlib = None  # type: ignore

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    faiss = None  # type: ignore

from app.core.config import get_settings
from app.models import (
    Cluster,
    ClusterMetrics,
    Embedding,
    Projection,
    Response,
    ResponseHull,
    ResponseSegment,
    Run,
    RunStatus,
    RunProvenance,
    ProjectionCache,
    AnnIndex,
    EmbeddingCache,
    SegmentInsight,
    SegmentEdge,
)
from app.schemas import RunCreateRequest, RunSummary, RunMetrics, RunUpdateRequest, SampleRequest, UMAPParams
from app.services.openai_client import EmbeddingBatch, OpenAIService
from app.services.projection import (
    FEATURE_VERSION,
    build_feature_matrix,
    cluster_with_fallback,
    compute_pca_projection,
    compute_tsne,
    compute_umap,
    _l2_normalise,
)
from app.services.cluster_metrics import (
    ClusterMetricsResult,
    compute_cluster_metrics,
)
from app.services.segmentation import SegmentDraft, make_segment_drafts
from app.utils.text import normalise_for_embedding, compute_simhash64


@dataclass(slots=True)
class ResponseFeatureContext:
    responses: list[Response]
    response_ids: list[UUID]
    feature_matrix: np.ndarray
    vectors: list[np.ndarray]
    cluster_labels: dict[UUID, int]
    umap_coords_2d: dict[UUID, np.ndarray]
    umap_coords_3d: dict[UUID, np.ndarray]


@dataclass(slots=True)
class ProjectionCachePayload:
    method: str
    requested_params: dict[str, Any]
    resolved_params: dict[str, Any]
    feature_version: str
    coords_2d: np.ndarray
    coords_3d: np.ndarray
    response_ids: list[UUID]
    warnings: list[str]
    total_count: int
    point_count: int
    is_subsample: bool
    subsample_strategy: str | None
    trustworthiness_2d: float | None
    trustworthiness_3d: float | None
    continuity_2d: float | None
    continuity_3d: float | None
    cached_at: datetime | None = None
from app.db.session import init_db

_SETTINGS = get_settings()
_LOGGER = logging.getLogger(__name__)

INDEX_DIR = Path(_SETTINGS.ann_index_dir).resolve()
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class RunStageTimer:
    """Utility to capture stage-level timings for a sampling run."""

    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._wall_start = datetime.utcnow()
        self._stages: list[dict[str, Any]] = []

    @asynccontextmanager
    async def track(self, name: str):
        start_counter = time.perf_counter()
        start_wall = datetime.utcnow()
        try:
            yield
        finally:
            end_counter = time.perf_counter()
            end_wall = datetime.utcnow()
            self._stages.append(
                {
                    "name": name,
                    "duration_ms": round((end_counter - start_counter) * 1000.0, 3),
                    "offset_ms": round((start_counter - self._origin) * 1000.0, 3),
                    "started_at": start_wall.isoformat(timespec="milliseconds") + "Z",
                    "finished_at": end_wall.isoformat(timespec="milliseconds") + "Z",
                }
            )

    def snapshot(self) -> dict[str, Any]:
        total_ms = round((time.perf_counter() - self._origin) * 1000.0, 3)
        finished_at = datetime.utcnow()
        return {
            "total_duration_ms": total_ms,
            "stages": list(self._stages),
            "started_at": self._wall_start.isoformat(timespec="milliseconds") + "Z",
            "finished_at": finished_at.isoformat(timespec="milliseconds") + "Z",
        }


class RunService:
    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self._openai = openai_service or OpenAIService()

    def _normalise_projection_method(self, method: str | None) -> str:
        if not method:
            return "umap"
        value = method.strip().lower()
        if value not in {"umap", "tsne", "pca"}:
            raise ValueError("method must be 'umap', 'tsne', or 'pca'")
        return value

    def _base_projection_params(
        self,
        run: Run,
        method: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        overrides = overrides or {}
        method = method.lower()
        if method == "umap":
            n_neighbors = self._safe_int(
                overrides.get("n_neighbors"),
                run.umap_n_neighbors or _SETTINGS.umap_default_n_neighbors,
            )
            min_dist = self._safe_float(
                overrides.get("min_dist"),
                run.umap_min_dist if run.umap_min_dist is not None else _SETTINGS.umap_default_min_dist,
            )
            metric = str(overrides.get("metric", run.umap_metric or _SETTINGS.umap_default_metric)).lower()
            seed = self._safe_int(
                overrides.get("seed"),
                run.umap_seed if run.umap_seed is not None else _SETTINGS.umap_default_seed,
            )
            return {
                "n_neighbors": max(2, n_neighbors),
                "min_dist": min_dist,
                "metric": metric,
                "seed": seed,
            }
        if method == "tsne":
            perplexity = self._safe_float(overrides.get("perplexity"), 30.0)
            learning_rate = overrides.get("learning_rate", "auto")
            n_iter = self._safe_int(overrides.get("n_iter"), 1000)
            early = self._safe_float(overrides.get("early_exaggeration"), 12.0)
            seed = self._safe_int(
                overrides.get("seed"),
                run.umap_seed if run.umap_seed is not None else _SETTINGS.umap_default_seed,
            )
            metric = str(overrides.get("metric", "euclidean")).lower()
            if isinstance(learning_rate, str) and learning_rate.strip().lower() == "auto":
                lr_value: float | str = "auto"
            else:
                lr_value = self._safe_float(learning_rate, 200.0)
            return {
                "perplexity": max(2.0, perplexity),
                "learning_rate": lr_value,
                "n_iter": max(250, n_iter),
                "early_exaggeration": max(1.0, early),
                "seed": seed,
                "metric": metric,
            }
        if method == "pca":
            n_components = self._safe_int(overrides.get("n_components"), 50)
            seed = self._safe_int(
                overrides.get("seed"),
                run.umap_seed if run.umap_seed is not None else _SETTINGS.umap_default_seed,
            )
            metric = str(overrides.get("metric", "euclidean")).lower()
            return {
                "n_components": max(3, n_components),
                "seed": seed,
                "metric": metric,
            }
        raise ValueError(f"Unsupported projection method: {method}")

    def _resolve_projection_params(
        self,
        method: str,
        base: dict[str, Any],
        *,
        sample_count: int,
        feature_dim: int,
    ) -> dict[str, Any]:
        resolved = dict(base)
        method = method.lower()
        if method == "umap":
            resolved["n_neighbors"] = int(max(2, resolved.get("n_neighbors", 30)))
            resolved["min_dist"] = float(max(0.0, min(resolved.get("min_dist", 0.3), 0.99)))
            resolved["metric"] = resolved.get("metric", "cosine")
            resolved["seed"] = int(resolved.get("seed", _SETTINGS.umap_default_seed))
            effective = min(
                max(5, resolved["n_neighbors"]),
                max(2, sample_count - 1),
            )
            resolved["effective_n_neighbors"] = int(effective)
            return resolved
        if method == "tsne":
            resolved["seed"] = int(resolved.get("seed", _SETTINGS.umap_default_seed))
            resolved["metric"] = resolved.get("metric", "euclidean")
            max_perplexity = max(2.0, (sample_count - 1) / 3.0) if sample_count > 1 else 2.0
            resolved["perplexity"] = float(min(max(2.0, resolved.get("perplexity", 30.0)), max_perplexity))
            learning_rate = resolved.get("learning_rate", "auto")
            if isinstance(learning_rate, str):
                lr_lower = learning_rate.strip().lower()
                resolved["learning_rate"] = "auto" if lr_lower == "auto" else max(10.0, self._safe_float(learning_rate, 200.0))
            else:
                resolved["learning_rate"] = max(10.0, float(learning_rate))
            resolved["n_iter"] = int(max(250, resolved.get("n_iter", 1000)))
            resolved["early_exaggeration"] = float(max(1.0, resolved.get("early_exaggeration", 12.0)))
            return resolved
        if method == "pca":
            resolved["seed"] = int(resolved.get("seed", _SETTINGS.umap_default_seed))
            resolved["metric"] = resolved.get("metric", "euclidean")
            max_components = min(
                resolved.get("n_components", 50),
                sample_count,
                feature_dim or resolved.get("n_components", 50),
            )
            resolved["n_components"] = int(max(3, max_components))
            return resolved
        raise ValueError(f"Unsupported projection method: {method}")

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _hash_projection_params(self, method: str, params: dict[str, Any]) -> str:
        payload = {
            "method": method,
            "params": params,
            "feature_version": FEATURE_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    async def _load_response_feature_context(
        self,
        session,
        run: Run,
    ) -> ResponseFeatureContext:
        responses_result = await session.exec(
            select(Response)
            .where(Response.run_id == run.id)
            .order_by(Response.index)
        )
        responses = responses_result.scalars().all()
        if not responses:
            raise ValueError("Run has no responses")
        response_ids = [response.id for response in responses]

        embeddings_result = await session.exec(
            select(Embedding).where(Embedding.response_id.in_(response_ids))
        )
        embeddings_map = {
            embedding.response_id: embedding
            for embedding in embeddings_result.scalars().all()
        }
        missing_embeddings = [rid for rid in response_ids if rid not in embeddings_map]
        if missing_embeddings:
            raise ValueError("Embeddings missing for responses")

        vectors: list[np.ndarray] = []
        for response in responses:
            record = embeddings_map[response.id]
            array = np.frombuffer(record.vector, dtype=np.float32)
            if record.dim and record.dim > 0 and array.size >= record.dim:
                array = array[: record.dim]
            vectors.append(array.astype(np.float32, copy=False))

        texts = [response.raw_text or "" for response in responses]
        feature_matrix = build_feature_matrix(
            texts,
            vectors,
            prompt_embedding=None,
            keyword_axes=_SETTINGS.segment_keyword_axes,
        )
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        cluster_result = await session.exec(
            select(Cluster).where(Cluster.response_id.in_(response_ids))
        )
        cluster_labels: dict[UUID, int] = {}
        for cluster in cluster_result.scalars().all():
            try:
                label_value = int(cluster.label) if cluster.label is not None else -1
            except (TypeError, ValueError):
                label_value = -1
            cluster_labels[cluster.response_id] = label_value

        projection_result = await session.exec(
            select(Projection).where(
                Projection.response_id.in_(response_ids),
                Projection.method == "umap",
            )
        )
        coords_2d: dict[UUID, np.ndarray] = {}
        coords_3d: dict[UUID, np.ndarray] = {}
        for projection in projection_result.scalars().all():
            if projection.dim == 2:
                coords_2d[projection.response_id] = np.array(
                    [float(projection.x), float(projection.y)],
                    dtype=np.float32,
                )
            elif projection.dim == 3:
                coords_3d[projection.response_id] = np.array(
                    [
                        float(projection.x),
                        float(projection.y),
                        float(projection.z or 0.0),
                    ],
                    dtype=np.float32,
                )

        return ResponseFeatureContext(
            responses=responses,
            response_ids=response_ids,
            feature_matrix=feature_matrix,
            vectors=vectors,
            cluster_labels=cluster_labels,
            umap_coords_2d=coords_2d,
            umap_coords_3d=coords_3d,
        )

    def _stratified_sample_indices(
        self,
        context: ResponseFeatureContext,
        limit: int,
        seed: int,
    ) -> list[int]:
        total = len(context.response_ids)
        if total <= limit:
            return list(range(total))
        rng = random.Random(seed)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, response_id in enumerate(context.response_ids):
            label = context.cluster_labels.get(response_id, -1)
            buckets[label].append(index)
        selected: list[int] = []
        for indices in buckets.values():
            if not indices:
                continue
            share = max(1, round(len(indices) / total * limit))
            rng.shuffle(indices)
            selected.extend(indices[:share])
        if len(selected) > limit:
            rng.shuffle(selected)
            selected = selected[:limit]
        unique_selected = sorted(set(selected))
        if len(unique_selected) < limit:
            remaining = [idx for idx in range(total) if idx not in unique_selected]
            rng.shuffle(remaining)
            needed = limit - len(unique_selected)
            unique_selected.extend(sorted(remaining[:needed]))
        return sorted(unique_selected[:limit])

    def _extract_layout(
        self,
        context: ResponseFeatureContext,
        indices: Sequence[int],
        *,
        dim: int,
    ) -> np.ndarray | None:
        source = context.umap_coords_2d if dim == 2 else context.umap_coords_3d
        coords: list[np.ndarray] = []
        for index in indices:
            response_id = context.response_ids[index]
            array = source.get(response_id)
            if array is None:
                return None
            coords.append(array)
        if not coords:
            return None
        return np.asarray(coords, dtype=np.float32)

    async def _persist_projection_cache(
        self,
        session,
        run_id: UUID,
        payload: ProjectionCachePayload,
        params_hash: str,
    ) -> ProjectionCache:
        coords_2d = np.asarray(payload.coords_2d, dtype=np.float32)
        coords_3d = np.asarray(payload.coords_3d, dtype=np.float32)
        await session.execute(
            delete(ProjectionCache).where(
                ProjectionCache.run_id == run_id,
                ProjectionCache.method == payload.method,
                ProjectionCache.params_hash == params_hash,
                ProjectionCache.feature_version == payload.feature_version,
            )
        )
        record = ProjectionCache(
            run_id=run_id,
            method=payload.method,
            params_hash=params_hash,
            params_json=json.dumps(
                {
                    "requested": payload.requested_params,
                    "resolved": payload.resolved_params,
                },
                sort_keys=True,
            ),
            feature_version=payload.feature_version,
            coord_dtype="float32",
            point_count=payload.point_count,
            total_count=payload.total_count,
            is_subsample=payload.is_subsample,
            subsample_strategy=payload.subsample_strategy,
            coords_2d=coords_2d.tobytes() if coords_2d.size else None,
            coords_3d=coords_3d.tobytes() if coords_3d.size else None,
            response_ids_json=json.dumps([str(rid) for rid in payload.response_ids]),
            trustworthiness_2d=payload.trustworthiness_2d,
            trustworthiness_3d=payload.trustworthiness_3d,
            continuity_2d=payload.continuity_2d,
            continuity_3d=payload.continuity_3d,
            warnings_json=json.dumps(payload.warnings),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    def _payload_from_cache(self, cache: ProjectionCache) -> ProjectionCachePayload:
        raw_params = cache.params_json or "{}"
        try:
            params_payload = json.loads(raw_params)
        except (TypeError, json.JSONDecodeError):
            params_payload = {}
        if isinstance(params_payload, dict) and "requested" in params_payload:
            requested = params_payload.get("requested", {})
            resolved = params_payload.get("resolved", requested)
        else:
            requested = params_payload if isinstance(params_payload, dict) else {}
            resolved = requested
        try:
            response_ids = [UUID(value) for value in json.loads(cache.response_ids_json)]
        except Exception:
            response_ids = []
        try:
            warnings = json.loads(cache.warnings_json) if cache.warnings_json else []
        except json.JSONDecodeError:
            warnings = []
        dtype = np.dtype(cache.coord_dtype or "float32")
        coords_2d = (
            np.frombuffer(cache.coords_2d, dtype=dtype).reshape(cache.point_count, 2)
            if cache.coords_2d
            else np.zeros((0, 2), dtype=np.float32)
        )
        coords_3d = (
            np.frombuffer(cache.coords_3d, dtype=dtype).reshape(cache.point_count, 3)
            if cache.coords_3d
            else np.zeros((0, 3), dtype=np.float32)
        )
        payload = ProjectionCachePayload(
            method=cache.method,
            requested_params=requested,
            resolved_params=resolved,
            feature_version=cache.feature_version,
            coords_2d=np.asarray(coords_2d, dtype=np.float32),
            coords_3d=np.asarray(coords_3d, dtype=np.float32),
            response_ids=response_ids,
            warnings=warnings,
            total_count=cache.total_count,
            point_count=cache.point_count,
            is_subsample=cache.is_subsample,
            subsample_strategy=cache.subsample_strategy,
            trustworthiness_2d=cache.trustworthiness_2d,
            trustworthiness_3d=cache.trustworthiness_3d,
            continuity_2d=cache.continuity_2d,
            continuity_3d=cache.continuity_3d,
            cached_at=cache.created_at,
        )
        return payload

    def _maybe_tsne_preview(
        self,
        context: ResponseFeatureContext,
        run: Run,
        base_params: dict[str, Any],
    ) -> tuple[list[int], bool, str | None, list[str]]:
        total = len(context.response_ids)
        threshold = getattr(_SETTINGS, "tsne_preview_threshold", 12000)
        preview_limit = getattr(_SETTINGS, "tsne_preview_size", 10000)
        if total <= threshold:
            return list(range(total)), False, None, []
        seed = base_params.get("seed") or run.umap_seed or _SETTINGS.umap_default_seed
        indices = self._stratified_sample_indices(context, preview_limit, int(seed))
        strategy = f"cluster-proportional-{preview_limit}"
        warnings = [
            f"tsne-preview:{preview_limit}",
            f"tsne-total:{total}",
        ]
        return indices, True, strategy, warnings

    async def get_projection_variant(
        self,
        session,
        *,
        run_id: UUID,
        method: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[ProjectionCachePayload, bool]:
        run = await session.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")
        method_name = self._normalise_projection_method(method)
        base_params = self._base_projection_params(run, method_name, params)
        params_hash = self._hash_projection_params(method_name, base_params)
        cache_result = await session.exec(
            select(ProjectionCache).where(
                ProjectionCache.run_id == run_id,
                ProjectionCache.method == method_name,
                ProjectionCache.params_hash == params_hash,
                ProjectionCache.feature_version == FEATURE_VERSION,
            )
        )
        cache_entry = cache_result.first()
        if cache_entry:
            payload = self._payload_from_cache(cache_entry)
            return payload, True

        context = await self._load_response_feature_context(session, run)
        total_count = len(context.response_ids)
        indices = list(range(total_count))
        is_subsample = False
        subsample_strategy: str | None = None
        warnings: list[str] = []
        if method_name == "tsne":
            indices, is_subsample, subsample_strategy, warnings = self._maybe_tsne_preview(context, run, base_params)

        feature_subset = context.feature_matrix[indices, :]
        sample_count = feature_subset.shape[0]
        feature_dim = feature_subset.shape[1] if feature_subset.ndim == 2 else 0
        resolved_params = self._resolve_projection_params(
            method_name,
            base_params,
            sample_count=sample_count,
            feature_dim=feature_dim,
        )

        if method_name == "umap":
            result = compute_umap(
                feature_subset,
                random_state=int(resolved_params.get("seed", _SETTINGS.umap_default_seed)),
                n_neighbors=int(resolved_params.get("effective_n_neighbors", resolved_params["n_neighbors"])),
                min_dist=float(resolved_params["min_dist"]),
                metric=str(resolved_params["metric"]),
            )
        elif method_name == "tsne":
            init_2d = self._extract_layout(context, indices, dim=2)
            init_3d = self._extract_layout(context, indices, dim=3)
            result = compute_tsne(
                feature_subset,
                random_state=int(resolved_params.get("seed", _SETTINGS.umap_default_seed)),
                perplexity=float(resolved_params["perplexity"]),
                learning_rate=resolved_params["learning_rate"],
                n_iter=int(resolved_params["n_iter"]),
                init_2d=init_2d,
                init_3d=init_3d,
                early_exaggeration=float(resolved_params["early_exaggeration"]),
                metric=str(resolved_params.get("metric", "euclidean")),
            )
        elif method_name == "pca":
            result = compute_pca_projection(
                feature_subset,
                n_components=int(resolved_params["n_components"]),
                metric=str(resolved_params.get("metric", "euclidean")),
            )
        else:
            raise ValueError(f"Unsupported projection method: {method_name}")

        coords_2d = np.asarray(result.coords_2d, dtype=np.float32)
        coords_3d = np.asarray(result.coords_3d, dtype=np.float32)
        response_ids = [context.response_ids[index] for index in indices]
        payload = ProjectionCachePayload(
            method=method_name,
            requested_params=base_params,
            resolved_params=resolved_params,
            feature_version=FEATURE_VERSION,
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            response_ids=response_ids,
            warnings=warnings,
            total_count=total_count,
            point_count=sample_count,
            is_subsample=is_subsample,
            subsample_strategy=subsample_strategy,
            trustworthiness_2d=result.trustworthiness_2d,
            trustworthiness_3d=result.trustworthiness_3d,
            continuity_2d=result.continuity_2d,
            continuity_3d=result.continuity_3d,
        )
        cache_record = await self._persist_projection_cache(
            session,
            run_id,
            payload,
            params_hash,
        )
        payload.cached_at = cache_record.created_at
        return payload, False

    def _resolve_umap_params(
        self,
        payload: RunCreateRequest,
    ) -> tuple[int, float, str, int, str]:
        defaults = {
            "n_neighbors": _SETTINGS.umap_default_n_neighbors,
            "min_dist": _SETTINGS.umap_default_min_dist,
            "metric": _SETTINGS.umap_default_metric,
            "seed": _SETTINGS.umap_default_seed,
        }

        umap_payload = payload.umap

        requested_neighbors = (
            umap_payload.n_neighbors
            if umap_payload and umap_payload.n_neighbors is not None
            else defaults["n_neighbors"]
        )
        if requested_neighbors > 200:
            raise ValueError("umap.n_neighbors must be between 5 and 200")
        if requested_neighbors < 5:
            requested_neighbors = 5

        max_neighbors = max(2, payload.n - 1)
        if payload.n and requested_neighbors > max_neighbors:
            _LOGGER.warning(
                "Requested UMAP n_neighbors %s exceeds dataset maximum %s; layout will clamp at runtime",
                requested_neighbors,
                max_neighbors,
            )
        if payload.n and max_neighbors < 5:
            _LOGGER.warning(
                "Dataset size %s limits UMAP n_neighbors to %s; consider sampling more responses",
                payload.n,
                max_neighbors,
            )
        if payload.n:
            threshold = max(1, int(0.02 * payload.n))
            if requested_neighbors > threshold:
                _LOGGER.warning(
                    "UMAP n_neighbors=%s exceeds 2%% of N=%s; layout may lose locality",
                    requested_neighbors,
                    payload.n,
                )

        n_neighbors = min(200, requested_neighbors)

        min_dist = (
            umap_payload.min_dist
            if umap_payload and umap_payload.min_dist is not None
            else defaults["min_dist"]
        )
        if not (0.0 <= min_dist < 1.0):
            raise ValueError("umap.min_dist must be between 0.0 and 0.99")

        metric = (
            umap_payload.metric
            if umap_payload and umap_payload.metric is not None
            else defaults["metric"]
        )
        metric = str(metric).strip().lower() or defaults["metric"]

        seed = (
            umap_payload.seed
            if umap_payload and umap_payload.seed is not None
            else payload.seed
        )
        if seed is None:
            seed = defaults["seed"]
            seed_source = "default"
        else:
            seed_source = "ui"

        return n_neighbors, float(min_dist), metric, int(seed), seed_source
    def _resolve_cluster_metric(self, metric: str | None) -> str:
        allowed = {"euclidean", "l1", "l2", "manhattan", "minkowski", "seuclidean", "chebyshev", "canberra", "braycurtis", "mahalanobis", "haversine"}
        if metric:
            name = metric.strip().lower()
            if name in allowed:
                return name
        return "euclidean"

    def _resolve_cluster_config(
        self,
        payload: RunCreateRequest,
    ) -> tuple[str, Optional[int], Optional[int]]:
        algo = payload.cluster_algo or _SETTINGS.cluster_default_algo
        if algo not in {"hdbscan", "kmeans"}:
            algo = _SETTINGS.cluster_default_algo

        if algo != "hdbscan":
            return algo, None, None

        default_size = max(
            2,
            min(_SETTINGS.hdbscan_default_min_cluster_size, max(payload.n, 2)),
        )
        requested_size = payload.hdbscan_min_cluster_size or default_size
        min_cluster_size = max(2, min(int(requested_size), max(payload.n, 2)))

        default_samples = max(
            1,
            min(_SETTINGS.hdbscan_default_min_samples, min_cluster_size),
        )
        requested_samples = payload.hdbscan_min_samples or default_samples
        min_samples = max(1, min(int(requested_samples), min_cluster_size))

        return algo, min_cluster_size, min_samples

    async def _upsert_provenance(
        self,
        session,
        run: Run,
        provenance_payload: dict[str, Any],
    ) -> None:
        record = await session.exec(
            select(RunProvenance).where(RunProvenance.run_id == run.id)
        )
        existing = record.scalar_one_or_none()
        serialised = {
            "lib_versions_json": json.dumps(provenance_payload.get("lib_versions", {})),
            "feature_weights_json": json.dumps(
                provenance_payload.get("feature_weights", {})
            ),
            "input_space_json": json.dumps(provenance_payload.get("input_space", {})),
            "umap_params_json": json.dumps(provenance_payload.get("umap_params", {})),
            "cluster_params_json": json.dumps(
                provenance_payload.get("cluster_params", {})
            ),
        }
        base = {
            "run_id": run.id,
            "python_version": provenance_payload.get("python_version"),
            "node_version": provenance_payload.get("node_version"),
            "blas_impl": provenance_payload.get("blas_impl"),
            "openmp_threads": provenance_payload.get("openmp_threads"),
            "numba_version": provenance_payload.get("numba_version"),
            "numba_target": provenance_payload.get("numba_target"),
            "embedding_model": provenance_payload.get("embedding_model"),
            "embedding_dim": provenance_payload.get("embedding_dim"),
            "llm_model": provenance_payload.get("llm_model"),
            "temperature": provenance_payload.get("temperature"),
            "top_p": provenance_payload.get("top_p"),
            "max_tokens": provenance_payload.get("max_tokens"),
            "commit_sha": provenance_payload.get("commit_sha"),
            "env_label": provenance_payload.get("env_label"),
            "random_state_seed_source": provenance_payload.get(
                "random_state_seed_source"
            ),
        }
        base.update(serialised)

        if existing:
            for key, value in base.items():
                setattr(existing, key, value)
            session.add(existing)
        else:
            session.add(RunProvenance(**base))
        await session.commit()

    def _collect_provenance(
        self,
        run: Run,
        *,
        embedding_dim: int,
        umap_params: dict[str, Any],
        cluster_params: dict[str, Any],
    ) -> dict[str, Any]:
        lib_versions: dict[str, str] = {}
        for lib in (
            "umap-learn",
            "hdbscan",
            "numpy",
            "scikit-learn",
            "scipy",
            "pandas",
            "sqlmodel",
            "fastapi",
        ):
            try:
                lib_versions[lib] = importlib_metadata.version(lib)
            except importlib_metadata.PackageNotFoundError:
                continue

        blas_impl = None
        openmp_threads = None
        try:
            infos = threadpool_info()
            for info in infos:
                if "blas_info" in info.get("internal_api", "") or info.get("internal_api") == "blas" or info.get("library"):
                    blas_impl = info.get("internal_api") or info.get("library")
                if "num_threads" in info:
                    openmp_threads = info.get("num_threads")
        except Exception:
            blas_impl = None

        numba_version = None
        numba_target = None
        try:
            numba_version = importlib_metadata.version("numba")
            import numba  # type: ignore

            cfg = getattr(numba, "config", None)
            target_value = getattr(cfg, "TARGET", None) if cfg is not None else None
            if target_value is not None:
                numba_target = str(target_value)
        except Exception:
            numba_target = None

        node_version = os.environ.get("NODE_VERSION")
        if node_version is None:
            try:
                completed = subprocess.run(
                    ["node", "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                if completed.returncode == 0:
                    node_version = completed.stdout.strip()
            except Exception:
                node_version = None

        if openmp_threads is None:
            omp_env = os.environ.get("OMP_NUM_THREADS")
            if omp_env:
                try:
                    openmp_threads = int(omp_env)
                except ValueError:
                    openmp_threads = None

        feature_weights = {
            "embedding": 1.0,
            "tfidf": 0.4,
            "stats": 0.25,
            "keyword_axes": 1.0 if _SETTINGS.segment_keyword_axes else 0.0,
        }

        input_space = {
            "description": "cosine similarity on blended embedding/tfidf/stat features",
            "metric": run.umap_metric,
        }

        payload = {
            "python_version": platform.python_version(),
            "node_version": node_version,
            "blas_impl": blas_impl,
            "openmp_threads": openmp_threads,
            "numba_version": numba_version,
            "numba_target": numba_target,
            "lib_versions": lib_versions,
            "embedding_model": run.embedding_model,
            "embedding_dim": embedding_dim,
            "llm_model": run.model,
            "temperature": run.temperature,
            "top_p": run.top_p,
            "max_tokens": run.max_tokens,
            "feature_weights": feature_weights,
            "input_space": input_space,
            "umap_params": umap_params,
            "cluster_params": cluster_params,
            "commit_sha": self._get_commit_sha(),
            "env_label": _SETTINGS.default_env_label,
            "random_state_seed_source": run.random_state_seed_source,
        }
        return payload

    def _deserialize_cache_vector(self, cache: EmbeddingCache) -> np.ndarray | None:
        dtype = (cache.vector_dtype or "float16").lower()
        try:
            if dtype == "float16":
                arr = np.frombuffer(cache.vector, dtype=np.float16).astype(np.float32)
            elif dtype == "float32":
                arr = np.frombuffer(cache.vector, dtype=np.float32)
            else:
                arr = np.frombuffer(cache.vector, dtype=np.float32)
        except Exception:
            return None
        if cache.dim and arr.size != cache.dim:
            try:
                arr = arr[: cache.dim]
            except Exception:
                return None
        return arr.astype(np.float32, copy=False)

    async def _upsert_embedding_cache(
        self,
        session,
        *,
        vectors: dict[str, np.ndarray],
        model_id: str,
        preproc_version: str,
        provider: str = "openai",
        model_revision: str | None = None,
    ) -> None:
        if not vectors:
            return

        hashes = list(vectors.keys())
        stmt = select(EmbeddingCache).where(
            EmbeddingCache.text_hash.in_(hashes),
            EmbeddingCache.model_id == model_id,
            EmbeddingCache.preproc_version == preproc_version,
        )
        result = await session.exec(stmt)
        existing_map = {record.text_hash: record for record in result.scalars()}

        for text_hash, array in vectors.items():
            arr = np.asarray(array, dtype=np.float32)
            dim = int(arr.size)
            norm = float(np.linalg.norm(arr)) if dim else 0.0
            vector_bytes = arr.astype(np.float16).tobytes()

            cache_obj = existing_map.get(text_hash)
            if cache_obj is None:
                cache_obj = EmbeddingCache(
                    text_hash=text_hash,
                    model_id=model_id,
                    preproc_version=preproc_version,
                    provider=provider,
                    model_revision=model_revision,
                    vector=vector_bytes,
                    vector_dtype="float16",
                    vector_norm=norm,
                    dim=dim,
                )
                session.add(cache_obj)
            else:
                cache_obj.vector = vector_bytes
                cache_obj.vector_dtype = "float16"
                cache_obj.vector_norm = norm
                cache_obj.dim = dim
                cache_obj.provider = provider
                if model_revision:
                    cache_obj.model_revision = model_revision
        await session.commit()

    async def _embed_texts_with_cache(
        self,
        session,
        texts: Sequence[str],
        *,
        model_id: str,
        preproc_version: str,
        use_cache: bool,
        mark_duplicates: bool,
    ) -> tuple[EmbeddingBatch, list[dict[str, Any]]]:
        if not texts:
            return EmbeddingBatch(vectors=[], model=model_id, dim=0), []

        records: list[dict[str, Any]] = []
        hash_to_indices: dict[str, list[int]] = defaultdict(list)
        seen_hashes: set[str] = set()

        for index, raw_text in enumerate(texts):
            original = raw_text or ""
            collapsed, normalised = normalise_for_embedding(original)
            base = normalised or collapsed
            text_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
            record = {
                "index": index,
                "text": original,
                "hash": text_hash,
                "vector": None,
                "is_cached": False,
                "is_duplicate": False,
                "simhash64": compute_simhash64(collapsed.split()) if collapsed else None,
            }
            if mark_duplicates and not use_cache:
                if text_hash in seen_hashes:
                    record["is_duplicate"] = True
                seen_hashes.add(text_hash)
            records.append(record)
            hash_to_indices[text_hash].append(index)

        expected_dim: int | None = None
        provider = "openai"
        model_revision: str | None = None

        if use_cache:
            hashes = {record["hash"] for record in records}
            if hashes:
                stmt = select(EmbeddingCache).where(
                    EmbeddingCache.text_hash.in_(hashes),
                    EmbeddingCache.model_id == model_id,
                    EmbeddingCache.preproc_version == preproc_version,
                )
                cache_result = await session.exec(stmt)
                cache_map = {
                    cache.text_hash: cache for cache in cache_result.scalars()
                }
            else:
                cache_map = {}

            for text_hash, indices in hash_to_indices.items():
                cache = cache_map.get(text_hash)
                if not cache:
                    continue
                arr = self._deserialize_cache_vector(cache)
                if arr is None:
                    continue
                if expected_dim is not None and arr.size != expected_dim and expected_dim != 0:
                    continue
                if expected_dim is None:
                    expected_dim = arr.size
                for idx in indices:
                    records[idx]["vector"] = arr
                    records[idx]["is_cached"] = True

        embed_primary_indices: list[int] = []
        for text_hash, indices in hash_to_indices.items():
            primary_idx = indices[0]
            if records[primary_idx]["vector"] is None:
                embed_primary_indices.append(primary_idx)

        written_vectors: dict[str, np.ndarray] = {}
        if embed_primary_indices:
            embed_texts = [records[idx]["text"] for idx in embed_primary_indices]
            batch = await self._openai.embed_texts(embed_texts, model=model_id)
            provider = batch.provider
            model_revision = batch.model_revision
            embed_vectors = batch.vectors
            for target_idx, vector in zip(embed_primary_indices, embed_vectors, strict=False):
                arr = np.asarray(vector, dtype=np.float32)
                if expected_dim is None:
                    expected_dim = arr.size
                records[target_idx]["vector"] = arr
                written_vectors[records[target_idx]["hash"]] = arr
                for dup_idx in hash_to_indices[records[target_idx]["hash"]][1:]:
                    if records[dup_idx]["vector"] is None:
                        records[dup_idx]["vector"] = arr

        vectors: list[list[float]] = []
        for record in records:
            arr = record["vector"]
            if arr is None:
                arr = np.zeros(expected_dim or 0, dtype=np.float32)
            vector_list = arr.astype(np.float32, copy=False).tolist()
            record["vector_list"] = vector_list
            vectors.append(vector_list)

        dim = expected_dim or (len(vectors[0]) if vectors and vectors[0] else 0)
        embedding_batch = EmbeddingBatch(
            vectors=vectors,
            model=model_id,
            dim=dim,
            model_revision=model_revision,
            provider=provider,
        )

        if written_vectors:
            await self._upsert_embedding_cache(
                session,
                vectors=written_vectors,
                model_id=model_id,
                preproc_version=preproc_version,
                provider=provider,
                model_revision=model_revision,
            )

        return embedding_batch, records

    def _get_commit_sha(self) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def _select_ann_method(self, count: int) -> str:
        """Pick an ANN backend based on dataset size and available libraries."""

        if count == 0:
            return "none"

        # Prefer FAISS for large datasets when available.
        if faiss is not None:
            if count >= 20000:
                return "faiss_ivf"
            if count >= 3000:
                return "faiss"
        if hnswlib is not None and count >= 1000:
            return "hnsw"
        if AnnoyIndex is not None:
            return "annoy"
        return "sklearn"

    def _maybe_apply_pca(
        self,
        run: Run,
        features: np.ndarray,
        *,
        run_dir: Path,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Optionally reduce dimensionality before ANN and persist components."""

        original_dim = features.shape[1]
        if original_dim <= 64:
            return features, None

        if original_dim > 256:
            target_dim = 128
        elif original_dim > 128:
            target_dim = 96
        else:
            target_dim = 64

        if target_dim >= original_dim:
            return features, None

        max_components = min(target_dim, features.shape[0], original_dim)
        if max_components < 2:
            return features, None

        pca = PCA(n_components=max_components, random_state=_SETTINGS.umap_default_seed or 42)
        reduced = pca.fit_transform(features)

        run_dir.mkdir(parents=True, exist_ok=True)
        components_path = run_dir / "segment_pca.npz"
        np.savez(
            components_path,
            components=pca.components_.astype(np.float32),
            mean=pca.mean_.astype(np.float32),
            explained_variance=pca.explained_variance_ratio_.astype(np.float32),
        )

        metadata = {
            "transform_version": "pca-v1",
            "components_path": str(components_path),
            "original_dim": original_dim,
            "reduced_dim": int(max_components),
        }
        return reduced.astype(np.float32), metadata

    def _build_ann_index(
        self,
        run: Run,
        features: np.ndarray,
        segment_ids: list[UUID],
        *,
        neighbor_k: int,
    ) -> tuple[dict[UUID, list[tuple[UUID, float]]], dict[str, Any], Path]:
        count, dim = features.shape
        effective_k = max(1, min(neighbor_k, max(count - 1, 1)))

        run_dir = INDEX_DIR / str(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)

        if count == 0 or dim == 0:
            return {}, {"method": "none", "dim": dim, "k": neighbor_k}, run_dir / "segments.index"

        reduced_features, transform_meta = self._maybe_apply_pca(run, features, run_dir=run_dir)
        working = _l2_normalise(reduced_features.astype(np.float32))

        method = self._select_ann_method(count)
        neighbors_map: dict[UUID, list[tuple[UUID, float]]] = {}
        index_path = run_dir / f"segments_{method}.idx"
        params: dict[str, Any] = {
            "method": method,
            "dim": working.shape[1],
            "k": effective_k,
        }
        if transform_meta:
            params["transform"] = transform_meta

        # Persist reusable artefacts for warm-load queries
        ids_path = run_dir / "segment_ids.json"
        with ids_path.open("w", encoding="utf-8") as fh:
            json.dump([str(seg_id) for seg_id in segment_ids], fh)
        feature_store_path = run_dir / "segments_features.npy"
        np.save(feature_store_path, working.astype(np.float32))
        params["feature_store"] = feature_store_path.name
        params["segment_ids_file"] = ids_path.name

        if method == "annoy":
            if AnnoyIndex is None:
                raise RuntimeError("Annoy backend requested but annoy is not installed")
            trees = 40 if count > 2000 else 20
            if count > 10000:
                trees = 60
            index = AnnoyIndex(working.shape[1], "angular")
            for idx, vector in enumerate(working):
                index.add_item(idx, vector.astype(np.float32))
            index.build(trees)
            index.save(str(index_path))
            ids_and_dists = [index.get_nns_by_item(idx, effective_k + 1, include_distances=True) for idx in range(count)]
            params.update({"trees": trees, "metric": "angular"})
            for idx, ((ids, dists), seg_id) in enumerate(zip(ids_and_dists, segment_ids, strict=False)):
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, dist in zip(ids, dists):
                    if neighbor_idx == idx or neighbor_idx < 0 or neighbor_idx >= len(segment_ids):
                        continue
                    sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
                    entries.append((segment_ids[neighbor_idx], sim))
                    if len(entries) >= effective_k:
                        break
                neighbors_map[seg_id] = entries

        elif method == "hnsw":
            if hnswlib is None:
                raise RuntimeError("HNSW backend requested but hnswlib is not installed")
            index = hnswlib.Index(space="cosine", dim=working.shape[1])
            index.init_index(max_elements=count, ef_construction=200, M=16)
            index.add_items(working, np.arange(count))
            ef_query = min(2 * effective_k, 200)
            index.set_ef(ef_query)
            labels, distances = index.knn_query(working, k=effective_k + 1)
            index.save_index(str(index_path))
            params.update({"ef": ef_query, "M": 16})
            for query_idx, (label_row, dist_row, seg_id) in enumerate(zip(labels, distances, segment_ids, strict=False)):
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, dist in zip(label_row, dist_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= len(segment_ids):
                        continue
                    sim = max(0.0, 1.0 - float(dist))
                    entries.append((segment_ids[int(neighbor_idx)], sim))
                    if len(entries) >= effective_k:
                        break
                neighbors_map[seg_id] = entries

        elif method == "faiss_ivf":
            if faiss is None:
                raise RuntimeError("FAISS backend requested but faiss is not installed")
            dim = working.shape[1]
            nlist = min(4096, max(64, int(np.sqrt(count))))
            m = 8 if dim >= 64 else max(4, dim // 2)
            nbits = 8
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            faiss.normalize_L2(working)
            index.train(working)
            index.add(working)
            faiss.write_index(index, str(index_path))
            params.update({"metric": "cosine", "nlist": nlist, "m": m, "nbits": nbits})
            sims, indices = index.search(working, effective_k + 1)
            for query_idx, (idx_row, sim_row, seg_id) in enumerate(zip(indices, sims, segment_ids, strict=False)):
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, sim in zip(idx_row, sim_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= len(segment_ids):
                        continue
                    entries.append((segment_ids[int(neighbor_idx)], float(sim)))
                    if len(entries) >= effective_k:
                        break
                neighbors_map[seg_id] = entries

        elif method == "faiss":
            if faiss is None:
                raise RuntimeError("FAISS backend requested but faiss is not installed")
            index = faiss.IndexFlatIP(working.shape[1])
            index.add(working.astype(np.float32))
            sims, indices = index.search(working.astype(np.float32), effective_k + 1)
            faiss.write_index(index, str(index_path))
            params.update({"metric": "cosine"})
            for query_idx, (idx_row, sim_row, seg_id) in enumerate(zip(indices, sims, segment_ids, strict=False)):
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, sim in zip(idx_row, sim_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= len(segment_ids):
                        continue
                    entries.append((segment_ids[int(neighbor_idx)], float(sim)))
                    if len(entries) >= effective_k:
                        break
                neighbors_map[seg_id] = entries

        else:  # sklearn fallback
            from sklearn.neighbors import NearestNeighbors

            model = NearestNeighbors(n_neighbors=effective_k + 1, metric="cosine", algorithm="brute")
            model.fit(working)
            distances, indices = model.kneighbors(working)
            params.update({"metric": "cosine", "backend": "sklearn"})
            for query_idx, (idx_row, dist_row, seg_id) in enumerate(zip(indices, distances, segment_ids, strict=False)):
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, dist in zip(idx_row, dist_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= len(segment_ids):
                        continue
                    sim = max(0.0, 1.0 - float(dist))
                    entries.append((segment_ids[int(neighbor_idx)], sim))
                    if len(entries) >= effective_k:
                        break
                neighbors_map[seg_id] = entries
            # persist numpy arrays for reproducibility
            np.savez(run_dir / "segments_sklearn_neighbors.npz", indices=indices, distances=distances)
            index_path = run_dir / "segments_sklearn_neighbors.npz"

        # ensure all segments present; fall back to empty list if missing
        for seg_id in segment_ids:
            neighbors_map.setdefault(seg_id, [])

        return neighbors_map, params, index_path

    def _load_cached_neighbors(
        self,
        ann_record: AnnIndex,
        *,
        requested_k: int,
    ) -> dict[UUID, list[tuple[UUID, float]]]:
        params = json.loads(ann_record.params_json or "{}")
        method = params.get("method", "none")
        run_dir = Path(ann_record.index_path).resolve().parent
        ids_path = run_dir / params.get("segment_ids_file", "segment_ids.json")
        feature_path = run_dir / params.get("feature_store", "segments_features.npy")

        if not ids_path.exists():
            return {}
        try:
            stored_ids = [UUID(value) for value in json.loads(ids_path.read_text())]
        except Exception:
            return {}

        count = len(stored_ids)
        if count == 0:
            return {}

        dim = int(params.get("dim", 0))
        effective_k = max(1, min(requested_k, count - 1))

        features: np.ndarray | None = None
        if method not in {"annoy"} or feature_path.exists():
            try:
                features = np.load(feature_path)
            except Exception:
                features = None

        neighbor_map: dict[UUID, list[tuple[UUID, float]]] = {}

        if method == "annoy":
            if AnnoyIndex is None or dim == 0:
                return {}
            index = AnnoyIndex(dim, "angular")
            if not index.load(str(ann_record.index_path)):
                return {}
            for query_idx, seg_id in enumerate(stored_ids):
                ids, dists = index.get_nns_by_item(query_idx, effective_k + 1, include_distances=True)
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, dist in zip(ids, dists):
                    if neighbor_idx == query_idx or neighbor_idx < 0 or neighbor_idx >= count:
                        continue
                    sim = max(0.0, 1.0 - (dist ** 2) / 2.0)
                    entries.append((stored_ids[neighbor_idx], sim))
                    if len(entries) >= effective_k:
                        break
                neighbor_map[seg_id] = entries

        elif method == "hnsw":
            if hnswlib is None or features is None:
                return {}
            index = hnswlib.Index(space="cosine", dim=features.shape[1])
            index.load_index(str(ann_record.index_path))
            ef = min(400, max(2 * effective_k, 50))
            index.set_ef(ef)
            actual_k = min(effective_k + 1, count)
            labels, distances = index.knn_query(features, k=actual_k)
            for query_idx, (label_row, dist_row) in enumerate(zip(labels, distances, strict=False)):
                seg_id = stored_ids[query_idx]
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, dist in zip(label_row, dist_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= count:
                        continue
                    sim = max(0.0, 1.0 - float(dist))
                    entries.append((stored_ids[int(neighbor_idx)], sim))
                    if len(entries) >= effective_k:
                        break
                neighbor_map[seg_id] = entries

        elif method in {"faiss", "faiss_ivf"}:
            if faiss is None or features is None:
                return {}
            try:
                index = faiss.read_index(str(ann_record.index_path))
            except Exception:
                return {}
            if method == "faiss_ivf":
                nprobe = min(int(params.get("nlist", 8)), 32)
                try:
                    faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
                except Exception:
                    index.nprobe = nprobe
            actual_k = min(effective_k + 1, count)
            sims, indices = index.search(features.astype(np.float32), actual_k)
            for query_idx, (idx_row, sim_row) in enumerate(zip(indices, sims, strict=False)):
                seg_id = stored_ids[query_idx]
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx, sim in zip(idx_row, sim_row):
                    if int(neighbor_idx) == query_idx or int(neighbor_idx) < 0 or int(neighbor_idx) >= count:
                        continue
                    entries.append((stored_ids[int(neighbor_idx)], float(sim)))
                    if len(entries) >= effective_k:
                        break
                neighbor_map[seg_id] = entries

        elif method == "sklearn":
            actual_k = min(effective_k + 1, count)
            if features is None:
                return {}
            sims = features @ features.T
            for query_idx, seg_id in enumerate(stored_ids):
                order = np.argsort(-sims[query_idx])
                entries: list[tuple[UUID, float]] = []
                for neighbor_idx in order:
                    if neighbor_idx == query_idx:
                        continue
                    score = float(sims[query_idx, neighbor_idx])
                    entries.append((stored_ids[neighbor_idx], score))
                    if len(entries) >= effective_k:
                        break
                neighbor_map[seg_id] = entries

        else:
            return {}

        for seg_id in stored_ids:
            neighbor_map.setdefault(seg_id, [])

        return neighbor_map
    def _extract_segment_top_terms(
        self,
        segments: Sequence[SegmentDraft],
        *,
        top_k: int = 6,
    ) -> dict[UUID, list[dict[str, float]]]:
        texts = [seg.text for seg in segments]
        if not texts:
            return {}
        vectorizer = TfidfVectorizer(
            max_features=256,
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        result: dict[UUID, list[dict[str, float]]] = {}
        for idx, seg in enumerate(segments):
            row = matrix.getrow(idx)
            if row.nnz == 0:
                result[seg.id] = []
                continue
            data = row.data
            indices = row.indices
            order = np.argsort(data)[::-1][:top_k]
            terms = []
            for order_idx in order:
                term = feature_names[indices[order_idx]]
                weight = float(data[order_idx])
                terms.append({"term": term, "weight": weight})
            result[seg.id] = terms
        return result

    def _select_cluster_exemplars(
        self,
        segments: Sequence[SegmentDraft],
    ) -> dict[int, UUID]:
        best: dict[int, tuple[float, float, UUID]] = {}
        for seg in segments:
            label = seg.cluster_label
            if label is None or label < 0:
                continue
            probability = float(seg.cluster_probability or 0.0)
            similarity = float(seg.cluster_similarity or 0.0)
            record = best.get(label)
            if record is None or (probability, similarity) > record[:2]:
                best[label] = (probability, similarity, seg.id)
        return {label: record[2] for label, record in best.items()}

    async def _persist_segment_insights(
        self,
        session,
        segments: Sequence[SegmentDraft],
        neighbors_map: dict[UUID, list[tuple[UUID, float]]],
        top_terms_map: dict[UUID, list[dict[str, float]]],
        exemplar_map: dict[int, UUID],
        metrics_map: dict[UUID, dict[str, Any]],
    ) -> None:
        segment_ids = [seg.id for seg in segments]
        if not segment_ids:
            return

        await session.execute(
            delete(SegmentInsight).where(SegmentInsight.segment_id.in_(segment_ids))
        )

        records: list[SegmentInsight] = []
        for segment in segments:
            neighbors = [
                {"segment_id": str(neighbor_id), "similarity": sim}
                for neighbor_id, sim in neighbors_map.get(segment.id, [])
            ]
            metrics = metrics_map.get(segment.id, {})
            top_terms = top_terms_map.get(segment.id, [])

            cluster_label = segment.cluster_label if segment.cluster_label is not None else -1
            exemplar_id = None
            if cluster_label is not None and cluster_label >= 0:
                exemplar_id = exemplar_map.get(cluster_label)

            records.append(
                SegmentInsight(
                    segment_id=segment.id,
                    top_terms_json=json.dumps(top_terms),
                    neighbors_json=json.dumps(neighbors),
                    cluster_exemplar_id=exemplar_id,
                    metrics_json=json.dumps(metrics),
                )
            )

        if records:
            session.add_all(records)
        await session.commit()

    async def _persist_ann_metadata(
        self,
        session,
        run_id: UUID,
        params: dict[str, Any],
        path: Path,
        vector_count: int,
    ) -> None:
        await session.execute(delete(AnnIndex).where(AnnIndex.run_id == run_id))
        session.add(
            AnnIndex(
                run_id=run_id,
                method=params.get("method", "annoy"),
                params_json=json.dumps(params),
                vector_count=vector_count,
                index_path=str(path),
            )
        )
        await session.commit()

    def _build_segment_edges(
        self,
        run_id: UUID,
        neighbors_map: dict[UUID, list[tuple[UUID, float]]],
        *,
        k: int,
        threshold: float = 0.0,
    ) -> list[SegmentEdge]:
        edges: dict[tuple[UUID, UUID], float] = {}
        for source_id, neighbors in neighbors_map.items():
            count = 0
            for neighbor_id, similarity in neighbors:
                if similarity < threshold or source_id == neighbor_id:
                    continue
                key = tuple(sorted((source_id, neighbor_id)))
                if key in edges:
                    edges[key] = max(edges[key], similarity)
                else:
                    edges[key] = similarity
                count += 1
                if count >= k:
                    break
        return [
            SegmentEdge(run_id=run_id, source_id=pair[0], target_id=pair[1], score=float(score))
            for pair, score in edges.items()
        ]

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
        chunk_size = max(2, chunk_size)

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
        preproc_version = (
            payload.preproc_version or _SETTINGS.embedding_preproc_version
        )

        n_neighbors, min_dist, metric, umap_seed, seed_source = self._resolve_umap_params(payload)

        cluster_algo, min_cluster_size, min_samples = self._resolve_cluster_config(payload)

        run = Run(
            prompt=payload.prompt,
            n=payload.n,
            model=payload.model,
            temperature=payload.temperature,
            top_p=payload.top_p,
            seed=payload.seed,
            max_tokens=payload.max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            system_prompt=system_prompt,
            embedding_model=embedding_model,
            use_cache=payload.use_cache,
            preproc_version=preproc_version,
            umap_n_neighbors=n_neighbors,
            umap_min_dist=min_dist,
            umap_metric=metric,
            umap_seed=umap_seed,
            random_state_seed_source=seed_source,
            cluster_algo=cluster_algo,
            hdbscan_min_cluster_size=min_cluster_size if cluster_algo == "hdbscan" else None,
            hdbscan_min_samples=min_samples if cluster_algo == "hdbscan" else None,
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
        self,
        session,
        *,
        run_id: UUID,
        payload: RunUpdateRequest,
    ) -> Run:
        run = await session.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")

        run.notes = payload.notes

        session.add(run)
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

        if not getattr(run, "preproc_version", None):
            run.preproc_version = _SETTINGS.embedding_preproc_version

        if getattr(run, "use_cache", None) is None:
            run.use_cache = True

        if not run.chunk_size or run.chunk_size < 2:
            run.chunk_size = max(2, _SETTINGS.segment_word_window)

        if run.chunk_overlap is None:
            run.chunk_overlap = _SETTINGS.segment_word_overlap

        if run.chunk_overlap is not None and run.chunk_size:
            max_overlap = max(run.chunk_size - 1, 0)
            run.chunk_overlap = min(run.chunk_overlap, max_overlap)

        if getattr(run, "umap_n_neighbors", None) is None:
            run.umap_n_neighbors = _SETTINGS.umap_default_n_neighbors

        if getattr(run, "umap_min_dist", None) is None:
            run.umap_min_dist = _SETTINGS.umap_default_min_dist

        if not getattr(run, "umap_metric", None):
            run.umap_metric = _SETTINGS.umap_default_metric

        if getattr(run, "umap_seed", None) is None:
            run.umap_seed = _SETTINGS.umap_default_seed
            if run.random_state_seed_source not in {"ui", "env"}:
                run.random_state_seed_source = "default"

        if not getattr(run, "cluster_algo", None):
            run.cluster_algo = _SETTINGS.cluster_default_algo
        if run.cluster_algo == "hdbscan":
            if getattr(run, "hdbscan_min_cluster_size", None) is None:
                run.hdbscan_min_cluster_size = _SETTINGS.hdbscan_default_min_cluster_size
            if getattr(run, "hdbscan_min_samples", None) is None:
                run.hdbscan_min_samples = _SETTINGS.hdbscan_default_min_samples

        options = sample_request or SampleRequest()

        telemetry = RunStageTimer()
        current_stage = "initializing"

        try:
            async with telemetry.track("prepare-run"):
                if options.overwrite_previous or options.force_refresh:
                    await self._clear_existing(session, run_id)

                run.status = RunStatus.PENDING
                run.error_message = None
                run.updated_at = datetime.utcnow()
                await session.commit()

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

            async with telemetry.track("request-completions"):
                chat_results = await self._openai.sample_chat(
                    prompt=run.prompt,
                    n=run.n,
                    model=run.model,
                    temperature=run.temperature,
                    top_p=run.top_p,
                    seed=run.seed,
                    jitter_token=options.jitter_prompt_token,
                    max_tokens=run.max_tokens,
                )
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

        segment_drafts: list[SegmentDraft] = []
        total_segments = 0
        segment_edge_count = 0

        try:
            async with telemetry.track("persist-responses"):
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
                session.add_all(responses)
                await session.commit()
                for response in responses:
                    await session.refresh(response)

                response_ids = [response.id for response in responses]
                prompt_embedding = await self._embed_prompt(run.prompt)

            current_stage = "responses-received"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message=f"Received {len(responses)} responses from {run.model}",
                percent=0.18,
                metadata={"responses": len(responses)},
            )

    
            if options.include_segments:
                current_stage = "segmenting-responses"

                await self._update_progress(
                    session,
                    run,
                    stage=current_stage,
                    message="Segmenting responses into discourse windows",
                    percent=0.26,
                    metadata={
                        "responses": len(responses),
                        "chunk_size": run.chunk_size,
                        "chunk_overlap": run.chunk_overlap,
                    },
                )

                segment_drafts: list[SegmentDraft] = []
                responses_total = max(1, len(responses))
                segment_progress_span = 0.04
                last_ratio = 0.0

                async with telemetry.track("segment-responses"):
                    for index, response in enumerate(responses, start=1):
                        drafts = make_segment_drafts(
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
                        segment_drafts.extend(drafts)

                        ratio = index / responses_total
                        if ratio - last_ratio >= 0.05 or index == responses_total:
                            segment_percent = 0.26 + segment_progress_span * ratio
                            await self._update_progress(
                                session,
                                run,
                                stage=current_stage,
                                message=f"Segmenting responses ({index}/{responses_total})",
                                percent=segment_percent,
                                metadata={
                                    "responses_processed": index,
                                    "responses_total": responses_total,
                                    "segments_accumulated": len(segment_drafts),
                                },
                            )
                            last_ratio = ratio

                segment_drafts = self._ensure_minimum_segments(responses, segment_drafts)
                total_segments = len(segment_drafts)

                await self._update_progress(
                    session,
                    run,
                    stage=current_stage,
                    message=f"Segmented {total_segments} discourse windows",
                    percent=0.30,
                    metadata={
                        "responses_total": len(responses),
                        "segments": total_segments,
                    },
                )

                if options.include_discourse_tags and self._openai.is_configured and segment_drafts:
                    current_stage = "tagging-discourse"

                    await self._update_progress(
                        session,
                        run,
                        stage=current_stage,
                        message="Requesting discourse role annotations",
                        percent=0.32,
                        metadata={"segments": total_segments},
                    )

                    tag_batch_size = max(1, min(50, total_segments))
                    processed_segments = 0
                    batch_count = (total_segments + tag_batch_size - 1) // tag_batch_size
                    tag_progress_span = 0.05

                    async with telemetry.track("discourse-tagging"):
                        for batch_index, start_idx in enumerate(range(0, total_segments, tag_batch_size)):
                            end_idx = min(start_idx + tag_batch_size, total_segments)
                            batch = segment_drafts[start_idx:end_idx]
                            roles = await self._openai.discourse_tag_segments([seg.text for seg in batch])
                            for seg, role in zip(batch, roles, strict=False):
                                if role:
                                    seg.role = role

                            processed_segments = end_idx
                            ratio = processed_segments / total_segments
                            tag_percent = min(0.37, 0.32 + tag_progress_span * ratio)
                            await self._update_progress(
                                session,
                                run,
                                stage=current_stage,
                                message=f"Tagging discourse roles ({processed_segments}/{total_segments})",
                                percent=tag_percent,
                                metadata={
                                    "segments_tagged": processed_segments,
                                    "segments_total": total_segments,
                                    "batch_index": batch_index + 1,
                                    "batch_count": batch_count,
                                },
                            )

            else:
                segment_drafts = []
                total_segments = 0

            current_stage = "embedding-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Embedding responses",
                percent=0.38,
                metadata={
                    "responses": len(responses),
                    "embedding_model": run.embedding_model,
                },
            )

            async with telemetry.track("embed-responses"):
                response_embeddings, _ = await self._embed_responses(session, run, responses)
            segment_embeddings: EmbeddingBatch | None = None

            if segment_drafts:
                current_stage = "embedding-segments"

                await self._update_progress(
                    session,
                    run,
                    stage=current_stage,
                    message="Embedding discourse segments",
                    percent=0.46,
                    metadata={
                        "segments": total_segments,
                        "chunk_size": run.chunk_size,
                        "chunk_overlap": run.chunk_overlap,
                    },
                )
                async with telemetry.track("embed-segments"):
                    segment_embeddings, segment_records = await self._embed_segments(
                        session,
                        run,
                        segment_drafts,
                    )
                    for seg, record in zip(segment_drafts, segment_records, strict=False):
                        seg.embedding = record["vector_list"]
                        seg.text_hash = record["hash"]
                        seg.is_cached = record["is_cached"]
                        seg.is_duplicate = record["is_duplicate"]
                        seg.simhash64 = record["simhash64"]

                    self._apply_prompt_similarity(segment_drafts, prompt_embedding)

            current_stage = "analysing-segments"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Analysing segment manifold",
                percent=0.58,
                metadata={"segments": total_segments},
            )

            async with telemetry.track("segment-analysis"):
                segment_feature_matrix = build_feature_matrix(
                    [seg.text for seg in segment_drafts],
                    segment_embeddings.vectors,
                    prompt_embedding=prompt_embedding,
                    keyword_axes=_SETTINGS.segment_keyword_axes,
                )
    
                normalized_segment_features = _l2_normalise(
                    np.asarray(segment_feature_matrix, dtype=float)
                )
                segment_ids = [seg.id for seg in segment_drafts]
                base_neighbor_k = min(
                    max(10, run.umap_n_neighbors),
                    max(2, len(segment_drafts) - 1),
                )
    
                if normalized_segment_features.size and len(segment_drafts) >= 2:
                    neighbors_map, ann_params, ann_path = self._build_ann_index(
                        run,
                        normalized_segment_features,
                        segment_ids,
                        neighbor_k=base_neighbor_k,
                    )
                    await self._persist_ann_metadata(
                        session,
                        run.id,
                        ann_params,
                        ann_path,
                        vector_count=len(segment_drafts),
                    )
                else:
                    neighbors_map = {}
                    await session.execute(delete(AnnIndex).where(AnnIndex.run_id == run.id))
                    await session.commit()
    
                segment_projection = compute_umap(
                    segment_feature_matrix,
                    random_state=run.umap_seed or _SETTINGS.umap_default_seed,
                    n_neighbors=min(base_neighbor_k, max(2, len(segment_drafts) - 1)),
                    min_dist=run.umap_min_dist,
                    metric=run.umap_metric,
                )
    
                min_cluster, min_samples = self._suggest_cluster_params(len(segment_drafts))
                segment_metric = self._resolve_cluster_metric(getattr(run, "umap_metric", None))
                segment_clustering = cluster_with_fallback(
                    segment_feature_matrix,
                    coords_3d=segment_projection.coords_3d,
                    similarity_basis=segment_embeddings.vectors,
                    min_cluster_size=min_cluster,
                    min_samples=min_samples,
                    metric=segment_metric,
                )
                self._apply_segment_analysis(segment_drafts, segment_projection, segment_clustering)
    
                top_terms_map = self._extract_segment_top_terms(segment_drafts)
                exemplar_map = self._select_cluster_exemplars(segment_drafts)
    
                feature_lookup = {
                    segment_ids[idx]: normalized_segment_features[idx]
                    for idx in range(len(segment_ids))
                }
    
                metrics_map: dict[UUID, dict[str, Any]] = {}
                for seg in segment_drafts:
                    neighbors = neighbors_map.get(seg.id, [])
                    sim_to_nn = neighbors[0][1] if neighbors else None
                    sim_to_exemplar = None
                    if seg.cluster_label is not None and seg.cluster_label >= 0:
                        exemplar_id = exemplar_map.get(seg.cluster_label)
                        if (
                            exemplar_id
                            and exemplar_id in feature_lookup
                            and seg.id in feature_lookup
                        ):
                            sim_to_exemplar = float(
                                np.clip(
                                    feature_lookup[seg.id] @ feature_lookup[exemplar_id],
                                    -1.0,
                                    1.0,
                                )
                            )
                    metrics_map[seg.id] = {
                        "sim_to_exemplar": sim_to_exemplar,
                        "sim_to_nn": sim_to_nn,
                    }
    
                default_edge_k = min(15, max(2, len(segment_drafts) - 1))
                segment_edges = self._build_segment_edges(
                    run.id,
                    neighbors_map,
                    k=default_edge_k,
                    threshold=_SETTINGS.segment_similarity_threshold,
                )
                segment_edge_count = len(segment_edges)
                await self._persist_segment_edges(session, run.id, segment_edges)
                response_hulls = self._build_response_hulls(segment_drafts)
                await self._persist_response_hulls(session, response_ids, response_hulls)
                await self._persist_segments(session, response_ids, segment_drafts)
                await self._persist_segment_insights(
                    session,
                    segment_drafts,
                    neighbors_map,
                    top_terms_map,
                    exemplar_map,
                    metrics_map,
                )
                if not segment_drafts:
                    await self._persist_segment_edges(session, run.id, [])
                    await self._persist_response_hulls(session, response_ids, [])
                    await self._persist_segments(session, response_ids, [])
                    await session.execute(
                        delete(SegmentInsight).where(
                            SegmentInsight.segment_id.in_([seg.id for seg in segment_drafts])
                        )
                    )
                    await session.commit()
                    await session.execute(delete(AnnIndex).where(AnnIndex.run_id == run.id))
                    await session.commit()
        
            if segment_drafts and segment_embeddings is not None:
                aggregated = self._aggregate_response_embeddings(responses, segment_drafts, response_embeddings)
                if aggregated:
                    vectors = [vec.tolist() for vec in aggregated]
                    dim = len(vectors[0]) if vectors else segment_embeddings.dim
                    response_embeddings = EmbeddingBatch(vectors=vectors, model=segment_embeddings.model, dim=dim)
    
            current_stage = "analysing-responses"

            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Building response manifold",
                percent=0.7,
                metadata={"responses": len(responses)},
            )

            async with telemetry.track("response-analysis"):
                texts = [response.raw_text or "" for response in responses]
                feature_matrix = build_feature_matrix(
                    texts,
                    response_embeddings.vectors,
                    prompt_embedding=prompt_embedding,
                    keyword_axes=_SETTINGS.segment_keyword_axes,
                )
        
                effective_neighbors = min(
                    max(5, run.umap_n_neighbors),
                    max(2, feature_matrix.shape[0] - 1),
                )
                projection_result = compute_umap(
                    feature_matrix,
                    random_state=run.umap_seed or _SETTINGS.umap_default_seed,
                    n_neighbors=effective_neighbors,
                    min_dist=run.umap_min_dist,
                    metric=run.umap_metric,
                )
                if projection_result.trustworthiness_2d is None:
                    projection_result.trustworthiness_2d = 1.0
                if projection_result.trustworthiness_3d is None:
                    projection_result.trustworthiness_3d = 1.0
                if projection_result.continuity_2d is None:
                    projection_result.continuity_2d = 1.0
                if projection_result.continuity_3d is None:
                    projection_result.continuity_3d = 1.0
                base_params = self._base_projection_params(run, "umap")
                resolved_params = self._resolve_projection_params(
                    "umap",
                    base_params,
                    sample_count=feature_matrix.shape[0],
                    feature_dim=feature_matrix.shape[1] if feature_matrix.ndim == 2 else 0,
                )
                umap_cache_payload = ProjectionCachePayload(
                    method="umap",
                    requested_params=base_params,
                    resolved_params=resolved_params,
                    feature_version=FEATURE_VERSION,
                    coords_2d=np.asarray(projection_result.coords_2d, dtype=np.float32),
                    coords_3d=np.asarray(projection_result.coords_3d, dtype=np.float32),
                    response_ids=[response.id for response in responses],
                    warnings=[],
                    total_count=feature_matrix.shape[0],
                    point_count=feature_matrix.shape[0],
                    is_subsample=False,
                    subsample_strategy=None,
                    trustworthiness_2d=projection_result.trustworthiness_2d,
                    trustworthiness_3d=projection_result.trustworthiness_3d,
                    continuity_2d=projection_result.continuity_2d,
                    continuity_3d=projection_result.continuity_3d,
                )
                umap_params_hash = self._hash_projection_params("umap", base_params)
                await self._persist_projection_cache(
                    session,
                    run.id,
                    umap_cache_payload,
                    umap_params_hash,
                )
        
                min_cluster, min_samples = self._effective_cluster_params(
                    run,
                    feature_matrix.shape[0],
                )
                if run.cluster_algo == "hdbscan":
                    run.hdbscan_min_cluster_size = min_cluster
                    run.hdbscan_min_samples = min_samples
    
                cluster_algo = getattr(run, "cluster_algo", _SETTINGS.cluster_default_algo)
                cluster_metric = self._resolve_cluster_metric(getattr(run, "umap_metric", None))
                clustering_result = cluster_with_fallback(
                    feature_matrix,
                    coords_3d=projection_result.coords_3d,
                    similarity_basis=response_embeddings.vectors,
                    min_cluster_size=min_cluster,
                    min_samples=min_samples,
                    metric=cluster_metric,
                    algo=cluster_algo,
                )
    
                similarity_basis = (
                    response_embeddings.vectors if response_embeddings.vectors else None
                )
                metrics_result = compute_cluster_metrics(
                    feature_matrix,
                    projection_result.coords_3d,
                    projection_result.coords_2d,
                    clustering_result,
                    algo=cluster_algo,
                    min_cluster_size=min_cluster if cluster_algo == "hdbscan" else None,
                    min_samples=min_samples if cluster_algo == "hdbscan" else None,
                    metric=cluster_metric,
                    similarity_basis=similarity_basis,
                )
    
                run.trustworthiness_2d = projection_result.trustworthiness_2d
                run.trustworthiness_3d = projection_result.trustworthiness_3d
                run.continuity_2d = projection_result.continuity_2d
                run.continuity_3d = projection_result.continuity_3d
        
                cluster_labels = np.asarray(
                    getattr(clustering_result, "labels", np.array([])), dtype=int
                )
                cluster_count = int(
                    len({int(label) for label in cluster_labels if int(label) >= 0})
                )
        
                umap_params_payload = {
                    "n_neighbors": run.umap_n_neighbors,
                    "effective_n_neighbors": effective_neighbors,
                    "min_dist": run.umap_min_dist,
                    "metric": run.umap_metric,
                    "seed": run.umap_seed,
                    "seed_source": run.random_state_seed_source,
                }
        
                cluster_params_payload = {
                    "method": clustering_result.method,
                    "algo": cluster_algo,
                    "metric": cluster_metric,
                    "min_cluster_size": min_cluster,
                    "min_samples": min_samples,
                    "clusters": cluster_count,
                }
        
            async with telemetry.track("persist-artifacts"):
                await self._persist_embeddings(session, responses, response_embeddings)
                await self._persist_projection(session, responses, projection_result)
                await self._persist_clusters(session, responses, clustering_result)
                await self._persist_cluster_metrics(session, run.id, metrics_result)
    
                embedding_dim = response_embeddings.dim or (
                    len(response_embeddings.vectors[0])
                    if response_embeddings.vectors
                    else 0
                )
        
                provenance_payload = self._collect_provenance(
                    run,
                    embedding_dim=embedding_dim,
                    umap_params=umap_params_payload,
                    cluster_params=cluster_params_payload,
                )
                await self._upsert_provenance(session, run, provenance_payload)
        
            current_stage = "persisting-artifacts"
    
            await self._update_progress(
                session,
                run,
                stage=current_stage,
                message="Persisting embeddings, projections, and clusters",
                percent=0.85,
                metadata={
                    "responses": len(responses),
                    "segments": total_segments,
                    "chunk_size": run.chunk_size,
                    "chunk_overlap": run.chunk_overlap,
                },
            )
    
            run.status = RunStatus.COMPLETED
            run.error_message = None
            await self._update_progress(
                session,
                run,
                stage="completed",
                message="Run completed successfully",
                percent=1.0,
                metadata={
                    "responses": len(responses),
                    "segments": total_segments,
                    "clusters": cluster_count,
                    "segment_edges": segment_edge_count,
                    "trustworthiness_2d": run.trustworthiness_2d,
                    "continuity_2d": run.continuity_2d,
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
        finally:
            snapshot = telemetry.snapshot()
            run.processing_time_ms = snapshot.get("total_duration_ms")
            run.timings_json = json.dumps(snapshot)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                _LOGGER.warning("Failed to persist timing telemetry for run %s", run.id, exc_info=True)

    async def build_segment_graph(
        self,
        session,
        run: Run,
        *,
        mode: str,
        neighbor_k: int,
        threshold: float,
        limit: int = 50_000,
    ) -> dict[str, Any]:
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = 0.0
        threshold_value = max(0.0, min(1.0, threshold_value))

        segments_result = await session.exec(
            select(ResponseSegment)
            .join(Response, ResponseSegment.response_id == Response.id)
            .where(Response.run_id == run.id)
        )
        segments = segments_result.scalars().all()
        node_count = len(segments)
        if node_count <= 1:
            return {
                "mode": "simplified" if mode == "simplified" else "full",
                "edges": [],
                "auto_simplified": False,
                "k": 0,
                "threshold": threshold_value,
                "node_count": node_count,
            }

        actual_k = max(1, min(int(neighbor_k), node_count - 1))

        ann_result = await session.exec(select(AnnIndex).where(AnnIndex.run_id == run.id))
        ann_record = ann_result.scalar_one_or_none()

        neighbors_map: dict[UUID, list[tuple[UUID, float]]] = {}
        if ann_record is not None:
            try:
                neighbors_map = self._load_cached_neighbors(ann_record, requested_k=actual_k)
            except Exception as exc:  # pragma: no cover - diagnostics only
                _LOGGER.warning("Failed to load ANN index for run %s: %s", run.id, exc)

        edge_rows = await session.exec(select(SegmentEdge).where(SegmentEdge.run_id == run.id))
        stored_edge_map = {tuple(sorted((edge.source_id, edge.target_id))): float(edge.score) for edge in edge_rows.scalars() if edge.source_id and edge.target_id}

        if not neighbors_map and stored_edge_map:
            adjacency: dict[UUID, list[tuple[UUID, float]]] = defaultdict(list)
            for (lhs, rhs), score in stored_edge_map.items():
                adjacency[lhs].append((rhs, score))
                adjacency[rhs].append((lhs, score))
            for key, entries in adjacency.items():
                entries.sort(key=lambda item: item[1], reverse=True)
                adjacency[key] = entries[:actual_k]
            neighbors_map = dict(adjacency)

        base_edges = self._build_edge_map_from_neighbors(
            neighbors_map,
            threshold=threshold_value,
            limit_per_node=actual_k,
        ) if neighbors_map else {}

        if not base_edges and stored_edge_map:
            base_edges = stored_edge_map.copy()

        cluster_nodes: dict[int, list[UUID]] = defaultdict(list)
        for segment in segments:
            if segment.cluster_label is not None and segment.cluster_label >= 0:
                cluster_nodes[int(segment.cluster_label)].append(segment.id)

        resolved_mode = "simplified" if mode == "simplified" else "full"

        if resolved_mode == "simplified" and base_edges:
            mutual_edges = self._build_mutual_edge_map(neighbors_map, threshold=threshold_value) if neighbors_map else {}
            mst_edges = self._cluster_mst_edges(cluster_nodes, base_edges)
            final_edges = dict(mst_edges)
            for pair, score in mutual_edges.items():
                existing = final_edges.get(pair)
                if existing is None or score > existing:
                    final_edges[pair] = score
            for pair, score in sorted(base_edges.items(), key=lambda item: item[1], reverse=True):
                if pair not in final_edges:
                    final_edges[pair] = score
            self._ensure_minimum_connections(segments, neighbors_map, final_edges, threshold_value)
        else:
            final_edges = base_edges.copy()

        if not final_edges and stored_edge_map:
            final_edges = stored_edge_map.copy()

        edge_items = sorted(final_edges.items(), key=lambda item: item[1], reverse=True)
        auto_simplified = False
        if limit > 0 and len(edge_items) > limit:
            auto_simplified = True
            edge_items = edge_items[:limit]

        edges_payload = [(pair[0], pair[1], float(score)) for pair, score in edge_items]

        return {
            "mode": resolved_mode,
            "edges": edges_payload,
            "auto_simplified": auto_simplified,
            "k": actual_k,
            "threshold": threshold_value,
            "node_count": node_count,
        }


    async def _embed_prompt(self, prompt: str | None) -> list[float] | None:
        if not prompt:
            return None
        try:
            batch = await self._openai.embed_texts([prompt])
        except Exception:
            return None
        return batch.vectors[0] if batch.vectors else None

    async def _embed_responses(
        self,
        session,
        run: Run,
        responses: Sequence[Response],
    ) -> tuple[EmbeddingBatch, list[dict[str, Any]]]:
        texts = [response.raw_text or "" for response in responses]
        batch, records = await self._embed_texts_with_cache(
            session,
            texts,
            model_id=run.embedding_model,
            preproc_version=run.preproc_version,
            use_cache=run.use_cache,
            mark_duplicates=False,
        )
        return batch, records

    async def _embed_segments(
        self,
        session,
        run: Run,
        segments: Sequence[SegmentDraft],
    ) -> tuple[EmbeddingBatch, list[dict[str, Any]]]:
        texts = [segment.text for segment in segments]
        batch, records = await self._embed_texts_with_cache(
            session,
            texts,
            model_id=run.embedding_model,
            preproc_version=run.preproc_version,
            use_cache=run.use_cache,
            mark_duplicates=True,
        )
        return batch, records

    async def recompute_clusters(
        self,
        session,
        *,
        run_id: UUID,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        algo: str | None = None,
    ) -> Run:
        run = await session.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")

        if algo:
            algo_clean = algo.strip().lower()
            if algo_clean not in {"hdbscan", "kmeans"}:
                raise ValueError("algo must be 'hdbscan' or 'kmeans'")
            run.cluster_algo = algo_clean

        cluster_algo = getattr(run, "cluster_algo", _SETTINGS.cluster_default_algo)

        if cluster_algo == "hdbscan":
            if min_cluster_size is not None:
                run.hdbscan_min_cluster_size = max(2, int(min_cluster_size))
            if min_samples is not None:
                run.hdbscan_min_samples = max(1, int(min_samples))

        responses_result = await session.exec(
            select(Response)
            .where(Response.run_id == run_id)
            .order_by(Response.index)
        )
        responses = responses_result.scalars().all()
        if not responses:
            raise ValueError("Run has no responses")

        response_ids = [response.id for response in responses]

        embeddings_result = await session.exec(
            select(Embedding).where(Embedding.response_id.in_(response_ids))
        )
        embeddings_map = {
            embedding.response_id: embedding
            for embedding in embeddings_result.scalars().all()
        }
        missing_embeddings = [resp.id for resp in responses if resp.id not in embeddings_map]
        if missing_embeddings:
            raise ValueError("Embeddings missing for responses")

        vectors: list[np.ndarray] = []
        for response in responses:
            record = embeddings_map[response.id]
            array = np.frombuffer(record.vector, dtype=np.float32)
            if record.dim and array.size != record.dim:
                array = array[: record.dim]
            vectors.append(array.astype(np.float32, copy=False))

        texts = [response.raw_text or "" for response in responses]
        feature_matrix = build_feature_matrix(
            texts,
            vectors,
            prompt_embedding=None,
            keyword_axes=_SETTINGS.segment_keyword_axes,
        )

        projection_result = await session.exec(
            select(Projection).where(Projection.response_id.in_(response_ids))
        )
        coords3_map: dict[UUID, tuple[float, float, float]] = {}
        coords2_map: dict[UUID, tuple[float, float]] = {}
        for projection in projection_result.scalars().all():
            if projection.dim == 3:
                coords3_map[projection.response_id] = (
                    float(projection.x),
                    float(projection.y),
                    float(projection.z or 0.0),
                )
            elif projection.dim == 2:
                coords2_map[projection.response_id] = (
                    float(projection.x),
                    float(projection.y),
                )
        coords_3d = np.array(
            [coords3_map.get(response.id, (0.0, 0.0, 0.0)) for response in responses],
            dtype=float,
        )
        coords_2d = np.array(
            [coords2_map.get(response.id, (0.0, 0.0)) for response in responses],
            dtype=float,
        )

        min_cluster, min_samples_eff = self._effective_cluster_params(
            run,
            feature_matrix.shape[0],
        )
        if cluster_algo == "hdbscan":
            run.hdbscan_min_cluster_size = min_cluster
            run.hdbscan_min_samples = min_samples_eff

        cluster_metric = self._resolve_cluster_metric(getattr(run, "umap_metric", None))
        clustering_result = cluster_with_fallback(
            feature_matrix,
            coords_3d=coords_3d,
            similarity_basis=vectors,
            min_cluster_size=min_cluster,
            min_samples=min_samples_eff,
            metric=cluster_metric,
            algo=cluster_algo,
        )

        metrics_result = compute_cluster_metrics(
            feature_matrix,
            coords_3d,
            coords_2d,
            clustering_result,
            algo=cluster_algo,
            min_cluster_size=min_cluster if cluster_algo == "hdbscan" else None,
            min_samples=min_samples_eff if cluster_algo == "hdbscan" else None,
            metric=cluster_metric,
            similarity_basis=vectors,
        )

        await self._persist_clusters(session, responses, clustering_result)
        await self._persist_cluster_metrics(session, run.id, metrics_result)

        run.updated_at = datetime.utcnow()
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run

    async def compute_run_metrics(
        self,
        session,
        run_id: UUID,
    ) -> RunMetrics:
        run = await session.get(Run, run_id)
        if run is None:
            raise ValueError("Run not found")

        base_stmt = (
            select(func.count())
            .select_from(ResponseSegment)
            .join(Response, ResponseSegment.response_id == Response.id)
            .where(Response.run_id == run_id)
        )

        total_segments = int((await session.exec(base_stmt)).scalar_one())
        cached_stmt = base_stmt.where(ResponseSegment.is_cached.is_(True))
        cached_segments = int((await session.exec(cached_stmt)).scalar_one())
        duplicate_stmt = base_stmt.where(ResponseSegment.is_duplicate.is_(True))
        duplicate_segments = int((await session.exec(duplicate_stmt)).scalar_one())

        hit_rate = (cached_segments / total_segments * 100.0) if total_segments else 0.0

        metrics_row = await session.exec(
            select(ClusterMetrics).where(ClusterMetrics.run_id == run_id)
        )
        cluster_metrics = metrics_row.scalar_one_or_none()

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

        def _safe_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number

        return RunMetrics(
            run_id=run_id,
            total_segments=total_segments,
            cached_segments=cached_segments,
            duplicate_segments=duplicate_segments,
            cache_hit_rate=float(hit_rate),
            processing_time_ms=processing_time_ms,
            stage_timings=stage_timings,
            silhouette_embed=_safe_float(getattr(cluster_metrics, "silhouette_embed", None)),
            silhouette_feature=_safe_float(getattr(cluster_metrics, "silhouette_feature", None)),
            davies_bouldin=_safe_float(getattr(cluster_metrics, "davies_bouldin", None)),
            calinski_harabasz=_safe_float(getattr(cluster_metrics, "calinski_harabasz", None)),
            n_clusters=getattr(cluster_metrics, "n_clusters", None),
            n_noise=getattr(cluster_metrics, "n_noise", None),
        )


    async def backfill_embedding_cache(
        self,
        session,
        *,
        limit: int = 200,
    ) -> int:
        if limit <= 0:
            return 0
        if not self._openai.is_configured:
            return 0

        stmt = (
            select(
                ResponseSegment.text,
                ResponseSegment.text_hash,
                Run.embedding_model,
                Run.preproc_version,
            )
            .join(Response, ResponseSegment.response_id == Response.id)
            .join(Run, Response.run_id == Run.id)
            .where(ResponseSegment.text_hash.isnot(None))
        )

        result = await session.exec(stmt.limit(limit * 4))
        rows = result.all()
        combo_map: dict[tuple[str, str, str], str] = {}
        for text, text_hash, model_id, preproc_version in rows:
            if not text_hash or not model_id:
                continue
            key = (text_hash, model_id, preproc_version or _SETTINGS.embedding_preproc_version)
            if key not in combo_map:
                combo_map[key] = text or ""

        if not combo_map:
            return 0

        processed = 0
        grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for (text_hash, model_id, preproc_version), text in combo_map.items():
            grouped[(model_id, preproc_version)][text_hash] = text

        for (model_id, preproc_version), hash_to_text in grouped.items():
            hashes = list(hash_to_text.keys())
            stmt_existing = select(EmbeddingCache.text_hash).where(
                EmbeddingCache.text_hash.in_(hashes),
                EmbeddingCache.model_id == model_id,
                EmbeddingCache.preproc_version == preproc_version,
            )
            existing = await session.exec(stmt_existing)
            existing_hashes = {row[0] for row in existing.all()}

            missing = [h for h in hashes if h not in existing_hashes]
            if not missing:
                continue
            if processed >= limit:
                break
            remaining = limit - processed
            if len(missing) > remaining:
                missing = missing[:remaining]
            texts = [hash_to_text[h] for h in missing]
            await self._embed_texts_with_cache(
                session,
                texts,
                model_id=model_id,
                preproc_version=preproc_version,
                use_cache=False,
                mark_duplicates=False,
            )
            processed += len(missing)
            if processed >= limit:
                break

        return processed

    def _apply_prompt_similarity(self, segments: Sequence[SegmentDraft], prompt_embedding: list[float] | None) -> None:
        if not prompt_embedding or not segments:
            return
        prompt_vec = np.asarray(prompt_embedding, dtype=float)
        norm = np.linalg.norm(prompt_vec)
        if norm == 0:
            return
        prompt_vec /= norm
        vectors = [np.asarray(seg.embedding, dtype=float) for seg in segments if seg.embedding]
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

    def _apply_segment_analysis(self, segments: Sequence[SegmentDraft], projection, clustering) -> None:
        coords3d = projection.coords_3d
        coords2d = projection.coords_2d
        labels = clustering.labels
        probabilities = clustering.probabilities
        similarities = clustering.per_point_similarity
        outliers = getattr(clustering, "outlier_scores", np.zeros_like(probabilities))
        silhouettes = getattr(clustering, "silhouette_scores", np.zeros_like(probabilities))
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
        default_dim = fallback.dim if fallback.dim else (fallback_map[responses[0].id].shape[0] if responses and responses[0].id in fallback_map else 1536)

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

    async def _persist_embeddings(self, session, responses: Sequence[Response], embeddings: EmbeddingBatch) -> None:
        if not responses:
            return
        await session.execute(delete(Embedding).where(Embedding.response_id.in_([r.id for r in responses])))
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
                delete(ResponseSegment).where(ResponseSegment.response_id.in_(response_ids))
            )
        if not segments:
            await session.commit()
            return

        entries = []
        for seg in segments:
            vector_bytes = (
                np.asarray(seg.embedding, dtype=np.float32).tobytes() if seg.embedding else None
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
                    text_hash=seg.text_hash,
                    is_cached=seg.is_cached,
                    is_duplicate=seg.is_duplicate,
                    simhash64=seg.simhash64,
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
        await session.execute(delete(Projection).where(Projection.response_id.in_([r.id for r in responses])))
        for response, coords3d, coords2d in zip(responses, projection.coords_3d, projection.coords_2d, strict=False):
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
        await session.execute(delete(Cluster).where(Cluster.response_id.in_([r.id for r in responses])))
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

    async def _persist_cluster_metrics(
        self,
        session,
        run_id: UUID,
        metrics: ClusterMetricsResult,
    ) -> None:
        await session.execute(
            delete(ClusterMetrics).where(ClusterMetrics.run_id == run_id)
        )
        entry = ClusterMetrics(
            run_id=run_id,
            algo=metrics.algo,
            params_json=json.dumps(metrics.params) if metrics.params else None,
            silhouette_embed=metrics.silhouette_embed,
            silhouette_feature=metrics.silhouette_feature,
            davies_bouldin=metrics.davies_bouldin,
            calinski_harabasz=metrics.calinski_harabasz,
            n_clusters=metrics.n_clusters,
            n_noise=metrics.n_noise,
            stability_json=json.dumps(metrics.stability) if metrics.stability else None,
            sweep_json=json.dumps(metrics.sweep) if metrics.sweep else None,
        )
        session.add(entry)
        await session.commit()

    def _build_edge_map_from_neighbors(
        self,
        neighbors_map: dict[UUID, list[tuple[UUID, float]]],
        *,
        threshold: float,
        limit_per_node: int,
    ) -> dict[tuple[UUID, UUID], float]:
        if limit_per_node <= 0:
            return {}
        edges: dict[tuple[UUID, UUID], float] = {}
        for source_id, neighbors in neighbors_map.items():
            count = 0
            for neighbor_id, similarity in neighbors:
                if similarity is None or similarity < threshold or source_id == neighbor_id:
                    continue
                key = tuple(sorted((source_id, neighbor_id)))
                value = float(similarity)
                existing = edges.get(key)
                if existing is None or value > existing:
                    edges[key] = value
                count += 1
                if count >= limit_per_node:
                    break
        return edges

    def _build_mutual_edge_map(
        self,
        neighbors_map: dict[UUID, list[tuple[UUID, float]]],
        *,
        threshold: float,
    ) -> dict[tuple[UUID, UUID], float]:
        if not neighbors_map:
            return {}
        mutual: dict[tuple[UUID, UUID], float] = {}
        for source_id, neighbors in neighbors_map.items():
            for neighbor_id, similarity in neighbors:
                if similarity is None or similarity < threshold or source_id == neighbor_id:
                    continue
                reverse_neighbors = neighbors_map.get(neighbor_id)
                if not reverse_neighbors:
                    continue
                reverse_similarity = None
                for candidate_id, candidate_sim in reverse_neighbors:
                    if candidate_id == source_id:
                        reverse_similarity = candidate_sim
                        break
                if reverse_similarity is None or reverse_similarity < threshold:
                    continue
                key = tuple(sorted((source_id, neighbor_id)))
                score = float(min(similarity, reverse_similarity))
                existing = mutual.get(key)
                if existing is None or score > existing:
                    mutual[key] = score
        return mutual

    def _cluster_mst_edges(
        self,
        cluster_nodes: dict[int, list[UUID]],
        base_edges: dict[tuple[UUID, UUID], float],
    ) -> dict[tuple[UUID, UUID], float]:
        mst_edges: dict[tuple[UUID, UUID], float] = {}
        for nodes in cluster_nodes.values():
            if len(nodes) <= 1:
                continue
            node_set = set(nodes)
            relevant = [
                (pair, similarity)
                for pair, similarity in base_edges.items()
                if pair[0] in node_set and pair[1] in node_set
            ]
            if not relevant:
                continue
            relevant.sort(key=lambda item: (1 - item[1], str(item[0][0]), str(item[0][1])))
            parent: dict[UUID, UUID] = {node: node for node in node_set}
            rank: dict[UUID, int] = {node: 0 for node in node_set}

            def find(node: UUID) -> UUID:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            def union(a: UUID, b: UUID) -> bool:
                root_a = find(a)
                root_b = find(b)
                if root_a == root_b:
                    return False
                if rank[root_a] < rank[root_b]:
                    parent[root_a] = root_b
                elif rank[root_a] > rank[root_b]:
                    parent[root_b] = root_a
                else:
                    parent[root_b] = root_a
                    rank[root_a] += 1
                return True

            edges_added = 0
            for (a, b), similarity in relevant:
                if union(a, b):
                    key = tuple(sorted((a, b)))
                    value = float(similarity)
                    existing = mst_edges.get(key)
                    if existing is None or value > existing:
                        mst_edges[key] = value
                    edges_added += 1
                    if edges_added >= len(node_set) - 1:
                        break
        return mst_edges

    def _ensure_minimum_connections(
        self,
        segments: Sequence[ResponseSegment],
        neighbors_map: dict[UUID, list[tuple[UUID, float]]],
        edge_map: dict[tuple[UUID, UUID], float],
        threshold: float,
    ) -> None:
        if not segments or not neighbors_map:
            return
        adjacency: dict[UUID, set[UUID]] = defaultdict(set)
        for source_id, target_id in edge_map:
            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)
        for segment in segments:
            seg_id = segment.id
            if adjacency.get(seg_id):
                continue
            candidates = neighbors_map.get(seg_id) or []
            for neighbor_id, similarity in candidates:
                if similarity is None or similarity < threshold or neighbor_id == seg_id:
                    continue
                key = tuple(sorted((seg_id, neighbor_id)))
                value = float(similarity)
                current = edge_map.get(key)
                if current is None or value > current:
                    edge_map[key] = value
                adjacency[seg_id].add(neighbor_id)
                adjacency[neighbor_id].add(seg_id)
                break

    async def _persist_segment_edges(self, session, run_id: UUID, edges: Sequence[SegmentEdge]) -> None:
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

    def _build_response_hulls(self, segments: Sequence[SegmentDraft]) -> list[ResponseHull]:
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

    def _effective_cluster_params(self, run: Run, count: int) -> tuple[int, int]:
        algo = getattr(run, "cluster_algo", _SETTINGS.cluster_default_algo)
        if algo != "hdbscan":
            return self._suggest_cluster_params(count)

        fallback_size = _SETTINGS.hdbscan_default_min_cluster_size
        fallback_samples = _SETTINGS.hdbscan_default_min_samples

        min_cluster = getattr(run, "hdbscan_min_cluster_size", None) or fallback_size
        min_samples = getattr(run, "hdbscan_min_samples", None) or fallback_samples

        if count <= 1:
            min_cluster = max(2, int(min_cluster))
        else:
            upper = max(2, count - 1) if count > 2 else max(2, count)
            min_cluster = max(2, min(int(min_cluster), upper))

        min_samples = max(1, min(int(min_samples), min_cluster))
        return min_cluster, min_samples

    async def _clear_existing(self, session, run_id: UUID) -> None:
        response_ids = await session.exec(select(Response.id).where(Response.run_id == run_id))
        ids = response_ids.scalars().all()
        if ids:
            await session.execute(delete(Cluster).where(Cluster.response_id.in_(ids)))
            await session.execute(delete(Projection).where(Projection.response_id.in_(ids)))
            await session.execute(delete(Embedding).where(Embedding.response_id.in_(ids)))
            await session.execute(delete(ResponseSegment).where(ResponseSegment.response_id.in_(ids)))
            await session.execute(delete(ResponseHull).where(ResponseHull.response_id.in_(ids)))
            await session.execute(delete(Response).where(Response.id.in_(ids)))
            await session.execute(delete(SegmentEdge).where(SegmentEdge.run_id == run_id))
            await session.execute(delete(ProjectionCache).where(ProjectionCache.run_id == run_id))
            await session.commit()




async def load_run_with_details(session, run_id: UUID):
    run = await session.get(Run, run_id)
    if not run:
        return None

    results = await session.exec(
        select(Response).where(Response.run_id == run_id).order_by(Response.index)
    )
    responses = results.scalars().all()

    provenance_result = await session.exec(
        select(RunProvenance).where(RunProvenance.run_id == run_id)
    )
    provenance = provenance_result.scalar_one_or_none()

    if not responses:
        return run, [], [], [], [], [], [], [], provenance

    response_ids = [response.id for response in responses]

    projection_rows = await session.exec(
        select(Projection).where(Projection.response_id.in_(response_ids)).order_by(Projection.response_id)
    )
    cluster_rows = await session.exec(
        select(Cluster).where(Cluster.response_id.in_(response_ids)).order_by(Cluster.response_id)
    )
    embedding_rows = await session.exec(
        select(Embedding).where(Embedding.response_id.in_(response_ids)).order_by(Embedding.response_id)
    )
    segment_rows = await session.exec(
        select(ResponseSegment)
        .where(ResponseSegment.response_id.in_(response_ids))
        .order_by(ResponseSegment.response_id, ResponseSegment.position)
    )
    edge_rows = await session.exec(select(SegmentEdge).where(SegmentEdge.run_id == run_id))
    hull_rows = await session.exec(
        select(ResponseHull).where(ResponseHull.response_id.in_(response_ids)).order_by(ResponseHull.response_id)
    )
    metrics_rows = await session.exec(
        select(ClusterMetrics).where(ClusterMetrics.run_id == run_id)
    )
    cluster_metrics = metrics_rows.scalar_one_or_none()

    return (
        run,
        responses,
        projection_rows.scalars().all(),
        cluster_rows.scalars().all(),
        embedding_rows.scalars().all(),
        segment_rows.scalars().all(),
        edge_rows.scalars().all(),
        hull_rows.scalars().all(),
        cluster_metrics,
        provenance,
    )

async def list_recent_runs(session, limit: int = 20) -> list[RunSummary]:
    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)
    try:
        results = await session.exec(stmt)
    except OperationalError as exc:
        message = str(getattr(exc, "orig", exc)).lower()
        if "no such table" in message:
            await init_db()
            return []
        raise
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

        processing_ms = None
        if getattr(run, "timings_json", None):
            try:
                parsed_timings = json.loads(run.timings_json)
            except json.JSONDecodeError:
                parsed_timings = None
            if isinstance(parsed_timings, dict):
                total_value = parsed_timings.get("total_duration_ms")
                if isinstance(total_value, (int, float)):
                    processing_ms = float(total_value)
        attr_duration = getattr(run, "processing_time_ms", None)
        if processing_ms is None and isinstance(attr_duration, (int, float)):
            processing_ms = float(attr_duration)

        summaries.append(
            RunSummary(
                id=run.id,
                prompt=run.prompt,
                n=run.n,
                model=run.model,
                chunk_size=run.chunk_size,
                chunk_overlap=run.chunk_overlap,
                system_prompt=run.system_prompt,
                embedding_model=run.embedding_model
                or _SETTINGS.openai_embedding_model,
                preproc_version=getattr(run, "preproc_version", None)
                or _SETTINGS.embedding_preproc_version,
                use_cache=getattr(run, "use_cache", True),
                cluster_algo=getattr(run, "cluster_algo", _SETTINGS.cluster_default_algo),
                hdbscan_min_cluster_size=getattr(run, "hdbscan_min_cluster_size", None),
                hdbscan_min_samples=getattr(run, "hdbscan_min_samples", None),
                umap=UMAPParams(
                    n_neighbors=getattr(
                        run,
                        "umap_n_neighbors",
                        _SETTINGS.umap_default_n_neighbors,
                    ),
                    min_dist=float(
                        getattr(
                            run,
                            "umap_min_dist",
                            _SETTINGS.umap_default_min_dist,
                        )
                    ),
                    metric=getattr(
                        run, "umap_metric", _SETTINGS.umap_default_metric
                    ),
                    seed=getattr(run, "umap_seed", None),
                    seed_source=getattr(
                        run, "random_state_seed_source", "default"
                    ),
                ),
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
                processing_time_ms=processing_ms,
                progress_stage=run.progress_stage,
                progress_message=run.progress_message,
                progress_percent=run.progress_percent,
                progress_metadata=metadata,
            )
        )
    return summaries
