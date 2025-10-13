"""Services for comparing two sampling runs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Sequence
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sqlalchemy import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.models import PointLink, Response, ResponseSegment, Run, RunPair
from app.schemas import (
    CompareRunsRequest,
    CompareRunsResponse,
    ComparisonLink,
    ComparisonMetrics,
    ComparisonPoint,
    ComparisonTransformComponent,
    ComparisonTransformSummary,
    MovementHistogramBin,
    RunResource,
    TopTermShift,
    UMAPParams,
)

_EPS = 1e-9
_DEFAULT_MAX_LINKS = 600


@dataclass(slots=True)
class SegmentRecord:
    """In-memory representation of a response segment for comparison workflows."""

    id: UUID
    run_id: UUID
    response_id: UUID
    response_index: int
    position: int
    text: str
    text_hash: Optional[str]
    cluster_label: Optional[int]
    cluster_probability: Optional[float]
    cluster_similarity: Optional[float]
    outlier_score: Optional[float]
    role: Optional[str]
    prompt_similarity: Optional[float]
    coords3d: NDArray[np.float64]
    coords2d: NDArray[np.float64]
    embedding: Optional[NDArray[np.float32]]
    feature_vector: Optional[NDArray[np.float32]]


@dataclass(slots=True)
class TransformResult:
    """Affine transform parameters derived from Procrustes alignment."""

    rotation: NDArray[np.float64]
    scale: float
    translation: NDArray[np.float64]
    rmsd: Optional[float]


@dataclass(slots=True)
class AnchorSet:
    """Anchor metadata used to derive alignment transforms."""

    kind: str
    left3d: NDArray[np.float64]
    right3d: NDArray[np.float64]
    left2d: NDArray[np.float64]
    right2d: NDArray[np.float64]
    left_ids: set[UUID]
    right_ids: set[UUID]

    @property
    def count(self) -> int:
        return int(self.left3d.shape[0])


class CompareService:
    """Coordinate cross-run alignment, linking, and diff metrics."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._settings = get_settings()
        
    def _serialise_for_json(self, value):
        """Normalise values for JSON persistence (datetimes, numpy types)."""
        from datetime import datetime
        import numpy as np

        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._serialise_for_json(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialise_for_json(val) for val in value]
        return value


    async def compare_runs(self, payload: CompareRunsRequest) -> CompareRunsResponse:
        if payload.left_run_id == payload.right_run_id:
            raise ValueError("Comparison requires two distinct runs")

        left_run = await self._session.get(Run, payload.left_run_id)
        right_run = await self._session.get(Run, payload.right_run_id)
        if left_run is None or right_run is None:
            missing = []
            if left_run is None:
                missing.append(str(payload.left_run_id))
            if right_run is None:
                missing.append(str(payload.right_run_id))
            raise ValueError(f"Run(s) not found: {', '.join(missing)}")

        left_segments = await self._load_segments(left_run.id)
        right_segments = await self._load_segments(right_run.id)

        left_resource = self._run_to_resource(left_run)
        right_resource = self._run_to_resource(right_run)

        shared_links = self._pair_segments_by_hash(left_segments, right_segments)
        shared_pairs = [(left, right) for left, right, _ in shared_links]

        nn_pairs = self._compute_feature_nn_pairs(
            left_segments,
            right_segments,
            max_pairs=payload.max_links or _DEFAULT_MAX_LINKS,
        )

        anchor_set = self._select_anchor_set(
            payload=payload,
            shared_pairs=shared_pairs,
            left_segments=left_segments,
            right_segments=right_segments,
            nn_pairs=nn_pairs,
        )

        transform_summary, transform3d, transform2d = self._compute_transforms(anchor_set)

        aligned_right_3d: dict[UUID, NDArray[np.float64]] = {}
        aligned_right_2d: dict[UUID, NDArray[np.float64]] = {}
        for segment in right_segments:
            aligned_right_3d[segment.id] = self._apply_transform(segment.coords3d, transform3d)
            aligned_right_2d[segment.id] = self._apply_transform(segment.coords2d, transform2d)

        anchor_left_ids = anchor_set.left_ids
        anchor_right_ids = anchor_set.right_ids

        points: list[ComparisonPoint] = []
        for segment in left_segments:
            coords3d = tuple(float(v) for v in segment.coords3d)
            coords2d = tuple(float(v) for v in segment.coords2d)
            points.append(
                ComparisonPoint(
                    id=segment.id,
                    run_id=segment.run_id,
                    response_id=segment.response_id,
                    response_index=segment.response_index,
                    segment_position=segment.position,
                    source="left",
                    text_hash=segment.text_hash,
                    cluster_label=segment.cluster_label,
                    cluster_probability=segment.cluster_probability,
                    cluster_similarity=segment.cluster_similarity,
                    role=segment.role,
                    prompt_similarity=segment.prompt_similarity,
                    text_preview=segment.text[:240],
                    coords_3d=coords3d,
                    coords_2d=coords2d,
                    aligned_coords_3d=coords3d,
                    aligned_coords_2d=coords2d,
                    is_anchor=segment.id in anchor_left_ids,
                )
            )

        for segment in right_segments:
            coords3d = tuple(float(v) for v in segment.coords3d)
            coords2d = tuple(float(v) for v in segment.coords2d)
            aligned3d = tuple(float(v) for v in aligned_right_3d[segment.id])
            aligned2d = tuple(float(v) for v in aligned_right_2d[segment.id])
            points.append(
                ComparisonPoint(
                    id=segment.id,
                    run_id=segment.run_id,
                    response_id=segment.response_id,
                    response_index=segment.response_index,
                    segment_position=segment.position,
                    source="right",
                    text_hash=segment.text_hash,
                    cluster_label=segment.cluster_label,
                    cluster_probability=segment.cluster_probability,
                    cluster_similarity=segment.cluster_similarity,
                    role=segment.role,
                    prompt_similarity=segment.prompt_similarity,
                    text_preview=segment.text[:240],
                    coords_3d=coords3d,
                    coords_2d=coords2d,
                    aligned_coords_3d=aligned3d,
                    aligned_coords_2d=aligned2d,
                    is_anchor=segment.id in anchor_right_ids,
                )
            )

        shared_movement = []
        comparison_links: list[ComparisonLink] = []
        shared_left_ids = {pair[0].id for pair in shared_pairs}
        shared_right_ids = {pair[1].id for pair in shared_pairs}
        for left_segment, right_segment, text_hash in shared_links:
            vector = aligned_right_3d[right_segment.id] - left_segment.coords3d
            distance = float(np.linalg.norm(vector))
            shared_movement.append(distance)
            comparison_links.append(
                ComparisonLink(
                    left_segment_id=left_segment.id,
                    right_segment_id=right_segment.id,
                    link_type="exact_hash",
                    distance=None,
                    text_hash=text_hash,
                    movement_vector=tuple(float(v) for v in vector),
                    movement_distance=distance,
                )
            )

        max_links = payload.max_links or _DEFAULT_MAX_LINKS
        remaining = max(0, max_links - len(comparison_links))
        if remaining and nn_pairs:
            nn_candidates = [
                pair
                for pair in nn_pairs
                if pair[0].id not in shared_left_ids and pair[1].id not in shared_right_ids
            ]
            nn_candidates = sorted(nn_candidates, key=lambda item: item[2], reverse=True)[:remaining]
            for left_segment, right_segment, similarity in nn_candidates:
                vector = aligned_right_3d[right_segment.id] - left_segment.coords3d
                comparison_links.append(
                    ComparisonLink(
                        left_segment_id=left_segment.id,
                        right_segment_id=right_segment.id,
                        link_type="nn",
                        distance=max(0.0, 1.0 - float(similarity)),
                        text_hash=None,
                        movement_vector=tuple(float(v) for v in vector),
                        movement_distance=float(np.linalg.norm(vector)),
                    )
                )

        ari, nmi = self._alignment_scores(shared_pairs)
        cluster_count_left, noise_left, mean_cluster_size_left, median_cluster_size_left, _ = self._cluster_stats(left_segments)
        cluster_count_right, noise_right, mean_cluster_size_right, median_cluster_size_right, _ = self._cluster_stats(right_segments)

        histogram = self._movement_histogram(shared_movement, bins=payload.histogram_bins)
        top_term_shifts = self._top_term_shifts(left_segments, right_segments)

        metrics = ComparisonMetrics(
            shared_segment_count=len(shared_pairs),
            linked_segment_count=len(comparison_links),
            alignment_method=transform_summary.method,
            ari=ari,
            nmi=nmi,
            mean_movement=self._safe_mean(shared_movement),
            median_movement=self._safe_median(shared_movement),
            max_movement=self._safe_max(shared_movement),
            movement_histogram=histogram,
            cluster_count_left=cluster_count_left,
            cluster_count_right=cluster_count_right,
            delta_cluster_count=cluster_count_right - cluster_count_left,
            mean_cluster_size_left=mean_cluster_size_left,
            mean_cluster_size_right=mean_cluster_size_right,
            median_cluster_size_left=median_cluster_size_left,
            median_cluster_size_right=median_cluster_size_right,
            noise_left=noise_left,
            noise_right=noise_right,
            top_term_shifts=top_term_shifts,
        )

        run_pair_id = await self._persist_comparison(
            payload=payload,
            left_run=left_run,
            right_run=right_run,
            summary=transform_summary,
            links=comparison_links,
        )

        return CompareRunsResponse(
            run_pair_id=run_pair_id,
            left_run=left_resource,
            right_run=right_resource,
            transforms=transform_summary,
            points=points,
            links=comparison_links,
            metrics=metrics,
            anchor_kind=anchor_set.kind,
        )

    async def _persist_comparison(
        self,
        *,
        payload: CompareRunsRequest,
        left_run: Run,
        right_run: Run,
        summary: ComparisonTransformSummary,
        links: Sequence[ComparisonLink],
    ) -> Optional[UUID]:
        if not payload.save:
            return None

        stmt = select(RunPair).where(
            RunPair.left_run_id == left_run.id,
            RunPair.right_run_id == right_run.id,
        )
        result = await self._session.exec(stmt)
        run_pair = result.scalar_one_or_none()

        transform_payload = self._serialise_for_json(summary.model_dump())
        if run_pair is not None:
            run_pair.alignment_method = summary.method
            run_pair.transform_json = transform_payload
            run_pair.created_at = datetime.utcnow()
        else:
            run_pair = RunPair(
                left_run_id=left_run.id,
                right_run_id=right_run.id,
                alignment_method=summary.method,
                transform_json=transform_payload,
            )
            self._session.add(run_pair)
        await self._session.flush()

        await self._session.exec(delete(PointLink).where(PointLink.run_pair_id == run_pair.id))
        link_models = [
            PointLink(
                run_pair_id=run_pair.id,
                left_segment_id=link.left_segment_id,
                right_segment_id=link.right_segment_id,
                link_type=link.link_type,
                distance=link.distance,
            )
            for link in links
        ]
        if link_models:
            self._session.add_all(link_models)
        await self._session.commit()
        return run_pair.id
    async def _load_segments(self, run_id: UUID) -> list[SegmentRecord]:
        stmt = (
            select(ResponseSegment, Response)
            .join(Response, ResponseSegment.response_id == Response.id)
            .where(Response.run_id == run_id)
        )
        result = await self._session.exec(stmt)
        records: list[SegmentRecord] = []
        for segment, response in result.all():
            embedding = self._decode_embedding(segment)
            feature_vector = self._build_feature_vector(segment, embedding)
            records.append(
                SegmentRecord(
                    id=segment.id,
                    run_id=run_id,
                    response_id=segment.response_id,
                    response_index=getattr(response, "index", 0),
                    position=segment.position,
                    text=segment.text or "",
                    text_hash=segment.text_hash,
                    cluster_label=segment.cluster_label,
                    cluster_probability=segment.cluster_probability,
                    cluster_similarity=segment.cluster_similarity,
                    outlier_score=segment.outlier_score,
                    role=segment.role,
                    prompt_similarity=segment.prompt_similarity,
                    coords3d=np.array([
                        float(segment.coord_x or 0.0),
                        float(segment.coord_y or 0.0),
                        float(segment.coord_z or 0.0),
                    ], dtype=np.float64),
                    coords2d=np.array([
                        float(segment.coord2_x or 0.0),
                        float(segment.coord2_y or 0.0),
                    ], dtype=np.float64),
                    embedding=embedding,
                    feature_vector=feature_vector,
                )
            )
        return records

    def _decode_embedding(self, segment: ResponseSegment) -> Optional[NDArray[np.float32]]:
        if not segment.embedding_vector:
            return None
        buffer = np.frombuffer(segment.embedding_vector, dtype=np.float16)
        if segment.embedding_dim and buffer.size != segment.embedding_dim:
            buffer = buffer[: segment.embedding_dim]
        return buffer.astype(np.float32)

    def _build_feature_vector(
        self,
        segment: ResponseSegment,
        embedding: Optional[NDArray[np.float32]],
    ) -> Optional[NDArray[np.float32]]:
        extras = np.array(
            [
                float(segment.prompt_similarity or 0.0),
                float(segment.cluster_probability or 0.0),
                float(segment.cluster_similarity or 0.0),
                float(segment.outlier_score or 0.0),
            ],
            dtype=np.float32,
        )
        if embedding is None:
            vector = extras.copy()
        else:
            vector = np.concatenate([embedding, extras])
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm < _EPS:
            return None
        vector /= norm
        return vector.astype(np.float32)

    def _pair_segments_by_hash(
        self,
        left_segments: Sequence[SegmentRecord],
        right_segments: Sequence[SegmentRecord],
    ) -> list[tuple[SegmentRecord, SegmentRecord, Optional[str]]]:
        left_map: dict[str, list[SegmentRecord]] = defaultdict(list)
        right_map: dict[str, list[SegmentRecord]] = defaultdict(list)
        for segment in left_segments:
            if segment.text_hash:
                left_map[segment.text_hash].append(segment)
        for segment in right_segments:
            if segment.text_hash:
                right_map[segment.text_hash].append(segment)

        shared_hashes = left_map.keys() & right_map.keys()
        pairs: list[tuple[SegmentRecord, SegmentRecord, Optional[str]]] = []
        for text_hash in sorted(shared_hashes):
            left_list = sorted(
                left_map[text_hash],
                key=lambda seg: (seg.response_index, seg.position),
            )
            right_list = sorted(
                right_map[text_hash],
                key=lambda seg: (seg.response_index, seg.position),
            )
            for left_segment, right_segment in zip(left_list, right_list):
                pairs.append((left_segment, right_segment, text_hash))
        return pairs

    def _compute_feature_nn_pairs(
        self,
        left_segments: Sequence[SegmentRecord],
        right_segments: Sequence[SegmentRecord],
        *,
        max_pairs: int,
    ) -> list[tuple[SegmentRecord, SegmentRecord, float]]:
        left_order = [seg for seg in left_segments if seg.feature_vector is not None]
        right_order = [seg for seg in right_segments if seg.feature_vector is not None]
        if not left_order or not right_order:
            return []

        left_matrix = np.stack([seg.feature_vector for seg in left_order])
        right_matrix = np.stack([seg.feature_vector for seg in right_order])

        similarity = left_matrix @ right_matrix.T
        left_best = similarity.argmax(axis=1)
        right_best = similarity.argmax(axis=0)

        pairs: list[tuple[SegmentRecord, SegmentRecord, float]] = []
        for left_idx, right_idx in enumerate(left_best):
            if right_best[right_idx] != left_idx:
                continue
            score = float(similarity[left_idx, right_idx])
            if math.isnan(score):
                continue
            pairs.append((left_order[left_idx], right_order[right_idx], score))

        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[:max_pairs]
    def _select_anchor_set(
        self,
        *,
        payload: CompareRunsRequest,
        shared_pairs: Sequence[tuple[SegmentRecord, SegmentRecord]],
        left_segments: Sequence[SegmentRecord],
        right_segments: Sequence[SegmentRecord],
        nn_pairs: Sequence[tuple[SegmentRecord, SegmentRecord, float]],
    ) -> AnchorSet:
        min_required = max(payload.min_shared, 3)
        if len(shared_pairs) >= min_required:
            left3d = np.stack([pair[0].coords3d for pair in shared_pairs])
            right3d = np.stack([pair[1].coords3d for pair in shared_pairs])
            left2d = np.stack([pair[0].coords2d for pair in shared_pairs])
            right2d = np.stack([pair[1].coords2d for pair in shared_pairs])
            return AnchorSet(
                kind="shared_hash",
                left3d=left3d,
                right3d=right3d,
                left2d=left2d,
                right2d=right2d,
                left_ids={pair[0].id for pair in shared_pairs},
                right_ids={pair[1].id for pair in shared_pairs},
            )

        left_centroids = self._cluster_centroids(left_segments)
        right_centroids = self._cluster_centroids(right_segments)
        common_labels = sorted(left_centroids.keys() & right_centroids.keys())
        if len(common_labels) >= 3:
            left3d = np.stack([left_centroids[label][0] for label in common_labels])
            right3d = np.stack([right_centroids[label][0] for label in common_labels])
            left2d = np.stack([left_centroids[label][1] for label in common_labels])
            right2d = np.stack([right_centroids[label][1] for label in common_labels])
            return AnchorSet(
                kind="cluster_centroid",
                left3d=left3d,
                right3d=right3d,
                left2d=left2d,
                right2d=right2d,
                left_ids=set(),
                right_ids=set(),
            )

        if nn_pairs:
            anchors = nn_pairs[: max(3, min(len(nn_pairs), 50))]
            left3d = np.stack([pair[0].coords3d for pair in anchors])
            right3d = np.stack([pair[1].coords3d for pair in anchors])
            left2d = np.stack([pair[0].coords2d for pair in anchors])
            right2d = np.stack([pair[1].coords2d for pair in anchors])
            return AnchorSet(
                kind="feature_nn",
                left3d=left3d,
                right3d=right3d,
                left2d=left2d,
                right2d=right2d,
                left_ids={pair[0].id for pair in anchors},
                right_ids={pair[1].id for pair in anchors},
            )

        zeros3d = np.zeros((0, 3), dtype=np.float64)
        zeros2d = np.zeros((0, 2), dtype=np.float64)
        return AnchorSet(
            kind="identity",
            left3d=zeros3d,
            right3d=zeros3d,
            left2d=zeros2d,
            right2d=zeros2d,
            left_ids=set(),
            right_ids=set(),
        )

    def _cluster_centroids(
        self,
        segments: Sequence[SegmentRecord],
    ) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]:
        groups: dict[int, list[SegmentRecord]] = defaultdict(list)
        for segment in segments:
            if segment.cluster_label is None or segment.cluster_label < 0:
                continue
            groups[int(segment.cluster_label)].append(segment)
        centroids: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        for label, members in groups.items():
            coords3d = np.stack([seg.coords3d for seg in members])
            coords2d = np.stack([seg.coords2d for seg in members])
            centroids[label] = (coords3d.mean(axis=0), coords2d.mean(axis=0))
        return centroids

    def _compute_transforms(
        self,
        anchor_set: AnchorSet,
    ) -> tuple[ComparisonTransformSummary, TransformResult, TransformResult]:
        transform3d = self._compute_transform(anchor_set.left3d, anchor_set.right3d, dims=3)
        transform2d = self._compute_transform(anchor_set.left2d, anchor_set.right2d, dims=2)

        components: dict[str, ComparisonTransformComponent] = {
            "3d": ComparisonTransformComponent(
                dimension="3d",
                rotation=transform3d.rotation.tolist(),
                scale=float(transform3d.scale),
                translation=transform3d.translation.tolist(),
                rmsd=transform3d.rmsd,
                anchor_count=anchor_set.count,
            ),
            "2d": ComparisonTransformComponent(
                dimension="2d",
                rotation=transform2d.rotation.tolist(),
                scale=float(transform2d.scale),
                translation=transform2d.translation.tolist(),
                rmsd=transform2d.rmsd,
                anchor_count=anchor_set.count,
            ),
        }

        summary = ComparisonTransformSummary(
            method="procrustes" if anchor_set.count else "identity",
            anchor_kind=anchor_set.kind,
            anchor_count=anchor_set.count,
            components=components,
        )
        return summary, transform3d, transform2d

    def _compute_transform(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        *,
        dims: int,
    ) -> TransformResult:
        if left.shape[0] == 0 or right.shape[0] == 0:
            return TransformResult(
                rotation=np.eye(dims, dtype=np.float64),
                scale=1.0,
                translation=np.zeros(dims, dtype=np.float64),
                rmsd=None,
            )

        left_mean = left.mean(axis=0)
        right_mean = right.mean(axis=0)
        left_centered = left - left_mean
        right_centered = right - right_mean

        norm_left = float(np.linalg.norm(left_centered))
        norm_right = float(np.linalg.norm(right_centered))
        if norm_left < _EPS or norm_right < _EPS:
            translation = left_mean - right_mean
            return TransformResult(
                rotation=np.eye(dims, dtype=np.float64),
                scale=1.0,
                translation=translation,
                rmsd=float(np.linalg.norm(translation)),
            )

        left_norm = left_centered / norm_left
        right_norm = right_centered / norm_right
        matrix = right_norm.T @ left_norm
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        scale = norm_left / norm_right
        translation = left_mean - scale * (right_mean @ rotation)
        aligned = scale * (right @ rotation) + translation
        rmsd = float(np.sqrt(np.mean(np.sum((aligned - left) ** 2, axis=1))))
        return TransformResult(rotation=rotation, scale=float(scale), translation=translation, rmsd=rmsd)

    def _apply_transform(
        self,
        coords: NDArray[np.float64],
        transform: TransformResult,
    ) -> NDArray[np.float64]:
        return transform.scale * (coords @ transform.rotation) + transform.translation
    def _alignment_scores(
        self,
        shared_pairs: Sequence[tuple[SegmentRecord, SegmentRecord]],
    ) -> tuple[Optional[float], Optional[float]]:
        if len(shared_pairs) < 2:
            return None, None
        left_labels = [
            int(pair[0].cluster_label) if pair[0].cluster_label is not None else -1
            for pair in shared_pairs
        ]
        right_labels = [
            int(pair[1].cluster_label) if pair[1].cluster_label is not None else -1
            for pair in shared_pairs
        ]
        try:
            ari = float(adjusted_rand_score(left_labels, right_labels))
        except Exception:
            ari = None
        try:
            nmi = float(normalized_mutual_info_score(left_labels, right_labels))
        except Exception:
            nmi = None
        return ari, nmi

    def _cluster_stats(
        self,
        segments: Sequence[SegmentRecord],
    ) -> tuple[int, int, Optional[float], Optional[float], Counter[int]]:
        labels = [
            int(segment.cluster_label)
            for segment in segments
            if segment.cluster_label is not None and segment.cluster_label >= 0
        ]
        counter: Counter[int] = Counter(labels)
        cluster_count = len(counter)
        noise = sum(
            1 for segment in segments if segment.cluster_label is None or segment.cluster_label < 0
        )
        mean_value = float(np.mean(list(counter.values()))) if counter else None
        median_value = float(np.median(list(counter.values()))) if counter else None
        return cluster_count, noise, mean_value, median_value, counter

    def _top_term_shifts(
        self,
        left_segments: Sequence[SegmentRecord],
        right_segments: Sequence[SegmentRecord],
        *,
        top_k: int = 6,
    ) -> list[TopTermShift]:
        left_terms = self._cluster_top_terms(left_segments, top_k=top_k)
        right_terms = self._cluster_top_terms(right_segments, top_k=top_k)
        shifts: list[TopTermShift] = []
        for label in sorted(left_terms.keys() & right_terms.keys()):
            left_set = set(left_terms[label])
            right_set = set(right_terms[label])
            union = left_set | right_set
            intersection = left_set & right_set
            jaccard = float(len(intersection) / len(union)) if union else None
            shifts.append(
                TopTermShift(
                    cluster_label=label,
                    left_terms=list(left_terms[label]),
                    right_terms=list(right_terms[label]),
                    shared_terms=sorted(intersection),
                    jaccard=jaccard,
                )
            )
        return shifts

    def _cluster_top_terms(
        self,
        segments: Sequence[SegmentRecord],
        *,
        top_k: int,
    ) -> dict[int, list[str]]:
        labelled_segments = [
            seg for seg in segments if seg.cluster_label is not None and seg.cluster_label >= 0
        ]
        if not labelled_segments:
            return {}
        texts = [seg.text for seg in labelled_segments]
        if not any(texts):
            return {}
        vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        cluster_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, segment in enumerate(labelled_segments):
            cluster_to_indices[int(segment.cluster_label)].append(idx)
        terms: dict[int, list[str]] = {}
        for label, indices in cluster_to_indices.items():
            rows = matrix[indices]
            mean_weights = np.asarray(rows.mean(axis=0)).ravel()
            if not np.any(mean_weights):
                continue
            order = mean_weights.argsort()[::-1]
            keywords = [feature_names[i] for i in order if mean_weights[i] > 0][:top_k]
            terms[label] = keywords
        return terms

    def _movement_histogram(
        self,
        movement: Sequence[float],
        *,
        bins: int,
    ) -> list[MovementHistogramBin]:
        if not movement or bins <= 0:
            return []
        counts, edges = np.histogram(movement, bins=bins)
        histogram: list[MovementHistogramBin] = []
        for idx, count in enumerate(counts):
            histogram.append(
                MovementHistogramBin(
                    start=float(edges[idx]),
                    end=float(edges[idx + 1]),
                    count=int(count),
                )
            )
        return histogram

    def _safe_mean(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.mean(values))

    def _safe_median(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.median(values))

    def _safe_max(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(max(values))

    def _run_to_resource(self, run: Run) -> RunResource:
        umap_params = UMAPParams(
            n_neighbors=getattr(run, "umap_n_neighbors", self._settings.umap_default_n_neighbors),
            min_dist=float(getattr(run, "umap_min_dist", self._settings.umap_default_min_dist)),
            metric=getattr(run, "umap_metric", self._settings.umap_default_metric),
            seed=getattr(run, "umap_seed", None),
            seed_source=getattr(run, "random_state_seed_source", "default"),
        )

        progress_metadata: Optional[dict[str, Any]] = None
        raw_metadata = getattr(run, "progress_metadata", None)
        if raw_metadata:
            try:
                progress_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                progress_metadata = None

        processing_time_ms: Optional[float] = None
        stage_timings: list[dict[str, Any]] = []
        raw_timings = getattr(run, "timings_json", None)
        if raw_timings:
            try:
                parsed = json.loads(raw_timings)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                stage_values = parsed.get("stages") or []
                total = parsed.get("total_duration_ms")
                if isinstance(total, (int, float)):
                    processing_time_ms = float(total)
            elif isinstance(parsed, list):
                stage_values = parsed
            else:
                stage_values = []
            stage_timings = [value for value in stage_values if isinstance(value, dict)]
        attr_time = getattr(run, "processing_time_ms", None)
        if processing_time_ms is None and isinstance(attr_time, (int, float)):
            processing_time_ms = float(attr_time)

        return RunResource(
            id=run.id,
            prompt=run.prompt,
            n=run.n,
            model=run.model,
            chunk_size=getattr(run, "chunk_size", None),
            chunk_overlap=getattr(run, "chunk_overlap", None),
            system_prompt=getattr(run, "system_prompt", None),
            embedding_model=getattr(run, "embedding_model", None) or self._settings.openai_embedding_model,
            preproc_version=getattr(run, "preproc_version", None) or self._settings.embedding_preproc_version,
            use_cache=bool(getattr(run, "use_cache", True)),
            cluster_algo=getattr(run, "cluster_algo", self._settings.cluster_default_algo),
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
