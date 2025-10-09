"""Cluster quality computation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from app.services.projection import ClusterResult, cluster_with_fallback, run_kmeans

_BOOTSTRAP_FRACTION_DEFAULT = 0.85
_BOOTSTRAP_ITERATIONS_DEFAULT = 20
_SWEEP_VARIATION = 0.2


@dataclass(slots=True)
class ClusterMetricsResult:
    algo: str
    params: dict[str, Any]
    silhouette_embed: Optional[float]
    silhouette_feature: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    n_clusters: int
    n_noise: int
    stability: Optional[dict[str, Any]]
    sweep: Optional[dict[str, Any]]


def _mean_for_mask(values: ArrayLike | None, mask: np.ndarray) -> Optional[float]:
    if values is None or mask.sum() == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    subset = arr[mask]
    if subset.size == 0:
        return None
    finite = subset[np.isfinite(subset)]
    if finite.size == 0:
        return None
    return float(np.clip(finite.mean(), -1.0, 1.0))


def _compute_embed_silhouette(
    coords_3d: np.ndarray | None,
    coords_2d: np.ndarray | None,
    labels: np.ndarray,
) -> Optional[float]:
    mask = labels >= 0
    if mask.sum() < 2:
        return None
    unique = np.unique(labels[mask])
    if unique.size < 2:
        return None
    coords: np.ndarray | None = None
    if coords_3d is not None and coords_3d.shape[0] == labels.size:
        coords = coords_3d[mask]
    elif coords_2d is not None and coords_2d.shape[0] == labels.size:
        coords = coords_2d[mask]
    if coords is None or coords.shape[0] <= len(unique):
        return None
    try:
        score = silhouette_score(coords, labels[mask], metric="euclidean")
    except Exception:
        return None
    if not np.isfinite(score):
        return None
    return float(np.clip(score, -1.0, 1.0))


def _safe_index_score(metric_fn, data: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if data.size == 0 or labels.size == 0:
        return None
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    try:
        score = metric_fn(data, labels)
    except Exception:
        return None
    if not np.isfinite(score):
        return None
    return float(score)


def _collect_persistence(cluster_result: ClusterResult) -> Optional[dict[str, float]]:
    metadata = cluster_result.metadata or {}
    persistence = metadata.get("persistence")
    if not isinstance(persistence, dict):
        return None
    formatted: dict[str, float] = {}
    for key, value in persistence.items():
        try:
            label = str(int(key))
        except Exception:
            label = str(key)
        try:
            formatted[label] = float(value)
        except Exception:
            continue
    return formatted or None


def _summarise_cluster_result(
    cluster_result: ClusterResult,
    feature_matrix: np.ndarray,
    coords_3d: np.ndarray | None,
    coords_2d: np.ndarray | None,
) -> dict[str, Any]:
    labels = np.asarray(cluster_result.labels, dtype=int)
    mask = labels >= 0
    n_noise = int(np.sum(labels < 0))
    unique = np.unique(labels[mask]) if mask.any() else np.array([], dtype=int)
    n_clusters = int(unique.size)
    silhouette_feature = _mean_for_mask(cluster_result.silhouette_scores, mask)
    embed_silhouette = _compute_embed_silhouette(coords_3d, coords_2d, labels)
    if mask.any():
        feature_subset = feature_matrix[mask]
        label_subset = labels[mask]
    else:
        feature_subset = np.empty((0, feature_matrix.shape[1] if feature_matrix.ndim > 1 else 1))
        label_subset = np.empty((0,), dtype=int)
    davies = _safe_index_score(davies_bouldin_score, feature_subset, label_subset)
    calinski = _safe_index_score(calinski_harabasz_score, feature_subset, label_subset)
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_feature": silhouette_feature,
        "silhouette_embed": embed_silhouette,
        "davies_bouldin": davies,
        "calinski_harabasz": calinski,
    }


def _bootstrap_stability(
    feature_matrix: np.ndarray,
    coords_3d: np.ndarray | None,
    similarity_basis: np.ndarray | None,
    labels: np.ndarray,
    *,
    algo: str,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    metric: str,
    fraction: float,
    iterations: int,
    random_state: int,
) -> Optional[dict[str, Any]]:
    n = feature_matrix.shape[0]
    positive_clusters = {
        int(label): np.flatnonzero(labels == label)
        for label in np.unique(labels)
        if int(label) >= 0
    }
    if not positive_clusters or n < 4 or fraction <= 0 or iterations <= 0:
        return None
    rng = np.random.default_rng(random_state)
    scores: dict[int, list[float]] = {label: [] for label in positive_clusters}
    actual_iters = 0
    basis = similarity_basis if similarity_basis is not None and similarity_basis.shape[0] == n else None
    coords3d = coords_3d if coords_3d is not None and coords_3d.shape[0] == n else None
    for _ in range(iterations):
        sample_size = int(round(n * fraction))
        sample_size = max(3, min(sample_size, n))
        if sample_size >= n:
            sample_indices = np.arange(n)
        else:
            sample_indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        subset_matrix = feature_matrix[sample_indices]
        subset_coords3d = coords3d[sample_indices] if coords3d is not None else None
        subset_basis = basis[sample_indices] if basis is not None else None
        subset_min_cluster = min_cluster_size or 2
        subset_min_cluster = max(2, min(subset_min_cluster, subset_matrix.shape[0] - 1))
        subset_min_samples = min_samples or 1
        subset_min_samples = max(1, min(subset_min_samples, subset_min_cluster))
        if algo == "kmeans":
            n_clusters = max(1, len(positive_clusters))
            result = run_kmeans(
                subset_matrix,
                coords_3d=subset_coords3d,
                n_clusters=n_clusters,
                similarity_basis=subset_basis,
            )
        else:
            result = cluster_with_fallback(
                subset_matrix,
                coords_3d=subset_coords3d,
                similarity_basis=subset_basis,
                min_cluster_size=subset_min_cluster,
                min_samples=subset_min_samples,
                metric=metric,
                algo="hdbscan",
            )
        subset_labels = np.asarray(result.labels, dtype=int)
        cluster_map = {
            int(label): set(np.flatnonzero(subset_labels == label))
            for label in np.unique(subset_labels)
            if int(label) >= 0
        }
        if not cluster_map:
            continue
        actual_iters += 1
        for label, original_indices in positive_clusters.items():
            original_set = set(int(idx) for idx in original_indices.tolist())
            sample_positions = [
                pos
                for pos, global_idx in enumerate(sample_indices)
                if global_idx in original_set
            ]
            if not sample_positions:
                continue
            base_set = set(sample_positions)
            best_jaccard = 0.0
            for candidate_indices in cluster_map.values():
                if not candidate_indices:
                    continue
                intersection = base_set.intersection(candidate_indices)
                if not intersection:
                    continue
                union = base_set.union(candidate_indices)
                if not union:
                    continue
                score = len(intersection) / len(union)
                if score > best_jaccard:
                    best_jaccard = score
            scores[label].append(best_jaccard)
    if actual_iters == 0:
        return None
    cluster_stats = {
        str(label): {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "samples": len(values),
        }
        for label, values in scores.items()
        if values
    }
    if not cluster_stats:
        return None
    return {
        "mode": "bootstrap",
        "fraction": fraction,
        "iterations": actual_iters,
        "clusters": cluster_stats,
    }


def _variation_values(base: int, lower: int, upper: int, fraction: float) -> list[int]:
    if upper < lower:
        upper = lower
    values = {int(base)}
    delta = max(1, int(round(abs(base * fraction))))
    values.add(max(lower, min(upper, int(base - delta))))
    values.add(max(lower, min(upper, int(base + delta))))
    filtered = sorted(value for value in values if lower <= value <= upper)
    if not filtered:
        filtered = [max(lower, min(upper, int(base)))]
    return filtered


def _parameter_sweep(
    feature_matrix: np.ndarray,
    coords_3d: np.ndarray | None,
    coords_2d: np.ndarray | None,
    similarity_basis: np.ndarray | None,
    *,
    algo: str,
    metric: str,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    fraction: float,
) -> Optional[dict[str, Any]]:
    if algo != "hdbscan":
        return None
    n = feature_matrix.shape[0]
    if n < 5:
        return None
    baseline_size = min_cluster_size or max(2, int(round(n * 0.1)))
    baseline_samples = min_samples or max(1, min(baseline_size, int(round(baseline_size * 0.5))))
    size_values = _variation_values(baseline_size, lower=2, upper=max(2, n - 1), fraction=fraction)
    sample_values = _variation_values(baseline_samples, lower=1, upper=max(1, baseline_size), fraction=fraction)
    results: list[dict[str, Any]] = []
    for size_value in size_values:
        for sample_value in sample_values:
            result = cluster_with_fallback(
                feature_matrix,
                coords_3d=coords_3d,
                similarity_basis=similarity_basis,
                min_cluster_size=size_value,
                min_samples=sample_value,
                metric=metric,
                algo="hdbscan",
            )
            summary = _summarise_cluster_result(result, feature_matrix, coords_3d, coords_2d)
            results.append(
                {
                    "min_cluster_size": size_value,
                    "min_samples": sample_value,
                    "algo": result.method,
                    "n_clusters": summary["n_clusters"],
                    "n_noise": summary["n_noise"],
                    "silhouette_feature": summary["silhouette_feature"],
                    "silhouette_embed": summary["silhouette_embed"],
                    "davies_bouldin": summary["davies_bouldin"],
                    "calinski_harabasz": summary["calinski_harabasz"],
                }
            )
    return {
        "baseline": {
            "min_cluster_size": baseline_size,
            "min_samples": baseline_samples,
        },
        "points": results,
    }


def compute_cluster_metrics(
    feature_matrix: ArrayLike,
    coords_3d: ArrayLike | None,
    coords_2d: ArrayLike | None,
    cluster_result: ClusterResult,
    *,
    algo: str,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    metric: str = "euclidean",
    similarity_basis: ArrayLike | None = None,
    bootstrap_fraction: float = _BOOTSTRAP_FRACTION_DEFAULT,
    bootstrap_iterations: int = _BOOTSTRAP_ITERATIONS_DEFAULT,
    random_state: int = 42,
    enable_sweep: bool = True,
) -> ClusterMetricsResult:
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    coords3d = np.asarray(coords_3d, dtype=float) if coords_3d is not None else None
    if coords3d is not None and coords3d.ndim == 1:
        coords3d = coords3d.reshape(-1, 1)
    coords2d = np.asarray(coords_2d, dtype=float) if coords_2d is not None else None
    if coords2d is not None and coords2d.ndim == 1:
        coords2d = coords2d.reshape(-1, 1)
    sim_basis = np.asarray(similarity_basis, dtype=float) if similarity_basis is not None else None
    if sim_basis is not None and sim_basis.ndim == 1:
        sim_basis = sim_basis.reshape(-1, 1)

    summary = _summarise_cluster_result(cluster_result, matrix, coords3d, coords2d)
    labels = np.asarray(cluster_result.labels, dtype=int)

    stability_parts: dict[str, Any] = {}
    persistence = _collect_persistence(cluster_result)
    if persistence:
        stability_parts["persistence"] = persistence

    bootstrap = _bootstrap_stability(
        matrix,
        coords3d,
        sim_basis,
        labels,
        algo=algo,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        fraction=bootstrap_fraction,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )
    if bootstrap:
        stability_parts["bootstrap"] = bootstrap

    sweep = None
    if enable_sweep:
        sweep = _parameter_sweep(
            matrix,
            coords3d,
            coords2d,
            sim_basis,
            algo=algo,
            metric=metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            fraction=_SWEEP_VARIATION,
        )

    params = {
        "metric": metric,
        "min_cluster_size": int(min_cluster_size) if min_cluster_size is not None else None,
        "min_samples": int(min_samples) if min_samples is not None else None,
    }

    stability = stability_parts or None

    return ClusterMetricsResult(
        algo=algo,
        params=params,
        silhouette_embed=summary["silhouette_embed"],
        silhouette_feature=summary["silhouette_feature"],
        davies_bouldin=summary["davies_bouldin"],
        calinski_harabasz=summary["calinski_harabasz"],
        n_clusters=summary["n_clusters"],
        n_noise=summary["n_noise"],
        stability=stability,
        sweep=sweep,
    )


__all__ = ["ClusterMetricsResult", "compute_cluster_metrics"]
