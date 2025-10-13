import numpy as np

from app.services.projection import compute_umap, compute_tsne, compute_pca_projection, run_hdbscan


def test_compute_umap_returns_zero_for_single_point():
    data = np.array([[0.1, -0.2, 0.3]], dtype=float)
    result = compute_umap(data)
    assert result.coords_3d.shape == (1, 3)
    assert np.allclose(result.coords_3d, 0.0)
    assert result.coords_2d.shape == (1, 2)
    assert np.allclose(result.coords_2d, 0.0)


def test_compute_umap_is_deterministic_with_seed():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(25, 8))
    first = compute_umap(data, random_state=42)
    second = compute_umap(data, random_state=42)
    assert np.allclose(first.coords_3d, second.coords_3d)
    assert np.allclose(first.coords_2d, second.coords_2d)


def test_hdbscan_flags_noise_when_insufficient_points():
    data = np.array([[0.0, 0.1], [0.05, 0.08]])
    result = run_hdbscan(data, min_cluster_size=5)
    assert (result.labels == -1).all()
    assert np.allclose(result.probabilities, 0.0)


def test_hdbscan_computes_similarity_and_centroids():
    vectors = np.array(
        [
            [0.5, 0.1, 0.0],
            [0.45, 0.15, 0.02],
            [-0.5, -0.1, 0.0],
            [-0.48, -0.05, -0.02],
            [0.51, 0.05, -0.02],
        ]
    )
    coords = np.column_stack((vectors, np.zeros(len(vectors))))
    result = run_hdbscan(vectors, coords_3d=coords, min_cluster_size=2)
    assert set(result.centroid_xyz.keys()) == {0, 1}
    assert all(-1 <= label <= 1 for label in result.labels)
    assert result.per_point_similarity.shape[0] == vectors.shape[0]
    assert result.silhouette_scores.shape[0] == vectors.shape[0]
    for label, centroid in result.centroid_xyz.items():
        assert centroid.shape == (3,)



def test_compute_tsne_shapes():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(30, 6))
    result = compute_tsne(data, random_state=42, perplexity=5.0, n_iter=250)
    assert result.coords_2d.shape == (30, 2)
    assert result.coords_3d.shape == (30, 3)



def test_compute_pca_projection_handles_degenerate_matrix():
    data = np.zeros((5, 0), dtype=float)
    result = compute_pca_projection(data)
    assert result.coords_2d.shape == (5, 2)
    assert result.coords_3d.shape == (5, 3)
    assert np.allclose(result.coords_2d, 0.0)
    assert np.allclose(result.coords_3d, 0.0)
