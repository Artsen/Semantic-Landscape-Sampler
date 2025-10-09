# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Embedding cache with normalised hashing, float16 storage, vector norms, provider/revision tracking, and duplicate tagging when cache is disabled.
- Processing timeline telemetry capturing per-stage durations (sampling, segmentation, embeddings, UMAP, clustering, ANN, persistence) surfaced via API metrics and a new timeline panel in the UI.
- Run configuration options for cache usage, embedding model selection, preprocessing version, UMAP parameters (neighbours, min-dist, metric, seed), and cluster algorithm tuning.
- Run provenance records with runtime versions, BLAS/OpenMP info, library fingerprints, feature weights, seeds, commit SHA, and environment labels; provenance export and UI panel.
- Approximate nearest-neighbour index support (Annoy with hnswlib/FAISS fallbacks), simplified mutual-k graphs, neighbour queries, and persisted index metadata.
- Segment insights (top TF-IDF terms, neighbour previews, exemplar medoids, similarity metrics) powering enriched tooltips and detail panels.
- Fine-grained export scopes (run, cluster, selection, viewport) with schema versioning, optional provenance/vectors, and CSV/JSON/JSONL/Parquet streaming.
- Cluster metrics service producing silhouette (embedding + feature space), trustworthiness/continuity gauges, Davies-Bouldin, Calinski-Harabasz, and bootstrap stability summaries exposed via API/UI.
- UI controls for UMAP presets, trustworthiness/continuity gauges, ANN graph toggles, neighbour spokes, duplicate filters, cache badges, cluster tuning sliders, and provenance viewing.
- Model pricing utilities, system message and embedding selectors, and token/cost breakdown display in the metadata bar.

### Changed
- Shared centering & spread calculations now keep point clouds, similarity edges, parent threads, density meshes, and response hulls aligned when adjusting spread factors.
- Run history drawer, notes editor, and metadata bar updated to surface cache hit rates, cluster metrics, provenance, and recent run quick-load actions.
- Export pipeline refactored to stream large payloads efficiently and include provenance sidecars by default when requested.
- Documentation refreshed (README, CONTRIBUTING, notices) to reflect embedding cache, ANN graph, provenance, exports, and cluster metrics.

### Fixed
- `/run?limit=...` query joins no longer omit segments when run histories contain overlapping IDs.
- React-three-fiber warnings consolidated under local types; build still emits known upstream typing notices (tracked separately).

## [0.1.0] - 2025-09-20
### Added
- Initial release of Semantic Landscape Sampler with backend FastAPI service, frontend React visualiser, embeddings pipeline, clustering, and interactive point cloud.
