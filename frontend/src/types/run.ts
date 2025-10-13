/**
 * Shared TypeScript definitions mirroring backend run schemas for strong typing across the app.
 */

export type RunStatus = "pending" | "completed" | "failed";

export type UmapMetric = "cosine" | "euclidean" | "manhattan";

export interface UmapParams {
  n_neighbors: number;
  min_dist: number;
  metric: UmapMetric;
  seed?: number | null;
  seed_source: "default" | "ui" | "env" | string;
}

export interface UmapParamsRequest {
  n_neighbors?: number;
  min_dist?: number;
  metric?: UmapMetric;
  seed?: number | null;
}

export interface QualityGauge {
  trustworthiness_2d?: number | null;
  trustworthiness_3d?: number | null;
  continuity_2d?: number | null;
  continuity_3d?: number | null;
}

export type ProjectionMethod = "umap" | "tsne" | "pca";
export type ProjectionMode = "both" | "2d" | "3d";

export interface ProjectionVariantResponse {
  run_id: string;
  method: ProjectionMethod;
  requested_params: Record<string, unknown>;
  resolved_params: Record<string, unknown>;
  feature_version: string;
  point_count: number;
  total_count: number;
  is_subsample: boolean;
  subsample_strategy?: string | null;
  warnings: string[];
  response_ids: string[];
  coords_2d?: [number, number][];
  coords_3d?: [number, number, number][];
  trustworthiness_2d?: number | null;
  trustworthiness_3d?: number | null;
  continuity_2d?: number | null;
  continuity_3d?: number | null;
  from_cache: boolean;
  cached_at?: string | null;
}

export interface StageTiming {
  name: string;
  duration_ms: number;
  offset_ms?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface RunProvenance {
  run_id: string;
  created_at: string;
  python_version?: string | null;
  node_version?: string | null;
  blas_impl?: string | null;
  openmp_threads?: number | null;
  numba_version?: string | null;
  numba_target?: string | null;
  lib_versions: Record<string, unknown>;
  embedding_model?: string | null;
  embedding_dim?: number | null;
  llm_model?: string | null;
  temperature?: number | null;
  top_p?: number | null;
  max_tokens?: number | null;
  feature_weights: Record<string, unknown>;
  input_space: Record<string, unknown>;
  umap_params: Record<string, unknown>;
  cluster_params: Record<string, unknown>;
  commit_sha?: string | null;
  env_label?: string | null;
  random_state_seed_source?: string | null;
}

export type ExportScope = "run" | "cluster" | "selection" | "viewport";
export type ExportMode = "responses" | "segments";
export type ExportFormat = "json" | "jsonl" | "csv" | "parquet";
export type ExportInclude = "provenance" | "vectors" | "metadata";

export interface Viewport2D {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface Viewport3D extends Viewport2D {
  minZ: number;
  maxZ: number;
}

export interface ExportRequestOptions {
  runId: string;
  scope: ExportScope;
  mode: ExportMode;
  format: ExportFormat;
  clusterId?: number;
  selectionIds?: string[];
  include?: ExportInclude[];
  viewport?: {
    dimension: "2d" | "3d";
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    minZ?: number;
    maxZ?: number;
  };
}

export interface RunResource {
  id: string;
  prompt: string;
  n: number;
  model: string;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
  system_prompt?: string | null;
  embedding_model: string;
  preproc_version: string;
  use_cache: boolean;
  cluster_algo: string;
  hdbscan_min_cluster_size?: number | null;
  hdbscan_min_samples?: number | null;
  umap: UmapParams;
  temperature: number;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  notes?: string | null;
  progress_stage?: string | null;
  progress_message?: string | null;
  progress_percent?: number | null;
  progress_metadata?: Record<string, unknown> | null;
  processing_time_ms?: number | null;
  stage_timings?: StageTiming[];
}

export interface RunSummary {
  id: string;
  prompt: string;
  n: number;
  model: string;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
  system_prompt?: string | null;
  embedding_model: string;
  preproc_version: string;
  use_cache: boolean;
  cluster_algo: string;
  hdbscan_min_cluster_size?: number | null;
  hdbscan_min_samples?: number | null;
  umap: UmapParams;
  temperature: number;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  response_count: number;
  segment_count: number;
  notes?: string | null;
  progress_stage?: string | null;
  progress_message?: string | null;
  progress_percent?: number | null;
  progress_metadata?: Record<string, unknown> | null;
  processing_time_ms?: number | null;
}

export interface RunMetrics {
  run_id: string;
  total_segments: number;
  cached_segments: number;
  duplicate_segments: number;
  cache_hit_rate: number;
  processing_time_ms?: number | null;
  stage_timings: StageTiming[];
  silhouette_embed?: number | null;
  silhouette_feature?: number | null;
  davies_bouldin?: number | null;
  calinski_harabasz?: number | null;
  n_clusters?: number | null;
  n_noise?: number | null;
}

export interface UsageInfo {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
}

export interface ResponsePoint {
  id: string;
  index: number;
  text_preview: string;
  full_text: string;
  tokens?: number | null;
  finish_reason?: string | null;
  usage?: UsageInfo | null;
  cluster?: number | null;
  probability?: number | null;
  similarity_to_centroid?: number | null;
  outlier_score?: number | null;
  hidden?: boolean;
   prompt_tokens?: number | null;
   completion_tokens?: number | null;
   embedding_tokens?: number | null;
  completion_cost?: number | null;
  embedding_cost?: number | null;
  total_cost?: number | null;
  coords_3d: [number, number, number];
  coords_2d: [number, number];
}

export interface SegmentPoint {
  id: string;
  response_id: string;
  response_index: number;
  position: number;
  text: string;
  role?: string | null;
  tokens?: number | null;
  embedding_tokens?: number | null;
  embedding_cost?: number | null;
  prompt_similarity?: number | null;
  silhouette_score?: number | null;
  cluster?: number | null;
  probability?: number | null;
  similarity_to_centroid?: number | null;
  outlier_score?: number | null;
  text_hash?: string | null;
  is_cached: boolean;
  is_duplicate: boolean;
  simhash64?: number | null;
  coords_3d: [number, number, number];
  coords_2d: [number, number];
}

export interface ClusterSummary {
  label: number;
  size: number;
  centroid_xyz: [number, number, number];
  exemplar_ids: string[];
  average_similarity?: number | null;
  method: string;
  keywords: string[];
  noise: boolean;
}

export interface SegmentClusterSummary {
  label: number;
  size: number;
  exemplar_ids: string[];
  average_similarity?: number | null;
  method: string;
  keywords: string[];
  theme?: string | null;
  noise: boolean;
}

export interface SegmentEdge {
  source_id: string;
  target_id: string;
  score: number;
}

export interface SegmentGraphEdge {
  source: string;
  target: string;
  similarity: number;
}

export interface SegmentGraphResponse {
  mode: "full" | "simplified";
  edges: SegmentGraphEdge[];
  auto_simplified: boolean;
  k: number;
  threshold: number;
  node_count: number;
}

export interface SegmentTopTerm {
  term: string;
  weight: number;
}

export interface SegmentNeighborPreview {
  id: string;
  similarity: number;
  text: string;
  cluster: number | null;
}

export interface SegmentContextMetrics {
  sim_to_exemplar?: number | null;
  sim_to_nn?: number | null;
}

export interface SegmentContextResponse {
  segment_id: string;
  cluster_id?: number | null;
  cluster_exemplar_id?: string | null;
  exemplar_preview?: string | null;
  top_terms: SegmentTopTerm[];
  neighbors: SegmentNeighborPreview[];
  why_here: SegmentContextMetrics;
  preview: string;
}

export interface ResponseHull {
  response_id: string;
  coords_2d: Array<[number, number]>;
  coords_3d: Array<[number, number, number]>;
}

export interface RunCostSummary {
  model: string;
  embedding_model: string;
  completion_input_tokens: number;
  completion_output_tokens: number;
  completion_cost: number;
  embedding_tokens: number;
  embedding_cost: number;
  total_cost: number;
}

export interface ClusterStabilityBootstrap {
  mode: "bootstrap";
  fraction: number;
  iterations: number;
  clusters: Record<string, { mean: number; std: number; samples: number }>;
}

export interface ClusterStabilitySummary {
  persistence?: Record<string, number> | null;
  bootstrap?: ClusterStabilityBootstrap | null;
}

export interface ClusterParameterSweepPoint {
  min_cluster_size: number;
  min_samples: number;
  algo: string;
  n_clusters?: number | null;
  n_noise?: number | null;
  silhouette_feature?: number | null;
  silhouette_embed?: number | null;
  davies_bouldin?: number | null;
  calinski_harabasz?: number | null;
}

export interface ClusterParameterSweep {
  baseline: { min_cluster_size: number; min_samples: number | null };
  points: ClusterParameterSweepPoint[];
}

export interface ClusterMetricsSummary {
  algo: string;
  params: Record<string, unknown>;
  silhouette_embed?: number | null;
  silhouette_feature?: number | null;
  davies_bouldin?: number | null;
  calinski_harabasz?: number | null;
  n_clusters?: number | null;
  n_noise?: number | null;
  stability?: ClusterStabilitySummary | null;
  sweep?: ClusterParameterSweep | null;
  created_at: string;
}

export interface ClusterMetricsResponse extends ClusterMetricsSummary {
  run_id: string;
}

export interface RunResultsResponse {
  run: RunResource;
  points: ResponsePoint[];
  clusters: ClusterSummary[];
  segments: SegmentPoint[];
  segment_clusters: SegmentClusterSummary[];
  segment_edges: SegmentEdge[];
  response_hulls: ResponseHull[];
  prompt: string;
  model: string;
  system_prompt?: string | null;
  embedding_model: string;
  preproc_version: string;
  n: number;
  costs: RunCostSummary;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
  umap: UmapParams;
  quality: QualityGauge;
  projection_quality?: Partial<Record<ProjectionMethod, QualityGauge>>;
  cluster_metrics?: ClusterMetricsSummary | null;
  provenance?: RunProvenance | Record<string, unknown> | null;
}

export type ComparisonMode = "segments" | "responses";

export interface MovementHistogramBin {
  start: number;
  end: number;
  count: number;
}

export interface ComparisonTransformComponent {
  dimension: "2d" | "3d";
  rotation: number[][];
  scale: number;
  translation: number[];
  rmsd?: number | null;
  anchor_count: number;
}

export interface ComparisonTransformSummary {
  method: string;
  anchor_kind: string;
  anchor_count: number;
  components: Record<string, ComparisonTransformComponent>;
}

export interface ComparisonLink {
  left_segment_id: string;
  right_segment_id: string;
  link_type: "exact_hash" | "nn";
  distance?: number | null;
  text_hash?: string | null;
  movement_vector?: [number, number, number] | null;
  movement_distance?: number | null;
}

export interface ComparisonPoint {
  id: string;
  run_id: string;
  response_id: string;
  response_index?: number | null;
  segment_position?: number | null;
  source: "left" | "right";
  text_hash?: string | null;
  cluster_label?: number | null;
  cluster_probability?: number | null;
  cluster_similarity?: number | null;
  role?: string | null;
  prompt_similarity?: number | null;
  text_preview?: string | null;
  coords_3d: [number, number, number];
  coords_2d: [number, number];
  aligned_coords_3d: [number, number, number];
  aligned_coords_2d: [number, number];
  is_anchor: boolean;
}

export interface ComparisonMetrics {
  shared_segment_count: number;
  linked_segment_count: number;
  alignment_method: string;
  ari?: number | null;
  nmi?: number | null;
  mean_movement?: number | null;
  median_movement?: number | null;
  max_movement?: number | null;
  movement_histogram: MovementHistogramBin[];
  cluster_count_left: number;
  cluster_count_right: number;
  delta_cluster_count: number;
  mean_cluster_size_left?: number | null;
  mean_cluster_size_right?: number | null;
  median_cluster_size_left?: number | null;
  median_cluster_size_right?: number | null;
  noise_left: number;
  noise_right: number;
  top_term_shifts: {
    cluster_label: number;
    left_terms: string[];
    right_terms: string[];
    shared_terms: string[];
    jaccard?: number | null;
  }[];
}

export interface CompareRunsResponse {
  run_pair_id?: string | null;
  left_run: RunResource;
  right_run: RunResource;
  transforms: ComparisonTransformSummary;
  points: ComparisonPoint[];
  links: ComparisonLink[];
  metrics: ComparisonMetrics;
  anchor_kind: string;
}

export interface CompareRunsRequest {
  left_run_id: string;
  right_run_id: string;
  mode?: ComparisonMode;
  min_shared?: number;
  save?: boolean;
  histogram_bins?: number;
  max_links?: number;
}

export interface CreateRunPayload {
  prompt: string;
  n: number;
  model: string;
  temperature: number;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  chunk_size?: number | null;
  chunk_overlap?: number | null;
  system_prompt?: string | null;
  embedding_model?: string;
  preproc_version?: string | null;
  use_cache?: boolean;
  notes?: string | null;
  umap?: UmapParamsRequest;
}

export interface CreateRunResponse {
  run_id: string;
}

export interface SampleRunBody {
  jitter_prompt_token?: string | null;
  force_refresh?: boolean;
  overwrite_previous?: boolean;
  include_segments?: boolean;
  include_discourse_tags?: boolean;
}

export interface UpdateRunPayload {
  notes?: string | null;
}

