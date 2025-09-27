/**
 * Shared TypeScript definitions mirroring backend run schemas for strong typing across the app.
 */

export type RunStatus = "pending" | "completed" | "failed";

export interface RunResource {
  id: string;
  prompt: string;
  n: number;
  model: string;
  temperature: number;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  notes?: string | null;
}

export interface RunSummary {
  id: string;
  prompt: string;
  n: number;
  model: string;
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
  prompt_similarity?: number | null;
  silhouette_score?: number | null;
  cluster?: number | null;
  probability?: number | null;
  similarity_to_centroid?: number | null;
  outlier_score?: number | null;
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

export interface ResponseHull {
  response_id: string;
  coords_2d: Array<[number, number]>;
  coords_3d: Array<[number, number, number]>;
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
  n: number;
}

export interface CreateRunPayload {
  prompt: string;
  n: number;
  model: string;
  temperature: number;
  top_p?: number | null;
  seed?: number | null;
  max_tokens?: number | null;
  notes?: string | null;
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
