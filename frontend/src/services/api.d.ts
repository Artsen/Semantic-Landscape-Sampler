/**
 * REST client helpers for the backend run endpoints.
 *
 * Functions:
 *  - createRun(payload): POST /run.
 *  - sampleRun(runId, body): POST /run/{id}/sample.
 *  - fetchRunResults(runId): GET /run/{id}/results.
 *  - exportRunData(runId, format): Download the export payload as a blob.
 */
import type { CreateRunPayload, CreateRunResponse, ExportRequestOptions, RunMetrics, ClusterMetricsResponse, RunResource, RunResultsResponse, RunSummary, SampleRunBody, ProjectionVariantResponse, ProjectionMode, ProjectionMethod, SegmentContextResponse, SegmentGraphResponse, UpdateRunPayload, CompareRunsRequest, CompareRunsResponse } from "@/types/run";
export declare function createRun(payload: CreateRunPayload): Promise<CreateRunResponse>;
export declare function sampleRun(runId: string, body?: SampleRunBody): Promise<void>;
export declare function fetchRunResults(runId: string): Promise<RunResultsResponse>;
export declare function fetchRunMetrics(runId: string): Promise<RunMetrics>;
export declare function fetchClusterMetrics(runId: string): Promise<ClusterMetricsResponse>;
export interface ReclusterParams {
    minClusterSize?: number;
    minSamples?: number;
    algo?: "hdbscan" | "kmeans";
}
export declare function recomputeClusters(runId: string, params: ReclusterParams): Promise<RunResultsResponse>;
export declare function fetchRun(runId: string): Promise<RunResource>;
export declare function fetchRuns(limit?: number): Promise<RunSummary[]>;
export declare function updateRun(runId: string, payload: UpdateRunPayload): Promise<RunResource>;
export interface ExportDownload {
    blob: Blob;
    filename: string | null;
    contentType: string | null;
}
export declare function exportRunData(options: ExportRequestOptions): Promise<ExportDownload>;
export declare function fetchRunGraph(runId: string, options?: {
    mode?: "full" | "simplified";
    k?: number;
    sim?: number;
}): Promise<SegmentGraphResponse>;
export declare function fetchSegmentContext(segmentId: string, k?: number): Promise<SegmentContextResponse>;
export declare function fetchProjectionVariant(runId: string, options: {
    method: ProjectionMethod;
    mode?: ProjectionMode;
    params?: Record<string, unknown>;
}): Promise<ProjectionVariantResponse>;
export declare function compareRuns(payload: CompareRunsRequest): Promise<CompareRunsResponse>;

