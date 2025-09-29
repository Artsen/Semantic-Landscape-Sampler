/**
 * REST client helpers for the backend run endpoints.
 *
 * Functions:
 *  - createRun(payload): POST /run.
 *  - sampleRun(runId, body): POST /run/{id}/sample.
 *  - fetchRunResults(runId): GET /run/{id}/results.
 *  - exportRunData(runId, format): Download the export payload as a blob.
 */

import { z } from "zod";

import type {
  CreateRunPayload,
  CreateRunResponse,
  RunResource,
  RunResultsResponse,
  RunSummary,
  SampleRunBody,
  UpdateRunPayload,
} from "@/types/run";

const envSchema = z.object({
  VITE_API_BASE_URL: z.string().default("http://localhost:8000"),
});

const env = envSchema.parse(import.meta.env);

const API_BASE_URL = env.VITE_API_BASE_URL.replace(/\/$/, "");

async function handleResponse<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API error ${response.status}: ${body || response.statusText}`);
  }

  return (await response.json()) as T;
}

export async function createRun(payload: CreateRunPayload): Promise<CreateRunResponse> {
  return handleResponse<CreateRunResponse>(`${API_BASE_URL}/run`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function sampleRun(runId: string, body?: SampleRunBody): Promise<void> {
  await handleResponse(`${API_BASE_URL}/run/${runId}/sample`, {
    method: "POST",
    body: body ? JSON.stringify(body) : undefined,
  });
}

export async function fetchRunResults(runId: string): Promise<RunResultsResponse> {
  return handleResponse<RunResultsResponse>(`${API_BASE_URL}/run/${runId}/results`, {
    method: "GET",
  });
}

export async function fetchRun(runId: string): Promise<RunResource> {
  return handleResponse<RunResource>(`${API_BASE_URL}/run/${runId}`, {
    method: "GET",
  });
}

export async function fetchRuns(limit = 20): Promise<RunSummary[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  return handleResponse<RunSummary[]>(`${API_BASE_URL}/run?${params.toString()}`);
}

export async function updateRun(runId: string, payload: UpdateRunPayload): Promise<RunResource> {
  return handleResponse<RunResource>(`${API_BASE_URL}/run/${runId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function exportRunData(runId: string, format: "json" | "csv" = "json"): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/run/${runId}/export.${format}`);
  if (!response.ok) {
    throw new Error(`Failed to export run: ${response.statusText}`);
  }
  return await response.blob();
}
