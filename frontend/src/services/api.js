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
const envSchema = z.object({
    VITE_API_BASE_URL: z.string().default("http://localhost:8000"),
});
const env = envSchema.parse(import.meta.env);
const API_BASE_URL = env.VITE_API_BASE_URL.replace(/\/$/, "");
async function handleResponse(input, init) {
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
    return (await response.json());
}
export async function createRun(payload) {
    return handleResponse(`${API_BASE_URL}/run`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}
export async function sampleRun(runId, body) {
    await handleResponse(`${API_BASE_URL}/run/${runId}/sample`, {
        method: "POST",
        body: body ? JSON.stringify(body) : undefined,
    });
}
export async function fetchRunResults(runId) {
    return handleResponse(`${API_BASE_URL}/run/${runId}/results`, {
        method: "GET",
    });
}
export async function fetchRunMetrics(runId) {
    return handleResponse(`${API_BASE_URL}/run/${runId}/metrics`, {
        method: "GET",
    });
}
export async function fetchClusterMetrics(runId) {
    return handleResponse(`${API_BASE_URL}/run/${runId}/cluster-metrics`, {
        method: "GET",
    });
}
export async function recomputeClusters(runId, params) {
    const query = new URLSearchParams();
    if (typeof params.minClusterSize === "number" && Number.isFinite(params.minClusterSize)) {
        query.set("min_cluster_size", String(Math.max(2, Math.floor(params.minClusterSize))));
    }
    if (typeof params.minSamples === "number" && Number.isFinite(params.minSamples)) {
        query.set("min_samples", String(Math.max(1, Math.floor(params.minSamples))));
    }
    if (params.algo) {
        query.set("algo", params.algo);
    }
    const queryString = query.toString();
    const url = `${API_BASE_URL}/run/${runId}/clusters${queryString ? `?${queryString}` : ""}`;
    return handleResponse(url, { method: "GET" });
}
export async function fetchRun(runId) {
    return handleResponse(`${API_BASE_URL}/run/${runId}`, {
        method: "GET",
    });
}
export async function fetchRuns(limit = 20) {
    const params = new URLSearchParams({ limit: String(limit) });
    return handleResponse(`${API_BASE_URL}/run?${params.toString()}`);
}
export async function updateRun(runId, payload) {
    return handleResponse(`${API_BASE_URL}/run/${runId}`, {
        method: "PATCH",
        body: JSON.stringify(payload),
    });
}
function parseFilename(disposition) {
    if (!disposition) {
        return null;
    }
    const match = disposition.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
    if (!match) {
        return null;
    }
    const encoded = match[1] ?? match[2];
    if (!encoded) {
        return null;
    }
    try {
        return decodeURIComponent(encoded);
    }
    catch {
        return encoded;
    }
}
export async function exportRunData(options) {
    const params = new URLSearchParams();
    params.set("scope", options.scope);
    params.set("mode", options.mode);
    params.set("format", options.format);
    if (typeof options.clusterId === "number" && Number.isFinite(options.clusterId)) {
        params.set("cluster_id", String(options.clusterId));
    }
    if (options.include && options.include.length) {
        params.set("include", options.include.join(","));
    }
    if (options.viewport) {
        const { dimension, minX, maxX, minY, maxY, minZ, maxZ } = options.viewport;
        params.set("viewport_dim", dimension);
        params.set("viewport_min_x", String(minX));
        params.set("viewport_max_x", String(maxX));
        params.set("viewport_min_y", String(minY));
        params.set("viewport_max_y", String(maxY));
        if (typeof minZ === "number") {
            params.set("viewport_min_z", String(minZ));
        }
        if (typeof maxZ === "number") {
            params.set("viewport_max_z", String(maxZ));
        }
    }
    let method = "GET";
    let body;
    if (options.selectionIds && options.selectionIds.length) {
        body = JSON.stringify({ selection_ids: options.selectionIds });
        method = "POST";
    }
    const query = params.toString();
    const url = `${API_BASE_URL}/run/${options.runId}/export${query ? `?${query}` : ""}`;
    const response = await fetch(url, {
        method,
        headers: {
            "Content-Type": "application/json",
        },
        body,
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Failed to export dataset: ${response.status} ${text || response.statusText}`);
    }
    const blob = await response.blob();
    return {
        blob,
        filename: parseFilename(response.headers.get("content-disposition")),
        contentType: response.headers.get("content-type"),
    };
}
export async function fetchRunGraph(runId, options = {}) {
    const params = new URLSearchParams();
    if (options.mode) {
        params.set("mode", options.mode);
    }
    if (typeof options.k === "number" && Number.isFinite(options.k)) {
        params.set("k", String(Math.max(1, Math.round(options.k))));
    }
    if (typeof options.sim === "number" && Number.isFinite(options.sim)) {
        params.set("sim", String(Math.max(0, Math.min(1, options.sim))));
    }
    const suffix = params.toString();
    const url = suffix
        ? `${API_BASE_URL}/run/${runId}/graph?${suffix}`
        : `${API_BASE_URL}/run/${runId}/graph`;
    return handleResponse(url, { method: "GET" });
}
export async function fetchSegmentContext(segmentId, k = 8) {
    const params = new URLSearchParams({ k: String(Math.max(1, Math.round(k))) });
    const url = `${API_BASE_URL}/run/segments/${segmentId}/context?${params.toString()}`;
    return handleResponse(url, { method: "GET" });
}
export async function fetchProjectionVariant(runId, options) {
    const params = new URLSearchParams();
    params.set("method", options.method);
    params.set("mode", options.mode ?? "both");
    if (options.params) {
        params.set("params", JSON.stringify(options.params));
    }
    return handleResponse(`${API_BASE_URL}/run/${runId}/projection?${params.toString()}`, { method: "GET" });
}

export async function compareRuns(payload) {
    return handleResponse(`${API_BASE_URL}/compare`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

