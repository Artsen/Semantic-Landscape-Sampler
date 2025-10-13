/**
 * Custom hook orchestrating run creation, sampling, fetching, and export interactions.
 *
 * Returns:
 *  - generate(): Create a run, trigger sampling, and hydrate the store with results.
 *  - exportDataset(options): Download the current run payload or fine-grained subset.
 *  - isLoading / error / results: Reactive state derived from the run store and mutation lifecycle.
 */
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { createRun, exportRunData, fetchRun, fetchRunMetrics, fetchRunResults, sampleRun, } from "@/services/api";
import { useRunStore } from "@/store/runStore";
const POLL_INTERVAL_MS = 2000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const toError = (error, fallbackMessage) => error instanceof Error ? error : new Error(fallbackMessage);
export function useRunWorkflow() {
    const [error, setError] = useState(null);
    const { prompt, systemPrompt, n, temperature, topP, model, seed, maxTokens, embeddingModel, umapNNeighbors, umapMinDist, umapMetric, umapSeed, preprocVersion, useCache, chunkSize, chunkOverlap, jitterToken, startGeneration, finishGeneration, setResults, results, currentRunId, applyRunSummary, setCurrentRunId, setHistoryOpen, setProgress, setRunMetrics, levelMode, selectedPointIds, selectedSegmentIds, viewportBounds, exportIncludeProvenance, exportFormat, } = useRunStore();
    const pushProgress = (resourceProgress) => {
        setProgress({
            stage: resourceProgress.progress_stage ?? null,
            message: resourceProgress.progress_message ?? null,
            percent: resourceProgress.progress_percent ?? null,
            metadata: resourceProgress.progress_metadata ?? null,
        });
    };
    const waitForCompletion = async (runId) => {
        for (;;) {
            let resource;
            try {
                resource = await fetchRun(runId);
            }
            catch (err) {
                throw toError(err, "Failed to fetch run progress");
            }
            pushProgress(resource);
            if (resource.status === "completed") {
                return;
            }
            if (resource.status === "failed") {
                throw new Error(resource.error_message ?? "Run failed");
            }
            await sleep(POLL_INTERVAL_MS);
        }
    };
    const runAndPoll = async (payload, sampleBody) => {
        const createResponse = await createRun(payload);
        const runId = createResponse.run_id;
        const waitPromise = waitForCompletion(runId);
        const samplePromise = sampleRun(runId, sampleBody);
        const [waitOutcome, sampleOutcome] = await Promise.allSettled([
            waitPromise,
            samplePromise,
        ]);
        if (sampleOutcome.status === "rejected") {
            console.error("Sampling request failed", sampleOutcome.reason);
        }
        if (waitOutcome.status === "rejected") {
            throw toError(waitOutcome.reason, "Run failed");
        }
        const runResults = await fetchRunResults(runId);
        let metrics = null;
        try {
            metrics = await fetchRunMetrics(runId);
        }
        catch (err) {
            console.warn("Failed to fetch run metrics", err);
        }
        return { runId, runResults, runMetrics: metrics };
    };
    const loadMutation = useMutation({
        mutationKey: ["load-run-from-history"],
        mutationFn: async (summary) => {
            const runResults = await fetchRunResults(summary.id);
            let metrics = null;
            try {
                metrics = await fetchRunMetrics(summary.id);
            }
            catch (err) {
                console.warn("Failed to fetch run metrics", err);
            }
            return { summary, runResults, runMetrics: metrics };
        },
        onMutate: () => {
            setError(null);
            setRunMetrics(null);
            startGeneration();
        },
        onSuccess: ({ summary, runResults, runMetrics }) => {
            applyRunSummary(summary);
            setResults(runResults);
            setCurrentRunId(summary.id);
            if (runMetrics) {
                setRunMetrics(runMetrics);
            }
            finishGeneration(summary.id, runResults);
            setHistoryOpen(false);
        },
        onError: (err) => {
            const message = err instanceof Error ? err.message : "Failed to load run";
            setError(message);
            finishGeneration("");
        },
    });
    const mutation = useMutation({
        mutationKey: ["generate-run"],
        mutationFn: async () => {
            const payload = {
                prompt,
                n,
                model,
                temperature,
                top_p: topP,
                seed: seed ?? undefined,
                max_tokens: maxTokens ?? undefined,
                chunk_size: chunkSize ?? undefined,
                chunk_overlap: chunkOverlap ?? undefined,
                system_prompt: systemPrompt.trim() ? systemPrompt : undefined,
                embedding_model: embeddingModel,
                preproc_version: preprocVersion ?? undefined,
                use_cache: useCache,
                umap: {
                    n_neighbors: Math.min(Math.max(umapNNeighbors, Math.min(5, Math.max(2, n - 1))), Math.max(2, Math.min(200, n - 1))),
                    min_dist: umapMinDist,
                    metric: umapMetric,
                    seed: umapSeed ?? undefined,
                },
            };
            const sampleBody = {
                jitter_prompt_token: jitterToken ?? undefined,
                include_segments: true,
                include_discourse_tags: true,
            };
            return runAndPoll(payload, sampleBody);
        },
        onMutate: () => {
            setError(null);
            setRunMetrics(null);
            startGeneration();
        },
        onSuccess: ({ runId, runResults, runMetrics }) => {
            setResults(runResults);
            finishGeneration(runId, runResults);
            if (runMetrics) {
                setRunMetrics(runMetrics);
            }
        },
        onError: (err) => {
            const message = err instanceof Error ? err.message : "Failed to generate run";
            setError(message);
            finishGeneration("error");
        },
    });
    const duplicateMutation = useMutation({
        mutationKey: ["duplicate-run"],
        mutationFn: async (summary) => {
            const payload = {
                prompt: summary.prompt,
                system_prompt: summary.system_prompt ?? undefined,
                n: summary.n,
                model: summary.model,
                embedding_model: summary.embedding_model,
                preproc_version: summary.preproc_version ?? preprocVersion ?? undefined,
                use_cache: summary.use_cache ?? useCache,
                temperature: summary.temperature,
                top_p: summary.top_p ?? undefined,
                seed: summary.seed ?? undefined,
                max_tokens: summary.max_tokens ?? undefined,
                chunk_size: summary.chunk_size ?? undefined,
                chunk_overlap: summary.chunk_overlap ?? undefined,
            };
            const sampleBody = {
                include_segments: true,
                include_discourse_tags: true,
            };
            return runAndPoll(payload, sampleBody);
        },
        onMutate: (summary) => {
            setError(null);
            applyRunSummary(summary);
            setRunMetrics(null);
            startGeneration();
        },
        onSuccess: ({ runId, runResults, runMetrics }) => {
            setResults(runResults);
            setCurrentRunId(runId);
            if (runMetrics) {
                setRunMetrics(runMetrics);
            }
            finishGeneration(runId, runResults);
        },
        onError: (err) => {
            const message = err instanceof Error ? err.message : "Failed to duplicate run";
            setError(message);
            finishGeneration("error");
        },
    });
    const generate = () => mutation.mutate();
    const loadFromHistory = (summary) => loadMutation.mutate(summary);
    const duplicateRun = (summary) => duplicateMutation.mutate(summary);
    const downloadBlob = (blob, filename) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };
    const deriveMode = (mode) => {
        if (mode) {
            return mode;
        }
        return levelMode === "segments" ? "segments" : "responses";
    };
    const deriveInclude = (include) => {
        if (include && include.length) {
            return include;
        }
        const tokens = [];
        if (exportIncludeProvenance) {
            tokens.push("provenance");
        }
        return tokens.length ? tokens : undefined;
    };
    const exportDataset = async ({ scope, format, mode, clusterId, selectionIds, viewport, include, }) => {
        const runId = currentRunId ?? results?.run.id;
        if (!runId) {
            throw new Error("No run results available for export");
        }
        const resolvedMode = deriveMode(mode);
        let requestedSelection = selectionIds;
        if (!requestedSelection && scope === "selection") {
            requestedSelection =
                resolvedMode === "segments" ? selectedSegmentIds : selectedPointIds;
        }
        if (scope === "selection" && (!requestedSelection || !requestedSelection.length)) {
            throw new Error("No selection to export");
        }
        if (scope === "cluster" && (clusterId == null || Number.isNaN(clusterId))) {
            throw new Error("Cluster exports require a cluster id");
        }
        let requestedViewport = viewport;
        if (scope === "viewport") {
            if (!requestedViewport && viewportBounds) {
                requestedViewport = {
                    dimension: viewportBounds.dimension,
                    minX: viewportBounds.minX,
                    maxX: viewportBounds.maxX,
                    minY: viewportBounds.minY,
                    maxY: viewportBounds.maxY,
                    ...(viewportBounds.dimension === "3d"
                        ? { minZ: viewportBounds.minZ, maxZ: viewportBounds.maxZ }
                        : {}),
                };
            }
            if (!requestedViewport) {
                throw new Error("Viewport bounds unavailable for export");
            }
        }
        const includeTokens = deriveInclude(include);
        const selectedFormat = format ?? exportFormat;
        const request = {
            runId,
            scope,
            format: selectedFormat,
            mode: resolvedMode,
            ...(clusterId != null ? { clusterId } : {}),
            ...(requestedSelection ? { selectionIds: requestedSelection } : {}),
            ...(requestedViewport ? { viewport: requestedViewport } : {}),
            ...(includeTokens ? { include: includeTokens } : {}),
        };
        try {
            const download = await exportRunData(request);
            const filename = download.filename ?? (() => {
                let suffix = scope;
                if (scope === "cluster" && clusterId != null) {
                    suffix = `${suffix}_${clusterId}`;
                }
                else if (scope === "selection") {
                    suffix = `${suffix}_${requestedSelection?.length ?? 0}`;
                }
                else if (scope === "viewport") {
                    suffix = `${suffix}_${requestedViewport?.dimension ?? "2d"}`;
                }
                return `run_${runId}__${suffix}.${selectedFormat}`;
            })();
            downloadBlob(download.blob, filename);
        }
        catch (err) {
            throw toError(err, "Failed to export dataset");
        }
    };
    const isLoading = mutation.isPending || loadMutation.isPending || duplicateMutation.isPending;
    return {
        generate,
        exportDataset,
        loadFromHistory,
        duplicateRun,
        isLoading,
        error,
        results,
    };
}
