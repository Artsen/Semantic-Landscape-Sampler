/**
 * Custom hook orchestrating run creation, sampling, fetching, and export interactions.
 *
 * Returns:
 *  - generate(): Create a run, trigger sampling, and hydrate the store with results.
 *  - exportDataset(options): Download the current run payload or fine-grained subset.
 *  - isLoading / error / results: Reactive state derived from the run store and mutation lifecycle.
 */

import { useMutation } from "@tanstack/react-query";
import { useCallback, useState } from "react";

import {
  createRun,
  exportRunData,
  fetchRun,
  fetchRunMetrics,
  fetchRunResults,
  sampleRun,
} from "@/services/api";
import type {
  CreateRunPayload,
  RunMetrics,
  RunResultsResponse,
  RunSummary,
  SampleRunBody,
  ExportFormat,
  ExportInclude,
  ExportMode,
  ExportScope,
  ExportRequestOptions,
} from "@/types/run";
import { useRunStore } from "@/store/runStore";

const POLL_INTERVAL_MS = 2000;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const toError = (error: unknown, fallbackMessage: string) =>
  error instanceof Error ? error : new Error(fallbackMessage);

export function useRunWorkflow() {
  const [error, setError] = useState<string | null>(null);
  const {
    prompt,
    systemPrompt,
    n,
    temperature,
    topP,
    model,
    seed,
    maxTokens,
    embeddingModel,
    umapNNeighbors,
    umapMinDist,
    umapMetric,
    umapSeed,
    preprocVersion,
    useCache,
    chunkSize,
    chunkOverlap,
    jitterToken,
    startGeneration,
    finishGeneration,
    setResults,
    results,
    currentRunId,
    applyRunSummary,
    setCurrentRunId,
    setHistoryOpen,
    setProgress,
    setRunMetrics,
    levelMode,
    selectedPointIds,
    selectedSegmentIds,
    viewportBounds,
    exportIncludeProvenance,
    exportFormat,
  } = useRunStore();

  const pushProgress = (resourceProgress: {
    progress_stage?: string | null;
    progress_message?: string | null;
    progress_percent?: number | null;
    progress_metadata?: Record<string, unknown> | null;
  }) => {
    setProgress({
      stage: resourceProgress.progress_stage ?? null,
      message: resourceProgress.progress_message ?? null,
      percent: resourceProgress.progress_percent ?? null,
      metadata: resourceProgress.progress_metadata ?? null,
    });
  };

  const updateRunUrl = useCallback((runId: string | null) => {
    if (typeof window === "undefined") {
      return;
    }
    const url = new URL(window.location.href);
    if (runId) {
      url.searchParams.set("run", runId);
    } else {
      url.searchParams.delete("run");
    }
    window.history.replaceState(null, "", url.toString());
  }, []);

  const waitForCompletion = async (runId: string) => {
    for (;;) {
      let resource;
      try {
        resource = await fetchRun(runId);
      } catch (err) {
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

  const runAndPoll = async (
    payload: CreateRunPayload,
    sampleBody: SampleRunBody,
  ): Promise<{
    runId: string;
    runResults: RunResultsResponse;
    runMetrics: RunMetrics | null;
  }> => {
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
    let metrics: RunMetrics | null = null;
    try {
      metrics = await fetchRunMetrics(runId);
    } catch (err) {
      console.warn("Failed to fetch run metrics", err);
    }

    return { runId, runResults, runMetrics: metrics };
  };

  const loadMutation = useMutation({
    mutationKey: ["load-run-from-history"],
    mutationFn: async (summary: RunSummary) => {
      const runResults = await fetchRunResults(summary.id);
      let metrics: RunMetrics | null = null;
      try {
        metrics = await fetchRunMetrics(summary.id);
      } catch (err) {
        console.warn("Failed to fetch run metrics", err);
      }
      return { summary, runResults, runMetrics: metrics };
    },
    onMutate: (summary) => {
      setError(null);
      setRunMetrics(null);
      if (summary) {
        updateRunUrl(summary.id);
      }
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
      updateRunUrl(summary.id);
      setHistoryOpen(false);
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error ? err.message : "Failed to load run";
      setError(message);
      finishGeneration("");
    },
  });

  const mutation = useMutation({
    mutationKey: ["generate-run"],
    mutationFn: async () => {
      const payload: CreateRunPayload = {
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
          n_neighbors: Math.min(
            Math.max(umapNNeighbors, Math.min(5, Math.max(2, n - 1))),
            Math.max(2, Math.min(200, n - 1)),
          ),
          min_dist: umapMinDist,
          metric: umapMetric,
          seed: umapSeed ?? undefined,
        },
      };

      const sampleBody: SampleRunBody = {
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
      updateRunUrl(runId);
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error ? err.message : "Failed to generate run";
      setError(message);
      finishGeneration("error");
    },
  });

  const duplicateMutation = useMutation({
    mutationKey: ["duplicate-run"],
    mutationFn: async (summary: RunSummary) => {
      const payload: CreateRunPayload = {
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

      const sampleBody: SampleRunBody = {
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
      updateRunUrl(runId);
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error ? err.message : "Failed to duplicate run";
      setError(message);
      finishGeneration("error");
    },
  });

  const generate = () => mutation.mutate();

  const loadFromHistory = (summary: RunSummary) => loadMutation.mutate(summary);

  const loadRunById = useCallback(
    async (runId: string) => {
      setError(null);
      setRunMetrics(null);
      startGeneration();
      try {
        const runResults = await fetchRunResults(runId);
        let metrics: RunMetrics | null = null;
        try {
          metrics = await fetchRunMetrics(runId);
        } catch (err) {
          console.warn("Failed to fetch run metrics", err);
        }

        const run = runResults.run;
        const summary: RunSummary = {
          id: run.id,
          prompt: run.prompt,
          n: run.n,
          model: run.model,
          chunk_size: run.chunk_size ?? null,
          chunk_overlap: run.chunk_overlap ?? null,
          system_prompt: run.system_prompt ?? null,
          embedding_model: run.embedding_model,
          preproc_version: run.preproc_version,
          use_cache: run.use_cache,
          cluster_algo: run.cluster_algo,
          hdbscan_min_cluster_size: run.hdbscan_min_cluster_size ?? null,
          hdbscan_min_samples: run.hdbscan_min_samples ?? null,
          umap: run.umap,
          temperature: run.temperature,
          top_p: run.top_p ?? null,
          seed: run.seed ?? null,
          max_tokens: run.max_tokens ?? null,
          status: run.status,
          created_at: run.created_at,
          updated_at: run.updated_at,
          response_count: runResults.points.length,
          segment_count: runResults.segments.length,
          notes: run.notes ?? null,
          progress_stage: run.progress_stage ?? null,
          progress_message: run.progress_message ?? null,
          progress_percent: run.progress_percent ?? null,
          progress_metadata: run.progress_metadata ?? null,
          processing_time_ms: run.processing_time_ms ?? null,
        };

        applyRunSummary(summary);
        setResults(runResults);
        setCurrentRunId(runId);
        if (metrics) {
          setRunMetrics(metrics);
        }
        finishGeneration(runId, runResults);
        updateRunUrl(runId);
        return { summary, runResults, runMetrics: metrics };
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed to load run";
        setError(message);
        finishGeneration("error");
        updateRunUrl(currentRunId ?? null);
        throw err;
      }
    },
    [applyRunSummary, currentRunId, finishGeneration, setCurrentRunId, setError, setResults, setRunMetrics, startGeneration, updateRunUrl],
  );

  const duplicateRun = (summary: RunSummary) =>
    duplicateMutation.mutate(summary);

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const deriveMode = (mode?: ExportMode): ExportMode => {
    if (mode) {
      return mode;
    }
    return levelMode === "segments" ? "segments" : "responses";
  };

  const deriveInclude = (include?: ExportInclude[]): ExportInclude[] | undefined => {
    if (include && include.length) {
      return include;
    }
    const tokens: ExportInclude[] = [];
    if (exportIncludeProvenance) {
      tokens.push("provenance");
    }
    return tokens.length ? tokens : undefined;
  };

  type ExportPayload = {
    scope: ExportScope;
    format?: ExportFormat;
    mode?: ExportMode;
    clusterId?: number;
    selectionIds?: string[];
    viewport?: ExportRequestOptions["viewport"];
    include?: ExportInclude[];
  };

  const exportDataset = async ({
    scope,
    format,
    mode,
    clusterId,
    selectionIds,
    viewport,
    include,
  }: ExportPayload) => {
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

    const request: ExportRequestOptions = {
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
        } else if (scope === "selection") {
          suffix = `${suffix}_${requestedSelection?.length ?? 0}`;
        } else if (scope === "viewport") {
          suffix = `${suffix}_${requestedViewport?.dimension ?? "2d"}`;
        }
        return `run_${runId}__${suffix}.${selectedFormat}`;
      })();
      downloadBlob(download.blob, filename);
    } catch (err) {
      throw toError(err, "Failed to export dataset");
    }
  };

  const isLoading =
    mutation.isPending || loadMutation.isPending || duplicateMutation.isPending;
  const isLoadingHistory = loadMutation.isPending;
  const isDuplicating = duplicateMutation.isPending;

  return {
    generate,
    exportDataset,
    loadFromHistory,
    duplicateRun,
    loadRunById,
    isLoading,
    isLoadingHistory,
    isDuplicating,
    error,
    results,
  };
}

export type RunWorkflow = ReturnType<typeof useRunWorkflow>;