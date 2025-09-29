/**
 * Custom hook orchestrating run creation, sampling, fetching, and export interactions.
 *
 * Returns:
 *  - generate(): Create a run, trigger sampling, and hydrate the store with results.
 *  - exportRun(format): Download the current run payload as JSON or CSV.
 *  - isLoading / error / results: Reactive state derived from the run store and mutation lifecycle.
 */

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import {
  createRun,
  exportRunData,
  fetchRun,
  fetchRunResults,
  sampleRun,
} from "@/services/api";
import type { CreateRunPayload, RunSummary, SampleRunBody } from "@/types/run";
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

  const runAndPoll = async (payload: CreateRunPayload, sampleBody: SampleRunBody) => {
    const { run_id } = await createRun(payload);

    const waitPromise = waitForCompletion(run_id);
    const samplePromise = sampleRun(run_id, sampleBody);

    const [waitOutcome, sampleOutcome] = await Promise.allSettled([waitPromise, samplePromise]);

    if (sampleOutcome.status === "rejected") {
      console.error("Sampling request failed", sampleOutcome.reason);
    }
    if (waitOutcome.status === "rejected") {
      throw toError(waitOutcome.reason, "Run failed");
    }

    const runResults = await fetchRunResults(run_id);
    return { runId: run_id, runResults };
  };

  const loadMutation = useMutation({
    mutationKey: ["load-run-from-history"],
    mutationFn: async (summary: RunSummary) => {
      const runResults = await fetchRunResults(summary.id);
      return { summary, runResults };
    },
    onMutate: () => {
      setError(null);
      startGeneration();
    },
    onSuccess: ({ summary, runResults }) => {
      applyRunSummary(summary);
      setResults(runResults);
      setCurrentRunId(summary.id);
      finishGeneration(summary.id, runResults);
      setHistoryOpen(false);
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to load run";
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
      startGeneration();
    },
    onSuccess: ({ runId, runResults }) => {
      setResults(runResults);
      finishGeneration(runId, runResults);
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to generate run";
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
        temperature: summary.temperature,
        top_p: summary.top_p ?? undefined,
        seed: summary.seed ?? undefined,
        max_tokens: summary.max_tokens ?? undefined,
        chunk_size: summary.chunk_size ?? undefined,
        chunk_overlap: summary.chunk_overlap ?? undefined,
      };
      const sampleBody: SampleRunBody = { include_segments: true, include_discourse_tags: true };
      return runAndPoll(payload, sampleBody);
    },
    onMutate: (summary) => {
      setError(null);
      applyRunSummary(summary);
      startGeneration();
    },
    onSuccess: ({ runId, runResults }) => {
      setResults(runResults);
      finishGeneration(runId, runResults);
      setCurrentRunId(runId);
      setHistoryOpen(false);
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : "Failed to duplicate run";
      setError(message);
      finishGeneration("error");
    },
  });

  const triggerGenerate = () => {
    if (!prompt.trim()) {
      setError("Prompt cannot be empty.");
      return;
    }
    mutation.mutate();
  };

  const loadFromHistory = (summary: RunSummary) => {
    loadMutation.mutate(summary);
  };

  const duplicateRun = (summary: RunSummary) => {
    duplicateMutation.mutate(summary);
  };

  const handleExport = async (format: "json" | "csv") => {
    const runId = currentRunId;
    if (!runId) {
      throw new Error("No run to export yet");
    }
    const blob = await exportRunData(runId, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "semantic-landscape-" + runId + "." + format;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return {
    generate: triggerGenerate,
    exportRun: handleExport,
    isLoading: mutation.isPending || duplicateMutation.isPending,
    error,
    results,
    loadFromHistory,
    duplicateRun,
    isLoadingHistory: loadMutation.isPending,
    isDuplicating: duplicateMutation.isPending,
  };
}

export type RunWorkflow = ReturnType<typeof useRunWorkflow>;
