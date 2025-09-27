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

import { createRun, exportRunData, fetchRunResults, sampleRun } from "@/services/api";
import type { CreateRunPayload, RunSummary } from "@/types/run";
import { useRunStore } from "@/store/runStore";

export function useRunWorkflow() {
  const [error, setError] = useState<string | null>(null);
  const {
    prompt,
    n,
    temperature,
    topP,
    model,
    seed,
    maxTokens,
    jitterToken,
    startGeneration,
    finishGeneration,
    setResults,
    results,
    currentRunId,
    applyRunSummary,
    setCurrentRunId,
    setHistoryOpen,
  } = useRunStore();

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
      };
      const { run_id } = await createRun(payload);
      await sampleRun(run_id, {
        jitter_prompt_token: jitterToken ?? undefined,
        include_segments: true,
        include_discourse_tags: true,
      });
      const runResults = await fetchRunResults(run_id);
      return { runId: run_id, runResults };
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
        n: summary.n,
        model: summary.model,
        temperature: summary.temperature,
        top_p: summary.top_p ?? undefined,
        seed: summary.seed ?? undefined,
        max_tokens: summary.max_tokens ?? undefined,
      };
      const { run_id } = await createRun(payload);
      await sampleRun(run_id, {
        include_segments: true,
        include_discourse_tags: true,
      });
      const runResults = await fetchRunResults(run_id);
      return { runId: run_id, runResults };
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
    link.download = `semantic-landscape-${runId}.${format}`;
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
