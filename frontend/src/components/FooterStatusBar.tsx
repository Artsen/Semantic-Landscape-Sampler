import { useMemo } from "react";

import { useRunStore } from "@/store/runStore";
import { formatDuration } from "@/utils/formatDuration";

export function FooterStatusBar() {
  const { runMetrics, results, isGenerating, projectionMethod } = useRunStore((state) => ({
    runMetrics: state.runMetrics,
    results: state.results,
    isGenerating: state.isGenerating,
    projectionMethod: state.projectionMethod,
  }));

  const processingMs = useMemo(() => {
    if (runMetrics?.processing_time_ms != null) {
      return runMetrics.processing_time_ms;
    }
    if (results?.run?.processing_time_ms != null) {
      return results.run.processing_time_ms;
    }
    return null;
  }, [runMetrics?.processing_time_ms, results?.run?.processing_time_ms]);

  const cacheHit = runMetrics ? `${runMetrics.cache_hit_rate.toFixed(1)}%` : "--";
  const seed = results?.run?.seed != null ? `Seed ${results.run.seed}` : "Seed --";
  const algo = results?.run?.cluster_algo ? results.run.cluster_algo.toUpperCase() : "Algo --";
  const status = isGenerating ? "Sampling responses" : results?.run?.status ?? "Idle";

  const stageTimings = runMetrics?.stage_timings ?? results?.run?.stage_timings ?? [];
  const timeline = stageTimings.length
    ? stageTimings
        .map((stage) => `${stage.name.replace(/-/g, " ")}: ${formatDuration(stage.duration_ms ?? 0)}`)
        .join(" • ")
    : "--";

  return (
    <footer className="flex h-8 items-center gap-4 border-t border-border bg-panel px-4 text-xs text-text-dim">
      <span className="rounded-full border border-border px-2 py-[1px] text-text">{status}</span>
      <span>Cache {cacheHit}</span>
      <span>{seed}</span>
      <span>{algo}</span>
      <span>{projectionMethod.toUpperCase()}</span>
      {processingMs != null ? <span>Total {formatDuration(processingMs)}</span> : null}
      <span className="truncate">{timeline}</span>
    </footer>
  );
}

