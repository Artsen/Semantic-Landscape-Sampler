/**
 * Displays metadata badges for the currently loaded run.
 */

import { memo, useMemo } from "react";

import { useRunStore } from "@/store/runStore";

const badgeHints: Record<string, string> = {
  Model: "Chat completion model used to generate the answers.",
  Embedding: "Embedding model used to place responses in the map.",
  Chunk: "Word window used when slicing responses into smaller segments.",
  Temp: "Sampling temperature controlling randomness (0 = focused, 1 = exploratory).",
  'Top-p': "Cumulative probability threshold for token sampling.",
  Seed: "Seed passed to the LLM for reproducible sampling when supported.",
  'Max tokens': "Maximum number of completion tokens requested per response.",
  'UMAP n': "Neighbour count steering how local or global the projection will be.",
  'UMAP dist': "Target distance between close points in the 2D/3D layout.",
  'UMAP metric': "Similarity measure used when building the projection.",
  'UMAP seed': "Random seed for the UMAP layout to keep runs repeatable.",
  Samples: "Number of completions requested for this run.",
  Status: "Latest backend status for this run.",
  'Total cost': "Estimated combined completion and embedding cost in USD.",
  Processing: "Total time to generate this map. Hover for a breakdown by stage.",
  'Cache Hit': "Share of segments served from the local embedding cache.",
  Cached: "Count of cached segments over the total processed segments.",
  Duplicates: "Segments flagged as duplicates within this run."
};

const stageLabels: Record<string, string> = {
  'prepare-run': 'Preparation',
  'request-completions': 'LLM sampling',
  'persist-responses': 'Persist responses',
  'segment-responses': 'Segment responses',
  'discourse-tagging': 'Discourse tagging',
  'embed-responses': 'Embed responses',
  'embed-segments': 'Embed segments',
  'segment-analysis': 'Segment analysis',
  'response-analysis': 'Response analysis',
  'persist-artifacts': 'Persist artifacts',
};

const gaugeHints: Record<string, string> = {
  'Trust 2D': "How well the 2D layout preserves neighbourhoods from the embedding space.",
  'Trust 3D': "Neighbourhood preservation for the 3D projection.",
  'Cont. 2D': "Continuity score indicating how few distant points were forced together in 2D.",
  'Cont. 3D': "Continuity score for the 3D projection."
};

const badgeClass = "inline-flex items-center gap-1 rounded-full border border-slate-700/60 bg-slate-900/70 px-2 py-[2px] text-[10px] uppercase tracking-wide text-slate-300";
const dateFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

const formatDuration = (ms?: number | null): string => {
  if (ms == null || Number.isNaN(ms)) {
    return '--';
  }
  if (ms >= 1000) {
    const seconds = ms / 1000;
    return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
  }
  if (ms >= 100) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${ms.toFixed(1)}ms`;
};

export const RunMetadataBar = memo(function RunMetadataBar() {
  const { results, runMetrics, projectionMethod, projectionVariants, projectionWarnings, isProjectionLoading } =
    useRunStore((state) => ({
      results: state.results,
      runMetrics: state.runMetrics,
      projectionMethod: state.projectionMethod,
      projectionVariants: state.projectionVariants,
      projectionWarnings: state.projectionWarnings,
      isProjectionLoading: state.isProjectionLoading,
    }));
  const run = results?.run;
  const quality = projectionVariants?.[projectionMethod]?.quality ?? results?.quality;
  const projectionState = projectionVariants?.[projectionMethod];
  const projectionMetadata = projectionState?.metadata;

  const badges = useMemo(() => {
    if (!run) {
      return [] as Array<{ label: string; value: string; title?: string }>;
    }
    const items: Array<{ label: string; value: string; title?: string }> = [
      { label: "Model", value: run.model },
      { label: "Embedding", value: run.embedding_model },
      { label: "Chunk", value: `${run.chunk_size ?? 3} w` },
      { label: "Temp", value: run.temperature.toFixed(2) },
    ];
    if (run.top_p != null) {
      items.push({ label: "Top-p", value: run.top_p.toFixed(2) });
    }
    if (run.seed != null) {
      items.push({ label: "Seed", value: String(run.seed) });
    }
    if (run.max_tokens != null) {
      items.push({ label: "Max tokens", value: String(run.max_tokens) });
    }
    items.push({ label: "UMAP n", value: String(run.umap.n_neighbors) });
    items.push({ label: "UMAP dist", value: run.umap.min_dist.toFixed(2) });
    items.push({ label: "UMAP metric", value: run.umap.metric });
    if (run.umap.seed != null) {
      items.push({ label: "UMAP seed", value: String(run.umap.seed) });
    }

    const projectionLabel = projectionMethod.toUpperCase();
    const projectionTitle: string[] = [];
    if (projectionMetadata) {
      projectionTitle.push(
        `Source: ${projectionMetadata.fromCache ? "cached" : "fresh"} - ${projectionMetadata.pointCount}/${projectionMetadata.totalCount} points`,
      );
      if (projectionMetadata.isSubsample && projectionMetadata.subsampleStrategy) {
        projectionTitle.push(`Subset: ${projectionMetadata.subsampleStrategy}`);
      }
      if (projectionMetadata.cachedAt) {
        projectionTitle.push(`Cached at ${projectionMetadata.cachedAt}`);
      }
    }
    if (projectionWarnings.length) {
      projectionTitle.push(...projectionWarnings);
    }
    if (isProjectionLoading) {
      projectionTitle.push("Loading layout...");
    }

    items.push({
      label: "Projection",
      value: isProjectionLoading ? `${projectionLabel}...` : projectionLabel,
      title: projectionTitle.length ? projectionTitle.join("\n") : undefined,
    });

    items.push({ label: "Samples", value: String(results?.n ?? run.n) });
    items.push({ label: "Status", value: run.status });
    if (results?.costs) {
      items.push({ label: "Total cost", value: `$${results.costs.total_cost.toFixed(6)}` });
    }
    if (runMetrics) {
      items.push({ label: "Cache Hit", value: `${runMetrics.cache_hit_rate.toFixed(1)}%` });
      items.push({ label: "Cached", value: `${runMetrics.cached_segments}/${runMetrics.total_segments}` });
      if (runMetrics.duplicate_segments > 0) {
        items.push({ label: "Duplicates", value: String(runMetrics.duplicate_segments) });
      }
    }

    const stageTimings = runMetrics?.stage_timings ?? run.stage_timings ?? [];
    const processingMs = runMetrics?.processing_time_ms ?? run.processing_time_ms ?? (stageTimings.length ? stageTimings.reduce((total, stage) => total + (stage.duration_ms ?? 0), 0) : null);
    if (processingMs != null) {
      const stageLines = stageTimings.map((stage) => {
        const friendly = stageLabels[stage.name] ?? stage.name.replace(/-/g, ' ');
        const offsetValue = stage.offset_ms ?? 0;
        const offsetLabel = offsetValue > 0 ? ` (start +${formatDuration(offsetValue)})` : '';
        return `${friendly}: ${formatDuration(stage.duration_ms ?? 0)}${offsetLabel}`;
      });
      const title = stageLines.length ? `Total ${formatDuration(processingMs)}
${stageLines.join('\n')}` : undefined;
      items.push({ label: "Processing", value: formatDuration(processingMs), title });
    }

    return items;
  }, [run, results, runMetrics, projectionMethod, projectionMetadata, projectionWarnings, isProjectionLoading]);

  const gauges = useMemo(() => {
    if (!quality) {
      return [] as Array<{ label: string; value: number | null | undefined }>;
    }
    return [
      { label: "Trust 2D", value: quality.trustworthiness_2d },
      { label: "Trust 3D", value: quality.trustworthiness_3d },
      { label: "Cont. 2D", value: quality.continuity_2d },
      { label: "Cont. 3D", value: quality.continuity_3d },
    ];
  }, [quality]);

  if (!run) {
    return null;
  }

  const renderGauge = (label: string, value: number | null | undefined) => {
    if (value == null || Number.isNaN(value)) {
      return (
    <div className="glass-panel flex items-center justify-between rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-3 text-xs text-slate-300">
      <div className="flex flex-col gap-2" title="Quick facts about the current run and its embedding quality.">
        <p className="text-[11px] text-slate-500">Hover any badge to learn how it influences the landscape.</p>
        <div className="flex flex-wrap gap-2">
          {badges.map((badge) => (
            <span
              key={badge.label}
              className={badgeClass}
              title={badge.title ?? badgeHints[badge.label] ?? "Metadata value"}
            >
              {badge.label}: {badge.value}
            </span>
          ))}
        </div>
        {gauges.length ? (
          <div className="flex flex-wrap gap-3">
            {gauges.map((item) => renderGauge(item.label, item.value))}
          </div>
        ) : null}
      </div>
      <span className="text-[11px] uppercase tracking-wide text-slate-500">
        Saved {dateFormatter.format(new Date(run.created_at))}
      </span>
    </div>
  );
});





