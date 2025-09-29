/**
 * Displays metadata badges for the currently loaded run.
 */

import { memo, useMemo } from "react";

import { useRunStore } from "@/store/runStore";

const badgeClass = "inline-flex items-center gap-1 rounded-full border border-slate-700/60 bg-slate-900/70 px-2 py-[2px] text-[10px] uppercase tracking-wide text-slate-300";
const dateFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

export const RunMetadataBar = memo(function RunMetadataBar() {
  const { results } = useRunStore((state) => ({ results: state.results }));
  const run = results?.run;

  const badges = useMemo(() => {
    if (!run) {
      return [] as Array<{ label: string; value: string }>;
    }
    const items: Array<{ label: string; value: string }> = [
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
    items.push({ label: "Samples", value: String(results?.n ?? run.n) });
    items.push({ label: "Status", value: run.status });
    if (results?.costs) {
      items.push({ label: "Total cost", value: `$${results.costs.total_cost.toFixed(6)}` });
    }
    return items;
  }, [run, results]);

  if (!run) {
    return null;
  }

  return (
    <div className="glass-panel flex items-center justify-between rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-3 text-xs text-slate-300">
      <div className="flex flex-wrap gap-2">
        {badges.map((badge) => (
          <span key={badge.label} className={badgeClass}>
            {badge.label}: {badge.value}
          </span>
        ))}
      </div>
      <span className="text-[11px] uppercase tracking-wide text-slate-500">
        Saved {dateFormatter.format(new Date(run.created_at))}
      </span>
    </div>
  );
});
