/**
 * Drawer overlay showing recently sampled runs with metadata and quick loading actions.
 *
 * Components:
 *  - RunHistoryDrawer: Fetches persisted runs from the backend, groups them by day,
 *    and lets users load a prior run without re-sampling.
 */

import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import type { RunWorkflow } from "@/hooks/useRunWorkflow";
import { fetchRuns } from "@/services/api";
import type { RunSummary } from "@/types/run";
import { useRunStore } from "@/store/runStore";

const dayFormatter = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
const timeFormatter = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });

const badgeClass =
  "inline-flex items-center gap-1 rounded-full border border-slate-700/60 bg-slate-900/70 px-2 py-[2px] text-[10px] uppercase tracking-wide text-slate-300";

function describeDate(timestamp: string): string {
  const date = new Date(timestamp);
  const today = new Date();
  const startOfDay = (input: Date) => new Date(input.getFullYear(), input.getMonth(), input.getDate());
  const todayKey = startOfDay(today).getTime();
  const targetKey = startOfDay(date).getTime();
  if (targetKey === todayKey) {
    return "Today";
  }
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (targetKey === startOfDay(yesterday).getTime()) {
    return "Yesterday";
  }
  const dayLabel = dayFormatter.format(date);
  if (today.getFullYear() === date.getFullYear()) {
    return dayLabel;
  }
  return `${dayLabel}, ${date.getFullYear()}`;
}

function truncatePrompt(prompt: string, maxLength = 140): string {
  if (prompt.length <= maxLength) {
    return prompt;
  }
  return `${prompt.slice(0, maxLength - 3)}...`;
}

function summariseNote(note: string, maxLength = 160): string {
  const trimmed = note.trim();
  if (trimmed.length <= maxLength) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxLength - 3)}...`;
}

function formatDuration(ms?: number | null): string {
  if (ms == null || Number.isNaN(ms)) {
    return 'N/A';
  }
  if (ms >= 1000) {
    const seconds = ms / 1000;
    return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
  }
  if (ms >= 100) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${ms.toFixed(1)}ms`;
}

type RunHistoryDrawerProps = {
  workflow: RunWorkflow;
};

export const RunHistoryDrawer = memo(function RunHistoryDrawer({ workflow }: RunHistoryDrawerProps) {
  const { isHistoryOpen, runHistory, setRunHistory, currentRunId, setHistoryOpen } = useRunStore((state) => ({
    isHistoryOpen: state.isHistoryOpen,
    runHistory: state.runHistory,
    setRunHistory: state.setRunHistory,
    currentRunId: state.currentRunId,
    setHistoryOpen: state.setHistoryOpen,
  }));
  const { loadFromHistory, duplicateRun, isLoadingHistory, isDuplicating } = workflow;
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const updateUrl = useCallback((runId: string | null) => {
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

  useEffect(() => {
    if (!isHistoryOpen) {
      return;
    }
    let cancelled = false;
    async function hydrateRuns() {
      setIsFetching(true);
      setFetchError(null);
      try {
        const runs = await fetchRuns(40);
        if (!cancelled) {
          setRunHistory(runs);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : "Failed to load run history";
          setFetchError(message);
        }
      } finally {
        if (!cancelled) {
          setIsFetching(false);
        }
      }
    }
    void hydrateRuns();
    return () => {
      cancelled = true;
    };
  }, [isHistoryOpen, setRunHistory]);

  const groupedRuns = useMemo(() => {
    if (!runHistory.length) {
      return [] as Array<{ label: string; runs: RunSummary[] }>;
    }
    const sorted = [...runHistory].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );
    const groups = new Map<string, { label: string; runs: RunSummary[] }>();
    for (const run of sorted) {
      const dateLabel = describeDate(run.created_at);
      if (!groups.has(dateLabel)) {
        groups.set(dateLabel, { label: dateLabel, runs: [] });
      }
      groups.get(dateLabel)?.runs.push(run);
    }
    return Array.from(groups.values());
  }, [runHistory]);

  const handleSelect = (run: RunSummary) => {
    if (currentRunId === run.id && !isLoadingHistory) {
      updateUrl(run.id);
      setHistoryOpen(false);
      return;
    }
    updateUrl(run.id);
    loadFromHistory(run);
  };

  const handleDuplicate = (run: RunSummary) => {
    duplicateRun(run);
  };

  if (!isHistoryOpen) {
    return null;
  }

  const historyCountLabel = `${runHistory.length} stored runs`;
  const statusBadges: string[] = [];
  if (isLoadingHistory) {
    statusBadges.push("Loading run...");
  }
  if (isDuplicating) {
    statusBadges.push("Re-running...");
  }

  const content = (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close run history overlay"
        onClick={() => setHistoryOpen(false)}
        className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm"
      />
      <aside className="relative z-10 flex h-full w-full max-w-md flex-col rounded-l-2xl border border-slate-800/70 bg-slate-950/95 p-6 shadow-2xl shadow-slate-950/70">
        <header className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.22rem] text-slate-500">Run history</p>
            <h2 className="text-lg font-semibold text-slate-100">Saved landscapes</h2>
            <p className="text-xs text-slate-500">Load a previous semantic layout without re-sampling.</p>
          </div>
          <button
            type="button"
            onClick={() => setHistoryOpen(false)}
            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
          >
            Close
          </button>
        </header>

        <div className="mt-4 flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500">
          <span>{isFetching ? "Refreshing..." : historyCountLabel}</span>
          {statusBadges.length ? (
            <div className="flex gap-3 text-cyan-200">
              {statusBadges.map((label) => (
                <span key={label}>{label}</span>
              ))}
            </div>
          ) : null}
        </div>
        {fetchError ? <p className="mt-3 text-xs text-rose-300">{fetchError}</p> : null}

        <div className="mt-4 flex-1 overflow-y-auto pr-2 text-sm">
          {groupedRuns.length === 0 && !isFetching ? (
            <p className="text-xs text-slate-500">
              Run a sample to populate history. Completed runs will appear here automatically.
            </p>
          ) : null}

          {groupedRuns.map((group) => (
            <section key={group.label} className="mb-6 space-y-3">
              <header className="text-[11px] uppercase tracking-[0.22rem] text-slate-500">{group.label}</header>
              <ul className="space-y-3">
                {group.runs.map((run) => {
                  const active = currentRunId === run.id;
                  const containerClass = active
                    ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-100"
                    : "border-slate-800/70 bg-slate-900/50 text-slate-200 hover:border-cyan-400/50 hover:text-cyan-100";
                  const cardClass = `rounded-xl border px-4 py-3 transition ${containerClass}`;
                  return (
                    <li key={run.id}>
                      <article className={cardClass}>
                        <div className="flex items-start justify-between gap-2">
                          <div className="max-w-[75%] space-y-2">
                            <p className="text-[13px] font-medium leading-snug">{truncatePrompt(run.prompt)}</p>
                            {run.notes ? (
                              <p className="text-[11px] italic text-slate-400">"{summariseNote(run.notes)}"</p>
                            ) : null}
                          </div>
                          <span className="text-[11px] uppercase tracking-wide text-slate-500">
                            {timeFormatter.format(new Date(run.created_at))}
                          </span>
                        </div>

                        <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
                          <span className={badgeClass}>Model {run.model}</span>
                          <span className={badgeClass}>Embedding {run.embedding_model}</span>
                          <span className={badgeClass}>Temp {run.temperature.toFixed(2)}</span>
                          {run.top_p != null ? <span className={badgeClass}>Top-p {run.top_p.toFixed(2)}</span> : null}
                          {run.seed != null ? <span className={badgeClass}>Seed {run.seed}</span> : null}
                          <span className={badgeClass}>UMAP n {run.umap.n_neighbors}</span>
                          <span className={badgeClass}>UMAP dist {run.umap.min_dist.toFixed(2)}</span>
                          <span className={badgeClass}>UMAP metric {run.umap.metric}</span>
                          <span className={badgeClass}>N {run.n}</span>
                          <span className={badgeClass}>Responses {run.response_count}</span>
                          {run.segment_count ? (
                            <span className={badgeClass}>Segments {run.segment_count}</span>
                          ) : null}
                          {run.chunk_size ? (
                            <span className={badgeClass}>Chunk {run.chunk_size} w</span>
                          ) : null}
                          {run.processing_time_ms != null ? (
                            <span className={badgeClass}>Processing {formatDuration(run.processing_time_ms)}</span>
                          ) : null}
                          <span className={badgeClass}>Status {run.status}</span>
                        </div>

                        <div className="mt-3 flex gap-2">
                          <button
                            type="button"
                            onClick={() => handleSelect(run)}
                            className="flex-1 rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-cyan-400 hover:text-cyan-100"
                            disabled={active && isLoadingHistory}
                          >
                            {active ? "Viewing" : "Load"}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDuplicate(run)}
                            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-emerald-400 hover:text-emerald-100 disabled:cursor-not-allowed disabled:opacity-60"
                            disabled={isDuplicating}
                          >
                            Re-run
                          </button>
                        </div>
                      </article>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      </aside>
    </div>
  );

  if (typeof document === "undefined") {
    return content;
  }

  return createPortal(content, document.body);

});
