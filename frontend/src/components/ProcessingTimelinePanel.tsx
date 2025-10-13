import { memo, useMemo } from "react";

import { InfoTooltip } from "@/components/InfoTooltip";
import { useRunStore } from "@/store/runStore";
import type { StageTiming } from "@/types/run";

const panelClass = "glass-panel flex flex-col gap-5 rounded-2xl border border-border/70 bg-panel-elev px-5 py-6 text-xs text-text";

const stageLabels: Record<string, string> = {
  "prepare-run": "Preparation",
  "request-completions": "LLM sampling",
  "persist-responses": "Persist responses",
  "segment-responses": "Segment responses",
  "discourse-tagging": "Discourse tagging",
  "embed-responses": "Embed responses",
  "embed-segments": "Embed segments",
  "segment-analysis": "Segment analysis",
  "response-analysis": "Response analysis",
  "persist-artifacts": "Persist artifacts",
};

const palette = [
  "#22c55e",
  "#0ea5e9",
  "#a855f7",
  "#f97316",
  "#f59e0b",
  "#ec4899",
  "#38bdf8",
  "#34d399",
  "#60a5fa",
  "#c084fc",
];

const formatDuration = (ms?: number | null): string => {
  if (ms == null || Number.isNaN(ms)) {
    return "--";
  }
  if (ms >= 60_000) {
    const minutes = Math.floor(ms / 60_000);
    const seconds = Math.round((ms % 60_000) / 1000);
    return `${minutes}m ${seconds}s`;
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

const parseTimestamp = (value?: string | null): Date | null => {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
};

const formatTime = (date?: Date | null) => {
  if (!date) {
    return null;
  }
  return date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
};

const coalesceStageTimings = (metricsStages?: StageTiming[], runStages?: StageTiming[]): StageTiming[] => {
  if (metricsStages && metricsStages.length) {
    return metricsStages;
  }
  if (runStages && runStages.length) {
    return runStages;
  }
  return [];
};

type Slice = {
  key: string;
  label: string;
  durationMs: number;
  percent: number;
  color: string;
};

const buildSlices = (stageTimings: StageTiming[], totalMs: number): Slice[] => {
  if (!stageTimings.length || totalMs <= 0) {
    return [];
  }

  return stageTimings
    .map((stage, index) => {
      const label = stageLabels[stage.name] ?? stage.name.replace(/-/g, " ");
      const durationMs = Math.max(0, stage.duration_ms ?? 0);
      const percent = totalMs > 0 ? (durationMs / totalMs) * 100 : 0;
      return {
        key: `${stage.name}-${index}`,
        label,
        durationMs,
        percent,
        color: palette[index % palette.length],
      };
    })
    .filter((slice) => slice.durationMs > 0)
    .sort((a, b) => b.durationMs - a.durationMs);
};

export const ProcessingBreakdownPanel = memo(function ProcessingBreakdownPanel() {
  const { runMetrics, results } = useRunStore((state) => ({
    runMetrics: state.runMetrics,
    results: state.results,
  }));

  const run = results?.run;
  const stageTimings = useMemo(
    () => coalesceStageTimings(runMetrics?.stage_timings, run?.stage_timings),
    [runMetrics?.stage_timings, run?.stage_timings],
  );

  const totalMs = useMemo(() => {
    if (runMetrics?.processing_time_ms != null) {
      return runMetrics.processing_time_ms;
    }
    if (run?.processing_time_ms != null) {
      return run.processing_time_ms;
    }
    if (stageTimings.length) {
      return stageTimings.reduce((total, stage) => total + (stage.duration_ms ?? 0), 0);
    }
    return 0;
  }, [runMetrics?.processing_time_ms, run?.processing_time_ms, stageTimings]);

  const startedAt = stageTimings.length
    ? parseTimestamp(stageTimings[0].started_at ?? run?.created_at)
    : parseTimestamp(run?.created_at);
  const finishedAt = stageTimings.length
    ? parseTimestamp(stageTimings[stageTimings.length - 1].finished_at ?? run?.updated_at)
    : parseTimestamp(run?.updated_at);

  const slices = useMemo(() => buildSlices(stageTimings, totalMs), [stageTimings, totalMs]);

  if (!totalMs && !stageTimings.length) {
    return (
      <div className={panelClass} title="Processing duration data will appear after a run completes.">
        <header className="flex items-start justify-between gap-2">
          <h2 className="text-sm font-semibold text-text">Processing Breakdown</h2>
          <InfoTooltip text="Displays how long each backend stage took once a run finishes." />
        </header>
        <p className="text-[11px] text-muted">
          Trigger a run to capture timing information. Stage percentages and summaries will appear here once data is available.
        </p>
      </div>
    );
  }

  const totalDisplay = formatDuration(totalMs, "short");
  const startedLabel = formatTime(startedAt);
  const finishedLabel = formatTime(finishedAt);

  const highlight = slices[0] ?? null;
  const secondary = slices.slice(1);

  return (
    <div className={panelClass} title="Processing timeline from trigger to completion.">
      <header className="flex items-start justify-between gap-4">
        <div className="space-y-3">
          <div>
            <p className="text-[10px] uppercase tracking-[0.24em] text-muted">Processing Breakdown</p>
            <div className="mt-2 flex items-baseline gap-3">
              <span className="text-3xl font-semibold tracking-tight text-text">{totalDisplay}</span>
              <span className="text-[10px] uppercase tracking-[0.22em] text-muted">total</span>
            </div>
            {startedLabel && finishedLabel ? (
              <p className="text-[11px] text-muted">{startedLabel} to {finishedLabel}</p>
            ) : null}
          </div>
          {highlight ? (
            <div className="flex items-center gap-2 text-[11px] text-muted">
              <span className="inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: highlight.color }} aria-hidden />
              <span>
                Longest stage · <span className="text-text font-medium">{highlight.label}</span> ({highlight.percent.toFixed(1)}%)
              </span>
            </div>
          ) : null}
        </div>
        <InfoTooltip text="End-to-end runtime split by backend stage." />
      </header>

      {highlight ? (
        <section className="rounded-2xl border border-border/60 bg-panel px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: highlight.color }} aria-hidden />
              <span className="text-sm font-semibold text-text">{highlight.label}</span>
            </div>
            <span className="text-xs text-muted">{highlight.percent.toFixed(1)}%</span>
          </div>
          <div className="mt-3 text-2xl font-semibold text-text">{formatDuration(highlight.durationMs)}</div>
          <p className="mt-1 text-[11px] text-muted">{(highlight.durationMs / 1000).toFixed(highlight.durationMs >= 1000 ? 1 : 2)}s of total runtime</p>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-border/40">
            <div
              className="h-full rounded-full"
              style={{ width: `${Math.min(100, Math.max(highlight.percent, 6))}%`, backgroundColor: highlight.color }}
            />
          </div>
        </section>
      ) : null}

      <section className="grid gap-2 sm:grid-cols-2">
        {secondary.map((slice) => (
          <article key={slice.key} className="rounded-xl border border-border/60 bg-panel px-3 py-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: slice.color }} aria-hidden />
                <span className="text-[12px] font-medium text-text">{slice.label}</span>
              </div>
              <span className="text-[11px] text-muted">{slice.percent.toFixed(1)}%</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between text-[11px] text-muted">
              <span>{formatDuration(slice.durationMs)}</span>
              <span>{(slice.durationMs / 1000).toFixed(slice.durationMs >= 1000 ? 1 : 2)}s</span>
            </div>
            <div className="mt-2 h-1 overflow-hidden rounded-full bg-border/30">
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.min(100, Math.max(slice.percent, 4))}%`, backgroundColor: slice.color }}
              />
            </div>
          </article>
        ))}
      </section>

      <footer className="flex flex-wrap gap-3 text-[11px] text-muted">
        {startedLabel ? <span>Started {startedLabel}</span> : null}
        {finishedLabel ? <span>Finished {finishedLabel}</span> : null}
        {run?.updated_at ? <span>Updated {formatTime(parseTimestamp(run?.updated_at))}</span> : null}
      </footer>
    </div>
  );
});

export const ProcessingTimelinePanel = ProcessingBreakdownPanel;
