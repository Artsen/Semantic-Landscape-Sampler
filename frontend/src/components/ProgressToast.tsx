/**
 * Floating overlay that surfaces backend progress updates while a run is executing.
 */

import { useMemo } from "react";

import { useRunStore } from "@/store/runStore";

const KEY_REPLACE: Record<string, string> = {
  n: "Requests",
  model: "Model",
  responses: "Responses",
  segments: "Segments",
  clusters: "Clusters",
  "segment_edges": "Edges",
  responses_processed: "Responses done",
  responses_total: "Responses total",
  segments_accumulated: "Segments built",
  segments_tagged: "Segments tagged",
  segments_total: "Segments total",
  batch_index: "Batch",
  batch_count: "Batches",
};

const formatKey = (key: string) => KEY_REPLACE[key] ?? key.replace(/_/g, " ");

const formatValue = (value: unknown) => {
  if (value == null) {
    return "";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(2);
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

export function ProgressToast() {
  const {
    progressStage,
    progressMessage,
    progressPercent,
    progressMetadata,
    isGenerating,
  } = useRunStore((state) => ({
    progressStage: state.progressStage,
    progressMessage: state.progressMessage,
    progressPercent: state.progressPercent,
    progressMetadata: state.progressMetadata,
    isGenerating: state.isGenerating,
  }));

  const visible = Boolean(progressMessage) || isGenerating;
  const percentLabel = useMemo(() => {
    if (progressPercent == null) {
      return null;
    }
    const clamped = Math.min(1, Math.max(0, progressPercent));
    return `${Math.round(clamped * 100)}%`;
  }, [progressPercent]);

  const details = useMemo(() => {
    if (!progressMetadata) {
      return [] as Array<[string, string]>;
    }
    return Object.entries(progressMetadata).map(([key, value]) => [
      formatKey(key),
      formatValue(value),
    ]);
  }, [progressMetadata]);

  if (!visible) {
    return null;
  }

  return (
    <div className="pointer-events-none fixed inset-0 z-50 flex items-start justify-center pt-16">
      <div className="pointer-events-auto w-full max-w-sm rounded-2xl border border-cyan-500/40 bg-slate-950/95 p-5 shadow-xl shadow-cyan-500/10 backdrop-blur-sm">
        <header className="flex items-center justify-between text-[11px] uppercase tracking-[0.22rem] text-cyan-200">
          <span>Run Progress</span>
          {percentLabel ? <span>{percentLabel}</span> : null}
        </header>
        <p className="mt-2 text-sm font-semibold text-cyan-100">
          {progressStage ?? (isGenerating ? "Processing" : "Idle")}
        </p>
        {progressMessage ? (
          <p className="mt-1 text-xs text-cyan-100/80">{progressMessage}</p>
        ) : null}
        {progressPercent != null ? (
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-cyan-500/20">
            <div
              className="h-full bg-cyan-400 transition-all duration-500"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(2, Math.min(1, Math.max(0, progressPercent)) * 100),
                )}%`,
              }}
            />
          </div>
        ) : null}
        {details.length ? (
          <ul className="mt-3 space-y-1 text-[11px] font-medium text-cyan-200/90">
            {details.map(([key, value]) => (
              <li key={key} className="flex justify-between gap-4">
                <span>{key}</span>
                <span className="text-cyan-100/90">{value}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}
