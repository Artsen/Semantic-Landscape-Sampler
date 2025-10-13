/**
 * Micro-panel summarising projection trustworthiness and continuity scores.
 */

import { memo, useMemo } from "react";

import { useRunStore } from "@/store/runStore";

type GaugeEntry = {
  label: string;
  value: number | null | undefined;
};

const formatGauge = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return (value * 100).toFixed(1) + "%";
};

export const ProjectionQualityPanel = memo(function ProjectionQualityPanel() {
  const { results } = useRunStore((state) => ({ results: state.results }));
  const baseQuality = results?.quality;
  const methodQualityMap = results?.projection_quality;

  const gauges = useMemo(() => {
    if (!baseQuality) {
      return [] as GaugeEntry[];
    }
    return [
      { label: "Trustworthiness (2D)", value: baseQuality.trustworthiness_2d },
      { label: "Trustworthiness (3D)", value: baseQuality.trustworthiness_3d },
      { label: "Continuity (2D)", value: baseQuality.continuity_2d },
      { label: "Continuity (3D)", value: baseQuality.continuity_3d },
    ];
  }, [baseQuality]);

  if (!gauges.length) {
    return (
      <section className="glass-panel rounded-2xl border border-border/60 bg-panel-elev px-4 py-4 text-sm text-text-dim">
        <header className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-text">Projection Quality</h3>
        </header>
        <p className="mt-2 text-xs text-muted">Quality metrics will appear after a run completes.</p>
      </section>
    );
  }

  const methodEntries = Object.entries(methodQualityMap ?? {});

  return (
    <section className="glass-panel flex flex-col gap-4 rounded-2xl border border-border/60 bg-panel-elev px-4 py-4 text-sm text-text">
      <header className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-text">Projection Quality</h3>
          <p className="text-xs text-muted">Trustworthiness and continuity based on blended features.</p>
        </div>
        <span className="rounded-full border border-border px-2 py-[1px] text-[11px] uppercase tracking-wide text-muted">UMAP</span>
      </header>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {gauges.map((entry) => (
          <div key={entry.label} className="rounded-xl border border-border/60 bg-panel px-3 py-2">
            <p className="text-[11px] uppercase tracking-wide text-muted">{entry.label}</p>
            <p className="mt-1 text-lg font-semibold text-text">{formatGauge(entry.value)}</p>
          </div>
        ))}
      </div>
      {methodEntries.length ? (
        <div className="space-y-2 text-xs text-muted">
          <p className="text-[11px] uppercase tracking-wide">Other layouts</p>
          <div className="grid gap-2">
            {methodEntries.map(([method, qualities]) => {
              if (!qualities) {
                return null;
              }
              const q = qualities as Record<string, number | null>;
              return (
                <div key={method} className="rounded-xl border border-border/60 bg-panel px-3 py-2">
                  <p className="text-sm font-medium text-text">{method.toUpperCase()}</p>
                  <div className="mt-1 grid grid-cols-2 gap-2 text-[11px]">
                    <span>Trust 2D {formatGauge(q.trustworthiness_2d)}</span>
                    <span>Trust 3D {formatGauge(q.trustworthiness_3d)}</span>
                    <span>Cont 2D {formatGauge(q.continuity_2d)}</span>
                    <span>Cont 3D {formatGauge(q.continuity_3d)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
});
