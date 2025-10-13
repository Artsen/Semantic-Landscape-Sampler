import { memo, useMemo } from "react";

import { useRunStore } from "@/store/runStore";
import { InfoTooltip } from "@/components/InfoTooltip";
import type { ClusterParameterSweepPoint, ClusterStabilityBootstrap } from "@/types/run";

const panelClass = "glass-panel flex flex-col gap-4 rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4";

const barBackground = "h-1.5 w-full rounded-full bg-slate-800/70";
const barForeground = "h-full rounded-full bg-cyan-400";

const formatPercent = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(1)}%`;
};

const clamp01 = (value: number | null | undefined) => {
  if (value == null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
};

const sortSweepPoints = (points: ClusterParameterSweepPoint[]) => {
  return [...points].sort((a, b) => {
    const aScore = a.silhouette_feature ?? a.silhouette_embed ?? -Infinity;
    const bScore = b.silhouette_feature ?? b.silhouette_embed ?? -Infinity;
    return bScore - aScore;
  });
};

export const ClusterMetricsPanel = memo(function ClusterMetricsPanel() {
  const {
    clusterMetrics,
    clusterAlgo,
    hdbscanMinClusterSize,
    hdbscanMinSamples,
    isRecomputingClusters,
  } = useRunStore((state) => ({
    clusterMetrics: state.clusterMetrics,
    clusterAlgo: state.clusterAlgo,
    hdbscanMinClusterSize: state.hdbscanMinClusterSize,
    hdbscanMinSamples: state.hdbscanMinSamples,
    isRecomputingClusters: state.isRecomputingClusters,
  }));

  const silhouettes = useMemo(
    () => [
      { label: "Embedding", value: clusterMetrics?.silhouette_embed ?? null },
      { label: "Feature", value: clusterMetrics?.silhouette_feature ?? null },
    ],
    [clusterMetrics],
  );

  const bootstrap = clusterMetrics?.stability?.bootstrap as ClusterStabilityBootstrap | undefined;
  const persistence = clusterMetrics?.stability?.persistence ?? null;

  const sweepPoints = useMemo(() => {
    if (!clusterMetrics?.sweep) {
      return [] as ClusterParameterSweepPoint[];
    }
    return sortSweepPoints(clusterMetrics.sweep.points).slice(0, 4);
  }, [clusterMetrics]);

  if (!clusterMetrics) {
    return (
      <div className={panelClass} title="Metrics summarising how reliable the current clustering is.">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200">Cluster Metrics <InfoTooltip text="Snapshot of silhouettes, stability, and sweep results." className="ml-1" /></h2>
          {isRecomputingClusters ? (
            <span className="text-xs text-cyan-300">Recomputing…</span>
          ) : null}
        </div>
        <p className="text-xs text-slate-500">
          Metrics will appear after sampling. Adjust the cluster parameters and run a recompute to populate this panel.
        </p>
      </div>
    );
  }

  const renderSilhouette = ({ label, value }: { label: string; value: number | null }) => {
    const width = `${(clamp01(value ?? 0) * 100).toFixed(1)}%`;
    return (
      <div
        key={label}
        className="flex flex-col gap-1"
        title="Silhouette compares cluster cohesion to separation; higher is better."
      >
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>{label}</span>
          <span className="text-slate-200">{formatPercent(value)}</span>
        </div>
        <div className={barBackground}>
          <div className={barForeground} style={{ width }} />
        </div>
      </div>
    );
  };

  const renderBootstrap = () => {
    if (!bootstrap || !Object.keys(bootstrap.clusters).length) {
      return <p className="text-xs text-slate-500">Bootstrap stability pending.</p>;
    }
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>Bootstrap (fraction {Math.round(bootstrap.fraction * 100)}%, {bootstrap.iterations} iters)</span>
        </div>
        <div className="flex flex-col gap-1.5">
          {Object.entries(bootstrap.clusters).map(([label, stats]) => {
            const width = `${Math.min(100, Math.max(0, stats.mean * 100)).toFixed(1)}%`;
            return (
              <div key={label} className="flex items-center gap-2 text-[11px] text-slate-300">
                <span className="w-10 rounded bg-slate-800/60 px-2 py-[2px] text-center text-[10px] text-slate-200">
                  #{label}
                </span>
                <div className="flex-1">
                  <div className={barBackground}>
                    <div className="h-full rounded-full bg-amber-400/80" style={{ width }} />
                  </div>
                </div>
                <span className="w-20 text-right text-slate-400">
                  {stats.mean.toFixed(2)} ± {stats.std.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderPersistence = () => {
    if (!persistence || !Object.keys(persistence).length) {
      return null;
    }
    return (
      <div className="flex flex-wrap gap-2 text-[11px] text-slate-400">
        {Object.entries(persistence).map(([label, score]) => (
          <span key={label} className="rounded-full border border-slate-800/70 bg-slate-900/70 px-2 py-[2px] text-slate-200">
            #{label} persistence {score.toFixed(2)}
          </span>
        ))}
      </div>
    );
  };

  const renderSweep = () => {
    if (!clusterMetrics.sweep || !sweepPoints.length) {
      return <p className="text-xs text-slate-500">No sweep results captured yet.</p>;
    }
    return (
      <div className="flex flex-col gap-2">
        <div className="text-[11px] text-slate-400">
          Baseline: {clusterMetrics.sweep.baseline.min_cluster_size} min size / {clusterMetrics.sweep.baseline.min_samples ?? "—"} min samples
        </div>
        <div className="grid grid-cols-1 gap-1 text-[11px] text-slate-300">
          {sweepPoints.map((point) => (
            <div
              key={`${point.min_cluster_size}-${point.min_samples}-${point.algo}`}
              className="rounded-lg border border-slate-800/60 bg-slate-900/60 px-2 py-1"
            >
              <div className="flex items-center justify-between text-slate-200">
                <span>
                  {point.algo} ({point.min_cluster_size}/{point.min_samples})
                </span>
                <span>{formatPercent(point.silhouette_feature ?? point.silhouette_embed ?? null)}</span>
              </div>
              <div className="text-[10px] text-slate-500">
                DBI {point.davies_bouldin?.toFixed(2) ?? "—"}  CHI {point.calinski_harabasz?.toFixed(1) ?? "—"}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={panelClass} title="Metrics summarising how reliable the current clustering is.">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Cluster Metrics</h2>
          <p className="text-xs text-slate-500">
            Current: {clusterAlgo.toUpperCase()} • min size {hdbscanMinClusterSize} • min samples {hdbscanMinSamples}
          </p>
        </div>
        {isRecomputingClusters ? (
          <span className="text-xs text-cyan-300">Recomputing…</span>
        ) : null}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="text-[12px] font-semibold text-slate-300">Silhouette Scores <InfoTooltip text="Higher silhouette values mean clusters are compact and well separated." className="ml-1" /></h3>
          <div className="mt-2 flex flex-col gap-2">
            {silhouettes.map(renderSilhouette)}
          </div>
        </div>
        <div>
          <h3 className="text-[12px] font-semibold text-slate-300">Cluster Indices <InfoTooltip text="Secondary metrics that double-check cohesion and separation." className="ml-1" /></h3>
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-400">
            <span className="rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200">
              DBI {clusterMetrics.davies_bouldin?.toFixed(2) ?? "—"}
            </span>
            <span className="rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200">
              CHI {clusterMetrics.calinski_harabasz?.toFixed(1) ?? "—"}
            </span>
            <span className="rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200">
              Clusters {clusterMetrics.n_clusters ?? "—"}
            </span>
            <span className="rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200">
              Noise {clusterMetrics.n_noise ?? "—"}
            </span>
          </div>
        </div>
        <div>
          <h3 className="text-[12px] font-semibold text-slate-300">Stability <InfoTooltip text="Bootstrap and persistence scores show how durable each cluster is." className="ml-1" /></h3>
          <div className="mt-2 flex flex-col gap-2">
            {renderBootstrap()}
            {renderPersistence()}
          </div>
        </div>
        <div>
          <h3 className="text-[12px] font-semibold text-slate-300">Parameter Sweep Top Picks <InfoTooltip text="Suggested HDBSCAN settings ranked by silhouette." className="ml-1" /></h3>
          <div className="mt-2">{renderSweep()}</div>
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Silhouettes in 2D can be inflated; check feature-space silhouette too.
      </p>
    </div>
  );
});
