/**
 * Legend for response and segment clusters displayed beside the point cloud.
 *
 * Components:
 *  - ClusterLegend: Lists clusters with keywords, counts, toggles, and export quick actions.
 */

import { memo } from "react";

import type { ClusterSummary, SegmentClusterSummary } from "@/types/run";
import { useRunStore } from "@/store/runStore";
import { InfoTooltip } from "@/components/InfoTooltip";

interface ClusterLegendProps {
  responseClusters: ClusterSummary[];
  segmentClusters: SegmentClusterSummary[];
  onExportCluster?: (payload: { label: number; mode: "responses" | "segments" }) => void;
  exportDisabled?: boolean;
  formatLabel?: string;
}

const METHOD_LABELS: Record<string, string> = {
  hdbscan: "HDBSCAN",
  kmeans: "K-Means",
};

export const ClusterLegend = memo(function ClusterLegend({
  responseClusters,
  segmentClusters,
  onExportCluster,
  exportDisabled,
  formatLabel,
}: ClusterLegendProps) {
  const { clusterPalette, clusterVisibility, toggleCluster, setHoveredCluster } = useRunStore((state) => ({
    clusterPalette: state.clusterPalette,
    clusterVisibility: state.clusterVisibility,
    toggleCluster: state.toggleCluster,
    setHoveredCluster: state.setHoveredCluster,
  }));

  const exportBadge = formatLabel ? formatLabel.toUpperCase() : null;

  if (!responseClusters.length && !segmentClusters.length) {
    return null;
  }

  return (
    <div className="glass-panel max-h-[360px] overflow-y-auto rounded-xl border border-slate-800/60 p-4 text-xs scrollbar-thin" title="Toggle or export clusters. Hover badges for quick explanations.">
      {responseClusters.length ? (
        <section className="space-y-3">
          <header className="flex items-center justify-between" title="Response-level clusters summarised by keywords and centroid similarity.">
            <span className="text-xs font-semibold uppercase tracking-[0.22rem] text-slate-400">Response clusters <InfoTooltip text="Click to hide or show all responses in this cluster." className="ml-1" /></span>
            <span className="text-[11px] text-slate-500">Toggle visibility</span>
          </header>
          <ul className="space-y-3">
            {responseClusters.map((cluster) => {
              const key = String(cluster.label);
              const visible = clusterVisibility[key] ?? true;
              const color = clusterPalette[key] ?? "#94a3b8";
              const method = METHOD_LABELS[cluster.method] ?? cluster.method;
              const title = cluster.noise ? "Noise / outliers" : `Cluster ${cluster.label}`;
              return (
                <li key={`response-${cluster.label}`} className="flex items-center gap-3">
                  <div className={`flex w-full flex-col gap-2 rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2 transition ${visible ? "shadow-inner shadow-cyan-500/10" : "opacity-50"}`}>
                    <button
                      type="button"
                      onClick={() => toggleCluster(cluster.label)}
                      onMouseEnter={() => setHoveredCluster(cluster.label)}
                      onMouseLeave={() => setHoveredCluster(null)}
                      className="flex items-start justify-between gap-3 text-left"
                      title="Click to hide or show this cluster in the point cloud."
                    >
                      <div className="flex items-start gap-3">
                        <span className="mt-1 inline-flex h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
                        <div className="space-y-1">
                          <p className="font-medium text-slate-100">{title}</p>
                          <p className="text-[11px] text-slate-400">
                            {cluster.size} samples · {method}
                            {cluster.average_similarity != null ? ` · avg sim ${cluster.average_similarity.toFixed(2)}` : ""}
                          </p>
                          {cluster.keywords.length ? (
                            <p className="text-[11px] text-slate-500">{cluster.keywords.slice(0, 4).join(", ")}</p>
                          ) : null}
                        </div>
                      </div>
                      <div className="text-right text-[10px] text-slate-500">
                        <p>Centroid</p>
                        <p>{cluster.centroid_xyz.map((value) => value.toFixed(2)).join(", ")}</p>
                      </div>
                    </button>
                    <div className="flex items-center justify-between text-[10px] text-slate-500">
                      {exportBadge ? (
                        <span className="rounded-full border border-slate-700/60 px-2 py-[1px] text-[9px] uppercase tracking-wide text-slate-400">
                          {exportBadge}
                        </span>
                      ) : (
                        <span />
                      )}
                      <button
                        type="button"
                        onClick={() => onExportCluster?.({ label: cluster.label, mode: "responses" })}
                        disabled={exportDisabled || !onExportCluster}
                        className="rounded-full border border-slate-700/60 px-2 py-[2px] text-[10px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        Export
                      </button>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      {segmentClusters.length ? (
        <section className="mt-5 space-y-3">
          <header className="flex items-center justify-between" title="Segment-level clusters showing recurring threads inside responses.">
            <span className="text-xs font-semibold uppercase tracking-[0.22rem] text-slate-400">Segment themes <InfoTooltip text="Snippets grouped by topic; great for spotting recurring motifs." className="ml-1" /></span>
            <span className="text-[11px] text-slate-500">Clustered micro-ideas</span>
          </header>
          <ul className="space-y-3">
            {segmentClusters.map((cluster) => {
              const title = cluster.noise ? "Orphan fragments" : `Group ${cluster.label}`;
              return (
                <li key={`segment-${cluster.label}`} className="rounded-lg border border-slate-800/70 bg-slate-900/30 px-3 py-2">
                  <p className="font-medium text-slate-100">{title}</p>
                  <p className="text-[11px] text-slate-400">
                    {cluster.size} segments
                    {cluster.average_similarity != null ? ` · avg sim ${cluster.average_similarity.toFixed(2)}` : ""}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {(cluster.theme ?? cluster.keywords.slice(0, 4).join(", ")) || "—"}
                  </p>
                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500">
                    {exportBadge ? (
                      <span className="rounded-full border border-slate-700/60 px-2 py-[1px] text-[9px] uppercase tracking-wide text-slate-400">
                        {exportBadge}
                      </span>
                    ) : (
                      <span />
                    )}
                    <button
                      type="button"
                      onClick={() => onExportCluster?.({ label: cluster.label, mode: "segments" })}
                      disabled={exportDisabled || !onExportCluster}
                      className="rounded-full border border-slate-700/60 px-2 py-[2px] text-[10px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-60"
                      title="Download every response assigned to this cluster."
                    >
                      Export
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
    </div>
  );
});
