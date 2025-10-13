/**
 * Focused panel for projection and clustering adjustments without leaving the main canvas.
 */

import { useCallback, type ChangeEvent } from "react";

import { useRunStore } from "@/store/runStore";

export function ClusterTuningPanel() {
  const {
    results,
    clusterAlgo,
    hdbscanMinClusterSize,
    hdbscanMinSamples,
    isRecomputingClusters,
    setClusterParams,
    recomputeClusters,
  } = useRunStore((state) => ({
    results: state.results,
    clusterAlgo: state.clusterAlgo,
    hdbscanMinClusterSize: state.hdbscanMinClusterSize,
    hdbscanMinSamples: state.hdbscanMinSamples,
    isRecomputingClusters: state.isRecomputingClusters,
    setClusterParams: state.setClusterParams,
    recomputeClusters: state.recomputeClusters,
  }));

  const handleAlgoChange = useCallback(
    (value: string) => {
      if (value === "hdbscan" || value === "kmeans") {
        setClusterParams({ algo: value });
      }
    },
    [setClusterParams],
  );

  const handleMinClusterSizeChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      if (Number.isNaN(value)) {
        return;
      }
      setClusterParams({ minClusterSize: Math.max(2, Math.floor(value)) });
    },
    [setClusterParams],
  );

  const handleMinSamplesChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      if (Number.isNaN(value)) {
        return;
      }
      setClusterParams({ minSamples: Math.max(1, Math.floor(value)) });
    },
    [setClusterParams],
  );

  const handleRecompute = useCallback(() => {
    recomputeClusters().catch((error) => {
      console.error("Recompute clusters failed", error);
    });
  }, [recomputeClusters]);

  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Clustering</h3>
        <label className="space-y-1 text-xs">
          <span className="uppercase tracking-wide text-muted">Algorithm</span>
          <select
            value={clusterAlgo}
            onChange={(event) => handleAlgoChange(event.target.value)}
            className="w-full rounded-xl border border-border bg-panel-elev p-2 text-sm text-text focus:border-accent focus:outline-none"
          >
            <option value="hdbscan">HDBSCAN</option>
            <option value="kmeans">KMeans</option>
          </select>
        </label>
      </section>

      {clusterAlgo === "hdbscan" ? (
        <section className="grid gap-3 sm:grid-cols-2">
          <label className="space-y-1 text-xs">
            <span className="uppercase tracking-wide text-muted">Min cluster size</span>
            <input
              type="number"
              min={2}
              value={hdbscanMinClusterSize ?? 2}
              onChange={handleMinClusterSizeChange}
              className="w-full rounded-xl border border-border bg-panel-elev p-2 text-sm text-text focus:border-accent focus:outline-none"
            />
          </label>
          <label className="space-y-1 text-xs">
            <span className="uppercase tracking-wide text-muted">Min samples</span>
            <input
              type="number"
              min={1}
              value={hdbscanMinSamples ?? 1}
              onChange={handleMinSamplesChange}
              className="w-full rounded-xl border border-border bg-panel-elev p-2 text-sm text-text focus:border-accent focus:outline-none"
            />
          </label>
        </section>
      ) : (
        <p className="text-xs text-muted">
          KMeans recomputes centroids using the active embedding blend. Adjust K by exporting and editing offline (coming soon).
        </p>
      )}

      <section className="space-y-2">
        <button
          type="button"
          onClick={handleRecompute}
          disabled={isRecomputingClusters || !results}
          className="inline-flex w-full items-center justify-center rounded-xl border border-accent/50 bg-accent/10 px-3 py-2 text-sm font-semibold text-accent transition hover:border-accent hover:bg-accent/20 disabled:cursor-not-allowed disabled:border-border disabled:text-muted"
        >
          {isRecomputingClusters ? "Recomputing..." : "Apply clustering"}
        </button>
        <p className="text-[11px] text-muted">
          Applies without re-sampling. Inspect silhouettes and stability in the analytics tab after recomputing.
        </p>
      </section>
    </div>
  );
}
