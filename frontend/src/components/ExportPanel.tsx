/**
 * Export configuration surface for choosing scope, format, and includes.
 */

import { useMemo, useState } from "react";
import clsx from "clsx";

import type { RunWorkflow } from "@/hooks/useRunWorkflow";
import { useRunStore } from "@/store/runStore";
import type { ExportFormat, ExportInclude, ExportMode, ExportScope } from "@/types/run";

type ExportPanelProps = {
  workflow: RunWorkflow;
  onComplete?: () => void;
};

const scopeOptions: Array<{ value: ExportScope; label: string; description: string }> = [
  { value: "run", label: "Entire run", description: "All responses, segments, and metadata" },
  { value: "cluster", label: "Cluster", description: "Responses or segments within a cluster" },
  { value: "selection", label: "Selection", description: "Currently selected responses or segments" },
  { value: "viewport", label: "Viewport", description: "Points within the current camera bounds" },
];

const formatOptions: Array<{ value: ExportFormat; label: string }> = [
  { value: "json", label: "JSON" },
  { value: "jsonl", label: "JSONL" },
  { value: "csv", label: "CSV" },
  { value: "parquet", label: "Parquet" },
];

export function ExportPanel({ workflow, onComplete }: ExportPanelProps) {
  const {
    results,
    exportFormat,
    setExportFormat,
    exportIncludeProvenance,
    setExportIncludeProvenance,
    levelMode,
    selectedPointIds,
    selectedSegmentIds,
  } = useRunStore((state) => ({
    results: state.results,
    exportFormat: state.exportFormat,
    setExportFormat: state.setExportFormat,
    exportIncludeProvenance: state.exportIncludeProvenance,
    setExportIncludeProvenance: state.setExportIncludeProvenance,
    levelMode: state.levelMode,
    selectedPointIds: state.selectedPointIds,
    selectedSegmentIds: state.selectedSegmentIds,
  }));

  const defaultMode: ExportMode = levelMode === "segments" ? "segments" : "responses";
  const [scope, setScope] = useState<ExportScope>("run");
  const [mode, setMode] = useState<ExportMode>(defaultMode);
  const [includeVectors, setIncludeVectors] = useState(false);
  const [includeMetadata, setIncludeMetadata] = useState(false);
  const [clusterId, setClusterId] = useState<number | null>(null);
  const [isExporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const clusters = results?.clusters ?? [];
  const clusterOptions = useMemo(
    () =>
      clusters.map((cluster) => ({
        id: cluster.label,
        name: "Cluster #" + cluster.label,
        size: cluster.size ?? 0,
      })),
    [clusters],
  );

  const selectionCount = mode === "segments" ? selectedSegmentIds.length : selectedPointIds.length;
  const selectionEmpty = selectionCount === 0;

  const includeTokens = useMemo(() => {
    const tokens: ExportInclude[] = [];
    if (exportIncludeProvenance) {
      tokens.push("provenance");
    }
    if (includeVectors) {
      tokens.push("vectors");
    }
    if (includeMetadata) {
      tokens.push("metadata");
    }
    return tokens;
  }, [exportIncludeProvenance, includeMetadata, includeVectors]);

  const handleScopeChange = (value: ExportScope) => {
    setScope(value);
    if (value === "cluster") {
      setClusterId((current) => {
        if (current != null) {
          return current;
        }
        return clusters.length ? clusters[0].label : null;
      });
    }
  };

  const handleExport = async () => {
    setError(null);
    if (scope === "selection" && selectionEmpty) {
      setError("Nothing selected to export");
      return;
    }
    if (scope === "cluster" && (clusterId == null || Number.isNaN(clusterId))) {
      setError("Choose a cluster to export");
      return;
    }
    setExporting(true);
    try {
      await workflow.exportDataset({
        scope,
        format: exportFormat,
        mode,
        ...(scope === "cluster" && clusterId != null ? { clusterId } : {}),
        include: includeTokens.length ? includeTokens : undefined,
      });
      if (onComplete) {
        onComplete();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed";
      setError(message);
    } finally {
      setExporting(false);
    }
  };

  const scopeHint = useMemo(() => {
    if (scope === "selection") {
      if (selectionEmpty) {
        return "Select responses or segments to enable selection exports.";
      }
      return String(selectionCount) + " items selected.";
    }
    if (scope === "cluster") {
      if (!clusters.length) {
        return "Run clustering to enable cluster exports.";
      }
      const selectedCluster = clusters.find((cluster) => cluster.label === clusterId);
      if (!selectedCluster) {
        return "Choose a cluster to export.";
      }
      const count = selectedCluster.size ?? 0;
      return String(count) + " items in cluster #" + selectedCluster.label + ".";
    }
    return null;
  }, [clusterId, clusters, scope, selectionCount, selectionEmpty]);

  return (
    <div className="flex flex-col gap-4 text-sm">
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Scope</h3>
        <div className="grid gap-2">
          {scopeOptions.map((option) => (
            <label
              key={option.value}
              className={clsx(
                "flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-2 text-text transition",
                scope === option.value ? "border-accent bg-accent/10" : "border-border/60 bg-panel-elev",
              )}
            >
              <input
                type="radio"
                name="export-scope"
                value={option.value}
                checked={scope === option.value}
                onChange={() => handleScopeChange(option.value)}
                className="mt-1 h-4 w-4 accent-accent"
              />
              <span>
                <span className="text-sm font-medium">{option.label}</span>
                <span className="block text-xs text-muted">{option.description}</span>
              </span>
            </label>
          ))}
        </div>
        {scope === "cluster" ? (
          <div className="space-y-2 rounded-xl border border-border/60 bg-panel-elev p-3">
            <label className="text-xs uppercase tracking-wide text-muted">Cluster</label>
            {clusterOptions.length ? (
              <select
                value={clusterId ?? ""}
                onChange={(event) => setClusterId(Number(event.target.value))}
                className="w-full rounded-lg border border-border bg-panel p-2 text-sm text-text focus:border-accent focus:outline-none"
              >
                {clusterOptions.map((cluster) => (
                  <option key={cluster.id} value={cluster.id}>
                    {cluster.name + " • " + cluster.size + " items"}
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-xs text-muted">No clusters computed yet.</p>
            )}
          </div>
        ) : null}
        {scopeHint ? <p className="text-xs text-muted">{scopeHint}</p> : null}
      </section>

      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Dataset</h3>
        <div className="flex gap-2">
          <button
            type="button"
            className={clsx(
              "flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition",
              mode === "responses" ? "border-accent bg-accent/10 text-text" : "border-border bg-panel-elev text-text",
            )}
            onClick={() => setMode("responses")}
          >
            Responses
          </button>
          <button
            type="button"
            className={clsx(
              "flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition",
              mode === "segments" ? "border-accent bg-accent/10 text-text" : "border-border bg-panel-elev text-text",
            )}
            onClick={() => setMode("segments")}
          >
            Segments
          </button>
        </div>
      </section>

      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Format</h3>
        <div className="grid gap-2">
          {formatOptions.map((option) => (
            <label
              key={option.value}
              className={clsx(
                "flex items-center justify-between rounded-lg border px-3 py-2 text-sm transition",
                exportFormat === option.value ? "border-accent bg-accent/10 text-text" : "border-border bg-panel-elev text-text",
              )}
            >
              <span>{option.label}</span>
              <input
                type="radio"
                checked={exportFormat === option.value}
                onChange={() => setExportFormat(option.value)}
                className="h-4 w-4 accent-accent"
              />
            </label>
          ))}
        </div>
      </section>

      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Include</h3>
        <label className="flex items-center gap-3 text-sm">
          <input
            type="checkbox"
            checked={exportIncludeProvenance}
            onChange={(event) => setExportIncludeProvenance(event.target.checked)}
            className="h-4 w-4 accent-accent"
          />
          Provenance metadata
        </label>
        <label className="flex items-center gap-3 text-sm">
          <input
            type="checkbox"
            checked={includeVectors}
            onChange={(event) => setIncludeVectors(event.target.checked)}
            className="h-4 w-4 accent-accent"
          />
          Embedding vectors
        </label>
        <label className="flex items-center gap-3 text-sm">
          <input
            type="checkbox"
            checked={includeMetadata}
            onChange={(event) => setIncludeMetadata(event.target.checked)}
            className="h-4 w-4 accent-accent"
          />
          Extended metadata
        </label>
      </section>

      {error ? <p className="text-xs text-danger">{error}</p> : null}

      <div className="mt-2 flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={handleExport}
          disabled={isExporting || (scope === "cluster" && !clusters.length)}
          className="rounded-lg border border-accent bg-accent/10 px-4 py-2 text-sm font-semibold text-accent transition hover:bg-accent/20 disabled:cursor-not-allowed disabled:border-border disabled:text-muted"
        >
          {isExporting ? "Preparing export..." : "Start export"}
        </button>
      </div>
    </div>
  );
}
