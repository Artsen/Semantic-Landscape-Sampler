
/**
 * Control surface for sampling parameters, visual filters, exports, and diagnostics.
 */
import { memo, useMemo, useState, useCallback, type ReactNode } from "react";

import { InfoTooltip } from "@/components/InfoTooltip";
import { useRunWorkflow } from "@/hooks/useRunWorkflow";
import { useRunStore, type UmapPreset } from "@/store/runStore";
import type { ExportFormat, ProjectionMethod, UmapMetric } from "@/types/run";

type SectionTone = "instant" | "rerun";

type ControlSectionProps = {
  title: string;
  description?: string;
  tone?: SectionTone;
  defaultOpen?: boolean;
  children: ReactNode;
};

function toneBadge(tone: SectionTone) {
  if (tone === "rerun") {
    return {
      label: "Requires regenerate",
      className: "border-rose-400/40 bg-rose-500/10 text-rose-200",
    };
  }
  return {
    label: "Instant update",
    className: "border-emerald-400/40 bg-emerald-500/10 text-emerald-200",
  };
}

function ControlSection({ title, description, tone = "instant", defaultOpen = true, children }: ControlSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const badge = toneBadge(tone);

  return (
    <section className="rounded-xl border border-slate-800/60 bg-slate-950/50 shadow-sm shadow-slate-950/40">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-slate-900/60 focus:outline-none focus-visible:ring focus-visible:ring-cyan-400/40"
      >
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-200">{title}</p>
          {description ? <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">{description}</p> : null}
        </div>
        <div className="flex items-center gap-2">
          <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${badge.className}`}>
            {badge.label}
          </span>
          <svg
            className={`h-4 w-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M5 7.5L10 12.5L15 7.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </button>
      {open ? <div className="border-t border-slate-800/60 px-4 pb-4 pt-3 space-y-4">{children}</div> : null}
    </section>
  );
}

const MODEL_OPTIONS = [
  { label: "GPT-4.1 Mini", value: "gpt-4.1-mini" },
  { label: "GPT-4o", value: "gpt-4o" },
  { label: "GPT-5 Codex", value: "gpt-5-codex" },
  { label: "GPT-4 Turbo", value: "gpt-4-turbo" },
];

const EMBEDDING_OPTIONS = [
  { label: "text-embedding-3-large", value: "text-embedding-3-large" },
  { label: "text-embedding-3-small", value: "text-embedding-3-small" },
  { label: "text-embedding-ada-002", value: "text-embedding-ada-002" },
];

const PROJECTION_OPTIONS: Array<{ value: ProjectionMethod; label: string; description: string }> = [
  { value: "umap", label: "UMAP", description: "Balanced local/global structure; best default for semantic clustering." },
  { value: "tsne", label: "t-SNE", description: "Sharper local clusters; great for deep dives, less stable between runs." },
  { value: "pca", label: "PCA", description: "Linear baseline to sanity-check embeddings and variance." },
];

const UMAP_PRESETS: Array<{ value: UmapPreset; label: string; description: string }> = [
  { value: "balanced", label: "Balanced", description: "Good separation with contextual smoothing." },
  { value: "tight", label: "Local", description: "Emphasise micro-clusters and nuanced phrasing." },
  { value: "global", label: "Global", description: "Hold macro structure steady for comparisons." },
];

const METRIC_OPTIONS: Array<{ value: UmapMetric; label: string }> = [
  { value: "cosine", label: "Cosine" },
  { value: "euclidean", label: "Euclidean" },
  { value: "manhattan", label: "Manhattan" },
];

const TEMPERATURE_MAX = 2;
const TOP_P_MAX = 1;
const N_MAX = 500;
const CHUNK_MIN = 2;
const CHUNK_MAX = 200;
const POINT_SIZE_MIN = 1;
const POINT_SIZE_MAX = 24;
const POINT_SPREAD_MAX = 8;
const EDGE_K_MIN = 5;
const EDGE_K_MAX = 40;
export const ControlsPanel = memo(function ControlsPanel() {
  const [showUmapHelp, setShowUmapHelp] = useState(false);
  const { generate, exportDataset, error, isLoading } = useRunWorkflow();
  const {
    prompt,
    setPrompt,
    systemPrompt,
    setSystemPrompt,
    n,
    setN,
    temperature,
    setTemperature,
    topP,
    setTopP,
    model,
    setModel,
    embeddingModel,
    setEmbeddingModel,
    seed,
    setSeed,
    maxTokens,
    setMaxTokens,
    jitterToken,
    setJitterToken,
    useCache,
    setUseCache,
    chunkSize,
    setChunkSize,
    chunkOverlap,
    setChunkOverlap,
    projectionMethod,
    setProjectionMethod,
    projectionWarnings,
    isProjectionLoading,
    umapNNeighbors,
    setUmapNNeighbors,
    umapMinDist,
    setUmapMinDist,
    umapMetric,
    setUmapMetric,
    umapSeed,
    setUmapSeed,
    umapPreset,
    setUmapPreset,
    viewMode,
    setViewMode,
    levelMode,
    setLevelMode,
    pointSize,
    setPointSize,
    spreadFactor,
    setSpreadFactor,
    showDensity,
    setShowDensity,
    showEdges,
    setShowEdges,
    showParentThreads,
    setShowParentThreads,
    showNeighborSpokes,
    setShowNeighborSpokes,
    showDuplicatesOnly,
    setShowDuplicatesOnly,
    graphEdgeK,
    setGraphEdgeK,
    graphEdgeThreshold,
    setGraphEdgeThreshold,
    segmentGraphAutoSimplified,
    segmentGraphError,
    results,
    roleVisibility,
    toggleRole,
    setRolesVisibility,
    selectTopOutliers,
    selectTopSegmentOutliers,
    exportFormat,
    setExportFormat,
    exportIncludeProvenance,
    setExportIncludeProvenance,
    viewportBounds,
    setHistoryOpen,
    isHistoryOpen,
    isGenerating,
    clusterAlgo,
    hdbscanMinClusterSize,
    hdbscanMinSamples,
    setClusterParams,
    recomputeClusters,
    isRecomputingClusters,
  } = useRunStore((state) => ({
    prompt: state.prompt,
    setPrompt: state.setPrompt,
    systemPrompt: state.systemPrompt,
    setSystemPrompt: state.setSystemPrompt,
    n: state.n,
    setN: state.setN,
    temperature: state.temperature,
    setTemperature: state.setTemperature,
    topP: state.topP,
    setTopP: state.setTopP,
    model: state.model,
    setModel: state.setModel,
    embeddingModel: state.embeddingModel,
    setEmbeddingModel: state.setEmbeddingModel,
    seed: state.seed,
    setSeed: state.setSeed,
    maxTokens: state.maxTokens,
    setMaxTokens: state.setMaxTokens,
    jitterToken: state.jitterToken,
    setJitterToken: state.setJitterToken,
    useCache: state.useCache,
    setUseCache: state.setUseCache,
    chunkSize: state.chunkSize,
    setChunkSize: state.setChunkSize,
    chunkOverlap: state.chunkOverlap,
    setChunkOverlap: state.setChunkOverlap,
    projectionMethod: state.projectionMethod,
    setProjectionMethod: state.setProjectionMethod,
    projectionWarnings: state.projectionWarnings,
    isProjectionLoading: state.isProjectionLoading,
    umapNNeighbors: state.umapNNeighbors,
    setUmapNNeighbors: state.setUmapNNeighbors,
    umapMinDist: state.umapMinDist,
    setUmapMinDist: state.setUmapMinDist,
    umapMetric: state.umapMetric,
    setUmapMetric: state.setUmapMetric,
    umapSeed: state.umapSeed,
    setUmapSeed: state.setUmapSeed,
    umapPreset: state.umapPreset,
    setUmapPreset: state.setUmapPreset,
    viewMode: state.viewMode,
    setViewMode: state.setViewMode,
    levelMode: state.levelMode,
    setLevelMode: state.setLevelMode,
    pointSize: state.pointSize,
    setPointSize: state.setPointSize,
    spreadFactor: state.spreadFactor,
    setSpreadFactor: state.setSpreadFactor,
    showDensity: state.showDensity,
    setShowDensity: state.setShowDensity,
    showEdges: state.showEdges,
    setShowEdges: state.setShowEdges,
    showParentThreads: state.showParentThreads,
    setShowParentThreads: state.setShowParentThreads,
    showNeighborSpokes: state.showNeighborSpokes,
    setShowNeighborSpokes: state.setShowNeighborSpokes,
    showDuplicatesOnly: state.showDuplicatesOnly,
    setShowDuplicatesOnly: state.setShowDuplicatesOnly,
    graphEdgeK: state.graphEdgeK,
    setGraphEdgeK: state.setGraphEdgeK,
    graphEdgeThreshold: state.graphEdgeThreshold,
    setGraphEdgeThreshold: state.setGraphEdgeThreshold,
    segmentGraphAutoSimplified: state.segmentGraphAutoSimplified,
    segmentGraphError: state.segmentGraphError,
    results: state.results,
    roleVisibility: state.roleVisibility,
    toggleRole: state.toggleRole,
    setRolesVisibility: state.setRolesVisibility,
    selectTopOutliers: state.selectTopOutliers,
    selectTopSegmentOutliers: state.selectTopSegmentOutliers,
    exportFormat: state.exportFormat,
    setExportFormat: state.setExportFormat,
    exportIncludeProvenance: state.exportIncludeProvenance,
    setExportIncludeProvenance: state.setExportIncludeProvenance,
    viewportBounds: state.viewportBounds,
    setHistoryOpen: state.setHistoryOpen,
    isHistoryOpen: state.isHistoryOpen,
    isGenerating: state.isGenerating,
    clusterAlgo: state.clusterAlgo,
    hdbscanMinClusterSize: state.hdbscanMinClusterSize,
    hdbscanMinSamples: state.hdbscanMinSamples,
    setClusterParams: state.setClusterParams,
    recomputeClusters: state.recomputeClusters,
    isRecomputingClusters: state.isRecomputingClusters,
  }));

  const isBusy = isGenerating || isLoading;
  const promptMissing = !prompt.trim();
  const chunkSizeValue = Math.min(Math.max(chunkSize ?? CHUNK_MIN, CHUNK_MIN), CHUNK_MAX);
  const overlapMax = Math.max(0, chunkSizeValue - 1);
  const overlapValue = Math.min(Math.max(chunkOverlap ?? 0, 0), overlapMax);
  const neighborCap = results?.points?.length
    ? Math.max(EDGE_K_MIN, Math.min(EDGE_K_MAX, Math.max(EDGE_K_MIN, results.points.length - 1)))
    : EDGE_K_MAX;
  const viewportAvailable = Boolean(viewportBounds);
  const thresholdPercent = Math.round((graphEdgeThreshold ?? 0) * 100);
  const exportModeLabel = levelMode === "segments" ? "segments" : "responses";
  const projectionWarningsList = projectionWarnings ?? [];
  const cacheButtonClass = `rounded-full border px-3 py-1 text-[11px] uppercase tracking-wide transition ${
    useCache ? "border-cyan-500 bg-cyan-500/10 text-cyan-200" : "border-slate-700/60 text-slate-200 hover:border-cyan-400 hover:text-cyan-200"
  }`;
  const roles = useMemo(() => {
    if (!results) {
      return [] as string[];
    }
    const tokens = new Set<string>();
    results.segments.forEach((segment) => {
      if (segment.role) {
        tokens.add(segment.role.toLowerCase());
      }
    });
    return Array.from(tokens).sort();
  }, [results]);
  const allRolesActive = roles.every((role) => roleVisibility[role] ?? true);
  const activeRoleCount = roles.filter((role) => roleVisibility[role] ?? true).length;
  const graphStatusLabel = segmentGraphAutoSimplified ? "Graph detail: simplified" : "Graph detail: full";
  const graphControlsDisabled = !results || results.points.length < EDGE_K_MIN;

  const handleExportRun = useCallback(async () => {
    try {
      await exportDataset({
        scope: "run",
        format: exportFormat,
        mode: levelMode === "segments" ? "segments" : "responses",
        include: exportIncludeProvenance ? ["provenance"] : undefined,
      });
    } catch (err) {
      console.error(err);
    }
  }, [exportDataset, exportFormat, exportIncludeProvenance, levelMode]);

  const handleExportViewport = useCallback(async () => {
    try {
      await exportDataset({
        scope: "viewport",
        format: exportFormat,
        mode: levelMode === "segments" ? "segments" : "responses",
        include: exportIncludeProvenance ? ["provenance"] : undefined,
      });
    } catch (err) {
      console.error(err);
    }
  }, [exportDataset, exportFormat, exportIncludeProvenance, levelMode]);

  const handleProjectionMethodChange = useCallback(
    (method: ProjectionMethod) => {
      setProjectionMethod(method).catch((err) => console.error(err));
    },
    [setProjectionMethod],
  );

  const handleGraphEdgeKChange = useCallback(
    (value: number) => {
      const clamped = Math.min(Math.max(value, EDGE_K_MIN), neighborCap);
      setGraphEdgeK(clamped).catch((err) => console.error(err));
    },
    [neighborCap, setGraphEdgeK],
  );

  const handleGraphEdgeThresholdChange = useCallback(
    (value: number) => {
      const clamped = Math.min(Math.max(value, 0), 1);
      setGraphEdgeThreshold(clamped).catch((err) => console.error(err));
    },
    [setGraphEdgeThreshold],
  );

  const handleRecomputeClusters = useCallback(() => {
    recomputeClusters().catch((err) => console.error(err));
  }, [recomputeClusters]);

  const handleGenerate = useCallback(() => {
    generate();
  }, [generate]);
  return (
    <aside className="flex h-full min-h-0 w-[380px] shrink-0 flex-col border-l border-slate-900/80 bg-slate-950/90 text-sm text-slate-100">
      <div className="border-b border-slate-900/60 px-5 pb-4 pt-5">
        <h1 className="text-lg font-semibold tracking-tight">Semantic Landscape Sampler</h1>
        <p className="mt-1 text-[11px] text-slate-500">Explore diverse LLM responses, surface clusters, and export spatial analytics.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setHistoryOpen(!isHistoryOpen)}
            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
          >
            {isHistoryOpen ? "Hide run history" : "Show run history"}
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 space-y-4 overflow-y-auto px-5 pb-6 pt-4">
        <ControlSection
          title="Run Setup"
          description="Configure prompts and sampling parameters. Changes here require regenerating the landscape."
          tone="rerun"
        >
          <div className="space-y-2">
            <label className="text-xs uppercase tracking-[0.22rem] text-slate-400">Prompt</label>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={5}
              className="w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40"
              placeholder="Describe the question you want the models to explore."
            />
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs uppercase tracking-[0.22rem] text-slate-400">
              System message
              <InfoTooltip text="Set tone, persona, or safety constraints. Leave blank to keep the default assistant behaviour." />
            </label>
            <textarea
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              rows={3}
              className="w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40"
              placeholder="Optional: reinforce persona, guardrails, or instructions."
            />
          </div>

          <div className="space-y-3 rounded-xl border border-slate-800/60 bg-slate-900/40 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-slate-100">Embedding cache</p>
                <p className="text-[11px] text-slate-500">Reuse embeddings for repeated segments to reduce latency and cost.</p>
              </div>
              <button type="button" onClick={() => setUseCache(!useCache)} className={cacheButtonClass}>
                {useCache ? "Enabled" : "Disabled"}
              </button>
            </div>
            <p className="text-[11px] text-slate-500">Cached segments are flagged in the details drawer. Disable for fresh embeddings.</p>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Model</span>
              <select
                value={model}
                onChange={(event) => setModel(event.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                {MODEL_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Embedding model</span>
              <select
                value={embeddingModel}
                onChange={(event) => setEmbeddingModel(event.target.value)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                {EMBEDDING_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Samples (N)</label>
              <p className="text-[11px] text-slate-500">Collect more responses for broader coverage.</p>
              <input
                type="range"
                min={1}
                max={N_MAX}
                value={n}
                onChange={(event) => setN(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{n}</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Temperature</label>
              <p className="text-[11px] text-slate-500">Lower values stay focused; higher encourages exploration.</p>
              <input
                type="range"
                min={0}
                max={TEMPERATURE_MAX}
                step={0.05}
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{temperature.toFixed(2)}</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Top-p</label>
              <p className="text-[11px] text-slate-500">Combine with temperature to control adventurous wording.</p>
              <input
                type="range"
                min={0}
                max={TOP_P_MAX}
                step={0.05}
                value={topP}
                onChange={(event) => setTopP(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{topP.toFixed(2)}</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Jitter token</label>
              <p className="text-[11px] text-slate-500">Inject per-sample noise. Leave blank for automatic jitter.</p>
              <input
                type="text"
                value={jitterToken ?? ""}
                onChange={(event) => setJitterToken(event.target.value ? event.target.value : null)}
                placeholder="auto"
                className="mt-2 w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              />
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Seed</span>
              <input
                type="number"
                min={0}
                value={seed ?? ""}
                onChange={(event) => setSeed(event.target.value ? Number(event.target.value) : null)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              />
            </label>
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Max tokens</span>
              <input
                type="number"
                min={64}
                max={4096}
                value={maxTokens ?? ""}
                onChange={(event) => setMaxTokens(event.target.value ? Number(event.target.value) : null)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              />
            </label>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Segment chunk size</label>
              <p className="text-[11px] text-slate-500">Controls sentence splitting for segment analysis.</p>
              <input
                type="range"
                min={CHUNK_MIN}
                max={CHUNK_MAX}
                value={chunkSizeValue}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  const clamped = Math.min(Math.max(value, CHUNK_MIN), CHUNK_MAX);
                  setChunkSize(clamped);
                  if (overlapValue > clamped - 1) {
                    setChunkOverlap(Math.max(0, clamped - 1));
                  }
                }}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{chunkSizeValue} tokens</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Segment overlap</label>
              <p className="text-[11px] text-slate-500">Increase to preserve context between neighbouring segments.</p>
              <input
                type="range"
                min={0}
                max={overlapMax}
                value={overlapValue}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  const clamped = Math.min(Math.max(value, 0), overlapMax);
                  setChunkOverlap(clamped);
                }}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">
                {overlapValue} tokens (max {overlapMax})
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={isBusy || promptMissing}
              className="inline-flex items-center justify-center rounded-full border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-cyan-200 transition hover:border-cyan-400 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:border-slate-700/60 disabled:bg-slate-800/40 disabled:text-slate-500"
            >
              {isBusy ? "Working..." : "Generate landscape"}
            </button>
            {error ? <p className="text-[11px] text-rose-400">{error}</p> : null}
            {promptMissing ? <p className="text-[11px] text-slate-500">Enter a prompt to enable generation.</p> : null}
          </div>
        </ControlSection>
        <ControlSection
          title="Projection & Layout"
          description="Switch reductions, reuse cached layouts, and tune UMAP without re-embedding."
        >
          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Projection method</span>
              <select
                value={projectionMethod}
                onChange={(event) => handleProjectionMethodChange(event.target.value as ProjectionMethod)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                {PROJECTION_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="space-y-2 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Method overview</span>
              <p className="text-[11px] leading-relaxed text-slate-500">
                {PROJECTION_OPTIONS.find((option) => option.value === projectionMethod)?.description}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase tracking-wide text-slate-400">UMAP presets</span>
              <button
                type="button"
                onClick={() => setShowUmapHelp((value) => !value)}
                className="text-[11px] text-cyan-300 transition hover:text-cyan-200"
              >
                {showUmapHelp ? "Hide tips" : "Show tips"}
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {UMAP_PRESETS.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  onClick={() => setUmapPreset(preset.value)}
                  className={`rounded-full border px-3 py-1 text-[11px] transition ${
                    umapPreset === preset.value
                      ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                  }`}
                  title={preset.description}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            {showUmapHelp ? (
              <ul className="space-y-1 rounded-lg border border-slate-800/60 bg-slate-900/40 p-3 text-[11px] text-slate-400">
                <li>• Balanced keeps macro topology stable for comparisons.</li>
                <li>• Local sharpens clusters; great for rhetorical nuance.</li>
                <li>• Global widens spacing for cross-run overlays.</li>
              </ul>
            ) : null}
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">UMAP n_neighbors</label>
              <input
                type="range"
                min={2}
                max={200}
                value={umapNNeighbors}
                onChange={(event) => setUmapNNeighbors(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{umapNNeighbors}</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">UMAP min_dist</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={umapMinDist}
                onChange={(event) => setUmapMinDist(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{umapMinDist.toFixed(2)}</span>
            </div>
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">UMAP metric</span>
              <select
                value={umapMetric}
                onChange={(event) => setUmapMetric(event.target.value as UmapMetric)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                {METRIC_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">UMAP seed</span>
              <input
                type="number"
                min={0}
                value={umapSeed ?? ""}
                onChange={(event) => setUmapSeed(event.target.value ? Number(event.target.value) : null)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
                placeholder="auto"
              />
            </label>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <span className="text-xs uppercase tracking-wide text-slate-400">View mode</span>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => setViewMode("3d")}
                  className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                    viewMode === "3d"
                      ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                  }`}
                >
                  3D scene
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("2d")}
                  className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                    viewMode === "2d"
                      ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                  }`}
                >
                  2D slice
                </button>
              </div>
            </div>
            <div>
              <span className="text-xs uppercase tracking-wide text-slate-400">Data mode</span>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => setLevelMode("responses")}
                  className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                    levelMode === "responses"
                      ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                  }`}
                >
                  Responses
                </button>
                <button
                  type="button"
                  onClick={() => setLevelMode("segments")}
                  className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${
                    levelMode === "segments"
                      ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"
                  }`}
                >
                  Segments
                </button>
              </div>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Point size</label>
              <input
                type="range"
                min={POINT_SIZE_MIN}
                max={POINT_SIZE_MAX}
                value={pointSize}
                onChange={(event) => setPointSize(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{pointSize.toFixed(0)}</span>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Spread factor</label>
              <input
                type="range"
                min={0}
                max={POINT_SPREAD_MAX}
                step={0.1}
                value={spreadFactor}
                onChange={(event) => setSpreadFactor(Number(event.target.value))}
                className="mt-2 w-full"
              />
              <span className="text-xs text-slate-400">{spreadFactor.toFixed(1)}</span>
            </div>
          </div>

          {isProjectionLoading ? <p className="text-[11px] text-cyan-300">Projection recalculating...</p> : null}

          {projectionWarningsList.length > 0 ? (
            <div className="space-y-1 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-[11px] text-amber-200">
              <p className="font-semibold uppercase tracking-[0.2rem]">Projection warnings</p>
              <ul className="list-disc pl-4 leading-relaxed text-amber-100">
                {projectionWarningsList.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </ControlSection>
        <ControlSection
          title="Visibility & Filters"
          description="Toggle overlays, similarity graphs, discourse roles, and duplicate highlighting."
        >
          <div className="grid gap-2">
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showDensity}
                onChange={(event) => setShowDensity(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Density overlay
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showEdges}
                onChange={(event) => setShowEdges(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Similarity edges
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showParentThreads}
                onChange={(event) => setShowParentThreads(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Parent threads
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showNeighborSpokes}
                onChange={(event) => setShowNeighborSpokes(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Neighbor spokes
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={showDuplicatesOnly}
                onChange={(event) => setShowDuplicatesOnly(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Show duplicates only
            </label>
          </div>

          <div className="space-y-2 rounded-lg border border-slate-800/60 bg-slate-900/40 p-3">
            <div className="flex items-center justify-between text-xs">
              <span className="uppercase tracking-wide text-slate-400">Similarity graph</span>
              <span className="text-slate-500">{graphStatusLabel}</span>
            </div>
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500">Neighbor count (k)</label>
              <input
                type="range"
                min={EDGE_K_MIN}
                max={neighborCap}
                disabled={graphControlsDisabled}
                value={Math.min(Math.max(graphEdgeK ?? EDGE_K_MIN, EDGE_K_MIN), neighborCap)}
                onChange={(event) => handleGraphEdgeKChange(Number(event.target.value))}
                className="mt-2 w-full disabled:opacity-50"
              />
              <span className="text-[11px] text-slate-500">
                {graphControlsDisabled ? "Run a sample to enable edge tuning." : `${graphEdgeK ?? EDGE_K_MIN} neighbors`}
              </span>
            </div>
            <div>
              <label className="text-[11px] uppercase tracking-wide text-slate-500">Similarity threshold</label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                disabled={graphControlsDisabled}
                value={Math.min(Math.max(graphEdgeThreshold ?? 0, 0), 1)}
                onChange={(event) => handleGraphEdgeThresholdChange(Number(event.target.value))}
                className="mt-2 w-full disabled:opacity-50"
              />
              <span className="text-[11px] text-slate-500">{thresholdPercent}% cosine</span>
            </div>
            {segmentGraphError ? <p className="text-[11px] text-rose-300">Edge simplification error: {segmentGraphError}</p> : null}
          </div>

          {roles.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="uppercase tracking-wide text-slate-400">Discourse roles</span>
                <span className="text-slate-500">
                  {activeRoleCount}/{roles.length} active
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {roles.map((role) => {
                  const active = roleVisibility[role] ?? true;
                  return (
                    <button
                      key={role}
                      type="button"
                      onClick={() => toggleRole(role)}
                      className={`rounded-full border px-3 py-1 text-[11px] capitalize transition ${
                        active
                          ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                          : "border-slate-700/60 text-slate-400 hover:border-cyan-400 hover:text-cyan-200"
                      }`}
                    >
                      {role}
                    </button>
                  );
                })}
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setRolesVisibility(roles, true)}
                  className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
                  disabled={roles.length === 0 || allRolesActive}
                >
                  Show all
                </button>
                <button
                  type="button"
                  onClick={() => setRolesVisibility(roles, false)}
                  className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
                  disabled={roles.length === 0 || !allRolesActive}
                >
                  Hide all
                </button>
              </div>
            </div>
          ) : (
            <p className="text-[11px] text-slate-500">Run segment mode to unlock role filtering.</p>
          )}
        </ControlSection>
        <ControlSection
          title="Selection Shortcuts"
          description="Highlight representative outliers for quick reviews."
        >
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={!results}
              onClick={() => selectTopOutliers()}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500"
            >
              Select top response outliers
            </button>
            <button
              type="button"
              disabled={!results}
              onClick={() => selectTopSegmentOutliers()}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500"
            >
              Select top segment outliers
            </button>
          </div>
          <p className="text-[11px] text-slate-500">Outlier search runs on the active level mode.</p>
        </ControlSection>

        <ControlSection
          title="Export"
          description="Download raw responses, segments, hulls, and provenance for downstream analysis."
        >
          <div className="grid gap-3 md:grid-cols-2">
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Format</span>
              <select
                value={exportFormat}
                onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                <option value="json">JSON</option>
                <option value="jsonl">JSONL</option>
                <option value="csv">CSV</option>
                <option value="parquet">Parquet</option>
              </select>
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={exportIncludeProvenance}
                onChange={(event) => setExportIncludeProvenance(event.target.checked)}
                className="h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40"
              />
              Include provenance metadata
            </label>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleExportRun}
              disabled={isBusy || !results}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-200 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500"
            >
              Export {exportModeLabel}
            </button>
            <button
              type="button"
              onClick={handleExportViewport}
              disabled={isBusy || !viewportAvailable}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-200 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500"
            >
              Export current viewport
            </button>
          </div>
          {!viewportAvailable ? (
            <p className="text-[11px] text-slate-500">Pan or zoom the scene to define a viewport export.</p>
          ) : null}
        </ControlSection>

        <ControlSection
          title="Cluster Tuning"
          description="Adjust clustering parameters and recompute without calling the LLM."
        >
          <div className="space-y-3">
            <label className="space-y-1 text-xs">
              <span className="uppercase tracking-wide text-slate-400">Algorithm</span>
              <select
                value={clusterAlgo}
                onChange={(event) => setClusterParams({ algo: event.target.value as "hdbscan" | "kmeans" })}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
              >
                <option value="hdbscan">HDBSCAN</option>
                <option value="kmeans">KMeans</option>
              </select>
            </label>

            {clusterAlgo === "hdbscan" ? (
              <div className="grid gap-3 md:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="uppercase tracking-wide text-slate-400">Min cluster size</span>
                  <input
                    type="number"
                    min={2}
                    value={hdbscanMinClusterSize ?? 2}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (!Number.isNaN(value)) {
                        setClusterParams({ minClusterSize: Math.max(2, value) });
                      }
                    }}
                    className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
                  />
                </label>
                <label className="space-y-1 text-xs">
                  <span className="uppercase tracking-wide text-slate-400">Min samples</span>
                  <input
                    type="number"
                    min={1}
                    value={hdbscanMinSamples ?? 1}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (!Number.isNaN(value)) {
                        setClusterParams({ minSamples: Math.max(1, value) });
                      }
                    }}
                    className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
                  />
                </label>
              </div>
            ) : (
              <p className="text-[11px] text-slate-500">
                KMeans recomputes centroid assignments using the active embeddings.
              </p>
            )}

            <button
              type="button"
              onClick={handleRecomputeClusters}
              disabled={isRecomputingClusters || !results}
              className="inline-flex items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs font-medium text-cyan-200 transition hover:border-cyan-400 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:bg-slate-900/40 disabled:text-slate-500"
            >
              {isRecomputingClusters ? "Recomputing..." : "Recompute clusters"}
            </button>
            <p className="text-[11px] text-slate-500">
              Compare silhouette scores and centroid similarity across projection and embedding spaces.
            </p>
          </div>
        </ControlSection>
      </div>
    </aside>
  );
});




