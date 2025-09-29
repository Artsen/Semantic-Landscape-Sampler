/**
 * Control surface for sampling parameters, filters, exports, and quick selections.
 *
 * Components:
 *  - ControlsPanel: Binds UI controls to Zustand actions for configuring and triggering runs.
 */

import { memo, useMemo } from "react";

import type { RunWorkflow } from "@/hooks/useRunWorkflow";
import { useRunStore } from "@/store/runStore";

const MODEL_OPTIONS = [
  { label: "GPT-5", value: "gpt-5" },
  { label: "GPT-5 Mini", value: "gpt-5-mini" },
  { label: "GPT-5 Nano", value: "gpt-5-nano" },
  { label: "GPT-5 Chat Latest", value: "gpt-5-chat-latest" },
  { label: "GPT-5 Codex", value: "gpt-5-codex" },
  { label: "GPT-4.1", value: "gpt-4.1" },
  { label: "GPT-4.1 Mini", value: "gpt-4.1-mini" },
  { label: "GPT-4.1 Nano", value: "gpt-4.1-nano" },
  { label: "GPT-4o", value: "gpt-4o" },
  { label: "GPT-4o (2024-05-13)", value: "gpt-4o-2024-05-13" },
  { label: "GPT-4o Mini", value: "gpt-4o-mini" },
  { label: "o1", value: "o1" },
  { label: "o1 Pro", value: "o1-pro" },
  { label: "o3 Pro", value: "o3-pro" },
  { label: "o3", value: "o3" },
  { label: "o3 Deep Research", value: "o3-deep-research" },
  { label: "o4 Mini", value: "o4-mini" },
  { label: "o4 Mini Deep Research", value: "o4-mini-deep-research" },
  { label: "o3 Mini", value: "o3-mini" },
  { label: "o1 Mini", value: "o1-mini" },
  { label: "Codex Mini Latest", value: "codex-mini-latest" },
];

const EMBEDDING_OPTIONS = [
  { label: "text-embedding-3-large", value: "text-embedding-3-large" },
  { label: "text-embedding-3-small", value: "text-embedding-3-small" },
  { label: "text-embedding-ada-002", value: "text-embedding-ada-002" },
];

const TEMPERATURE_MAX = 2;
const TOP_P_MAX = 1;
const N_MAX = 500;
const POINT_SPREAD_MAX = 8;
const CHUNK_MIN = 1;
const CHUNK_MAX = 30;

type ControlsPanelProps = {
  workflow: RunWorkflow;
};

export const ControlsPanel = memo(function ControlsPanel({ workflow }: ControlsPanelProps) {
  const {
    prompt,
    systemPrompt,
    n,
    temperature,
    topP,
    model,
    seed,
    maxTokens,
    embeddingModel,
    chunkSize,
    chunkOverlap,
    pointSize,
    spreadFactor,
    showDensity,
    showEdges,
    showParentThreads,
    viewMode,
    levelMode,
    jitterToken,
    isHistoryOpen,
    isGenerating,
    results,
    progressStage,
    progressMessage,
    progressPercent,
    progressMetadata,
    roleVisibility,
    setPrompt,
    setSystemPrompt,
    setN,
    setTemperature,
    setTopP,
    setModel,
    setSeed,
    setMaxTokens,
    setEmbeddingModel,
    setChunkSize,
    setPointSize,
    setSpreadFactor,
    setShowDensity,
    setShowEdges,
    setShowParentThreads,
    setViewMode,
    setLevelMode,
    setJitterToken,
    setHistoryOpen,
    selectTopOutliers,
    selectTopSegmentOutliers,
    toggleRole,
    setRolesVisibility,
  } = useRunStore((state) => ({
    prompt: state.prompt,
    systemPrompt: state.systemPrompt,
    n: state.n,
    temperature: state.temperature,
    topP: state.topP,
    model: state.model,
    seed: state.seed,
    maxTokens: state.maxTokens,
    embeddingModel: state.embeddingModel,
    chunkSize: state.chunkSize,
    chunkOverlap: state.chunkOverlap,
    pointSize: state.pointSize,
    spreadFactor: state.spreadFactor,
    showDensity: state.showDensity,
    showEdges: state.showEdges,
    showParentThreads: state.showParentThreads,
    viewMode: state.viewMode,
    levelMode: state.levelMode,
    jitterToken: state.jitterToken,
    isHistoryOpen: state.isHistoryOpen,
    isGenerating: state.isGenerating,
    results: state.results,
    progressStage: state.progressStage,
    progressMessage: state.progressMessage,
    progressPercent: state.progressPercent,
    progressMetadata: state.progressMetadata,
    roleVisibility: state.roleVisibility,
    setPrompt: state.setPrompt,
    setSystemPrompt: state.setSystemPrompt,
    setN: state.setN,
    setTemperature: state.setTemperature,
    setTopP: state.setTopP,
    setModel: state.setModel,
    setSeed: state.setSeed,
    setMaxTokens: state.setMaxTokens,
    setEmbeddingModel: state.setEmbeddingModel,
    setChunkSize: state.setChunkSize,
    setChunkOverlap: state.setChunkOverlap,
    setPointSize: state.setPointSize,
    setSpreadFactor: state.setSpreadFactor,
    setHistoryOpen: state.setHistoryOpen,
    setShowDensity: state.setShowDensity,
    setShowEdges: state.setShowEdges,
    setShowParentThreads: state.setShowParentThreads,
    setViewMode: state.setViewMode,
    setLevelMode: state.setLevelMode,
    setJitterToken: state.setJitterToken,
    selectTopOutliers: state.selectTopOutliers,
    selectTopSegmentOutliers: state.selectTopSegmentOutliers,
    toggleRole: state.toggleRole,
    setRolesVisibility: state.setRolesVisibility,
  }));
  const { generate, exportRun, error, isLoading } = workflow;

  const createExportHandler = (format: "json" | "csv") => async () => {
    try {
      await exportRun(format);
    } catch (err) {
      console.error(err);
    }
  };

  const isBusy = isGenerating || isLoading;
  const progressPercentValue = useMemo(() => {
    if (progressPercent == null) {
      return null;
    }
    return Math.max(0, Math.min(1, progressPercent));
  }, [progressPercent]);

  const progressPercentLabel = useMemo(() => {
    if (progressPercentValue == null) {
      return null;
    }
    return Math.round(progressPercentValue * 100) + "%";
  }, [progressPercentValue]);

  const progressStageLabel = useMemo(() => {
    if (!progressStage) {
      return null;
    }
    return progressStage
      .replace(/[-_]/g, " ")
      .replace(/\w/g, (char) => char.toUpperCase());
  }, [progressStage]);

  const progressDetails = useMemo(() => {
    if (!progressMetadata) {
      return [] as Array<[string, unknown]>;
    }
    return Object.entries(progressMetadata)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .slice(0, 6);
  }, [progressMetadata]);

  const formatProgressKey = (key: string) =>
    key
      .replace(/[-_]/g, " ")
      .replace(/\w/g, (char) => char.toUpperCase());

  const formatProgressValue = (value: unknown): string => {
    if (typeof value === "number") {
      return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
    }
    if (typeof value === "boolean") {
      return value ? "yes" : "no";
    }
    if (Array.isArray(value)) {
      return value.length ? value.join(", ") : "—";
    }
    return String(value);
  };

  const seedValue = seed ?? "";

  const densityLabel = useMemo(
    () => (showDensity ? "Disable Density Overlay" : "Enable Density Overlay"),
    [showDensity],
  );
  const edgesLabel = useMemo(
    () => (showEdges ? "Hide Similarity Edges" : "Show Similarity Edges"),
    [showEdges],
  );
  const parentThreadsLabel = useMemo(
    () => (showParentThreads ? "Hide Parent Threads" : "Show Parent Threads"),
    [showParentThreads],
  );
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
  const hasRoles = roles.length > 0;
  const allRolesActive = useMemo(() => {
    if (!hasRoles) {
      return true;
    }
    return roles.every((role) => roleVisibility[role] ?? true);
  }, [roles, roleVisibility, hasRoles]);
  const activeRoleCount = roles.filter((role) => roleVisibility[role] ?? true).length;

  return (
    <aside className="glass-panel flex h-full w-[360px] shrink-0 flex-col overflow-hidden border border-slate-800/50 p-6 text-sm">
      <div className="shrink-0">
        <header className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">Semantic Landscape Sampler</h1>
          <p className="text-slate-400">
            Explore diverse LLM responses, compare clusters, and surface exemplars with interactive spatial tooling.
          </p>
        </header>
        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={() => setHistoryOpen(!isHistoryOpen)}
            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
          >
            {isHistoryOpen ? 'Hide Run History' : 'Show Run History'}
          </button>
        </div>
      </div>

      {progressMessage ? (
        <section className="mt-4 rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-4 text-xs text-cyan-100">
          <header className="flex items-center justify-between text-[11px] uppercase tracking-[0.22rem] text-cyan-200">
            <span>Backend Progress</span>
            {progressPercentLabel ? <span>{progressPercentLabel}</span> : null}
          </header>
          <p className="mt-2 text-sm font-semibold text-cyan-100">
            {progressStageLabel ?? (isBusy ? "Processing" : "Status")}
          </p>
          <p className="mt-1 text-xs text-cyan-100/80">{progressMessage}</p>
          {progressPercentValue != null ? (
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-cyan-500/20">
              <div
                className="h-full bg-cyan-400 transition-all duration-500"
                style={{ width: Math.min(100, Math.max(2, progressPercentValue * 100)) + "%" }}
              />
            </div>
          ) : null}
          {progressDetails.length ? (
            <ul className="mt-3 space-y-1 text-[11px] font-medium">
              {progressDetails.map(([key, value]) => (
                <li key={key} className="flex justify-between gap-4">
                  <span className="text-cyan-200">{formatProgressKey(key)}</span>
                  <span className="text-cyan-100/90">{formatProgressValue(value)}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}

      <div className="mt-4 flex-1 overflow-y-auto pr-2 pb-6 scrollbar-thin space-y-6">
      <section className="space-y-2">
        <label className="text-xs uppercase tracking-[0.22rem] text-slate-400">Prompt</label>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          rows={5}
          className="w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40"
        />
      </section>

      <section className="mt-4 space-y-2">
        <label className="text-xs uppercase tracking-[0.22rem] text-slate-400">System Message</label>
        <textarea
          value={systemPrompt}
          onChange={(event) => setSystemPrompt(event.target.value)}
          rows={3}
          placeholder="Leave blank to use the default system instructions."
          className="w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40"
        />
      </section>

      <section className="mt-4 space-y-2">
        <label className="text-xs uppercase tracking-[0.22rem] text-slate-400">Segment Chunk Size</label>
        <input
          type="range"
          min={CHUNK_MIN}
          max={CHUNK_MAX}
          step={1}
          value={chunkSize}
          onChange={(event) => setChunkSize(Number(event.target.value))}
          className="mt-2 w-full"
        />
        <span className="text-xs text-slate-400">{chunkSize} word{chunkSize === 1 ? "" : "s"} per chunk</span>
      </section>

      <section className="space-y-2">
        <label className="text-xs uppercase tracking-[0.22rem] text-slate-400">Segment Overlap</label>
        <input
          type="range"
          min={0}
          max={Math.max(0, chunkSize - 1)}
          step={1}
          value={Math.min(chunkOverlap, Math.max(0, chunkSize - 1))}
          onChange={(event) => setChunkOverlap(Number(event.target.value))}
          className="mt-2 w-full"
          disabled={chunkSize <= 1}
        />
        <span className="text-xs text-slate-400">{chunkOverlap} word{chunkOverlap === 1 ? "" : "s"} overlap</span>
      </section>

        <section className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Samples (N)</label>
            <input
              type="range"
              min={1}
              max={N_MAX}
              step={1}
              value={n}
              onChange={(event) => setN(Number(event.target.value))}
              className="mt-2 w-full"
            />
            <span className="text-xs text-slate-400">{n}</span>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Temperature</label>
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
            <label className="text-xs uppercase tracking-wide text-slate-400">Point Size</label>
            <input
              type="range"
              min={0.02}
              max={0.18}
              step={0.01}
              value={pointSize}
              onChange={(event) => setPointSize(Number(event.target.value))}
              className="mt-2 w-full"
            />
            <span className="text-xs text-slate-400">{pointSize.toFixed(2)}</span>
          <label className="mt-4 block text-xs uppercase tracking-wide text-slate-400">Point Spread</label>
          <input
            type="range"
            min={0.6}
            max={POINT_SPREAD_MAX}
            step={0.1}
            value={spreadFactor}
              onChange={(event) => setSpreadFactor(Number(event.target.value))}
              className="mt-2 w-full"
            />
            <span className="text-xs text-slate-400">{spreadFactor.toFixed(2)}</span>
          </div>
        </section>

        <section className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setLevelMode("responses")}
            className={`rounded-lg border px-3 py-2 text-xs font-medium transition focus:outline-none focus:ring-1 ${
              levelMode === "responses"
                ? "border-cyan-400/80 bg-cyan-500/10 text-cyan-200"
                : "border-slate-700/60 bg-slate-900/60 text-slate-300 hover:border-cyan-400/50 hover:text-cyan-200"
            }`}
            disabled={isBusy}
          >
            Responses View
          </button>
          <button
            type="button"
            onClick={() => setLevelMode("segments")}
            className={`rounded-lg border px-3 py-2 text-xs font-medium transition focus:outline-none focus:ring-1 ${
              levelMode === "segments"
                ? "border-cyan-400/80 bg-cyan-500/10 text-cyan-200"
                : "border-slate-700/60 bg-slate-900/60 text-slate-300 hover:border-cyan-400/50 hover:text-cyan-200"
            }`}
            disabled={isBusy}
          >
            Segment Mesh
          </button>
        </section>

        <section className="space-y-4">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Model</label>
            <select
              value={model}
              onChange={(event) => setModel(event.target.value)}
              className="mt-2 w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
            >
              {MODEL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Embedding model</label>
            <select
              value={embeddingModel}
              onChange={(event) => setEmbeddingModel(event.target.value)}
              className="mt-2 w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
            >
              {EMBEDDING_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <label className="space-y-1">
              <span className="block text-xs uppercase tracking-wide text-slate-400">Seed</span>
              <input
                type="number"
                value={seedValue}
                min={0}
                onChange={(event) => setSeed(event.target.value ? Number(event.target.value) : null)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-slate-100 focus:border-cyan-400 focus:outline-none"
              />
            </label>
            <label className="space-y-1">
              <span className="block text-xs uppercase tracking-wide text-slate-400">Max tokens</span>
              <input
                type="number"
                value={maxTokens ?? ""}
                min={64}
                max={4096}
                onChange={(event) => setMaxTokens(event.target.value ? Number(event.target.value) : null)}
                className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-slate-100 focus:border-cyan-400 focus:outline-none"
              />
            </label>
          </div>
          <label className="space-y-1 text-sm">
            <span className="block text-xs uppercase tracking-wide text-slate-400">Jitter token</span>
            <input
              type="text"
              value={jitterToken ?? ""}
              placeholder="auto"
              onChange={(event) => setJitterToken(event.target.value || null)}
              className="w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-slate-100 focus:border-cyan-400 focus:outline-none"
            />
          </label>
        </section>

        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setViewMode(viewMode === "3d" ? "2d" : "3d")}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              {viewMode === "3d" ? "Switch to 2D" : "Switch to 3D"}
            </button>
            <button
              type="button"
              onClick={() => setShowEdges(!showEdges)}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              {edgesLabel}
            </button>
            <button
              type="button"
              onClick={() => setShowParentThreads(!showParentThreads)}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              {parentThreadsLabel}
            </button>
            <button
              type="button"
              onClick={() => setShowDensity(!showDensity)}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              {densityLabel}
            </button>
          </div>
          <div className="flex flex-wrap justify-between gap-2">
            <button
              type="button"
              onClick={createExportHandler("json")}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-cyan-400"
              disabled={isBusy}
            >
              Export JSON
            </button>
            <button
              type="button"
              onClick={createExportHandler("csv")}
              className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-cyan-400"
              disabled={isBusy}
            >
              Export CSV
            </button>
          </div>
        </section>

        {results?.points.length ? (
          <section className="space-y-2 rounded-xl border border-slate-800/60 bg-slate-900/40 p-3 text-xs text-slate-200">
            <header className="flex items-center justify-between text-[11px] uppercase tracking-[0.22rem] text-slate-400">
              <span>Insights</span>
              <span className="text-[10px] text-slate-500">Quick selections</span>
            </header>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => selectTopOutliers()}
                className="rounded-full border border-amber-400/30 px-3 py-1 text-[11px] text-amber-200 transition hover:border-amber-300 hover:text-amber-100"
                disabled={isBusy}
              >
                Select top outlier responses
              </button>
              <button
                type="button"
                onClick={() => selectTopSegmentOutliers()}
                className="rounded-full border border-cyan-400/30 px-3 py-1 text-[11px] text-cyan-200 transition hover:border-cyan-300 hover:text-cyan-100"
                disabled={isBusy || !results?.segments.length}
              >
                Spotlight divergent segments
              </button>
            </div>
          </section>
        ) : null}

        {hasRoles ? (
          <section className="space-y-2 rounded-xl border border-slate-800/60 bg-slate-900/40 p-3 text-xs text-slate-200">
            <header className="flex items-center justify-between text-[11px] uppercase tracking-[0.22rem] text-slate-400">
              <span>Semantic filters</span>
              <span className="text-[10px] text-slate-500">{activeRoleCount}/{roles.length} active</span>
            </header>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setRolesVisibility(roles, true)}
                className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
                disabled={allRolesActive || !roles.length}
              >
                Show all
              </button>
              <button
                type="button"
                onClick={() => setRolesVisibility(roles, false)}
                className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-rose-400 hover:text-rose-200"
                disabled={!roles.length}
              >
                Hide all
              </button>
              {roles.map((role) => {
                const active = roleVisibility[role] ?? true;
                return (
                  <button
                    key={role}
                    type="button"
                    onClick={() => toggleRole(role)}
                    className={`rounded-full border px-3 py-1 text-[11px] transition focus:outline-none focus:ring-1 ${
                      active
                        ? "border-cyan-400/50 bg-cyan-500/10 text-cyan-200"
                        : "border-slate-700/60 bg-slate-900/60 text-slate-400 hover:border-cyan-400/40 hover:text-cyan-200"
                    }`}
                  >
                    {role}
                  </button>
                );
              })}
            </div>
          </section>
        ) : null}

        <section className="space-y-3">
          <button
            type="button"
            className="w-full rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 px-4 py-2 font-semibold text-slate-50 shadow-lg shadow-cyan-500/30 transition active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-60"
            onClick={generate}
            disabled={isBusy}
          >
            {isBusy ? "Generating..." : "Generate Landscape"}
          </button>
          {error ? <p className="text-xs text-rose-300">{error}</p> : null}
          <p className="text-[11px] leading-relaxed text-slate-500">
            Tip: Shift + drag inside the scene to lasso select multiple responses. Hover any node to peek the answer.
          </p>
        </section>
      </div>
    </aside>
  );
});

