import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Control surface for sampling parameters, visual filters, exports, and diagnostics.
 */
import { memo, useMemo, useState, useCallback } from "react";
import { InfoTooltip } from "@/components/InfoTooltip";
import { useRunWorkflow } from "@/hooks/useRunWorkflow";
import { useRunStore } from "@/store/runStore";
function toneBadge(tone) {
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
function ControlSection({ title, description, tone = "instant", defaultOpen = true, children }) {
    const [open, setOpen] = useState(defaultOpen);
    const badge = toneBadge(tone);
    return (_jsxs("section", { className: "rounded-xl border border-slate-800/60 bg-slate-950/50 shadow-sm shadow-slate-950/40", children: [_jsxs("button", { type: "button", onClick: () => setOpen((value) => !value), className: "flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-slate-900/60 focus:outline-none focus-visible:ring focus-visible:ring-cyan-400/40", children: [_jsxs("div", { className: "min-w-0", children: [_jsx("p", { className: "text-xs font-semibold uppercase tracking-wide text-slate-200", children: title }), description ? _jsx("p", { className: "mt-0.5 text-[11px] leading-relaxed text-slate-500", children: description }) : null] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: `rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${badge.className}`, children: badge.label }), _jsx("svg", { className: `h-4 w-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`, viewBox: "0 0 20 20", fill: "none", stroke: "currentColor", strokeWidth: "1.5", children: _jsx("path", { d: "M5 7.5L10 12.5L15 7.5", strokeLinecap: "round", strokeLinejoin: "round" }) })] })] }), open ? _jsx("div", { className: "border-t border-slate-800/60 px-4 pb-4 pt-3 space-y-4", children: children }) : null] }));
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
const PROJECTION_OPTIONS = [
    { value: "umap", label: "UMAP", description: "Balanced local/global structure; best default for semantic clustering." },
    { value: "tsne", label: "t-SNE", description: "Sharper local clusters; great for deep dives, less stable between runs." },
    { value: "pca", label: "PCA", description: "Linear baseline to sanity-check embeddings and variance." },
];
const UMAP_PRESETS = [
    { value: "balanced", label: "Balanced", description: "Good separation with contextual smoothing." },
    { value: "tight", label: "Local", description: "Emphasise micro-clusters and nuanced phrasing." },
    { value: "global", label: "Global", description: "Hold macro structure steady for comparisons." },
];
const METRIC_OPTIONS = [
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
    const { prompt, setPrompt, systemPrompt, setSystemPrompt, n, setN, temperature, setTemperature, topP, setTopP, model, setModel, embeddingModel, setEmbeddingModel, seed, setSeed, maxTokens, setMaxTokens, jitterToken, setJitterToken, useCache, setUseCache, chunkSize, setChunkSize, chunkOverlap, setChunkOverlap, projectionMethod, setProjectionMethod, projectionWarnings, isProjectionLoading, umapNNeighbors, setUmapNNeighbors, umapMinDist, setUmapMinDist, umapMetric, setUmapMetric, umapSeed, setUmapSeed, umapPreset, setUmapPreset, viewMode, setViewMode, levelMode, setLevelMode, pointSize, setPointSize, spreadFactor, setSpreadFactor, showDensity, setShowDensity, showEdges, setShowEdges, showParentThreads, setShowParentThreads, showNeighborSpokes, setShowNeighborSpokes, showDuplicatesOnly, setShowDuplicatesOnly, graphEdgeK, setGraphEdgeK, graphEdgeThreshold, setGraphEdgeThreshold, segmentGraphAutoSimplified, segmentGraphError, results, roleVisibility, toggleRole, setRolesVisibility, selectTopOutliers, selectTopSegmentOutliers, exportFormat, setExportFormat, exportIncludeProvenance, setExportIncludeProvenance, viewportBounds, setHistoryOpen, isHistoryOpen, isGenerating, clusterAlgo, hdbscanMinClusterSize, hdbscanMinSamples, setClusterParams, recomputeClusters, isRecomputingClusters, } = useRunStore((state) => ({
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
    const cacheButtonClass = `rounded-full border px-3 py-1 text-[11px] uppercase tracking-wide transition ${useCache ? "border-cyan-500 bg-cyan-500/10 text-cyan-200" : "border-slate-700/60 text-slate-200 hover:border-cyan-400 hover:text-cyan-200"}`;
    const roles = useMemo(() => {
        if (!results) {
            return [];
        }
        const tokens = new Set();
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
        }
        catch (err) {
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
        }
        catch (err) {
            console.error(err);
        }
    }, [exportDataset, exportFormat, exportIncludeProvenance, levelMode]);
    const handleProjectionMethodChange = useCallback((method) => {
        setProjectionMethod(method).catch((err) => console.error(err));
    }, [setProjectionMethod]);
    const handleGraphEdgeKChange = useCallback((value) => {
        const clamped = Math.min(Math.max(value, EDGE_K_MIN), neighborCap);
        setGraphEdgeK(clamped).catch((err) => console.error(err));
    }, [neighborCap, setGraphEdgeK]);
    const handleGraphEdgeThresholdChange = useCallback((value) => {
        const clamped = Math.min(Math.max(value, 0), 1);
        setGraphEdgeThreshold(clamped).catch((err) => console.error(err));
    }, [setGraphEdgeThreshold]);
    const handleRecomputeClusters = useCallback(() => {
        recomputeClusters().catch((err) => console.error(err));
    }, [recomputeClusters]);
    const handleGenerate = useCallback(() => {
        generate();
    }, [generate]);
    return (_jsxs("aside", { className: "flex h-full min-h-0 w-[380px] shrink-0 flex-col border-l border-slate-900/80 bg-slate-950/90 text-sm text-slate-100", children: [_jsxs("div", { className: "border-b border-slate-900/60 px-5 pb-4 pt-5", children: [_jsx("h1", { className: "text-lg font-semibold tracking-tight", children: "Semantic Landscape Sampler" }), _jsx("p", { className: "mt-1 text-[11px] text-slate-500", children: "Explore diverse LLM responses, surface clusters, and export spatial analytics." }), _jsx("div", { className: "mt-3 flex flex-wrap gap-2", children: _jsx("button", { type: "button", onClick: () => setHistoryOpen(!isHistoryOpen), className: "rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", children: isHistoryOpen ? "Hide run history" : "Show run history" }) })] }), _jsxs("div", { className: "flex-1 min-h-0 space-y-4 overflow-y-auto px-5 pb-6 pt-4", children: [_jsxs(ControlSection, { title: "Run Setup", description: "Configure prompts and sampling parameters. Changes here require regenerating the landscape.", tone: "rerun", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("label", { className: "text-xs uppercase tracking-[0.22rem] text-slate-400", children: "Prompt" }), _jsx("textarea", { value: prompt, onChange: (event) => setPrompt(event.target.value), rows: 5, className: "w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40", placeholder: "Describe the question you want the models to explore." })] }), _jsxs("div", { className: "space-y-2", children: [_jsxs("label", { className: "flex items-center gap-2 text-xs uppercase tracking-[0.22rem] text-slate-400", children: ["System message", _jsx(InfoTooltip, { text: "Set tone, persona, or safety constraints. Leave blank to keep the default assistant behaviour." })] }), _jsx("textarea", { value: systemPrompt, onChange: (event) => setSystemPrompt(event.target.value), rows: 3, className: "w-full resize-none rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 text-sm text-slate-100 outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/40", placeholder: "Optional: reinforce persona, guardrails, or instructions." })] }), _jsxs("div", { className: "space-y-3 rounded-xl border border-slate-800/60 bg-slate-900/40 p-3", children: [_jsxs("div", { className: "flex items-center justify-between gap-3", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm font-semibold text-slate-100", children: "Embedding cache" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Reuse embeddings for repeated segments to reduce latency and cost." })] }), _jsx("button", { type: "button", onClick: () => setUseCache(!useCache), className: cacheButtonClass, children: useCache ? "Enabled" : "Disabled" })] }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Cached segments are flagged in the details drawer. Disable for fresh embeddings." })] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Model" }), _jsx("select", { value: model, onChange: (event) => setModel(event.target.value), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: MODEL_OPTIONS.map((option) => (_jsx("option", { value: option.value, children: option.label }, option.value))) })] }), _jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Embedding model" }), _jsx("select", { value: embeddingModel, onChange: (event) => setEmbeddingModel(event.target.value), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: EMBEDDING_OPTIONS.map((option) => (_jsx("option", { value: option.value, children: option.label }, option.value))) })] })] }), _jsxs("div", { className: "grid gap-4 md:grid-cols-2", children: [_jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Samples (N)" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Collect more responses for broader coverage." }), _jsx("input", { type: "range", min: 1, max: N_MAX, value: n, onChange: (event) => setN(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: n })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Temperature" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Lower values stay focused; higher encourages exploration." }), _jsx("input", { type: "range", min: 0, max: TEMPERATURE_MAX, step: 0.05, value: temperature, onChange: (event) => setTemperature(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: temperature.toFixed(2) })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Top-p" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Combine with temperature to control adventurous wording." }), _jsx("input", { type: "range", min: 0, max: TOP_P_MAX, step: 0.05, value: topP, onChange: (event) => setTopP(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: topP.toFixed(2) })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Jitter token" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Inject per-sample noise. Leave blank for automatic jitter." }), _jsx("input", { type: "text", value: jitterToken ?? "", onChange: (event) => setJitterToken(event.target.value ? event.target.value : null), placeholder: "auto", className: "mt-2 w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none" })] })] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Seed" }), _jsx("input", { type: "number", min: 0, value: seed ?? "", onChange: (event) => setSeed(event.target.value ? Number(event.target.value) : null), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none" })] }), _jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Max tokens" }), _jsx("input", { type: "number", min: 64, max: 4096, value: maxTokens ?? "", onChange: (event) => setMaxTokens(event.target.value ? Number(event.target.value) : null), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none" })] })] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Segment chunk size" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Controls sentence splitting for segment analysis." }), _jsx("input", { type: "range", min: CHUNK_MIN, max: CHUNK_MAX, value: chunkSizeValue, onChange: (event) => {
                                                    const value = Number(event.target.value);
                                                    const clamped = Math.min(Math.max(value, CHUNK_MIN), CHUNK_MAX);
                                                    setChunkSize(clamped);
                                                    if (overlapValue > clamped - 1) {
                                                        setChunkOverlap(Math.max(0, clamped - 1));
                                                    }
                                                }, className: "mt-2 w-full" }), _jsxs("span", { className: "text-xs text-slate-400", children: [chunkSizeValue, " tokens"] })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Segment overlap" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Increase to preserve context between neighbouring segments." }), _jsx("input", { type: "range", min: 0, max: overlapMax, value: overlapValue, onChange: (event) => {
                                                    const value = Number(event.target.value);
                                                    const clamped = Math.min(Math.max(value, 0), overlapMax);
                                                    setChunkOverlap(clamped);
                                                }, className: "mt-2 w-full" }), _jsxs("span", { className: "text-xs text-slate-400", children: [overlapValue, " tokens (max ", overlapMax, ")"] })] })] }), _jsxs("div", { className: "flex flex-wrap items-center gap-3 pt-2", children: [_jsx("button", { type: "button", onClick: handleGenerate, disabled: isBusy || promptMissing, className: "inline-flex items-center justify-center rounded-full border border-cyan-500/40 bg-cyan-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-cyan-200 transition hover:border-cyan-400 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:border-slate-700/60 disabled:bg-slate-800/40 disabled:text-slate-500", children: isBusy ? "Working..." : "Generate landscape" }), error ? _jsx("p", { className: "text-[11px] text-rose-400", children: error }) : null, promptMissing ? _jsx("p", { className: "text-[11px] text-slate-500", children: "Enter a prompt to enable generation." }) : null] })] }), _jsxs(ControlSection, { title: "Projection & Layout", description: "Switch reductions, reuse cached layouts, and tune UMAP without re-embedding.", children: [_jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Projection method" }), _jsx("select", { value: projectionMethod, onChange: (event) => handleProjectionMethodChange(event.target.value), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: PROJECTION_OPTIONS.map((option) => (_jsx("option", { value: option.value, children: option.label }, option.value))) })] }), _jsxs("div", { className: "space-y-2 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Method overview" }), _jsx("p", { className: "text-[11px] leading-relaxed text-slate-500", children: PROJECTION_OPTIONS.find((option) => option.value === projectionMethod)?.description })] })] }), _jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-400", children: "UMAP presets" }), _jsx("button", { type: "button", onClick: () => setShowUmapHelp((value) => !value), className: "text-[11px] text-cyan-300 transition hover:text-cyan-200", children: showUmapHelp ? "Hide tips" : "Show tips" })] }), _jsx("div", { className: "flex flex-wrap gap-2", children: UMAP_PRESETS.map((preset) => (_jsx("button", { type: "button", onClick: () => setUmapPreset(preset.value), className: `rounded-full border px-3 py-1 text-[11px] transition ${umapPreset === preset.value
                                                ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"}`, title: preset.description, children: preset.label }, preset.value))) }), showUmapHelp ? (_jsxs("ul", { className: "space-y-1 rounded-lg border border-slate-800/60 bg-slate-900/40 p-3 text-[11px] text-slate-400", children: [_jsx("li", { children: "\uFFFD Balanced keeps macro topology stable for comparisons." }), _jsx("li", { children: "\uFFFD Local sharpens clusters; great for rhetorical nuance." }), _jsx("li", { children: "\uFFFD Global widens spacing for cross-run overlays." })] })) : null] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "UMAP n_neighbors" }), _jsx("input", { type: "range", min: 2, max: 200, value: umapNNeighbors, onChange: (event) => setUmapNNeighbors(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: umapNNeighbors })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "UMAP min_dist" }), _jsx("input", { type: "range", min: 0, max: 1, step: 0.01, value: umapMinDist, onChange: (event) => setUmapMinDist(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: umapMinDist.toFixed(2) })] }), _jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "UMAP metric" }), _jsx("select", { value: umapMetric, onChange: (event) => setUmapMetric(event.target.value), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: METRIC_OPTIONS.map((option) => (_jsx("option", { value: option.value, children: option.label }, option.value))) })] }), _jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "UMAP seed" }), _jsx("input", { type: "number", min: 0, value: umapSeed ?? "", onChange: (event) => setUmapSeed(event.target.value ? Number(event.target.value) : null), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", placeholder: "auto" })] })] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("div", { children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-400", children: "View mode" }), _jsxs("div", { className: "mt-2 flex gap-2", children: [_jsx("button", { type: "button", onClick: () => setViewMode("3d"), className: `flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${viewMode === "3d"
                                                            ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                            : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"}`, children: "3D scene" }), _jsx("button", { type: "button", onClick: () => setViewMode("2d"), className: `flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${viewMode === "2d"
                                                            ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                            : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"}`, children: "2D slice" })] })] }), _jsxs("div", { children: [_jsx("span", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Data mode" }), _jsxs("div", { className: "mt-2 flex gap-2", children: [_jsx("button", { type: "button", onClick: () => setLevelMode("responses"), className: `flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${levelMode === "responses"
                                                            ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                            : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"}`, children: "Responses" }), _jsx("button", { type: "button", onClick: () => setLevelMode("segments"), className: `flex-1 rounded-lg border px-3 py-2 text-xs font-medium transition ${levelMode === "segments"
                                                            ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                            : "border-slate-700/60 text-slate-300 hover:border-cyan-400 hover:text-cyan-200"}`, children: "Segments" })] })] })] }), _jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Point size" }), _jsx("input", { type: "range", min: POINT_SIZE_MIN, max: POINT_SIZE_MAX, value: pointSize, onChange: (event) => setPointSize(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: pointSize.toFixed(0) })] }), _jsxs("div", { children: [_jsx("label", { className: "text-xs uppercase tracking-wide text-slate-400", children: "Spread factor" }), _jsx("input", { type: "range", min: 0, max: POINT_SPREAD_MAX, step: 0.1, value: spreadFactor, onChange: (event) => setSpreadFactor(Number(event.target.value)), className: "mt-2 w-full" }), _jsx("span", { className: "text-xs text-slate-400", children: spreadFactor.toFixed(1) })] })] }), isProjectionLoading ? _jsx("p", { className: "text-[11px] text-cyan-300", children: "Projection recalculating..." }) : null, projectionWarningsList.length > 0 ? (_jsxs("div", { className: "space-y-1 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-[11px] text-amber-200", children: [_jsx("p", { className: "font-semibold uppercase tracking-[0.2rem]", children: "Projection warnings" }), _jsx("ul", { className: "list-disc pl-4 leading-relaxed text-amber-100", children: projectionWarningsList.map((warning) => (_jsx("li", { children: warning }, warning))) })] })) : null] }), _jsxs(ControlSection, { title: "Visibility & Filters", description: "Toggle overlays, similarity graphs, discourse roles, and duplicate highlighting.", children: [_jsxs("div", { className: "grid gap-2", children: [_jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: showDensity, onChange: (event) => setShowDensity(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Density overlay"] }), _jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: showEdges, onChange: (event) => setShowEdges(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Similarity edges"] }), _jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: showParentThreads, onChange: (event) => setShowParentThreads(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Parent threads"] }), _jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: showNeighborSpokes, onChange: (event) => setShowNeighborSpokes(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Neighbor spokes"] }), _jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: showDuplicatesOnly, onChange: (event) => setShowDuplicatesOnly(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Show duplicates only"] })] }), _jsxs("div", { className: "space-y-2 rounded-lg border border-slate-800/60 bg-slate-900/40 p-3", children: [_jsxs("div", { className: "flex items-center justify-between text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Similarity graph" }), _jsx("span", { className: "text-slate-500", children: graphStatusLabel })] }), _jsxs("div", { children: [_jsx("label", { className: "text-[11px] uppercase tracking-wide text-slate-500", children: "Neighbor count (k)" }), _jsx("input", { type: "range", min: EDGE_K_MIN, max: neighborCap, disabled: graphControlsDisabled, value: Math.min(Math.max(graphEdgeK ?? EDGE_K_MIN, EDGE_K_MIN), neighborCap), onChange: (event) => handleGraphEdgeKChange(Number(event.target.value)), className: "mt-2 w-full disabled:opacity-50" }), _jsx("span", { className: "text-[11px] text-slate-500", children: graphControlsDisabled ? "Run a sample to enable edge tuning." : `${graphEdgeK ?? EDGE_K_MIN} neighbors` })] }), _jsxs("div", { children: [_jsx("label", { className: "text-[11px] uppercase tracking-wide text-slate-500", children: "Similarity threshold" }), _jsx("input", { type: "range", min: 0, max: 1, step: 0.05, disabled: graphControlsDisabled, value: Math.min(Math.max(graphEdgeThreshold ?? 0, 0), 1), onChange: (event) => handleGraphEdgeThresholdChange(Number(event.target.value)), className: "mt-2 w-full disabled:opacity-50" }), _jsxs("span", { className: "text-[11px] text-slate-500", children: [thresholdPercent, "% cosine"] })] }), segmentGraphError ? _jsxs("p", { className: "text-[11px] text-rose-300", children: ["Edge simplification error: ", segmentGraphError] }) : null] }), roles.length > 0 ? (_jsxs("div", { className: "space-y-2", children: [_jsxs("div", { className: "flex items-center justify-between text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Discourse roles" }), _jsxs("span", { className: "text-slate-500", children: [activeRoleCount, "/", roles.length, " active"] })] }), _jsx("div", { className: "flex flex-wrap gap-2", children: roles.map((role) => {
                                            const active = roleVisibility[role] ?? true;
                                            return (_jsx("button", { type: "button", onClick: () => toggleRole(role), className: `rounded-full border px-3 py-1 text-[11px] capitalize transition ${active
                                                    ? "border-cyan-500/60 bg-cyan-500/15 text-cyan-200"
                                                    : "border-slate-700/60 text-slate-400 hover:border-cyan-400 hover:text-cyan-200"}`, children: role }, role));
                                        }) }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { type: "button", onClick: () => setRolesVisibility(roles, true), className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", disabled: roles.length === 0 || allRolesActive, children: "Show all" }), _jsx("button", { type: "button", onClick: () => setRolesVisibility(roles, false), className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", disabled: roles.length === 0 || !allRolesActive, children: "Hide all" })] })] })) : (_jsx("p", { className: "text-[11px] text-slate-500", children: "Run segment mode to unlock role filtering." }))] }), _jsxs(ControlSection, { title: "Selection Shortcuts", description: "Highlight representative outliers for quick reviews.", children: [_jsxs("div", { className: "flex flex-wrap gap-2", children: [_jsx("button", { type: "button", disabled: !results, onClick: () => selectTopOutliers(), className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500", children: "Select top response outliers" }), _jsx("button", { type: "button", disabled: !results, onClick: () => selectTopSegmentOutliers(), className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500", children: "Select top segment outliers" })] }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Outlier search runs on the active level mode." })] }), _jsxs(ControlSection, { title: "Export", description: "Download raw responses, segments, hulls, and provenance for downstream analysis.", children: [_jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Format" }), _jsxs("select", { value: exportFormat, onChange: (event) => setExportFormat(event.target.value), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: [_jsx("option", { value: "json", children: "JSON" }), _jsx("option", { value: "jsonl", children: "JSONL" }), _jsx("option", { value: "csv", children: "CSV" }), _jsx("option", { value: "parquet", children: "Parquet" })] })] }), _jsxs("label", { className: "flex items-center gap-2 text-xs text-slate-300", children: [_jsx("input", { type: "checkbox", checked: exportIncludeProvenance, onChange: (event) => setExportIncludeProvenance(event.target.checked), className: "h-3 w-3 rounded border border-slate-700/60 bg-slate-900/60 text-cyan-400 focus:ring-cyan-400/40" }), "Include provenance metadata"] })] }), _jsxs("div", { className: "flex flex-wrap gap-2", children: [_jsxs("button", { type: "button", onClick: handleExportRun, disabled: isBusy || !results, className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-200 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500", children: ["Export ", exportModeLabel] }), _jsx("button", { type: "button", onClick: handleExportViewport, disabled: isBusy || !viewportAvailable, className: "rounded-full border border-slate-700/60 px-3 py-1 text-[11px] text-slate-200 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:text-slate-500", children: "Export current viewport" })] }), !viewportAvailable ? (_jsx("p", { className: "text-[11px] text-slate-500", children: "Pan or zoom the scene to define a viewport export." })) : null] }), _jsx(ControlSection, { title: "Cluster Tuning", description: "Adjust clustering parameters and recompute without calling the LLM.", children: _jsxs("div", { className: "space-y-3", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Algorithm" }), _jsxs("select", { value: clusterAlgo, onChange: (event) => setClusterParams({ algo: event.target.value }), className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none", children: [_jsx("option", { value: "hdbscan", children: "HDBSCAN" }), _jsx("option", { value: "kmeans", children: "KMeans" })] })] }), clusterAlgo === "hdbscan" ? (_jsxs("div", { className: "grid gap-3 md:grid-cols-2", children: [_jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Min cluster size" }), _jsx("input", { type: "number", min: 2, value: hdbscanMinClusterSize ?? 2, onChange: (event) => {
                                                        const value = Number(event.target.value);
                                                        if (!Number.isNaN(value)) {
                                                            setClusterParams({ minClusterSize: Math.max(2, value) });
                                                        }
                                                    }, className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none" })] }), _jsxs("label", { className: "space-y-1 text-xs", children: [_jsx("span", { className: "uppercase tracking-wide text-slate-400", children: "Min samples" }), _jsx("input", { type: "number", min: 1, value: hdbscanMinSamples ?? 1, onChange: (event) => {
                                                        const value = Number(event.target.value);
                                                        if (!Number.isNaN(value)) {
                                                            setClusterParams({ minSamples: Math.max(1, value) });
                                                        }
                                                    }, className: "w-full rounded-lg border border-slate-700/60 bg-slate-900/60 p-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none" })] })] })) : (_jsx("p", { className: "text-[11px] text-slate-500", children: "KMeans recomputes centroid assignments using the active embeddings." })), _jsx("button", { type: "button", onClick: handleRecomputeClusters, disabled: isRecomputingClusters || !results, className: "inline-flex items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs font-medium text-cyan-200 transition hover:border-cyan-400 hover:bg-cyan-500/20 disabled:cursor-not-allowed disabled:border-slate-800/60 disabled:bg-slate-900/40 disabled:text-slate-500", children: isRecomputingClusters ? "Recomputing..." : "Recompute clusters" }), _jsx("p", { className: "text-[11px] text-slate-500", children: "Compare silhouette scores and centroid similarity across projection and embedding spaces." })] }) })] })] }));
});
