import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Side panel showing detailed metrics for selected or hovered responses and segments.
 *
 * Components:
 *  - PointDetailsPanel: Chooses between response or segment detail stacks.
 *  - ResponseDetails / SegmentDetails: Present per-item metrics, raw text, and contextual badges.
 */
import { memo, useMemo, useState } from "react";
import { useRunStore } from "@/store/runStore";
import { useSegmentContext } from "@/hooks/useSegmentContext";
export const PointDetailsPanel = memo(function PointDetailsPanel({ workflow }) {
    const { results, levelMode, selectedPointIds, selectedSegmentIds, hoveredPointId, hoveredSegmentId, setSelectedPoints, setSelectedSegments, setFocusedResponse, isGenerating, exportFormat, } = useRunStore((state) => ({
        results: state.results,
        levelMode: state.levelMode,
        selectedPointIds: state.selectedPointIds,
        selectedSegmentIds: state.selectedSegmentIds,
        hoveredPointId: state.hoveredPointId,
        hoveredSegmentId: state.hoveredSegmentId,
        setSelectedPoints: state.setSelectedPoints,
        setSelectedSegments: state.setSelectedSegments,
        setFocusedResponse: state.setFocusedResponse,
        isGenerating: state.isGenerating,
        exportFormat: state.exportFormat,
    }));
    const [searchTerm, setSearchTerm] = useState("");
    const { exportDataset } = workflow;
    const exportBadge = exportFormat.toUpperCase();
    const points = results?.points ?? [];
    const segments = results?.segments ?? [];
    const responseMap = useMemo(() => new Map(points.map((point) => [point.id, point])), [points]);
    const segmentMap = useMemo(() => new Map(segments.map((segment) => [segment.id, segment])), [segments]);
    const query = searchTerm.trim().toLowerCase();
    const responseMatches = useMemo(() => {
        if (!query) {
            return [];
        }
        const matches = points.filter((point) => {
            const preview = (point.text_preview || "").toLowerCase();
            const text = (point.full_text || "").toLowerCase();
            return preview.includes(query) || text.includes(query);
        });
        return matches.slice(0, 20);
    }, [points, query]);
    const segmentMatches = useMemo(() => {
        if (!query) {
            return [];
        }
        const matches = segments.filter((segment) => {
            const text = segment.text.toLowerCase();
            const role = segment.role ? segment.role.toLowerCase() : "";
            return text.includes(query) || (role && role.includes(query));
        });
        return matches.slice(0, 25);
    }, [segments, query]);
    const showSearchResults = query.length > 0;
    const showingSegments = levelMode === "segments";
    const activeIds = showingSegments
        ? selectedSegmentIds.length
            ? selectedSegmentIds
            : hoveredSegmentId
                ? [hoveredSegmentId]
                : []
        : selectedPointIds.length
            ? selectedPointIds
            : hoveredPointId
                ? [hoveredPointId]
                : [];
    const selectionCount = showingSegments ? selectedSegmentIds.length : selectedPointIds.length;
    const clearSelection = () => {
        if (showingSegments) {
            setSelectedSegments([]);
        }
        else {
            setSelectedPoints([]);
        }
    };
    const handleExportSelection = async () => {
        if (selectionCount === 0) {
            return;
        }
        try {
            await exportDataset({ scope: "selection" });
        }
        catch (err) {
            console.error(err);
        }
    };
    const activeSegments = showingSegments
        ? activeIds.map((id) => segmentMap.get(id)).filter(Boolean)
        : [];
    const activeResponses = showingSegments
        ? []
        : activeIds.map((id) => responseMap.get(id)).filter(Boolean);
    let headerTitle;
    const showClearSelection = !showSearchResults && activeIds.length > 0;
    const showClearSearch = showSearchResults && searchTerm.length > 0;
    if (showingSegments) {
        if (showSearchResults) {
            headerTitle = `Segment matches: ${segmentMatches.length}`;
        }
        else if (selectedSegmentIds.length) {
            headerTitle = `${selectedSegmentIds.length} segments`;
        }
        else if (hoveredSegmentId) {
            headerTitle = "Hovered segment";
        }
        else {
            headerTitle = "No segment focus";
        }
    }
    else if (showSearchResults) {
        headerTitle = `Response matches: ${responseMatches.length}`;
    }
    else if (selectedPointIds.length) {
        headerTitle = `${selectedPointIds.length} responses`;
    }
    else if (hoveredPointId) {
        headerTitle = "Hovered response";
    }
    else {
        headerTitle = "No response focus";
    }
    const handleResponseSearchSelect = (point) => {
        setSelectedPoints([point.id]);
        setFocusedResponse(point.id);
    };
    const bodyContent = (() => {
        if (!results) {
            return _jsx("p", { className: "text-xs text-slate-500", children: "Run a sample to inspect details." });
        }
        if (showSearchResults) {
            return (_jsx("div", { className: "space-y-3", children: showingSegments
                    ? segmentMatches.map((segment) => (_jsx(SegmentDetails, { segment: segment, parent: responseMap.get(segment.response_id) }, segment.id)))
                    : responseMatches.map((point) => (_jsx("button", { type: "button", className: "w-full text-left", onClick: () => handleResponseSearchSelect(point), children: _jsx(ResponseDetails, { point: point }) }, point.id))) }));
        }
        if (showingSegments) {
            if (!activeSegments.length) {
                return _jsx("p", { className: "text-xs text-slate-500", children: "Hover or select segments to view details." });
            }
            return activeSegments.map((segment) => (_jsx(SegmentDetails, { segment: segment, parent: responseMap.get(segment.response_id) }, segment.id)));
        }
        if (!activeResponses.length) {
            return _jsx("p", { className: "text-xs text-slate-500", children: "Hover or select responses to view details." });
        }
        return activeResponses.map((point) => _jsx(ResponseDetails, { point: point }, point.id));
    })();
    return (_jsxs("aside", { className: "glass-panel flex w-full max-h-[70vh] flex-col overflow-hidden rounded-2xl border border-slate-800/70 bg-slate-950/75 p-4 text-sm text-slate-200 xl:w-[360px]", children: [_jsxs("header", { className: "flex items-center justify-between gap-3", children: [_jsx("h2", { className: "text-base font-semibold text-slate-100", children: headerTitle }), _jsxs("div", { className: "flex items-center gap-2 text-[11px]", children: [showClearSelection ? (_jsxs("button", { type: "button", onClick: handleExportSelection, className: "rounded-lg border border-cyan-500/40 px-3 py-1 text-xs text-cyan-200 transition hover:border-cyan-300 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-60", disabled: isGenerating || selectionCount === 0, children: ["Export selection (", exportBadge, ")"] })) : null, showClearSelection ? (_jsx("button", { type: "button", onClick: clearSelection, className: "rounded-lg border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", children: "Clear" })) : null, showClearSearch ? (_jsx("button", { type: "button", onClick: () => setSearchTerm(""), className: "rounded-lg border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", children: "Clear search" })) : null] })] }), _jsx("div", { className: "mt-3", children: _jsx("input", { type: "search", value: searchTerm, onChange: (event) => setSearchTerm(event.target.value), placeholder: "Search responses or segments", className: "w-full rounded-xl border border-slate-800/60 bg-slate-900/50 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none" }) }), _jsx("div", { className: "mt-4 flex-1 min-h-0 space-y-4 overflow-y-auto pr-2 text-slate-200 scrollbar-thin pb-4", children: bodyContent })] }));
});
function ResponseDetails({ point }) {
    const clusterLabel = point.cluster ?? -1;
    const outlierScore = point.outlier_score ?? (clusterLabel === -1 ? 1 : undefined);
    const similarity = point.similarity_to_centroid;
    const similarityPct = similarity != null ? Math.round(((similarity + 1) / 2) * 100) : undefined;
    const outlierPct = outlierScore != null ? Math.round(outlierScore * 100) : undefined;
    const hasCompletionCost = point.completion_cost != null;
    const hasEmbeddingCost = point.embedding_cost != null;
    const completionCost = hasCompletionCost ? point.completion_cost ?? 0 : null;
    const embeddingCost = hasEmbeddingCost ? point.embedding_cost ?? 0 : null;
    const totalCost = point.total_cost ?? (completionCost ?? 0) + (embeddingCost ?? 0);
    return (_jsxs("article", { className: "space-y-3 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3", children: [_jsxs("header", { className: "flex items-center justify-between text-xs text-slate-400", children: [_jsxs("span", { children: ["Sample #", point.index] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("span", { children: ["Cluster ", clusterLabel === -1 ? "noise" : clusterLabel] }), outlierPct != null && outlierPct >= 25 ? (_jsxs("span", { className: "rounded-full border border-amber-400/30 bg-amber-400/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-amber-300", children: ["Outlier ", outlierPct, "%"] })) : null] })] }), _jsxs("div", { className: "flex items-center justify-between text-[11px] text-slate-400", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "inline-flex h-2 w-24 overflow-hidden rounded-full bg-slate-800", children: _jsx("span", { className: "block h-full bg-cyan-400", style: { width: `${similarityPct ?? 0}%` } }) }), _jsxs("span", { children: ["Similarity ", similarity != null ? similarity.toFixed(2) : "--"] })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "inline-flex h-2 w-20 overflow-hidden rounded-full bg-slate-800", children: _jsx("span", { className: "block h-full bg-amber-400", style: { width: `${outlierPct ?? 0}%` } }) }), _jsxs("span", { children: ["Outlier ", outlierScore != null ? outlierScore.toFixed(2) : "--"] })] })] }), _jsxs("div", { className: "grid grid-cols-2 gap-2 rounded-lg border border-slate-800/70 bg-slate-900/30 p-2 text-[11px] text-slate-300", children: [_jsxs("div", { children: [_jsx("p", { className: "font-semibold text-slate-200", children: "Tokens" }), _jsxs("p", { children: ["Prompt: ", point.prompt_tokens ?? "--"] }), _jsxs("p", { children: ["Completion: ", point.completion_tokens ?? "--"] }), _jsxs("p", { children: ["Embedding: ", point.embedding_tokens ?? "--"] })] }), _jsxs("div", { children: [_jsx("p", { className: "font-semibold text-slate-200", children: "Cost (USD)" }), _jsxs("p", { children: ["Completion: ", hasCompletionCost ? `$${(completionCost ?? 0).toFixed(6)}` : "--"] }), _jsxs("p", { children: ["Embedding: ", hasEmbeddingCost ? `$${(embeddingCost ?? 0).toFixed(6)}` : "--"] }), _jsxs("p", { children: ["Total: ", hasCompletionCost || hasEmbeddingCost ? `$${totalCost.toFixed(6)}` : "--"] })] })] }), _jsx("p", { className: "whitespace-pre-wrap text-sm leading-relaxed text-slate-100", children: point.full_text }), _jsxs("footer", { className: "grid grid-cols-3 gap-2 text-[11px] text-slate-400", children: [_jsxs("span", { children: ["Tokens: ", point.tokens ?? point.usage?.completion_tokens ?? "--"] }), _jsxs("span", { children: ["Prob: ", point.probability?.toFixed(2) ?? "--"] }), _jsxs("span", { children: ["Finish: ", point.finish_reason ?? "n/a"] })] })] }));
}
function SegmentDetails({ segment, parent }) {
    const clusterLabel = segment.cluster ?? -1;
    const outlierScore = segment.outlier_score ?? (clusterLabel === -1 ? 1 : undefined);
    const similarity = segment.similarity_to_centroid;
    const similarityPct = similarity != null ? Math.round(((similarity + 1) / 2) * 100) : undefined;
    const outlierPct = outlierScore != null ? Math.round(outlierScore * 100) : undefined;
    const hasEmbeddingCost = segment.embedding_cost != null;
    const embeddingCost = hasEmbeddingCost ? segment.embedding_cost ?? 0 : null;
    const { data: context, isFetching } = useSegmentContext(segment.id, {
        enabled: Boolean(segment.id),
        k: 8,
        staleTimeMs: 300_000,
    });
    return (_jsxs("article", { className: "space-y-3 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3", children: [_jsxs("header", { className: "flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400", children: [_jsxs("span", { children: ["Sample #", segment.response_index, " \u2013 segment #", segment.position + 1, segment.role ? ` - ${segment.role}` : ""] }), _jsxs("div", { className: "flex flex-wrap items-center gap-2", children: [_jsxs("span", { children: ["Cluster ", clusterLabel === -1 ? "noise" : clusterLabel] }), segment.is_cached ? (_jsx("span", { className: "rounded-full border border-emerald-400/40 bg-emerald-500/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-emerald-200", children: "Cached" })) : null, segment.is_duplicate ? (_jsx("span", { className: "rounded-full border border-amber-400/40 bg-amber-500/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-amber-200", children: "Duplicate" })) : null, outlierPct != null && outlierPct >= 25 ? (_jsxs("span", { className: "rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-cyan-200", children: ["Divergence ", outlierPct, "%"] })) : null] })] }), _jsxs("div", { className: "flex items-center justify-between text-[11px] text-slate-400", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "inline-flex h-2 w-24 overflow-hidden rounded-full bg-slate-800", children: _jsx("span", { className: "block h-full bg-cyan-400", style: { width: `${similarityPct ?? 0}%` } }) }), _jsxs("span", { children: ["Similarity ", similarity != null ? similarity.toFixed(2) : "--"] })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "inline-flex h-2 w-20 overflow-hidden rounded-full bg-slate-800", children: _jsx("span", { className: "block h-full bg-amber-400", style: { width: `${outlierPct ?? 0}%` } }) }), _jsxs("span", { children: ["Outlier ", outlierScore != null ? outlierScore.toFixed(2) : "--"] })] })] }), _jsx("p", { className: "whitespace-pre-wrap text-sm leading-relaxed text-slate-100", children: segment.text }), context ? (_jsxs("div", { className: "space-y-2 rounded-lg border border-slate-800/70 bg-slate-900/30 p-3 text-[11px] text-slate-300", children: [context.top_terms.length ? (_jsxs("div", { children: [_jsx("p", { className: "text-[10px] uppercase tracking-wide text-slate-500", children: "Top terms" }), _jsx("div", { className: "mt-2 flex flex-wrap gap-1", children: context.top_terms.slice(0, 6).map((item) => (_jsx("span", { className: "rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2 py-[2px] text-[10px] text-cyan-100", title: "TF-IDF weight " + item.weight.toFixed(3), children: item.term }, segment.id + "-" + item.term))) })] })) : null, context.exemplar_preview ? (_jsxs("div", { className: "space-y-1", children: [_jsx("p", { className: "text-[10px] uppercase tracking-wide text-slate-500", children: "Closest exemplar" }), _jsx("p", { className: "text-slate-400", children: context.exemplar_preview.length > 160
                                    ? context.exemplar_preview.slice(0, 160) + "..."
                                    : context.exemplar_preview })] })) : null, context.why_here && (context.why_here.sim_to_exemplar != null || context.why_here.sim_to_nn != null) ? (_jsxs("div", { className: "flex flex-wrap gap-3 text-[10px] text-slate-400", children: [context.why_here.sim_to_exemplar != null ? (_jsxs("span", { children: ["Sim to exemplar ", context.why_here.sim_to_exemplar.toFixed(2)] })) : null, context.why_here.sim_to_nn != null ? (_jsxs("span", { children: ["Nearest neighbour ", context.why_here.sim_to_nn.toFixed(2)] })) : null] })) : null, context.neighbors.length ? (_jsxs("div", { className: "space-y-1", children: [_jsx("p", { className: "text-[10px] uppercase tracking-wide text-slate-500", children: "Nearest neighbours" }), _jsx("ul", { className: "space-y-1 text-[10px] text-slate-300", children: context.neighbors.slice(0, 5).map((neighbor) => (_jsxs("li", { children: [_jsx("span", { className: "text-cyan-200", children: neighbor.similarity.toFixed(2) }), " \u00B7 ", neighbor.text] }, neighbor.id))) })] })) : null] })) : isFetching ? (_jsx("p", { className: "text-[11px] text-slate-500", children: "Loading segment context\u2026" })) : null, _jsxs("footer", { className: "space-y-2 text-[11px] text-slate-400", children: [_jsxs("div", { className: "flex flex-wrap gap-3", children: [_jsxs("span", { children: ["Tokens: ", segment.tokens ?? "--"] }), _jsxs("span", { children: ["Prompt sim: ", segment.prompt_similarity != null ? segment.prompt_similarity.toFixed(2) : "--"] }), _jsxs("span", { children: ["Embedding cost: ", hasEmbeddingCost ? `$${(embeddingCost ?? 0).toFixed(6)}` : "--"] })] }), _jsxs("div", { className: "flex flex-wrap gap-3 text-slate-500", children: [_jsxs("span", { children: ["Cache: ", segment.is_cached ? "cached" : "new"] }), _jsxs("span", { children: ["Duplicate: ", segment.is_duplicate ? "yes" : "no"] })] }), _jsxs("div", { className: "flex flex-wrap gap-3 text-[10px] text-slate-600", children: [segment.text_hash ? (_jsxs("span", { children: ["Hash ", _jsxs("code", { children: [segment.text_hash.slice(0, 12), "\u2026"] })] })) : null, segment.simhash64 != null ? (_jsxs("span", { children: ["SimHash ", segment.simhash64] })) : null] }), parent ? (_jsxs("p", { className: "text-[11px] text-slate-500", children: ["Parent response #", parent.index, " \u2013 Cluster ", parent.cluster ?? "?"] })) : null] })] }));
}



