import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Drawer overlay showing recently sampled runs with metadata and quick loading actions.
 *
 * Components:
 *  - RunHistoryDrawer: Fetches persisted runs from the backend, groups them by day,
 *    and lets users load a prior run without re-sampling.
 */
import { memo, useEffect, useMemo, useState } from "react";
import { fetchRuns } from "@/services/api";
import { useRunStore } from "@/store/runStore";
const dayFormatter = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
const timeFormatter = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const badgeClass = "inline-flex items-center gap-1 rounded-full border border-slate-700/60 bg-slate-900/70 px-2 py-[2px] text-[10px] uppercase tracking-wide text-slate-300";
function describeDate(timestamp) {
    const date = new Date(timestamp);
    const today = new Date();
    const startOfDay = (input) => new Date(input.getFullYear(), input.getMonth(), input.getDate());
    const todayKey = startOfDay(today).getTime();
    const targetKey = startOfDay(date).getTime();
    if (targetKey === todayKey) {
        return "Today";
    }
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    if (targetKey === startOfDay(yesterday).getTime()) {
        return "Yesterday";
    }
    const dayLabel = dayFormatter.format(date);
    if (today.getFullYear() === date.getFullYear()) {
        return dayLabel;
    }
    return `${dayLabel}, ${date.getFullYear()}`;
}
function truncatePrompt(prompt, maxLength = 140) {
    if (prompt.length <= maxLength) {
        return prompt;
    }
    return `${prompt.slice(0, maxLength - 3)}...`;
}
function summariseNote(note, maxLength = 160) {
    const trimmed = note.trim();
    if (trimmed.length <= maxLength) {
        return trimmed;
    }
    return `${trimmed.slice(0, maxLength - 3)}...`;
}
function formatDuration(ms) {
    if (ms == null || Number.isNaN(ms)) {
        return '—';
    }
    if (ms >= 1000) {
        const seconds = ms / 1000;
        return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
    }
    if (ms >= 100) {
        return `${ms.toFixed(0)}ms`;
    }
    return `${ms.toFixed(1)}ms`;
}
export const RunHistoryDrawer = memo(function RunHistoryDrawer({ workflow }) {
    const { isHistoryOpen, runHistory, setRunHistory, currentRunId, setHistoryOpen } = useRunStore((state) => ({
        isHistoryOpen: state.isHistoryOpen,
        runHistory: state.runHistory,
        setRunHistory: state.setRunHistory,
        currentRunId: state.currentRunId,
        setHistoryOpen: state.setHistoryOpen,
    }));
    const { loadFromHistory, duplicateRun, isLoadingHistory, isDuplicating } = workflow;
    const [isFetching, setIsFetching] = useState(false);
    const [fetchError, setFetchError] = useState(null);
    useEffect(() => {
        if (!isHistoryOpen) {
            return;
        }
        let cancelled = false;
        async function hydrateRuns() {
            setIsFetching(true);
            setFetchError(null);
            try {
                const runs = await fetchRuns(40);
                if (!cancelled) {
                    setRunHistory(runs);
                }
            }
            catch (error) {
                if (!cancelled) {
                    const message = error instanceof Error ? error.message : "Failed to load run history";
                    setFetchError(message);
                }
            }
            finally {
                if (!cancelled) {
                    setIsFetching(false);
                }
            }
        }
        void hydrateRuns();
        return () => {
            cancelled = true;
        };
    }, [isHistoryOpen, setRunHistory]);
    const groupedRuns = useMemo(() => {
        if (!runHistory.length) {
            return [];
        }
        const sorted = [...runHistory].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        const groups = new Map();
        for (const run of sorted) {
            const dateLabel = describeDate(run.created_at);
            if (!groups.has(dateLabel)) {
                groups.set(dateLabel, { label: dateLabel, runs: [] });
            }
            groups.get(dateLabel)?.runs.push(run);
        }
        return Array.from(groups.values());
    }, [runHistory]);
    const handleSelect = (run) => {
        if (currentRunId === run.id && !isLoadingHistory) {
            setHistoryOpen(false);
            return;
        }
        loadFromHistory(run);
    };
    const handleDuplicate = (run) => {
        duplicateRun(run);
    };
    if (!isHistoryOpen) {
        return null;
    }
    const historyCountLabel = `${runHistory.length} stored runs`;
    const statusBadges = [];
    if (isLoadingHistory) {
        statusBadges.push("Loading run�");
    }
    if (isDuplicating) {
        statusBadges.push("Re-running�");
    }
    return (_jsxs("aside", { className: "absolute inset-0 z-30 flex flex-col rounded-2xl border border-slate-800/70 bg-slate-950/95 p-5 shadow-2xl", children: [_jsxs("header", { className: "flex items-start justify-between gap-4", children: [_jsxs("div", { children: [_jsx("p", { className: "text-[11px] uppercase tracking-[0.22rem] text-slate-500", children: "Run history" }), _jsx("h2", { className: "text-lg font-semibold text-slate-100", children: "Saved landscapes" }), _jsx("p", { className: "text-xs text-slate-500", children: "Load a previous semantic layout without re-sampling." })] }), _jsx("button", { type: "button", onClick: () => setHistoryOpen(false), className: "rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200", children: "Close" })] }), _jsxs("div", { className: "mt-4 flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500", children: [_jsx("span", { children: isFetching ? "Refreshing�" : historyCountLabel }), statusBadges.length ? (_jsx("div", { className: "flex gap-3 text-cyan-200", children: statusBadges.map((label) => (_jsx("span", { children: label }, label))) })) : null] }), fetchError ? _jsx("p", { className: "mt-3 text-xs text-rose-300", children: fetchError }) : null, _jsxs("div", { className: "mt-4 flex-1 overflow-y-auto pr-2 text-sm", children: [groupedRuns.length === 0 && !isFetching ? (_jsx("p", { className: "text-xs text-slate-500", children: "Run a sample to populate history. Completed runs will appear here automatically." })) : null, groupedRuns.map((group) => (_jsxs("section", { className: "mb-6 space-y-3", children: [_jsx("header", { className: "text-[11px] uppercase tracking-[0.22rem] text-slate-500", children: group.label }), _jsx("ul", { className: "space-y-3", children: group.runs.map((run) => {
                                    const active = currentRunId === run.id;
                                    const containerClass = active
                                        ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-100"
                                        : "border-slate-800/70 bg-slate-900/50 text-slate-200 hover:border-cyan-400/50 hover:text-cyan-100";
                                    const cardClass = `rounded-xl border px-4 py-3 transition ${containerClass}`;
                                    return (_jsx("li", { children: _jsxs("article", { className: cardClass, children: [_jsxs("div", { className: "flex items-start justify-between gap-2", children: [_jsxs("div", { className: "max-w-[75%] space-y-2", children: [_jsx("p", { className: "text-[13px] font-medium leading-snug", children: truncatePrompt(run.prompt) }), run.notes ? (_jsxs("p", { className: "text-[11px] italic text-slate-400", children: ["\"", summariseNote(run.notes), "\""] })) : null] }), _jsx("span", { className: "text-[11px] uppercase tracking-wide text-slate-500", children: timeFormatter.format(new Date(run.created_at)) })] }), _jsxs("div", { className: "mt-3 flex flex-wrap gap-2 text-[11px]", children: [_jsxs("span", { className: badgeClass, children: ["Model ", run.model] }), _jsxs("span", { className: badgeClass, children: ["Embedding ", run.embedding_model] }), _jsxs("span", { className: badgeClass, children: ["Temp ", run.temperature.toFixed(2)] }), run.top_p != null ? _jsxs("span", { className: badgeClass, children: ["Top-p ", run.top_p.toFixed(2)] }) : null, run.seed != null ? _jsxs("span", { className: badgeClass, children: ["Seed ", run.seed] }) : null, _jsxs("span", { className: badgeClass, children: ["UMAP n ", run.umap.n_neighbors] }), _jsxs("span", { className: badgeClass, children: ["UMAP dist ", run.umap.min_dist.toFixed(2)] }), _jsxs("span", { className: badgeClass, children: ["UMAP metric ", run.umap.metric] }), _jsxs("span", { className: badgeClass, children: ["N ", run.n] }), _jsxs("span", { className: badgeClass, children: ["Responses ", run.response_count] }), run.segment_count ? (_jsxs("span", { className: badgeClass, children: ["Segments ", run.segment_count] })) : null, run.chunk_size ? (_jsxs("span", { className: badgeClass, children: ["Chunk ", run.chunk_size, " w"] })) : null, run.processing_time_ms != null ? (_jsxs("span", { className: badgeClass, children: ["Processing ", formatDuration(run.processing_time_ms)] })) : null, _jsxs("span", { className: badgeClass, children: ["Status ", run.status] })] }), _jsxs("div", { className: "mt-3 flex gap-2", children: [_jsx("button", { type: "button", onClick: () => handleSelect(run), className: "flex-1 rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-cyan-400 hover:text-cyan-100", disabled: active && isLoadingHistory, children: active ? "Viewing" : "Load" }), _jsx("button", { type: "button", onClick: () => handleDuplicate(run), className: "rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-200 transition hover:border-emerald-400 hover:text-emerald-100 disabled:cursor-not-allowed disabled:opacity-60", disabled: isDuplicating, children: "Re-run" })] })] }) }, run.id));
                                }) })] }, group.label)))] })] }));
});
