import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Legend for response and segment clusters displayed beside the point cloud.
 *
 * Components:
 *  - ClusterLegend: Lists clusters with keywords, counts, toggles, and export quick actions.
 */
import { memo } from "react";
import { useRunStore } from "@/store/runStore";
import { InfoTooltip } from "@/components/InfoTooltip";
const METHOD_LABELS = {
    hdbscan: "HDBSCAN",
    kmeans: "K-Means",
};
export const ClusterLegend = memo(function ClusterLegend({ responseClusters, segmentClusters, onExportCluster, exportDisabled, formatLabel, }) {
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
    return (_jsxs("div", { className: "glass-panel max-h-[360px] overflow-y-auto rounded-xl border border-slate-800/60 p-4 text-xs scrollbar-thin", title: "Toggle or export clusters. Hover badges for quick explanations.", children: [responseClusters.length ? (_jsxs("section", { className: "space-y-3", children: [_jsxs("header", { className: "flex items-center justify-between", title: "Response-level clusters summarised by keywords and centroid similarity.", children: [_jsxs("span", { className: "text-xs font-semibold uppercase tracking-[0.22rem] text-slate-400", children: ["Response clusters ", _jsx(InfoTooltip, { text: "Click to hide or show all responses in this cluster.", className: "ml-1" })] }), _jsx("span", { className: "text-[11px] text-slate-500", children: "Toggle visibility" })] }), _jsx("ul", { className: "space-y-3", children: responseClusters.map((cluster) => {
                            const key = String(cluster.label);
                            const visible = clusterVisibility[key] ?? true;
                            const color = clusterPalette[key] ?? "#94a3b8";
                            const method = METHOD_LABELS[cluster.method] ?? cluster.method;
                            const title = cluster.noise ? "Noise / outliers" : `Cluster ${cluster.label}`;
                            return (_jsx("li", { className: "flex items-center gap-3", children: _jsxs("div", { className: `flex w-full flex-col gap-2 rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2 transition ${visible ? "shadow-inner shadow-cyan-500/10" : "opacity-50"}`, children: [_jsxs("button", { type: "button", onClick: () => toggleCluster(cluster.label), onMouseEnter: () => setHoveredCluster(cluster.label), onMouseLeave: () => setHoveredCluster(null), className: "flex items-start justify-between gap-3 text-left", title: "Click to hide or show this cluster in the point cloud.", children: [_jsxs("div", { className: "flex items-start gap-3", children: [_jsx("span", { className: "mt-1 inline-flex h-3 w-3 rounded-full", style: { backgroundColor: color } }), _jsxs("div", { className: "space-y-1", children: [_jsx("p", { className: "font-medium text-slate-100", children: title }), _jsxs("p", { className: "text-[11px] text-slate-400", children: [cluster.size, " samples \u00B7 ", method, cluster.average_similarity != null ? ` · avg sim ${cluster.average_similarity.toFixed(2)}` : ""] }), cluster.keywords.length ? (_jsx("p", { className: "text-[11px] text-slate-500", children: cluster.keywords.slice(0, 4).join(", ") })) : null] })] }), _jsxs("div", { className: "text-right text-[10px] text-slate-500", children: [_jsx("p", { children: "Centroid" }), _jsx("p", { children: cluster.centroid_xyz.map((value) => value.toFixed(2)).join(", ") })] })] }), _jsxs("div", { className: "flex items-center justify-between text-[10px] text-slate-500", children: [exportBadge ? (_jsx("span", { className: "rounded-full border border-slate-700/60 px-2 py-[1px] text-[9px] uppercase tracking-wide text-slate-400", children: exportBadge })) : (_jsx("span", {})), _jsx("button", { type: "button", onClick: () => onExportCluster?.({ label: cluster.label, mode: "responses" }), disabled: exportDisabled || !onExportCluster, className: "rounded-full border border-slate-700/60 px-2 py-[2px] text-[10px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-60", children: "Export" })] })] }) }, `response-${cluster.label}`));
                        }) })] })) : null, segmentClusters.length ? (_jsxs("section", { className: "mt-5 space-y-3", children: [_jsxs("header", { className: "flex items-center justify-between", title: "Segment-level clusters showing recurring threads inside responses.", children: [_jsxs("span", { className: "text-xs font-semibold uppercase tracking-[0.22rem] text-slate-400", children: ["Segment themes ", _jsx(InfoTooltip, { text: "Snippets grouped by topic; great for spotting recurring motifs.", className: "ml-1" })] }), _jsx("span", { className: "text-[11px] text-slate-500", children: "Clustered micro-ideas" })] }), _jsx("ul", { className: "space-y-3", children: segmentClusters.map((cluster) => {
                            const title = cluster.noise ? "Orphan fragments" : `Group ${cluster.label}`;
                            return (_jsxs("li", { className: "rounded-lg border border-slate-800/70 bg-slate-900/30 px-3 py-2", children: [_jsx("p", { className: "font-medium text-slate-100", children: title }), _jsxs("p", { className: "text-[11px] text-slate-400", children: [cluster.size, " segments", cluster.average_similarity != null ? ` · avg sim ${cluster.average_similarity.toFixed(2)}` : ""] }), _jsx("p", { className: "text-[11px] text-slate-500", children: (cluster.theme ?? cluster.keywords.slice(0, 4).join(", ")) || "—" }), _jsxs("div", { className: "mt-2 flex items-center justify-between text-[10px] text-slate-500", children: [exportBadge ? (_jsx("span", { className: "rounded-full border border-slate-700/60 px-2 py-[1px] text-[9px] uppercase tracking-wide text-slate-400", children: exportBadge })) : (_jsx("span", {})), _jsx("button", { type: "button", onClick: () => onExportCluster?.({ label: cluster.label, mode: "segments" }), disabled: exportDisabled || !onExportCluster, className: "rounded-full border border-slate-700/60 px-2 py-[2px] text-[10px] text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:opacity-60", title: "Download every response assigned to this cluster.", children: "Export" })] })] }, `segment-${cluster.label}`));
                        }) })] })) : null] }));
});
