import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { memo, useMemo } from "react";
import { useRunStore } from "@/store/runStore";
import { InfoTooltip } from "@/components/InfoTooltip";
const panelClass = "glass-panel flex flex-col gap-4 rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4";
const barBackground = "h-1.5 w-full rounded-full bg-slate-800/70";
const barForeground = "h-full rounded-full bg-cyan-400";
const formatPercent = (value) => {
    if (value == null || Number.isNaN(value)) {
        return "—";
    }
    return `${(value * 100).toFixed(1)}%`;
};
const clamp01 = (value) => {
    if (value == null || Number.isNaN(value)) {
        return 0;
    }
    return Math.max(0, Math.min(1, value));
};
const sortSweepPoints = (points) => {
    return [...points].sort((a, b) => {
        const aScore = a.silhouette_feature ?? a.silhouette_embed ?? -Infinity;
        const bScore = b.silhouette_feature ?? b.silhouette_embed ?? -Infinity;
        return bScore - aScore;
    });
};
export const ClusterMetricsPanel = memo(function ClusterMetricsPanel() {
    const { clusterMetrics, clusterAlgo, hdbscanMinClusterSize, hdbscanMinSamples, isRecomputingClusters, } = useRunStore((state) => ({
        clusterMetrics: state.clusterMetrics,
        clusterAlgo: state.clusterAlgo,
        hdbscanMinClusterSize: state.hdbscanMinClusterSize,
        hdbscanMinSamples: state.hdbscanMinSamples,
        isRecomputingClusters: state.isRecomputingClusters,
    }));
    const silhouettes = useMemo(() => [
        { label: "Embedding", value: clusterMetrics?.silhouette_embed ?? null },
        { label: "Feature", value: clusterMetrics?.silhouette_feature ?? null },
    ], [clusterMetrics]);
    const bootstrap = clusterMetrics?.stability?.bootstrap;
    const persistence = clusterMetrics?.stability?.persistence ?? null;
    const sweepPoints = useMemo(() => {
        if (!clusterMetrics?.sweep) {
            return [];
        }
        return sortSweepPoints(clusterMetrics.sweep.points).slice(0, 4);
    }, [clusterMetrics]);
    if (!clusterMetrics) {
        return (_jsxs("div", { className: panelClass, title: "Metrics summarising how reliable the current clustering is.", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("h2", { className: "text-sm font-semibold text-slate-200", children: ["Cluster Metrics ", _jsx(InfoTooltip, { text: "Snapshot of silhouettes, stability, and sweep results.", className: "ml-1" })] }), isRecomputingClusters ? (_jsx("span", { className: "text-xs text-cyan-300", children: "Recomputing\u2026" })) : null] }), _jsx("p", { className: "text-xs text-slate-500", children: "Metrics will appear after sampling. Adjust the cluster parameters and run a recompute to populate this panel." })] }));
    }
    const renderSilhouette = ({ label, value }) => {
        const width = `${(clamp01(value ?? 0) * 100).toFixed(1)}%`;
        return (_jsxs("div", { className: "flex flex-col gap-1", title: "Silhouette compares cluster cohesion to separation; higher is better.", children: [_jsxs("div", { className: "flex items-center justify-between text-[11px] text-slate-400", children: [_jsx("span", { children: label }), _jsx("span", { className: "text-slate-200", children: formatPercent(value) })] }), _jsx("div", { className: barBackground, children: _jsx("div", { className: barForeground, style: { width } }) })] }, label));
    };
    const renderBootstrap = () => {
        if (!bootstrap || !Object.keys(bootstrap.clusters).length) {
            return _jsx("p", { className: "text-xs text-slate-500", children: "Bootstrap stability pending." });
        }
        return (_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("div", { className: "flex items-center justify-between text-[11px] text-slate-400", children: _jsxs("span", { children: ["Bootstrap (fraction ", Math.round(bootstrap.fraction * 100), "%, ", bootstrap.iterations, " iters)"] }) }), _jsx("div", { className: "flex flex-col gap-1.5", children: Object.entries(bootstrap.clusters).map(([label, stats]) => {
                        const width = `${Math.min(100, Math.max(0, stats.mean * 100)).toFixed(1)}%`;
                        return (_jsxs("div", { className: "flex items-center gap-2 text-[11px] text-slate-300", children: [_jsxs("span", { className: "w-10 rounded bg-slate-800/60 px-2 py-[2px] text-center text-[10px] text-slate-200", children: ["#", label] }), _jsx("div", { className: "flex-1", children: _jsx("div", { className: barBackground, children: _jsx("div", { className: "h-full rounded-full bg-amber-400/80", style: { width } }) }) }), _jsxs("span", { className: "w-20 text-right text-slate-400", children: [stats.mean.toFixed(2), " \u00B1 ", stats.std.toFixed(2)] })] }, label));
                    }) })] }));
    };
    const renderPersistence = () => {
        if (!persistence || !Object.keys(persistence).length) {
            return null;
        }
        return (_jsx("div", { className: "flex flex-wrap gap-2 text-[11px] text-slate-400", children: Object.entries(persistence).map(([label, score]) => (_jsxs("span", { className: "rounded-full border border-slate-800/70 bg-slate-900/70 px-2 py-[2px] text-slate-200", children: ["#", label, " persistence ", score.toFixed(2)] }, label))) }));
    };
    const renderSweep = () => {
        if (!clusterMetrics.sweep || !sweepPoints.length) {
            return _jsx("p", { className: "text-xs text-slate-500", children: "No sweep results captured yet." });
        }
        return (_jsxs("div", { className: "flex flex-col gap-2", children: [_jsxs("div", { className: "text-[11px] text-slate-400", children: ["Baseline: ", clusterMetrics.sweep.baseline.min_cluster_size, " min size / ", clusterMetrics.sweep.baseline.min_samples ?? "—", " min samples"] }), _jsx("div", { className: "grid grid-cols-1 gap-1 text-[11px] text-slate-300", children: sweepPoints.map((point) => (_jsxs("div", { className: "rounded-lg border border-slate-800/60 bg-slate-900/60 px-2 py-1", children: [_jsxs("div", { className: "flex items-center justify-between text-slate-200", children: [_jsxs("span", { children: [point.algo, " (", point.min_cluster_size, "/", point.min_samples, ")"] }), _jsx("span", { children: formatPercent(point.silhouette_feature ?? point.silhouette_embed ?? null) })] }), _jsxs("div", { className: "text-[10px] text-slate-500", children: ["DBI ", point.davies_bouldin?.toFixed(2) ?? "—", "  CHI ", point.calinski_harabasz?.toFixed(1) ?? "—"] })] }, `${point.min_cluster_size}-${point.min_samples}-${point.algo}`))) })] }));
    };
    return (_jsxs("div", { className: panelClass, title: "Metrics summarising how reliable the current clustering is.", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-sm font-semibold text-slate-200", children: "Cluster Metrics" }), _jsxs("p", { className: "text-xs text-slate-500", children: ["Current: ", clusterAlgo.toUpperCase(), " \u2022 min size ", hdbscanMinClusterSize, " \u2022 min samples ", hdbscanMinSamples] })] }), isRecomputingClusters ? (_jsx("span", { className: "text-xs text-cyan-300", children: "Recomputing\u2026" })) : null] }), _jsxs("div", { className: "grid gap-4 md:grid-cols-2", children: [_jsxs("div", { children: [_jsxs("h3", { className: "text-[12px] font-semibold text-slate-300", children: ["Silhouette Scores ", _jsx(InfoTooltip, { text: "Higher silhouette values mean clusters are compact and well separated.", className: "ml-1" })] }), _jsx("div", { className: "mt-2 flex flex-col gap-2", children: silhouettes.map(renderSilhouette) })] }), _jsxs("div", { children: [_jsxs("h3", { className: "text-[12px] font-semibold text-slate-300", children: ["Cluster Indices ", _jsx(InfoTooltip, { text: "Secondary metrics that double-check cohesion and separation.", className: "ml-1" })] }), _jsxs("div", { className: "mt-2 flex flex-wrap gap-2 text-[11px] text-slate-400", children: [_jsxs("span", { className: "rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200", children: ["DBI ", clusterMetrics.davies_bouldin?.toFixed(2) ?? "—"] }), _jsxs("span", { className: "rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200", children: ["CHI ", clusterMetrics.calinski_harabasz?.toFixed(1) ?? "—"] }), _jsxs("span", { className: "rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200", children: ["Clusters ", clusterMetrics.n_clusters ?? "—"] }), _jsxs("span", { className: "rounded-full border border-slate-800/60 bg-slate-900/70 px-2 py-[2px] text-slate-200", children: ["Noise ", clusterMetrics.n_noise ?? "—"] })] })] }), _jsxs("div", { children: [_jsxs("h3", { className: "text-[12px] font-semibold text-slate-300", children: ["Stability ", _jsx(InfoTooltip, { text: "Bootstrap and persistence scores show how durable each cluster is.", className: "ml-1" })] }), _jsxs("div", { className: "mt-2 flex flex-col gap-2", children: [renderBootstrap(), renderPersistence()] })] }), _jsxs("div", { children: [_jsxs("h3", { className: "text-[12px] font-semibold text-slate-300", children: ["Parameter Sweep Top Picks ", _jsx(InfoTooltip, { text: "Suggested HDBSCAN settings ranked by silhouette.", className: "ml-1" })] }), _jsx("div", { className: "mt-2", children: renderSweep() })] })] }), _jsx("p", { className: "text-xs text-slate-500", children: "Silhouettes in 2D can be inflated; check feature-space silhouette too." })] }));
});
