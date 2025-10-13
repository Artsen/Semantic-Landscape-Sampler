/**
 * Side panel showing detailed metrics for selected or hovered responses and segments.
 *
 * Components:
 *  - PointDetailsPanel: Chooses between response or segment detail stacks.
 *  - ResponseDetails / SegmentDetails: Present per-item metrics, raw text, and contextual badges.
 */

import { memo, useMemo, useState } from "react";

import { useRunStore } from "@/store/runStore";
import type { RunStoreState } from "@/store/runStore";
import { useSegmentContext } from "@/hooks/useSegmentContext";
import type { RunWorkflow } from "@/hooks/useRunWorkflow";

type Results = NonNullable<RunStoreState["results"]>;
type ResponsePointType = Results["points"][number];
type SegmentPointType = Results["segments"][number];
type PointDetailsPanelProps = {
  workflow: RunWorkflow;
};

export const PointDetailsPanel = memo(function PointDetailsPanel({ workflow }: PointDetailsPanelProps) {
  const {
    results,
    levelMode,
    selectedPointIds,
    selectedSegmentIds,
    hoveredPointId,
    hoveredSegmentId,
    setSelectedPoints,
    setSelectedSegments,
    setFocusedResponse,
    isGenerating,
    exportFormat,
  } = useRunStore((state) => ({
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

  const points: ResponsePointType[] = results?.points ?? [];
  const segments: SegmentPointType[] = results?.segments ?? [];

  const responseMap = useMemo(() => new Map(points.map((point) => [point.id, point])), [points]);
  const segmentMap = useMemo(() => new Map(segments.map((segment) => [segment.id, segment])), [segments]);

  const query = searchTerm.trim().toLowerCase();

  const responseMatches = useMemo(() => {
    if (!query) {
      return [] as ResponsePointType[];
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
      return [] as SegmentPointType[];
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
    } else {
      setSelectedPoints([]);
    }
  };

  const handleExportSelection = async () => {
    if (selectionCount === 0) {
      return;
    }
    try {
      await exportDataset({ scope: "selection" });
    } catch (err) {
      console.error(err);
    }
  };

  const activeSegments = showingSegments
    ? (activeIds.map((id) => segmentMap.get(id)).filter(Boolean) as SegmentPointType[])
    : [];
  const activeResponses = showingSegments
    ? []
    : (activeIds.map((id) => responseMap.get(id)).filter(Boolean) as ResponsePointType[]);

  let headerTitle: string;
  const showClearSelection = !showSearchResults && activeIds.length > 0;
  const showClearSearch = showSearchResults && searchTerm.length > 0;

  if (showingSegments) {
    if (showSearchResults) {
      headerTitle = `Segment matches: ${segmentMatches.length}`;
    } else if (selectedSegmentIds.length) {
      headerTitle = `${selectedSegmentIds.length} segments`;
    } else if (hoveredSegmentId) {
      headerTitle = "Hovered segment";
    } else {
      headerTitle = "No segment focus";
    }
  } else if (showSearchResults) {
    headerTitle = `Response matches: ${responseMatches.length}`;
  } else if (selectedPointIds.length) {
    headerTitle = `${selectedPointIds.length} responses`;
  } else if (hoveredPointId) {
    headerTitle = "Hovered response";
  } else {
    headerTitle = "No response focus";
  }

  const handleResponseSearchSelect = (point: ResponsePointType) => {
    setSelectedPoints([point.id]);
    setFocusedResponse(point.id);
  };

  const bodyContent = (() => {
    if (!results) {
      return <p className="text-xs text-slate-500">Run a sample to inspect details.</p>;
    }

    if (showSearchResults) {
      return (
        <div className="space-y-3">
          {showingSegments
            ? segmentMatches.map((segment) => (
                <SegmentDetails
                  key={segment.id}
                  segment={segment}
                  parent={responseMap.get(segment.response_id)}
                />
              ))
            : responseMatches.map((point) => (
                <button
                  key={point.id}
                  type="button"
                  className="w-full text-left"
                  onClick={() => handleResponseSearchSelect(point)}
                >
                  <ResponseDetails point={point} />
                </button>
              ))}
        </div>
      );
    }

    if (showingSegments) {
      if (!activeSegments.length) {
        return <p className="text-xs text-slate-500">Hover or select segments to view details.</p>;
      }
      return activeSegments.map((segment) => (
        <SegmentDetails
          key={segment.id}
          segment={segment}
          parent={responseMap.get(segment.response_id)}
        />
      ));
    }

    if (!activeResponses.length) {
      return <p className="text-xs text-slate-500">Hover or select responses to view details.</p>;
    }

    return activeResponses.map((point) => <ResponseDetails key={point.id} point={point} />);
  })();

  return (
    <aside className="glass-panel flex w-full max-h-[70vh] flex-col overflow-hidden rounded-2xl border border-slate-800/70 bg-slate-950/75 p-4 text-sm text-slate-200 xl:w-[360px]">
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-slate-100">{headerTitle}</h2>
        <div className="flex items-center gap-2 text-[11px]">
          {showClearSelection ? (
            <button
              type="button"
              onClick={handleExportSelection}
              className="rounded-lg border border-cyan-500/40 px-3 py-1 text-xs text-cyan-200 transition hover:border-cyan-300 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isGenerating || selectionCount === 0}
            >
              Export selection ({exportBadge})
            </button>
          ) : null}
          {showClearSelection ? (
            <button
              type="button"
              onClick={clearSelection}
              className="rounded-lg border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              Clear
            </button>
          ) : null}
          {showClearSearch ? (
            <button
              type="button"
              onClick={() => setSearchTerm("")}
              className="rounded-lg border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-200"
            >
              Clear search
            </button>
          ) : null}
        </div>
      </header>

      <div className="mt-3">
        <input
          type="search"
          value={searchTerm}
          onChange={(event) => setSearchTerm(event.target.value)}
          placeholder="Search responses or segments"
          className="w-full rounded-xl border border-slate-800/60 bg-slate-900/50 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
        />
      </div>

      <div className="mt-4 flex-1 min-h-0 space-y-4 overflow-y-auto pr-2 text-slate-200 scrollbar-thin pb-4">
        {bodyContent}
      </div>
    </aside>
  );
});

type ResponseDetailsProps = {
  point: ResponsePointType;
};

function ResponseDetails({ point }: ResponseDetailsProps) {
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

  return (
    <article className="space-y-3 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3">
      <header className="flex items-center justify-between text-xs text-slate-400">
        <span>Sample #{point.index}</span>
        <div className="flex items-center gap-2">
          <span>Cluster {clusterLabel === -1 ? "noise" : clusterLabel}</span>
          {outlierPct != null && outlierPct >= 25 ? (
            <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-amber-300">
              Outlier {outlierPct}%
            </span>
          ) : null}
        </div>
      </header>

      <div className="flex items-center justify-between text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-24 overflow-hidden rounded-full bg-slate-800">
            <span className="block h-full bg-cyan-400" style={{ width: `${similarityPct ?? 0}%` }} />
          </span>
          <span>Similarity {similarity != null ? similarity.toFixed(2) : "--"}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-20 overflow-hidden rounded-full bg-slate-800">
            <span className="block h-full bg-amber-400" style={{ width: `${outlierPct ?? 0}%` }} />
          </span>
          <span>Outlier {outlierScore != null ? outlierScore.toFixed(2) : "--"}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 rounded-lg border border-slate-800/70 bg-slate-900/30 p-2 text-[11px] text-slate-300">
        <div>
          <p className="font-semibold text-slate-200">Tokens</p>
          <p>Prompt: {point.prompt_tokens ?? "--"}</p>
          <p>Completion: {point.completion_tokens ?? "--"}</p>
          <p>Embedding: {point.embedding_tokens ?? "--"}</p>
        </div>
        <div>
          <p className="font-semibold text-slate-200">Cost (USD)</p>
          <p>Completion: {hasCompletionCost ? `$${(completionCost ?? 0).toFixed(6)}` : "--"}</p>
          <p>Embedding: {hasEmbeddingCost ? `$${(embeddingCost ?? 0).toFixed(6)}` : "--"}</p>
          <p>Total: {hasCompletionCost || hasEmbeddingCost ? `$${totalCost.toFixed(6)}` : "--"}</p>
        </div>
      </div>

      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{point.full_text}</p>

      <footer className="grid grid-cols-3 gap-2 text-[11px] text-slate-400">
        <span>Tokens: {point.tokens ?? point.usage?.completion_tokens ?? "--"}</span>
        <span>Prob: {point.probability?.toFixed(2) ?? "--"}</span>
        <span>Finish: {point.finish_reason ?? "n/a"}</span>
      </footer>
    </article>
  );
}

type SegmentDetailsProps = {
  segment: SegmentPointType;
  parent?: ResponsePointType;
};

function SegmentDetails({ segment, parent }: SegmentDetailsProps) {
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

  return (
    <article className="space-y-3 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3">
      <header className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
        <span>
          Sample #{segment.response_index} - segment #{segment.position + 1}
          {segment.role ? ` - ${segment.role}` : ""}
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <span>Cluster {clusterLabel === -1 ? "noise" : clusterLabel}</span>
          {segment.is_cached ? (
            <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-emerald-200">
              Cached
            </span>
          ) : null}
          {segment.is_duplicate ? (
            <span className="rounded-full border border-amber-400/40 bg-amber-500/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-amber-200">
              Duplicate
            </span>
          ) : null}
          {outlierPct != null && outlierPct >= 25 ? (
            <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-2 py-[2px] text-[10px] font-semibold uppercase tracking-wide text-cyan-200">
              Divergence {outlierPct}%
            </span>
          ) : null}
        </div>
      </header>

      <div className="flex items-center justify-between text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-24 overflow-hidden rounded-full bg-slate-800">
            <span className="block h-full bg-cyan-400" style={{ width: `${similarityPct ?? 0}%` }} />
          </span>
          <span>Similarity {similarity != null ? similarity.toFixed(2) : "--"}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-2 w-20 overflow-hidden rounded-full bg-slate-800">
            <span className="block h-full bg-amber-400" style={{ width: `${outlierPct ?? 0}%` }} />
          </span>
          <span>Outlier {outlierScore != null ? outlierScore.toFixed(2) : "--"}</span>
        </div>
      </div>

      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{segment.text}</p>

      {context ? (
        <div className="space-y-2 rounded-lg border border-slate-800/70 bg-slate-900/30 p-3 text-[11px] text-slate-300">
          {context.top_terms.length ? (
            <div>
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Top terms</p>
              <div className="mt-2 flex flex-wrap gap-1">
                {context.top_terms.slice(0, 6).map((item) => (
                  <span
                    key={segment.id + "-" + item.term}
                    className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2 py-[2px] text-[10px] text-cyan-100"
                    title={"TF-IDF weight " + item.weight.toFixed(3)}
                  >
                    {item.term}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
          {context.exemplar_preview ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Closest exemplar</p>
              <p className="text-slate-400">
                {context.exemplar_preview.length > 160
                  ? context.exemplar_preview.slice(0, 160) + "..."
                  : context.exemplar_preview}
              </p>
            </div>
          ) : null}
          {context.why_here && (context.why_here.sim_to_exemplar != null || context.why_here.sim_to_nn != null) ? (
            <div className="flex flex-wrap gap-3 text-[10px] text-slate-400">
              {context.why_here.sim_to_exemplar != null ? (
                <span>Sim to exemplar {context.why_here.sim_to_exemplar.toFixed(2)}</span>
              ) : null}
              {context.why_here.sim_to_nn != null ? (
                <span>Nearest neighbour {context.why_here.sim_to_nn.toFixed(2)}</span>
              ) : null}
            </div>
          ) : null}
          {context.neighbors.length ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase tracking-wide text-slate-500">Nearest neighbours</p>
              <ul className="space-y-1 text-[10px] text-slate-300">
                {context.neighbors.slice(0, 5).map((neighbor) => (
                  <li key={neighbor.id}>
                    <span className="text-cyan-200">{neighbor.similarity.toFixed(2)}</span>  -  {neighbor.text}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : isFetching ? (
        <p className="text-[11px] text-slate-500">Loading segment context...</p>
      ) : null}

      <footer className="space-y-2 text-[11px] text-slate-400">
        <div className="flex flex-wrap gap-3">
          <span>Tokens: {segment.tokens ?? "--"}</span>
          <span>Prompt sim: {segment.prompt_similarity != null ? segment.prompt_similarity.toFixed(2) : "--"}</span>
          <span>Embedding cost: {hasEmbeddingCost ? `$${(embeddingCost ?? 0).toFixed(6)}` : "--"}</span>
        </div>
        <div className="flex flex-wrap gap-3 text-slate-500">
          <span>Cache: {segment.is_cached ? "cached" : "new"}</span>
          <span>Duplicate: {segment.is_duplicate ? "yes" : "no"}</span>
        </div>
        <div className="flex flex-wrap gap-3 text-[10px] text-slate-600">
          {segment.text_hash ? (
            <span>Hash <code>{segment.text_hash.slice(0, 12)}...</code></span>
          ) : null}
          {segment.simhash64 != null ? (
            <span>SimHash {segment.simhash64}</span>
          ) : null}
        </div>
        {parent ? (
          <p className="text-[11px] text-slate-500">
            Parent response #{parent.index} - Cluster {parent.cluster ?? "?"}
          </p>
        ) : null}
      </footer>
    </article>
  );
}






