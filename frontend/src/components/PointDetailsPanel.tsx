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

type Results = NonNullable<RunStoreState["results"]>;
type ResponsePointType = Results["points"][number];
type SegmentPointType = Results["segments"][number];

export const PointDetailsPanel = memo(function PointDetailsPanel() {
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
  }));

  const [searchTerm, setSearchTerm] = useState("");

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

  const clearSelection = () => {
    if (showingSegments) {
      setSelectedSegments([]);
    } else {
      setSelectedPoints([]);
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
    <aside className="glass-panel flex w-[360px] flex-col rounded-2xl border border-slate-800/70 bg-slate-950/75 p-4 text-sm text-slate-200">
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-slate-100">{headerTitle}</h2>
        <div className="flex items-center gap-2 text-[11px]">
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

  return (
    <article className="space-y-3 rounded-xl border border-slate-800/80 bg-slate-900/40 p-3">
      <header className="flex items-center justify-between text-xs text-slate-400">
        <span>
          Sample #{segment.response_index} · segment #{segment.position + 1}
          {segment.role ? ` · ${segment.role}` : ""}
        </span>
        <div className="flex items-center gap-2">
          <span>Cluster {clusterLabel === -1 ? "noise" : clusterLabel}</span>
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

      <footer className="space-y-1 text-[11px] text-slate-400">
        <div className="flex gap-3">
          <span>Tokens: {segment.tokens ?? "--"}</span>
          <span>Prompt sim: {segment.prompt_similarity != null ? segment.prompt_similarity.toFixed(2) : "--"}</span>
          <span>Embedding cost: {hasEmbeddingCost ? `$${(embeddingCost ?? 0).toFixed(6)}` : "--"}</span>
        </div>
        {parent ? (
          <p className="text-[11px] text-slate-500">
            Parent response #{parent.index} · Cluster {parent.cluster ?? "?"}
          </p>
        ) : null}
      </footer>
    </article>
  );
}



