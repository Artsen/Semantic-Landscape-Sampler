
import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  useComparisonStore,
  type ComparisonConfig,
  type ComparisonViewMode,
} from "@/store/comparisonStore";
import { useRunStore } from "@/store/runStore";
import { PointCloudScene } from "@/components/PointCloudScene";
import { fetchRunResults } from "@/services/api";
import type {
  CompareRunsResponse,
  ComparisonLink,
  ComparisonPoint,
  ComparisonTransformComponent,
  MovementHistogramBin,
  RunResultsResponse,
  ResponsePoint,
  SegmentPoint,
  SegmentEdge,
  ResponseHull,
  RunSummary,
} from "@/types/run";

interface HoverState {
  point: ComparisonPoint;
  clientX: number;
  clientY: number;
}

interface MovementVector {
  target: [number, number];
  link: ComparisonLink;
  distance: number;
}

interface ComparisonSceneData {
  responses: ResponsePoint[];
  segments: SegmentPoint[];
  segmentEdges: SegmentEdge[];
  responseHulls: ResponseHull[];
  clusterPalette: Record<string, string>;
}

export function CompareView() {
  const {
    config,
    result,
    isLoading,
    error,
    setLeftRunId,
    setRightRunId,
    swapRuns,
    setMode,
    setView,
    setMinShared,
    setHistogramBins,
    setMaxLinks,
    setSaveResults,
    setShowSharedOnly,
    setHighlightThreshold,
    setShowMovementVectors,
    setShowDensity,
    setShowSimilarityEdges,
    setShowParentThreads,
    setShowResponseHulls,
    setViewDimension,
    runComparison,
  } = useComparisonStore((state) => ({
    config: state.config,
    result: state.result,
    isLoading: state.isLoading,
    error: state.error,
    setLeftRunId: state.setLeftRunId,
    setRightRunId: state.setRightRunId,
    swapRuns: state.swapRuns,
    setMode: state.setMode,
    setView: state.setView,
    setMinShared: state.setMinShared,
    setHistogramBins: state.setHistogramBins,
    setMaxLinks: state.setMaxLinks,
    setSaveResults: state.setSaveResults,
    setShowSharedOnly: state.setShowSharedOnly,
    setHighlightThreshold: state.setHighlightThreshold,
    setShowMovementVectors: state.setShowMovementVectors,
    setShowDensity: state.setShowDensity,
    setShowSimilarityEdges: state.setShowSimilarityEdges,
    setShowParentThreads: state.setShowParentThreads,
    setShowResponseHulls: state.setShowResponseHulls,
    setViewDimension: state.setViewDimension,
    runComparison: state.runComparison,
  }));
  const runHistory = useRunStore((state) => state.runHistory ?? []);

  const leftRunId = config.leftRunId;
  const rightRunId = config.rightRunId;

  const leftResultsQuery = useQuery({
    queryKey: ["compare", "run-results", leftRunId],
    enabled: Boolean(leftRunId),
    queryFn: async () => {
      if (!leftRunId) {
        throw new Error("Missing left run id");
      }
      return fetchRunResults(leftRunId);
    },
    staleTime: 60_000,
  });

  const rightResultsQuery = useQuery({
    queryKey: ["compare", "run-results", rightRunId],
    enabled: Boolean(rightRunId),
    queryFn: async () => {
      if (!rightRunId) {
        throw new Error("Missing right run id");
      }
      return fetchRunResults(rightRunId);
    },
    staleTime: 60_000,
  });

  const [hoverState, setHoverState] = useState<HoverState | null>(null);
  const [activeLink, setActiveLink] = useState<ComparisonLink | null>(null);

  const linkMaps = useMemo(() => buildLinkMaps(result?.links ?? []), [result?.links]);

  const sceneData = useMemo(() => {
    if (!leftResultsQuery.data || !rightResultsQuery.data) {
      return null;
    }
    return prepareComparisonSceneData({
      mode: config.mode,
      view: config.view,
      showSharedOnly: config.showSharedOnly,
      comparison: result,
      left: leftResultsQuery.data,
      right: rightResultsQuery.data,
    });
  }, [
    config.mode,
    config.view,
    config.showSharedOnly,
    leftResultsQuery.data,
    rightResultsQuery.data,
    result,
  ]);

  const sceneLoading = leftResultsQuery.isLoading || rightResultsQuery.isLoading;
  const sceneError = leftResultsQuery.error || rightResultsQuery.error;

  const handleHover = useCallback(
    (state: HoverState | null) => {
      setHoverState(state);
      if (!state) {
        setActiveLink(null);
        return;
      }
      const link = findLinkForPoint(linkMaps, state.point.id);
      setActiveLink(link);
    },
    [linkMaps],
  );

  const handleCompareClick = useCallback(async () => {
    await runComparison();
  }, [runComparison]);

  const tooltipLines =
    hoverState && result ? buildTooltipContent(hoverState.point, result.links) : null;

  useEffect(() => {
    const store = useRunStore.getState();
    store.setShowDensity(config.showDensity);
    store.setShowEdges(config.mode === "segments" ? config.showSimilarityEdges : false);
    store.setShowParentThreads(
      config.mode === "segments" ? config.showParentThreads || config.showResponseHulls : false,
    );
  }, [
    config.mode,
    config.showDensity,
    config.showSimilarityEdges,
    config.showParentThreads,
    config.showResponseHulls,
  ]);

  useEffect(() => {
    useRunStore.getState().setViewMode(config.viewDimension);
  }, [config.viewDimension]);

  useEffect(() => {
    useRunStore.getState().setLevelMode(config.mode === "segments" ? "segments" : "responses");
  }, [config.mode]);

  useEffect(() => {
    if (!sceneData) {
      return;
    }
    const clusterKeys = Object.keys(sceneData.clusterPalette);
    const palette = sceneData.clusterPalette;
    const store = useRunStore.getState();
    store.setClusterPalette(palette);
    useRunStore.setState((state) => {
      const nextVisibility: Record<string, boolean> = { ...state.clusterVisibility };
      clusterKeys.forEach((key) => {
        if (nextVisibility[key] === undefined) {
          nextVisibility[key] = true;
        }
      });
      const existingKeys = Object.keys(nextVisibility);
      existingKeys.forEach((key) => {
        if (!clusterKeys.includes(key)) {
          delete nextVisibility[key];
        }
      });
      return { clusterVisibility: nextVisibility };
    });
    store.setSelectedPoints([]);
    store.setSelectedSegments([]);
    store.setHoveredPoint(undefined);
    store.setHoveredSegment(undefined);
    store.setHoveredCluster(null);
  }, [sceneData]);

  return (
    <div className="flex h-full flex-col gap-4">
      <ComparisonControls
        config={config}
        runHistory={runHistory}
        setLeftRunId={setLeftRunId}
        setRightRunId={setRightRunId}
        swapRuns={swapRuns}
        setMode={setMode}
        setView={setView}
        setMinShared={setMinShared}
        setHistogramBins={setHistogramBins}
        setMaxLinks={setMaxLinks}
        setSaveResults={setSaveResults}
        setShowSharedOnly={setShowSharedOnly}
        setHighlightThreshold={setHighlightThreshold}
        setShowMovementVectors={setShowMovementVectors}
        setShowDensity={setShowDensity}
        setShowSimilarityEdges={setShowSimilarityEdges}
        setShowParentThreads={setShowParentThreads}
        setShowResponseHulls={setShowResponseHulls}
        setViewDimension={setViewDimension}
        isLoading={isLoading}
        onCompare={handleCompareClick}
        error={error}
      />

      <ComparisonPointCloudPanel
        hasRunSelection={Boolean(leftRunId && rightRunId)}
        sceneData={sceneData}
        mode={config.mode}
        isLoading={sceneLoading}
        errorMessage={sceneError ? extractErrorMessage(sceneError) || "Failed to load run geometry" : null}
      />

      <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="relative min-h-[420px] overflow-hidden rounded-2xl border border-border bg-panel p-4">
          {result ? (
            <ComparisonVisualization
              config={config}
              result={result}
              activeLink={activeLink}
              onHover={handleHover}
            />
          ) : (
            <EmptyState />
          )}
        </div>

        <ComparisonMetricsPanel result={result} />
      </div>

      {hoverState && tooltipLines ? (
        <ComparisonTooltip hover={hoverState} tooltip={tooltipLines} />
      ) : null}
    </div>
  );
}
interface ComparisonControlsProps {
  config: ComparisonConfig;
  runHistory: RunSummary[];
  setLeftRunId: (runId: string | null) => void;
  setRightRunId: (runId: string | null) => void;
  swapRuns: () => void;
  setMode: (mode: ComparisonMode) => void;
  setView: (view: ComparisonViewMode) => void;
  setMinShared: (value: number) => void;
  setHistogramBins: (value: number) => void;
  setMaxLinks: (value: number) => void;
  setSaveResults: (value: boolean) => void;
  setShowSharedOnly: (value: boolean) => void;
  setHighlightThreshold: (value: number) => void;
  setShowMovementVectors: (value: boolean) => void;
  setShowDensity: (value: boolean) => void;
  setShowSimilarityEdges: (value: boolean) => void;
  setShowParentThreads: (value: boolean) => void;
  setShowResponseHulls: (value: boolean) => void;
  setViewDimension: (value: "2d" | "3d") => void;
  isLoading: boolean;
  onCompare: () => void;
  error?: string | null;
}

function ComparisonVisualization({
  config,
  result,
  activeLink,
  onHover,
}: {
  config: ComparisonConfig;
  result: CompareRunsResponse;
  activeLink: ComparisonLink | null;
  onHover: (hover: HoverState | null) => void;
}) {
  const points = result.points ?? [];
  const links = result.links ?? [];

  const leftPoints = useMemo(
    () => points.filter((point) => point.source === "left"),
    [points],
  );
  const rightPoints = useMemo(
    () => points.filter((point) => point.source === "right"),
    [points],
  );

  const sharedLinks = useMemo(
    () => links.filter((link) => link.link_type === "exact_hash"),
    [links],
  );
  const sharedLeftIds = useMemo(
    () => new Set(sharedLinks.map((link) => link.left_segment_id)),
    [sharedLinks],
  );
  const sharedRightIds = useMemo(
    () => new Set(sharedLinks.map((link) => link.right_segment_id)),
    [sharedLinks],
  );

  const transform2d = result.transforms?.components?.["2d"];

  const movementData = useMemo(() => {
    const distanceById = new Map<string, number>();
    const leftVectors = new Map<string, MovementVector>();
    const rightVectors = new Map<string, MovementVector>();

    if (!links.length) {
      return { distanceById, leftVectors, rightVectors };
    }

    const pointMap = new Map(points.map((point) => [point.id, point]));

    const rotation = transform2d?.rotation ?? [
      [1, 0],
      [0, 1],
    ];
    const scale =
      typeof transform2d?.scale === "number" && Number.isFinite(transform2d.scale)
        ? transform2d.scale
        : 1;
    const translation = transform2d?.translation ?? [0, 0];

    const rot = rotation.map((row) =>
      row.map((value) => (typeof value === "number" ? value : Number(value) || 0)),
    ) as [number, number][];
    const det = rot[0][0] * rot[1][1] - rot[0][1] * rot[1][0];
    const hasInverse = Math.abs(det) > 1e-9 && Math.abs(scale) > 1e-9;

    const inverse = hasInverse
      ? (coords: [number, number]): [number, number] => {
          const [x, y] = coords;
          const tx = x - translation[0];
          const ty = y - translation[1];
          const invScale = 1 / scale;
          const invRot: [number, number][] = [
            [rot[0][0], rot[1][0]],
            [rot[0][1], rot[1][1]],
          ];
          return [
            (tx * invRot[0][0] + ty * invRot[0][1]) * invScale,
            (tx * invRot[1][0] + ty * invRot[1][1]) * invScale,
          ];
        }
      : null;

    links.forEach((link) => {
      const distance = link.movement_distance ?? 0;
      distanceById.set(link.left_segment_id, distance);
      distanceById.set(link.right_segment_id, distance);

      const leftPoint = pointMap.get(link.left_segment_id);
      const rightPoint = pointMap.get(link.right_segment_id);
      if (!leftPoint || !rightPoint) {
        return;
      }

      if (rightPoint.aligned_coords_2d) {
        leftVectors.set(leftPoint.id, {
          target: rightPoint.aligned_coords_2d as [number, number],
          link,
          distance,
        });
      }

      if (inverse) {
        rightVectors.set(rightPoint.id, {
          target: inverse(leftPoint.coords_2d as [number, number]),
          link,
          distance,
        });
      }
    });

    return { distanceById, leftVectors, rightVectors };
  }, [links, points, transform2d]);

  const {
    distanceById: movementDistances,
    leftVectors: leftMovementVectors,
    rightVectors: rightMovementVectors,
  } = movementData;

  return (
    <>
      {config.view === "side-by-side" ? (
        <div className="flex h-full flex-col gap-4">
          <div className="grid h-full gap-4 lg:grid-cols-2">
            <ScatterPanel
              title={`Left � ${summariseRun(result.left_run)}`}
              points={leftPoints}
              color="#60a5fa"
              sharedIds={sharedLeftIds}
              movementDistances={movementDistances}
              movementVectors={leftMovementVectors}
              highlightThreshold={config.highlightThreshold}
              showSharedOnly={config.showSharedOnly}
              showMovementVectors={config.showMovementVectors}
              activeLinkId={activeLink?.left_segment_id ?? null}
              onHover={onHover}
            />
            <ScatterPanel
              title={`Right � ${summariseRun(result.right_run)}`}
              points={rightPoints}
              color="#f97316"
              sharedIds={sharedRightIds}
              movementDistances={movementDistances}
              movementVectors={rightMovementVectors}
              highlightThreshold={config.highlightThreshold}
              showSharedOnly={config.showSharedOnly}
              showMovementVectors={config.showMovementVectors}
              activeLinkId={activeLink?.right_segment_id ?? null}
              onHover={onHover}
            />
          </div>
        </div>
      ) : (
        <OverlayScatter
          leftPoints={leftPoints}
          rightPoints={rightPoints}
          links={links}
          sharedLeftIds={sharedLeftIds}
          sharedRightIds={sharedRightIds}
          movementDistances={movementDistances}
          highlightThreshold={config.highlightThreshold}
          showSharedOnly={config.showSharedOnly}
          showMovementVectors={config.showMovementVectors}
          activeLink={activeLink}
          onHover={onHover}
          leftLabel={summariseRun(result.left_run)}
          rightLabel={summariseRun(result.right_run)}
        />
      )}
    </>
  );
}

function ComparisonPointCloudPanel({
  hasRunSelection,
  sceneData,
  mode,
  isLoading,
  errorMessage,
}: {
  hasRunSelection: boolean;
  sceneData: ComparisonSceneData | null;
  mode: ComparisonMode;
  isLoading: boolean;
  errorMessage: string | null;
}) {
  const noSelection = !hasRunSelection;
  const hasPoints = Boolean(sceneData && (sceneData.responses.length || sceneData.segments.length));
  const showPlaceholder = !sceneData && !isLoading && !errorMessage;

  return (
    <section className="rounded-2xl border border-border bg-panel p-4">
      <header className="flex flex-col gap-1">
        <h3 className="text-sm font-semibold text-slate-100">Interactive point clouds</h3>
        <p className="text-xs text-slate-400">
          Explore the runs with the full 3D scene. Use the toggles above to adjust density, edges, and
          parent threads.
        </p>
      </header>
      <div className="relative mt-4 h-[420px] min-h-[320px] overflow-hidden rounded-xl border border-slate-800/60 bg-slate-950">
        {noSelection ? (
          <EmptyStateNotice message="Select two runs to load the point clouds." />
        ) : errorMessage ? (
          <EmptyStateNotice message={errorMessage} tone="error" />
        ) : sceneData && hasPoints ? (
          <PointCloudScene
            responses={sceneData.responses}
            segments={sceneData.segments}
            segmentEdges={sceneData.segmentEdges}
            responseHulls={sceneData.responseHulls}
          />
        ) : sceneData && !hasPoints ? (
          <EmptyStateNotice message="No shared points match the current filters." />
        ) : isLoading ? (
          <LoadingNotice message="Loading run geometry…" />
        ) : showPlaceholder ? (
          <EmptyStateNotice message="Run geometry will appear here once loaded." />
        ) : null}
      </div>
    </section>
  );
}

function EmptyStateNotice({
  message,
  tone = "neutral",
}: {
  message: string;
  tone?: "neutral" | "error";
}) {
  const toneClasses =
    tone === "error"
      ? "border-red-500/40 bg-red-500/10 text-red-300"
      : "border-border/60 bg-slate-900/40 text-slate-300";
  return (
    <div className={`absolute inset-0 flex items-center justify-center text-xs ${toneClasses}`}>
      <span className="rounded-md border px-3 py-2">{message}</span>
    </div>
  );
}

function LoadingNotice({ message }: { message: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="flex items-center gap-2 rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-xs text-accent">
        <span className="h-2 w-2 animate-ping rounded-full bg-accent" />
        {message}
      </div>
    </div>
  );
}

function ScatterPanel({
  title,
  points,
  color,
  sharedIds,
  movementDistances,
  movementVectors,
  highlightThreshold,
  showSharedOnly,
  showMovementVectors,
  activeLinkId,
  onHover,
}: {
  title: string;
  points: ComparisonPoint[];
  color: string;
  sharedIds: Set<string>;
  movementDistances: Map<string, number>;
  movementVectors: Map<string, MovementVector>;
  highlightThreshold: number;
  showSharedOnly: boolean;
  showMovementVectors: boolean;
  activeLinkId: string | null;
  onHover: (hover: HoverState | null) => void;
}) {
  const pointLookup = useMemo(() => new Map(points.map((point) => [point.id, point])), [points]);

  const visiblePoints = useMemo(
    () =>
      points.filter((point) => {
        if (showSharedOnly && !sharedIds.has(point.id)) {
          return false;
        }
        return true;
      }),
    [points, showSharedOnly, sharedIds],
  );
  const visiblePointIds = useMemo(
    () => new Set(visiblePoints.map((point) => point.id)),
    [visiblePoints],
  );

  const viewBounds = useMemo(() => computeViewBox(visiblePoints.map((point) => point.coords_2d)), [
    visiblePoints,
  ]);
  const radius = viewBounds.span > 0 ? Math.max(viewBounds.span / 150, 0.02) : 0.05;

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
        <span className="truncate">{title}</span>
        <span>{visiblePoints.length} pts</span>
      </div>
      <div className="relative flex-1">
        {visiblePoints.length ? (
          <svg
            className="h-full w-full"
            viewBox={viewBounds.viewBox}
            onMouseLeave={() => onHover(null)}
          >
            {
              showMovementVectors && movementVectors.size
                ? [...movementVectors.entries()].map(([pointId, vector]) => {
                    if (!visiblePointIds.has(pointId)) {
                      return null;
                    }
                    const source = pointLookup.get(pointId);
                    if (!source) {
                      return null;
                    }
                    const [x1, y1] = source.coords_2d as [number, number];
                    const [x2, y2] = vector.target;
                    const distance = vector.distance;
                    const belowThreshold =
                      distance < highlightThreshold && movementDistances.has(pointId);
                    const isActive =
                      activeLinkId !== null &&
                      (vector.link.left_segment_id === activeLinkId ||
                        vector.link.right_segment_id === activeLinkId);
                    return (
                      <line
                        key={`vector-${pointId}`}
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke={isActive ? "#fef3c7" : color}
                        strokeOpacity={belowThreshold ? 0.18 : isActive ? 0.9 : 0.4}
                        strokeWidth={isActive ? radius * 1.1 : radius * 0.6}
                        strokeDasharray={vector.link.link_type === "nn" ? "3 4" : undefined}
                      />
                    );
                  })
                : null
            }
            {visiblePoints.map((point) => {
              const [x, y] = point.coords_2d as [number, number];
              const distance = movementDistances.get(point.id) ?? 0;
              const isDimmed = distance < highlightThreshold && movementDistances.has(point.id);
              const isActive = activeLinkId !== null && point.id === activeLinkId;
              return (
                <circle
                  key={point.id}
                  cx={x}
                  cy={y}
                  r={radius * (isActive ? 1.6 : 1)}
                  fill={color}
                  fillOpacity={isDimmed ? 0.25 : 0.85}
                  stroke={isActive ? "#fef3c7" : "transparent"}
                  strokeWidth={isActive ? radius * 0.6 : 0}
                  onMouseEnter={(event) =>
                    onHover({ point, clientX: event.clientX, clientY: event.clientY })
                  }
                />
              );
            })}
          </svg>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500">
            No points to display
          </div>
        )}
      </div>
    </div>
  );
}

function OverlayScatter({
  leftPoints,
  rightPoints,
  links,
  sharedLeftIds,
  sharedRightIds,
  movementDistances,
  highlightThreshold,
  showSharedOnly,
  showMovementVectors,
  activeLink,
  onHover,
  leftLabel,
  rightLabel,
}: {
  leftPoints: ComparisonPoint[];
  rightPoints: ComparisonPoint[];
  links: ComparisonLink[];
  sharedLeftIds: Set<string>;
  sharedRightIds: Set<string>;
  movementDistances: Map<string, number>;
  highlightThreshold: number;
  showSharedOnly: boolean;
  showMovementVectors: boolean;
  activeLink: ComparisonLink | null;
  onHover: (hover: HoverState | null) => void;
  leftLabel: string;
  rightLabel: string;
}) {
  const alignedPoints = useMemo(() => {
    const annotated: Array<{ point: ComparisonPoint; position: [number, number] }> = [];
    for (const point of leftPoints) {
      annotated.push({ point, position: point.coords_2d });
    }
    for (const point of rightPoints) {
      annotated.push({ point, position: point.aligned_coords_2d });
    }
    return annotated;
  }, [leftPoints, rightPoints]);

  const viewBounds = useMemo(
    () => computeViewBox(alignedPoints.map((entry) => entry.position)),
    [alignedPoints],
  );
  const radius = viewBounds.span > 0 ? Math.max(viewBounds.span / 160, 0.02) : 0.05;

  const pointById = useMemo(() => {
    const map = new Map<string, { point: ComparisonPoint; position: [number, number] }>();
    alignedPoints.forEach((entry) => {
      map.set(entry.point.id, entry);
    });
    return map;
  }, [alignedPoints]);

  const visibleLinks = useMemo(
    () =>
      links.filter((link) => {
        if (showSharedOnly) {
          return (
            sharedLeftIds.has(link.left_segment_id) &&
            sharedRightIds.has(link.right_segment_id)
          );
        }
        return true;
      }),
    [links, showSharedOnly, sharedLeftIds, sharedRightIds],
  );

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
        <div className="flex items-center gap-4">
          <LegendSwatch color="#60a5fa" label={leftLabel} />
          <LegendSwatch color="#f97316" label={rightLabel} />
        </div>
        <span>{alignedPoints.length} pts</span>
      </div>
      <div className="relative flex-1">
        {alignedPoints.length ? (
          <svg
            className="h-full w-full"
            viewBox={viewBounds.viewBox}
            onMouseLeave={() => onHover(null)}
          >
            {showMovementVectors
              ? visibleLinks.map((link) => {
                  const leftEntry = pointById.get(link.left_segment_id);
                  const rightEntry = pointById.get(link.right_segment_id);
                  if (!leftEntry || !rightEntry) {
                    return null;
                  }
                  const movement = link.movement_distance ?? 0;
                  const representativeDistance =
                    movementDistances.get(link.left_segment_id) ??
                    movementDistances.get(link.right_segment_id) ??
                    movement;
                  const belowThreshold = representativeDistance < highlightThreshold;
                  const isActive =
                    activeLink?.left_segment_id === link.left_segment_id ||
                    activeLink?.right_segment_id === link.right_segment_id;
                  return (
                    <line
                      key={`link-${link.left_segment_id}-${link.right_segment_id}`}
                      x1={leftEntry.position[0]}
                      y1={leftEntry.position[1]}
                      x2={rightEntry.position[0]}
                      y2={rightEntry.position[1]}
                      stroke={isActive ? "#fef3c7" : "#94a3b8"}
                      strokeOpacity={belowThreshold ? 0.2 : isActive ? 0.95 : 0.45}
                      strokeWidth={isActive ? radius * 1.2 : radius * 0.6}
                      strokeDasharray={link.link_type === "nn" ? "3 4" : undefined}
                    />
                  );
                })
              : null}
            {alignedPoints.map(({ point, position }) => {
              const movement = movementDistances.get(point.id) ?? 0;
              const isDimmed = movement < highlightThreshold && movementDistances.has(point.id);
              const isActive =
                activeLink?.left_segment_id === point.id ||
                activeLink?.right_segment_id === point.id;
              const color = point.source === "left" ? "#60a5fa" : "#f97316";
              return (
                <circle
                  key={point.id}
                  cx={position[0]}
                  cy={position[1]}
                  r={radius * (isActive ? 1.6 : 1)}
                  fill={color}
                  fillOpacity={
                    point.source === "right"
                      ? isDimmed
                        ? 0.25
                        : 0.65
                      : isDimmed
                      ? 0.25
                      : 0.9
                  }
                  stroke={isActive ? "#fef3c7" : "transparent"}
                  strokeWidth={isActive ? radius * 0.6 : 0}
                  onMouseEnter={(event) =>
                    onHover({ point, clientX: event.clientX, clientY: event.clientY })
                  }
                />
              );
            })}
          </svg>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500">
            No points to display
          </div>
        )}
      </div>
    </div>
  );
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      <span className="max-w-[140px] truncate">{label}</span>
    </span>
  );
}

function ComparisonControls({
  config,
  runHistory,
  setLeftRunId,
  setRightRunId,
  swapRuns,
  setMode,
  setView,
  setMinShared,
  setHistogramBins,
  setMaxLinks,
  setSaveResults,
  setShowSharedOnly,
  setHighlightThreshold,
  setShowMovementVectors,
  setShowDensity,
  setShowSimilarityEdges,
  setShowParentThreads,
  setShowResponseHulls,
  setViewDimension,
  isLoading,
  onCompare,
  error,
}: ComparisonControlsProps) {
  return (
    <section className="rounded-2xl border border-border bg-panel p-4">
      <div className="flex flex-col gap-4">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-200">Compare runs</h2>
            <p className="text-xs text-slate-400">
              Select two runs to align their landscapes and explore how points moved.
            </p>
          </div>
          <button
            type="button"
            onClick={onCompare}
            disabled={isLoading || !config.leftRunId || !config.rightRunId}
            className="rounded-md border border-accent px-3 py-1 text-sm font-medium text-accent transition hover:bg-accent/10 disabled:cursor-not-allowed disabled:border-border disabled:text-border"
          >
            {isLoading ? "Comparing�" : "Compare"}
          </button>
        </header>

        {error ? (
          <p className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {error}
          </p>
        ) : null}

        <RunSelectors
          config={config}
          runHistory={runHistory}
          setLeftRunId={setLeftRunId}
          setRightRunId={setRightRunId}
          swapRuns={swapRuns}
        />

        <ViewToggles config={config} setView={setView} setMode={setMode} />

        <FilterControls
          config={config}
          setMinShared={setMinShared}
          setHistogramBins={setHistogramBins}
          setMaxLinks={setMaxLinks}
          setSaveResults={setSaveResults}
          setShowSharedOnly={setShowSharedOnly}
          setHighlightThreshold={setHighlightThreshold}
          setShowMovementVectors={setShowMovementVectors}
          setShowDensity={setShowDensity}
          setShowSimilarityEdges={setShowSimilarityEdges}
          setShowParentThreads={setShowParentThreads}
          setShowResponseHulls={setShowResponseHulls}
          setViewDimension={setViewDimension}
        />
      </div>
    </section>
  );
}

function RunSelectors({
  config,
  runHistory,
  setLeftRunId,
  setRightRunId,
  swapRuns,
}: {
  config: ComparisonConfig;
  runHistory: RunSummary[];
  setLeftRunId: (runId: string | null) => void;
  setRightRunId: (runId: string | null) => void;
  swapRuns: () => void;
}) {
  const options = runHistory.map((run) => ({
    id: run.id,
    label: run.prompt ?? run.id,
  }));

  return (
    <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
      <RunInput
        label="Left run"
        value={config.leftRunId ?? ""}
        onChange={setLeftRunId}
        options={options}
      />
      <RunInput
        label="Right run"
        value={config.rightRunId ?? ""}
        onChange={setRightRunId}
        options={options}
      />
      <button
        type="button"
        onClick={swapRuns}
        className="self-end rounded-md border border-border px-3 py-[6px] text-xs text-slate-300 transition hover:border-accent hover:text-accent"
      >
        Swap
      </button>
    </div>
  );
}

function RunInput({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string | null) => void;
  options: { id: string; label: string }[];
}) {
  const selectValue = options.find((option) => option.id === value) ? value : "";

  return (
    <label className="flex flex-col gap-1 text-xs text-slate-300">
      <span>{label}</span>
      <select
        value={selectValue}
        onChange={(event) => {
          const next = event.currentTarget.value.trim();
          onChange(next ? next : null);
        }}
        className="rounded-md border border-border bg-slate-900/40 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent"
      >
        <option value="">Select a run�</option>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </select>
      <input
        value={value}
        onChange={(event) => {
          const next = event.currentTarget.value.trim();
          onChange(next ? next : null);
        }}
        placeholder="Or paste run ID"
        className="rounded-md border border-border bg-slate-900/40 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent"
      />
    </label>
  );
}

function ViewToggles({
  config,
  setView,
  setMode,
}: {
  config: ComparisonConfig;
  setView: (view: ComparisonViewMode) => void;
  setMode: (mode: ComparisonMode) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-400">View</span>
        <ToggleGroup
          value={config.view}
          options={[
            { value: "side-by-side", label: "Side-by-side" },
            { value: "overlay", label: "Overlay" },
          ]}
          onChange={(value) => setView(value as ComparisonViewMode)}
        />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-400">Mode</span>
        <ToggleGroup
          value={config.mode}
          options={[
            { value: "segments", label: "Segments" },
            { value: "responses", label: "Responses" },
          ]}
          onChange={(value) => setMode(value as ComparisonMode)}
        />
      </div>
    </div>
  );
}

function ToggleGroup({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-border bg-slate-900/50 p-1 text-xs">
      {options.map((option) => {
        const isActive = value === option.value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={[
              "rounded px-2 py-1 transition",
              isActive ? "bg-accent/20 text-accent" : "text-slate-300 hover:bg-slate-800/80",
            ].join(" ")}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function FilterControls({
  config,
  setMinShared,
  setHistogramBins,
  setMaxLinks,
  setSaveResults,
  setShowSharedOnly,
  setHighlightThreshold,
  setShowMovementVectors,
  setShowDensity,
  setShowSimilarityEdges,
  setShowParentThreads,
  setShowResponseHulls,
  setViewDimension,
}: {
  config: ComparisonConfig;
  setMinShared: (value: number) => void;
  setHistogramBins: (value: number) => void;
  setMaxLinks: (value: number) => void;
  setSaveResults: (value: boolean) => void;
  setShowSharedOnly: (value: boolean) => void;
  setHighlightThreshold: (value: number) => void;
  setShowMovementVectors: (value: boolean) => void;
  setShowDensity: (value: boolean) => void;
  setShowSimilarityEdges: (value: boolean) => void;
  setShowParentThreads: (value: boolean) => void;
  setShowResponseHulls: (value: boolean) => void;
  setViewDimension: (value: "2d" | "3d") => void;
}) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <label className="flex flex-col gap-1 text-xs text-slate-300">
        <span>Minimum anchors</span>
        <input
          type="number"
          min={0}
          value={config.minShared}
          onChange={(event) => setMinShared(Number(event.currentTarget.value) || 0)}
          className="rounded-md border border-border bg-slate-900/40 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-slate-300">
        <span>Histogram bins</span>
        <input
          type="number"
          min={4}
          max={60}
          value={config.histogramBins}
          onChange={(event) => setHistogramBins(Number(event.currentTarget.value) || 10)}
          className="rounded-md border border-border bg-slate-900/40 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-slate-300">
        <span>Maximum links</span>
        <input
          type="number"
          min={50}
          value={config.maxLinks}
          onChange={(event) => setMaxLinks(Number(event.currentTarget.value) || 600)}
          className="rounded-md border border-border bg-slate-900/40 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-accent"
        />
      </label>
      <div className="flex flex-col gap-3 rounded-md border border-border/60 bg-slate-900/30 p-3 text-xs text-slate-300">
        <div>
          <span className="block text-[11px] uppercase tracking-wide text-slate-400">View dimension</span>
          <div className="mt-2 flex gap-2">
            {(
              [
                { label: "3D", value: "3d" },
                { label: "2D", value: "2d" },
              ] as const
            ).map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => setViewDimension(option.value)}
                className={`flex-1 rounded-md border px-3 py-1 text-[11px] font-medium transition ${
                  config.viewDimension === option.value
                    ? "border-accent bg-accent/20 text-accent"
                    : "border-border text-slate-400 hover:border-accent/60 hover:text-slate-200"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <label className="flex items-center justify-between gap-3">
          <span>Persist comparison</span>
          <input
            type="checkbox"
            checked={config.saveResults}
            onChange={(event) => setSaveResults(event.currentTarget.checked)}
            className="h-4 w-4"
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Shared points only</span>
          <input
            type="checkbox"
            checked={config.showSharedOnly}
            onChange={(event) => setShowSharedOnly(event.currentTarget.checked)}
            className="h-4 w-4"
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Show movement vectors</span>
          <input
            type="checkbox"
            checked={config.showMovementVectors}
            onChange={(event) => setShowMovementVectors(event.currentTarget.checked)}
            className="h-4 w-4"
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Density overlay</span>
          <input
            type="checkbox"
            checked={config.showDensity}
            onChange={(event) => setShowDensity(event.currentTarget.checked)}
            className="h-4 w-4"
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Similarity edges</span>
          <input
            type="checkbox"
            checked={config.showSimilarityEdges}
            onChange={(event) => setShowSimilarityEdges(event.currentTarget.checked)}
            className="h-4 w-4"
            disabled={config.mode !== "segments"}
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Parent threads</span>
          <input
            type="checkbox"
            checked={config.showParentThreads}
            onChange={(event) => setShowParentThreads(event.currentTarget.checked)}
            className="h-4 w-4"
            disabled={config.mode !== "segments"}
          />
        </label>
        <label className="flex items-center justify-between gap-3">
          <span>Response hulls</span>
          <input
            type="checkbox"
            checked={config.showResponseHulls}
            onChange={(event) => setShowResponseHulls(event.currentTarget.checked)}
            className="h-4 w-4"
            disabled={config.mode !== "segments"}
          />
        </label>
        <div className="flex flex-col gap-1">
          <span>Highlight threshold ({config.highlightThreshold.toFixed(2)})</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={config.highlightThreshold}
            onChange={(event) => setHighlightThreshold(Number(event.currentTarget.value))}
          />
        </div>
      </div>
    </div>
  );
}
function ComparisonMetricsPanel({ result }: { result: CompareRunsResponse | undefined }) {
  return (
    <aside className="flex h-full flex-col gap-3 rounded-2xl border border-border bg-panel p-4 text-xs text-slate-300">
      <h3 className="text-sm font-semibold text-slate-100">Diff metrics</h3>
      {result ? (
        <div className="grid gap-3">
          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <MetricCard label="Shared segments" value={result.metrics.shared_segment_count} />
            <MetricCard label="Linked segments" value={result.metrics.linked_segment_count} />
            <MetricCard label="ARI" value={formatNumber(result.metrics.ari)} />
            <MetricCard label="NMI" value={formatNumber(result.metrics.nmi)} />
            <MetricCard label="Mean movement" value={formatNumber(result.metrics.mean_movement)} />
            <MetricCard label="Median movement" value={formatNumber(result.metrics.median_movement)} />
            <MetricCard label="? clusters" value={result.metrics.delta_cluster_count} />
            <MetricCard
              label="Noise L/R"
              value={`${result.metrics.noise_left}/${result.metrics.noise_right}`}
            />
          </div>
          <div>
            <h4 className="mb-2 text-[11px] uppercase tracking-wide text-slate-400">
              Movement histogram
            </h4>
            <MovementHistogram bins={result.metrics.movement_histogram} />
          </div>
          {result.metrics.top_term_shifts.length ? (
            <div>
              <h4 className="mb-2 text-[11px] uppercase tracking-wide text-slate-400">
                Cluster theme shifts
              </h4>
              <div className="flex max-h-40 flex-col gap-2 overflow-y-auto pr-1">
                {result.metrics.top_term_shifts.map((shift) => (
                  <div
                    key={shift.cluster_label}
                    className="rounded-md border border-border/60 bg-slate-900/30 p-2 text-[11px]"
                  >
                    <div className="mb-1 text-slate-300">
                      Cluster {shift.cluster_label} � Jaccard {" "}
                      {shift.jaccard !== null && shift.jaccard !== undefined
                        ? shift.jaccard.toFixed(2)
                        : "--"}
                    </div>
                    <div className="grid gap-1 text-[10px] text-slate-400">
                      <div>
                        <span className="text-slate-500">Left:</span> {shift.left_terms.join(", ") || "--"}
                      </div>
                      <div>
                        <span className="text-slate-500">Right:</span> {shift.right_terms.join(", ") || "--"}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <p className="text-[11px] text-slate-500">Run a comparison to see metrics.</p>
      )}
    </aside>
  );
}

function MetricCard({
  label,
  value,
}: {
  label: string;
  value: number | string | null | undefined;
}) {
  return (
    <div className="rounded-md border border-border/60 bg-slate-900/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm text-slate-100">{value ?? "--"}</div>
    </div>
  );
}

function MovementHistogram({ bins }: { bins: MovementHistogramBin[] }) {
  if (!bins.length) {
    return <p className="text-[11px] text-slate-500">No movement data</p>;
  }
  const maxCount = Math.max(...bins.map((bin) => bin.count));
  return (
    <div className="flex h-24 items-end gap-[3px]">
      {bins.map((bin, index) => {
        const height = maxCount ? (bin.count / maxCount) * 100 : 0;
        return (
          <div
            key={`${bin.start}-${index}`}
            className="flex h-full w-full flex-col items-center justify-end text-[9px] text-slate-500"
          >
            <div
              className="w-full rounded-sm bg-accent/60"
              style={{ height: `${Math.max(4, height)}%` }}
              title={`${bin.start.toFixed(2)}-${bin.end.toFixed(2)}: ${bin.count}`}
            />
          </div>
        );
      })}
    </div>
  );
}

const RIGHT_CLUSTER_OFFSET = 10_000;

interface PrepareComparisonSceneParams {
  mode: ComparisonMode;
  view: ComparisonViewMode;
  showSharedOnly: boolean;
  left: RunResultsResponse;
  right: RunResultsResponse;
  comparison?: CompareRunsResponse;
}

function prepareComparisonSceneData({
  mode,
  view,
  showSharedOnly,
  left,
  right,
  comparison,
}: PrepareComparisonSceneParams): ComparisonSceneData {
  const comparisonPointMap = new Map<string, ComparisonPoint>();
  comparison?.points?.forEach((point) => {
    comparisonPointMap.set(point.id, point);
  });

  const transform2d = comparison?.transforms?.components?.["2d"];
  const transform3d = comparison?.transforms?.components?.["3d"];

  const sharedLeftIds = new Set<string>();
  const sharedRightIds = new Set<string>();
  if (showSharedOnly && comparison?.links) {
    comparison.links.forEach((link) => {
      sharedLeftIds.add(link.left_segment_id);
      sharedRightIds.add(link.right_segment_id);
    });
  }

  const includeLeft = (id: string | null | undefined): boolean => {
    if (!showSharedOnly) {
      return true;
    }
    return id != null && sharedLeftIds.has(id);
  };

  const includeRight = (id: string | null | undefined): boolean => {
    if (!showSharedOnly) {
      return true;
    }
    return id != null && sharedRightIds.has(id);
  };

  const responses: ResponsePoint[] = [];
  const segments: SegmentPoint[] = [];
  const responseHulls: ResponseHull[] = [];

  const leftClusterLabels = new Set<number>();
  const rightClusterLabels = new Set<number>();

  const leftResponsesSource = left.points ?? [];
  const rightResponsesSource = right.points ?? [];
  const leftSegmentsSource = left.segments ?? [];
  const rightSegmentsSource = right.segments ?? [];

  const statsLeftBase = mode === "segments" ? leftSegmentsSource : leftResponsesSource;
  const statsRightBase = mode === "segments" ? rightSegmentsSource : rightResponsesSource;

  const leftCenter3d = computeCenter3dFromPoints(statsLeftBase.map((point) => point.coords_3d));
  const rightCenter3d = computeCenter3dFromPoints(statsRightBase.map((point) => point.coords_3d));
  const leftCenter2d = computeCenter2dFromPoints(statsLeftBase.map((point) => point.coords_2d));
  const rightCenter2d = computeCenter2dFromPoints(statsRightBase.map((point) => point.coords_2d));

  const leftStats3d = computeAxisStats(statsLeftBase, (point) => point.coords_3d[0]);
  const rightStats3d = computeAxisStats(statsRightBase, (point) => point.coords_3d[0]);
  const leftStats2d = computeAxisStats(statsLeftBase, (point) => point.coords_2d[0]);
  const rightStats2d = computeAxisStats(statsRightBase, (point) => point.coords_2d[0]);

  const shift3d = computeSideBySideShift(leftStats3d.width, rightStats3d.width);
  const shift2d = computeSideBySideShift(leftStats2d.width, rightStats2d.width);

  const applySideBySide = view === "side-by-side";
  const leftShift3d = applySideBySide ? shift3d.left : 0;
  const rightShift3d = applySideBySide ? shift3d.right : 0;
  const leftShift2d = applySideBySide ? shift2d.left : 0;
  const rightShift2d = applySideBySide ? shift2d.right : 0;

  leftResponsesSource.forEach((point) => {
    if (!includeLeft(point.id)) {
      return;
    }
    const mappedCluster = mapClusterLabel(point.cluster, "left");
    if (mappedCluster != null) {
      leftClusterLabels.add(mappedCluster);
    }
    const coords3d = applySideBySide
      ? shiftVector3d(point.coords_3d, leftCenter3d, leftShift3d)
      : ([...point.coords_3d] as [number, number, number]);
    const coords2d = applySideBySide
      ? shiftVector2d(point.coords_2d, leftCenter2d, leftShift2d)
      : ([...point.coords_2d] as [number, number]);
    responses.push({
      ...point,
      cluster: mappedCluster,
      coords_3d: coords3d,
      coords_2d: coords2d,
    });
  });

  rightResponsesSource.forEach((point) => {
    if (!includeRight(point.id)) {
      return;
    }
    const mappedCluster = mapClusterLabel(point.cluster, "right");
    if (mappedCluster != null) {
      rightClusterLabels.add(mappedCluster);
    }
    const comparisonPoint = comparisonPointMap.get(point.id);
    let coords3d: [number, number, number];
    let coords2d: [number, number];
    if (view === "overlay") {
      if (comparisonPoint?.aligned_coords_3d) {
        coords3d = [...comparisonPoint.aligned_coords_3d] as [number, number, number];
      } else if (transform3d) {
        coords3d = applyTransformVector(point.coords_3d, transform3d) as [number, number, number];
      } else {
        coords3d = [...point.coords_3d] as [number, number, number];
      }
      if (comparisonPoint?.aligned_coords_2d) {
        coords2d = [...comparisonPoint.aligned_coords_2d] as [number, number];
      } else if (transform2d) {
        coords2d = applyTransformVector(point.coords_2d, transform2d) as [number, number];
      } else {
        coords2d = [...point.coords_2d] as [number, number];
      }
    } else {
      coords3d = shiftVector3d(point.coords_3d, rightCenter3d, rightShift3d);
      coords2d = shiftVector2d(point.coords_2d, rightCenter2d, rightShift2d);
    }
    responses.push({
      ...point,
      cluster: mappedCluster,
      coords_3d: coords3d,
      coords_2d: coords2d,
    });
  });

  const allowedLeftResponseIds = new Set<string>();
  const allowedRightResponseIds = new Set<string>();

  leftSegmentsSource.forEach((segment) => {
    if (!includeLeft(segment.id)) {
      return;
    }
    const mappedCluster = mapClusterLabel(segment.cluster, "left");
    if (mappedCluster != null) {
      leftClusterLabels.add(mappedCluster);
    }
    const coords3d = applySideBySide
      ? shiftVector3d(segment.coords_3d, leftCenter3d, leftShift3d)
      : ([...segment.coords_3d] as [number, number, number]);
    const coords2d = applySideBySide
      ? shiftVector2d(segment.coords_2d, leftCenter2d, leftShift2d)
      : ([...segment.coords_2d] as [number, number]);
    segments.push({
      ...segment,
      cluster: mappedCluster,
      coords_3d: coords3d,
      coords_2d: coords2d,
    });
    allowedLeftResponseIds.add(segment.response_id);
  });

  rightSegmentsSource.forEach((segment) => {
    if (!includeRight(segment.id)) {
      return;
    }
    const mappedCluster = mapClusterLabel(segment.cluster, "right");
    if (mappedCluster != null) {
      rightClusterLabels.add(mappedCluster);
    }
    const comparisonPoint = comparisonPointMap.get(segment.id);
    let coords3d: [number, number, number];
    let coords2d: [number, number];
    if (view === "overlay") {
      if (comparisonPoint?.aligned_coords_3d) {
        coords3d = [...comparisonPoint.aligned_coords_3d] as [number, number, number];
      } else if (transform3d) {
        coords3d = applyTransformVector(segment.coords_3d, transform3d) as [number, number, number];
      } else {
        coords3d = [...segment.coords_3d] as [number, number, number];
      }
      if (comparisonPoint?.aligned_coords_2d) {
        coords2d = [...comparisonPoint.aligned_coords_2d] as [number, number];
      } else if (transform2d) {
        coords2d = applyTransformVector(segment.coords_2d, transform2d) as [number, number];
      } else {
        coords2d = [...segment.coords_2d] as [number, number];
      }
    } else {
      coords3d = shiftVector3d(segment.coords_3d, rightCenter3d, rightShift3d);
      coords2d = shiftVector2d(segment.coords_2d, rightCenter2d, rightShift2d);
    }
    segments.push({
      ...segment,
      cluster: mappedCluster,
      coords_3d: coords3d,
      coords_2d: coords2d,
    });
    allowedRightResponseIds.add(segment.response_id);
  });

  const allowedSegmentIds = new Set(segments.map((segment) => segment.id));
  const combinedEdges = [
    ...(left.segment_edges ?? []),
    ...(right.segment_edges ?? []),
  ]
    .filter((edge) => allowedSegmentIds.has(edge.source_id) && allowedSegmentIds.has(edge.target_id))
    .map((edge) => ({ ...edge }));

  const restrictHulls = mode === "segments" && showSharedOnly;

  const leftHullResponses = restrictHulls ? allowedLeftResponseIds : undefined;
  left.response_hulls?.forEach((hull) => {
    if (leftHullResponses && !leftHullResponses.has(hull.response_id)) {
      return;
    }
    const coords3d = applySideBySide
      ? hull.coords_3d.map((coord) => shiftVector3d(coord, leftCenter3d, leftShift3d))
      : hull.coords_3d.map((coord) => [...coord] as [number, number, number]);
    const coords2d = applySideBySide
      ? hull.coords_2d.map((coord) => shiftVector2d(coord, leftCenter2d, leftShift2d))
      : hull.coords_2d.map((coord) => [...coord] as [number, number]);
    responseHulls.push({
      response_id: hull.response_id,
      coords_3d: coords3d,
      coords_2d: coords2d,
    });
  });

  const rightHullResponses = restrictHulls ? allowedRightResponseIds : undefined;
  right.response_hulls?.forEach((hull) => {
    if (rightHullResponses && !rightHullResponses.has(hull.response_id)) {
      return;
    }
    let coords3dList: Array<[number, number, number]>;
    let coords2dList: Array<[number, number]>;
    if (view === "overlay") {
      coords3dList = hull.coords_3d.map((coord) =>
        transform3d ? (applyTransformVector(coord, transform3d) as [number, number, number]) : ([...coord] as [number, number, number]),
      );
      coords2dList = hull.coords_2d.map((coord) =>
        transform2d ? (applyTransformVector(coord, transform2d) as [number, number]) : ([...coord] as [number, number]),
      );
    } else {
      coords3dList = hull.coords_3d.map((coord) => shiftVector3d(coord, rightCenter3d, rightShift3d));
      coords2dList = hull.coords_2d.map((coord) => shiftVector2d(coord, rightCenter2d, rightShift2d));
    }
    responseHulls.push({
      response_id: hull.response_id,
      coords_3d: coords3dList,
      coords_2d: coords2dList,
    });
  });

  const anchorPoints = mode === "segments" ? segments : responses;
  if (anchorPoints.length) {
    const anchorCenter3d = computeCenter3dFromPoints(anchorPoints.map((entry) => entry.coords_3d));
    const anchorCenter2d = computeCenter2dFromPoints(anchorPoints.map((entry) => entry.coords_2d));
    if (hasNonZeroVector(anchorCenter3d) || hasNonZeroVector2d(anchorCenter2d)) {
      responses.forEach((point) => {
        point.coords_3d = subtractVector3d(point.coords_3d, anchorCenter3d);
        point.coords_2d = subtractVector2d(point.coords_2d, anchorCenter2d);
      });
      segments.forEach((segment) => {
        segment.coords_3d = subtractVector3d(segment.coords_3d, anchorCenter3d);
        segment.coords_2d = subtractVector2d(segment.coords_2d, anchorCenter2d);
      });
      responseHulls.forEach((hull) => {
        hull.coords_3d = hull.coords_3d.map((coord) => subtractVector3d(coord, anchorCenter3d));
        hull.coords_2d = hull.coords_2d.map((coord) => subtractVector2d(coord, anchorCenter2d));
      });
    }
  }

  const clusterPalette = buildComparisonPalette(leftClusterLabels, rightClusterLabels);

  return {
    responses,
    segments,
    segmentEdges: combinedEdges,
    responseHulls,
    clusterPalette,
  };
}

function computeCenter3dFromPoints(points: Array<[number, number, number]>): [number, number, number] {
  if (!points.length) {
    return [0, 0, 0];
  }
  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  points.forEach(([x, y, z]) => {
    sumX += isFiniteNumber(x) ? x : 0;
    sumY += isFiniteNumber(y) ? y : 0;
    sumZ += isFiniteNumber(z) ? z : 0;
  });
  const divisor = points.length || 1;
  return [sumX / divisor, sumY / divisor, sumZ / divisor];
}

function computeCenter2dFromPoints(points: Array<[number, number]>): [number, number] {
  if (!points.length) {
    return [0, 0];
  }
  let sumX = 0;
  let sumY = 0;
  points.forEach(([x, y]) => {
    sumX += isFiniteNumber(x) ? x : 0;
    sumY += isFiniteNumber(y) ? y : 0;
  });
  const divisor = points.length || 1;
  return [sumX / divisor, sumY / divisor];
}

function computeAxisStats<T>(items: readonly T[], getter: (item: T) => number): {
  min: number;
  max: number;
  width: number;
  center: number;
} {
  if (!items.length) {
    return { min: 0, max: 0, width: 1, center: 0 };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  items.forEach((item) => {
    const value = getter(item);
    if (!isFiniteNumber(value)) {
      return;
    }
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  });
  if (!isFiniteNumber(min) || !isFiniteNumber(max)) {
    return { min: 0, max: 0, width: 1, center: 0 };
  }
  const width = Math.max(1e-6, max - min);
  return { min, max, width, center: (min + max) / 2 };
}

function computeSideBySideShift(leftWidth: number, rightWidth: number): {
  left: number;
  right: number;
} {
  const safeLeft = Math.max(1e-6, leftWidth);
  const safeRight = Math.max(1e-6, rightWidth);
  const margin = Math.max(safeLeft, safeRight) * 0.6 + 1;
  const left = -(safeRight / 2) - margin / 2;
  const right = safeLeft / 2 + margin / 2;
  return { left, right };
}

function mapClusterLabel(label: number | null | undefined, source: "left" | "right") {
  if (label == null) {
    return null;
  }
  if (source === "left") {
    return label;
  }
  if (label === -1) {
    return -1 - RIGHT_CLUSTER_OFFSET;
  }
  return label + RIGHT_CLUSTER_OFFSET;
}

function shiftVector3d(
  coords: [number, number, number],
  center: [number, number, number],
  shiftX: number,
): [number, number, number] {
  return [coords[0] - center[0] + shiftX, coords[1] - center[1], coords[2] - center[2]];
}

function shiftVector2d(
  coords: [number, number],
  center: [number, number],
  shiftX: number,
): [number, number] {
  return [coords[0] - center[0] + shiftX, coords[1] - center[1]];
}

function subtractVector3d(
  coords: [number, number, number],
  delta: [number, number, number],
): [number, number, number] {
  return [coords[0] - delta[0], coords[1] - delta[1], coords[2] - delta[2]];
}

function subtractVector2d(coords: [number, number], delta: [number, number]): [number, number] {
  return [coords[0] - delta[0], coords[1] - delta[1]];
}

function hasNonZeroVector(vector: [number, number, number]) {
  return vector.some((value) => Math.abs(value) > 1e-6);
}

function hasNonZeroVector2d(vector: [number, number]) {
  return vector.some((value) => Math.abs(value) > 1e-6);
}

function applyTransformVector(
  coords: number[] | [number, number] | [number, number, number],
  component: ComparisonTransformComponent,
): number[] {
  const dimension = component.rotation?.length ?? coords.length;
  const rotation = component.rotation ?? [];
  const scale = isFiniteNumber(component.scale) ? component.scale : 1;
  const translation = component.translation ?? [];
  const result = new Array(dimension).fill(0);
  for (let row = 0; row < dimension; row += 1) {
    let acc = 0;
    for (let col = 0; col < dimension; col += 1) {
      const matrixValue = rotation[row]?.[col];
      const coord = coords[col] ?? 0;
      acc += (isFiniteNumber(matrixValue) ? matrixValue : 0) * coord;
    }
    result[row] = acc * scale + (translation[row] ?? 0);
  }
  return result;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function buildComparisonPalette(
  leftLabels: Set<number>,
  rightLabels: Set<number>,
): Record<string, string> {
  const palette: Record<string, string> = {};
  const sortedLeft = Array.from(leftLabels).sort((a, b) => a - b);
  const sortedRight = Array.from(rightLabels).sort((a, b) => a - b);

  sortedLeft.forEach((label, index) => {
    if (label === -1) {
      palette[String(label)] = "#64748b";
      return;
    }
    palette[String(label)] = colorFromSequence(210, index);
  });

  sortedRight.forEach((label, index) => {
    if (label === -1 - RIGHT_CLUSTER_OFFSET) {
      palette[String(label)] = "#94a3b8";
      return;
    }
    palette[String(label)] = colorFromSequence(24, index);
  });

  return palette;
}

function colorFromSequence(baseHue: number, index: number): string {
  const hue = (baseHue + index * 18) % 360;
  return `hsl(${hue}deg 78% 62%)`;
}

function extractErrorMessage(error: unknown): string {
  if (!error) {
    return "";
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "Failed to load run geometry";
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-slate-400">
      <p>Select two runs and press Compare to generate an alignment.</p>
      <p className="text-xs text-slate-500">
        Side-by-side and overlay views will appear here when the comparison is ready.
      </p>
    </div>
  );
}

function ComparisonTooltip({ hover, tooltip }: { hover: HoverState; tooltip: string[] }) {
  return (
    <div
      className="pointer-events-none fixed z-50 min-w-[220px] max-w-sm rounded-md border border-border bg-slate-900/95 px-3 py-2 text-xs text-slate-200 shadow-lg"
      style={{ left: hover.clientX + 12, top: hover.clientY + 12 }}
    >
      {tooltip.map((line) => (
        <p key={line}>{line}</p>
      ))}
    </div>
  );
}

function buildTooltipContent(point: ComparisonPoint, links: ComparisonLink[]): string[] {
  const content: string[] = [];
  content.push(point.source === "left" ? "Left point" : "Right point");
  if (point.cluster_label !== undefined && point.cluster_label !== null) {
    content.push(`Cluster ${point.cluster_label}`);
  }
  if (point.text_preview) {
    content.push(point.text_preview);
  }
  const link = links.find(
    (item) => item.left_segment_id === point.id || item.right_segment_id === point.id,
  );
  if (link) {
    content.push(`Link: ${link.link_type === "exact_hash" ? "shared text" : "nearest neighbour"}`);
    if (link.movement_distance !== undefined && link.movement_distance !== null) {
      content.push(`Movement: ${link.movement_distance.toFixed(3)}`);
    }
  }
  return content;
}

function buildLinkMaps(links: ComparisonLink[]) {
  const byLeft = new Map<string, ComparisonLink>();
  const byRight = new Map<string, ComparisonLink>();
  for (const link of links) {
    byLeft.set(link.left_segment_id, link);
    byRight.set(link.right_segment_id, link);
  }
  return { byLeft, byRight };
}



function computeViewBox(points: Array<[number, number]>) {
  if (!points.length) {
    return { viewBox: "0 0 1 1", span: 1 };
  }
  let minX = points[0][0];
  let maxX = points[0][0];
  let minY = points[0][1];
  let maxY = points[0][1];
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const pad = Math.max(spanX, spanY) * 0.1;
  const viewBox = `${minX - pad} ${minY - pad} ${spanX + pad * 2} ${spanY + pad * 2}`;
  return { viewBox, span: Math.max(spanX, spanY) };
}

function formatNumber(value: number | string | null | undefined) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "string") {
    return value;
  }
  return value.toFixed(3);
}

function summariseRun(run: CompareRunsResponse["left_run"]) {
  const prompt = run.prompt?.trim();
  if (!prompt) {
    return run.id;
  }
  return prompt.length > 48 ? `${prompt.slice(0, 45)}�` : prompt;
}

type ComparisonMode = ComparisonConfig["mode"];
