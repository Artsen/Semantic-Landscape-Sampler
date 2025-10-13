/**































































 * Zustand store describing application state for prompts, results, and visualisation toggles.































































 */
import type { ClusterMetricsSummary, ExportFormat, ProjectionMethod, QualityGauge, RunMetrics, RunResultsResponse, RunSummary, SegmentEdge, SegmentPoint, UmapMetric, Viewport2D, Viewport3D } from "@/types/run";
declare const UMAP_PRESETS: {
    readonly tight: {
        readonly n_neighbors: 15;
        readonly min_dist: 0.05;
        readonly metric: UmapMetric;
    };
    readonly balanced: {
        readonly n_neighbors: 30;
        readonly min_dist: 0.3;
        readonly metric: UmapMetric;
    };
    readonly global: {
        readonly n_neighbors: 100;
        readonly min_dist: 0.6;
        readonly metric: UmapMetric;
    };
};
export type UmapPreset = keyof typeof UMAP_PRESETS | "custom";
export type SceneDimension = "3d" | "2d";
export type LevelMode = "responses" | "segments";
type ClusterVisibilityMap = Record<string, boolean>;
type RoleVisibilityMap = Record<string, boolean>;
type ProjectionVariantMetadata = {
    fromCache: boolean;
    cachedAt?: string | null;
    warnings: string[];
    requestedParams: Record<string, unknown>;
    resolvedParams: Record<string, unknown>;
    isSubsample: boolean;
    pointCount: number;
    totalCount: number;
    subsampleStrategy?: string | null;
};
type ProjectionVariantState = {
    coords: Record<string, {
        coords2d: [number, number];
        coords3d: [number, number, number];
    }>;
    subsetIds: string[] | null;
    metadata: ProjectionVariantMetadata;
    quality: QualityGauge;
};
type SelectionUpdater = (current: string[]) => string[];
type ViewportBounds2D = Viewport2D & {
    dimension: "2d";
    minZ?: number;
    maxZ?: number;
};
type ViewportBounds3D = Viewport3D & {
    dimension: "3d";
};
type ViewportBounds = ViewportBounds2D | ViewportBounds3D;
export interface RunStoreState {
    prompt: string;
    systemPrompt: string;
    n: number;
    temperature: number;
    topP: number;
    model: string;
    seed?: number | null;
    maxTokens?: number | null;
    embeddingModel: string;
    useCache: boolean;
    clusterAlgo: "hdbscan" | "kmeans";
    hdbscanMinClusterSize: number;
    hdbscanMinSamples: number;
    clusterMetrics: ClusterMetricsSummary | null;
    isRecomputingClusters: boolean;
    preprocVersion: string;
    chunkSize: number;
    chunkOverlap: number;
    umapNNeighbors: number;
    umapMinDist: number;
    umapMetric: UmapMetric;
    umapSeed?: number | null;
    umapPreset: UmapPreset;
    projectionMethod: ProjectionMethod;
    projectionVariants: Partial<Record<ProjectionMethod, ProjectionVariantState>>;
    projectionWarnings: string[];
    isProjectionLoading: boolean;
    projectionError: string | null;
    viewMode: SceneDimension;
    levelMode: LevelMode;
    pointSize: number;
    spreadFactor: number;
    showDensity: boolean;
    showEdges: boolean;
    showParentThreads: boolean;
    exportIncludeProvenance: boolean;
    exportFormat: ExportFormat;
    jitterToken?: string | null;
    isHistoryOpen: boolean;
    isGenerating: boolean;
    currentRunId?: string;
    results?: RunResultsResponse;
    hoveredPointId?: string;
    hoveredSegmentId?: string;
    focusedResponseId?: string;
    progressStage: string | null;
    progressMessage: string | null;
    progressPercent: number | null;
    progressMetadata: Record<string, unknown> | null;
    hoveredClusterLabel: number | null;
    runHistory: RunSummary[];
    selectedPointIds: string[];
    selectedSegmentIds: string[];
    clusterVisibility: ClusterVisibilityMap;
    clusterPalette: Record<string, string>;
    segmentEdges: SegmentEdge[];
    segmentGraphMode: "full" | "simplified";
    segmentGraphAutoSimplified: boolean;
    segmentGraphLoading: boolean;
    segmentGraphError: string | null;
    graphEdgeK: number;
    graphEdgeThreshold: number;
    simplifyEdges: boolean;
    showNeighborSpokes: boolean;
    viewportBounds: ViewportBounds | null;
    runMetrics?: RunMetrics | null;
    showDuplicatesOnly: boolean;
    roleVisibility: RoleVisibilityMap;
    setPrompt: (value: string) => void;
    setSystemPrompt: (value: string) => void;
    setN: (value: number) => void;
    setTemperature: (value: number) => void;
    setTopP: (value: number) => void;
    setModel: (value: string) => void;
    setSeed: (value: number | null) => void;
    setMaxTokens: (value: number | null) => void;
    setUseCache: (value: boolean) => void;
    setPreprocVersion: (value: string) => void;
    setEmbeddingModel: (value: string) => void;
    setChunkSize: (value: number) => void;
    setChunkOverlap: (value: number) => void;
    setUmapNNeighbors: (value: number) => void;
    setUmapMinDist: (value: number) => void;
    setUmapMetric: (value: UmapMetric) => void;
    setUmapSeed: (value: number | null) => void;
    setUmapPreset: (preset: UmapPreset) => void;
    setProjectionMethod: (method: ProjectionMethod) => Promise<void>;
    setViewMode: (mode: SceneDimension) => void;
    setLevelMode: (mode: LevelMode) => void;
    setPointSize: (value: number) => void;
    setSpreadFactor: (value: number) => void;
    setShowDensity: (value: boolean) => void;
    setShowEdges: (value: boolean) => void;
    setShowParentThreads: (value: boolean) => void;
    setExportIncludeProvenance: (value: boolean) => void;
    setExportFormat: (format: ExportFormat) => void;
    setHistoryOpen: (value: boolean) => void;
    setCurrentRunId: (runId: string | undefined) => void;
    applyRunSummary: (run: RunSummary) => void;
    setFocusedResponse: (value: string | undefined) => void;
    setRunHistory: (runs: RunSummary[]) => void;
    setRunNotes: (runId: string, notes: string | null) => void;
    setJitterToken: (token: string | null) => void;
    setHoveredPoint: (id: string | undefined) => void;
    setHoveredSegment: (id: string | undefined) => void;
    setHoveredCluster: (label: number | null) => void;
    setSelectedPoints: (payload: string[] | SelectionUpdater) => void;
    setSelectedSegments: (payload: string[] | SelectionUpdater) => void;
    toggleCluster: (label: number) => void;
    toggleRole: (role: string) => void;
    setRolesVisibility: (roles: string[], visible: boolean) => void;
    setRunMetrics: (metrics: RunMetrics | null) => void;
    setClusterMetrics: (metrics: ClusterMetricsSummary | null) => void;
    setClusterParams: (params: {
        algo?: "hdbscan" | "kmeans";
        minClusterSize?: number;
        minSamples?: number;
    }) => void;
    recomputeClusters: (params?: {
        minClusterSize?: number;
        minSamples?: number;
        algo?: "hdbscan" | "kmeans";
    }) => Promise<void>;
    setShowDuplicatesOnly: (value: boolean) => void;
    setClusterPalette: (palette: Record<string, string>) => void;
    setSegmentEdges: (edges: SegmentEdge[], meta?: {
        mode?: "full" | "simplified";
        autoSimplified?: boolean;
        k?: number;
        threshold?: number;
        error?: string | null;
        loading?: boolean;
    }) => void;
    refreshSegmentGraph: () => Promise<void>;
    setSimplifyEdges: (value: boolean) => Promise<void>;
    setGraphEdgeK: (value: number) => Promise<void>;
    setGraphEdgeThreshold: (value: number) => Promise<void>;
    setShowNeighborSpokes: (value: boolean) => void;
    setViewportBounds: (bounds: ViewportBounds | null) => void;
    selectTopOutliers: (count?: number) => void;
    selectTopSegmentOutliers: (count?: number) => void;
    setProgress: (payload: {
        stage?: string | null;
        message?: string | null;
        percent?: number | null;
        metadata?: Record<string, unknown> | null;
    }) => void;
    startGeneration: () => void;
    finishGeneration: (runId: string, results?: RunResultsResponse) => void;
    setResults: (results: RunResultsResponse) => void;
    reset: () => void;
}
export declare const useRunStore: import("zustand").UseBoundStore<Omit<import("zustand").StoreApi<RunStoreState>, "persist"> & {
    persist: {
        setOptions: (options: Partial<import("zustand/middleware").PersistOptions<RunStoreState, RunStoreState>>) => void;
        clearStorage: () => void;
        rehydrate: () => Promise<void> | void;
        hasHydrated: () => boolean;
        onHydrate: (fn: (state: RunStoreState) => void) => () => void;
        onFinishHydration: (fn: (state: RunStoreState) => void) => () => void;
        getOptions: () => Partial<import("zustand/middleware").PersistOptions<RunStoreState, RunStoreState>>;
    };
}>;
export declare function filterSegmentsByRole(segments: SegmentPoint[], roleVisibility: RoleVisibilityMap): SegmentPoint[];
export declare function filterEdgesByVisibility(edges: SegmentEdge[], visibleSegments: Set<string>): SegmentEdge[];
export {};
