/**
 * Zustand store describing application state for prompts, results, and visualisation toggles.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";

import type {
  ClusterSummary,
  RunResultsResponse,
  RunSummary,
  SegmentEdge,
  SegmentPoint,
} from "@/types/run";

export type SceneDimension = "3d" | "2d";
export type LevelMode = "responses" | "segments";

type ClusterVisibilityMap = Record<string, boolean>;
type RoleVisibilityMap = Record<string, boolean>;

type SelectionUpdater = (current: string[]) => string[];

export interface RunStoreState {
  prompt: string;
  n: number;
  temperature: number;
  topP: number;
  model: string;
  seed?: number | null;
  maxTokens?: number | null;
  viewMode: SceneDimension;
  levelMode: LevelMode;
  pointSize: number;
  spreadFactor: number;
  showDensity: boolean;
  showEdges: boolean;
  showParentThreads: boolean;
  jitterToken?: string | null;
  isHistoryOpen: boolean;
  isGenerating: boolean;
  currentRunId?: string;
  results?: RunResultsResponse;
  hoveredPointId?: string;
  hoveredSegmentId?: string;
  focusedResponseId?: string;
  hoveredClusterLabel: number | null;
  runHistory: RunSummary[];
  selectedPointIds: string[];
  selectedSegmentIds: string[];
  clusterVisibility: ClusterVisibilityMap;
  clusterPalette: Record<string, string>;
  roleVisibility: RoleVisibilityMap;
  setPrompt: (value: string) => void;
  setN: (value: number) => void;
  setTemperature: (value: number) => void;
  setTopP: (value: number) => void;
  setModel: (value: string) => void;
  setSeed: (value: number | null) => void;
  setMaxTokens: (value: number | null) => void;
  setViewMode: (mode: SceneDimension) => void;
  setLevelMode: (mode: LevelMode) => void;
  setPointSize: (value: number) => void;
  setSpreadFactor: (value: number) => void;
  setShowDensity: (value: boolean) => void;
  setShowEdges: (value: boolean) => void;
  setShowParentThreads: (value: boolean) => void;
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
  setClusterPalette: (palette: Record<string, string>) => void;
  selectTopOutliers: (count?: number) => void;
  selectTopSegmentOutliers: (count?: number) => void;
  startGeneration: () => void;
  finishGeneration: (runId: string, results?: RunResultsResponse) => void;
  setResults: (results: RunResultsResponse) => void;
  reset: () => void;
}

const defaultState: Pick<
  RunStoreState,
  | "prompt"
  | "n"
  | "temperature"
  | "topP"
  | "model"
  | "seed"
  | "maxTokens"
  | "viewMode"
  | "levelMode"
  | "pointSize"
  | "spreadFactor"
  | "showDensity"
  | "showEdges"
  | "showParentThreads"
  | "selectedPointIds"
  | "selectedSegmentIds"
  | "clusterVisibility"
  | "clusterPalette"
  | "roleVisibility"
  | "hoveredClusterLabel"
> = {
  prompt: "How will climate change transform urban living in the next decade?",
  n: 40,
  temperature: 0.9,
  topP: 1,
  model: "gpt-4.1-mini",
  seed: null,
  maxTokens: 800,
  viewMode: "3d",
  levelMode: "responses",
  pointSize: 0.06,
  spreadFactor: 1.4,
  showDensity: false,
  showEdges: true,
  showParentThreads: true,
  selectedPointIds: [],
  selectedSegmentIds: [],
  clusterVisibility: {},
  clusterPalette: {},
  roleVisibility: {},
  hoveredClusterLabel: null,
};

export const useRunStore = create<RunStoreState>()(
  persist(
    (set, get) => ({
      ...defaultState,
      jitterToken: null,
      isHistoryOpen: false,
      isGenerating: false,
      currentRunId: undefined,
      results: undefined,
      hoveredPointId: undefined,
      hoveredSegmentId: undefined,
      focusedResponseId: undefined,
      runHistory: [],
      setPrompt: (value) => set({ prompt: value }),
      setN: (value) => set({ n: value }),
      setTemperature: (value) => set({ temperature: value }),
      setTopP: (value) => set({ topP: value }),
      setModel: (value) => set({ model: value }),
      setSeed: (value) => set({ seed: value ?? null }),
      setMaxTokens: (value) => set({ maxTokens: value ?? null }),
      setViewMode: (mode) => set({ viewMode: mode }),
      setLevelMode: (mode) =>
        set({
          levelMode: mode,
          selectedPointIds: mode === "responses" ? get().selectedPointIds : [],
          selectedSegmentIds: mode === "segments" ? get().selectedSegmentIds : [],
        }),
      setPointSize: (value) => set({ pointSize: value }),
      setSpreadFactor: (value) => set({ spreadFactor: value }),
      setShowDensity: (value) => set({ showDensity: value }),
      setShowEdges: (value) => set({ showEdges: value }),
      setShowParentThreads: (value) => set({ showParentThreads: value }),
      setHistoryOpen: (value) => set({ isHistoryOpen: value }),
      setCurrentRunId: (runId) => set({ currentRunId: runId }),
      applyRunSummary: (run) =>
        set((state) => ({
          prompt: run.prompt,
          n: run.n,
          model: run.model,
          temperature: run.temperature,
          topP: run.top_p ?? state.topP,
          seed: run.seed ?? null,
          maxTokens: run.max_tokens ?? null,
        })),
      setFocusedResponse: (value) => set({ focusedResponseId: value }),
      setRunHistory: (runs) => set({ runHistory: runs }),
      setRunNotes: (runId, notes) =>
        set((state) => {
          const nextHistory = state.runHistory.map((summary) =>
            summary.id === runId ? { ...summary, notes: notes ?? null } : summary,
          );
          const nextResults =
            state.results && state.results.run.id === runId
              ? { ...state.results, run: { ...state.results.run, notes: notes ?? null } }
              : state.results;
          return { runHistory: nextHistory, results: nextResults };
        }),
      setJitterToken: (token) => set({ jitterToken: token }),
      setHoveredPoint: (id) => set({ hoveredPointId: id ?? undefined }),
      setHoveredSegment: (id) => set({ hoveredSegmentId: id ?? undefined }),
      setHoveredCluster: (label) => set({ hoveredClusterLabel: label }),
      setSelectedPoints: (payload) =>
        set((state) => ({
          selectedPointIds: typeof payload === "function" ? payload(state.selectedPointIds) : payload,
        })),
      setSelectedSegments: (payload) =>
        set((state) => ({
          selectedSegmentIds:
            typeof payload === "function" ? payload(state.selectedSegmentIds) : payload,
        })),
      toggleCluster: (label) => {
        const key = String(label);
        const visibility = { ...get().clusterVisibility };
        visibility[key] = visibility[key] ?? true;
        visibility[key] = !visibility[key];
        set({ clusterVisibility: visibility });
      },
      toggleRole: (role) => {
        const visibility = { ...get().roleVisibility };
        visibility[role] = !(visibility[role] ?? true);
        set({ roleVisibility: visibility });
      },
      setRolesVisibility: (roles, visible) => {
        set((state) => {
          const next = { ...state.roleVisibility };
          roles.forEach((role) => {
            next[role] = visible;
          });
          return { roleVisibility: next };
        });
      },
      setClusterPalette: (palette) => set({ clusterPalette: palette }),
      selectTopOutliers: (count = 8) => {
        const { results, levelMode } = get();
        if (!results) {
          return;
        }
        if (levelMode === "segments") {
          get().selectTopSegmentOutliers(count);
          return;
        }
        const ranked = [...results.points]
          .map((point) => ({
            id: point.id,
            score: point.outlier_score ?? (point.cluster === -1 ? 1 : 0),
          }))
          .filter((item) => item.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, count)
          .map((item) => item.id);
        if (ranked.length) {
          set({ selectedPointIds: ranked, levelMode: "responses" });
        }
      },
      selectTopSegmentOutliers: (count = 12) => {
        const { results } = get();
        if (!results) {
          return;
        }
        const ranked = [...results.segments]
          .map((segment) => {
            const noiseScore = segment.cluster === -1 ? 1 : undefined;
            const silhouette =
              segment.silhouette_score != null ? Math.abs(segment.silhouette_score) : undefined;
            const score = segment.outlier_score ?? noiseScore ?? silhouette ?? 0;
            return { id: segment.id, score };
          })
          .filter((item) => item.score > 0)
          .sort((a, b) => b.score - a.score)
          .slice(0, count)
          .map((item) => item.id);
        if (ranked.length) {
          set({ selectedSegmentIds: ranked, levelMode: "segments" });
        }
      },
      startGeneration: () =>
        set({
          isGenerating: true,
          selectedPointIds: [],
          selectedSegmentIds: [],
          hoveredPointId: undefined,
          hoveredSegmentId: undefined,
          focusedResponseId: undefined,
          hoveredClusterLabel: null,
        }),
      finishGeneration: (runId, results) =>
        set((state) => ({
          isGenerating: false,
          currentRunId: runId ? runId : state.currentRunId,
          results: results ?? state.results,
          focusedResponseId: undefined,
          hoveredClusterLabel: null,
        })),
      setResults: (results) => {
        const palette = buildClusterPalette(results.clusters);
        const visibility: ClusterVisibilityMap = {};
        results.clusters.forEach((cluster) => {
          visibility[String(cluster.label)] = true;
        });
        const roleVisibility: RoleVisibilityMap = {};
        results.segments
          .map((segment) => segment.role?.toLowerCase())
          .filter((role): role is string => Boolean(role))
          .forEach((role) => {
            roleVisibility[role] = roleVisibility[role] ?? true;
          });
        set({
          results,
          clusterPalette: palette,
          clusterVisibility: visibility,
          roleVisibility,
          selectedPointIds: [],
          selectedSegmentIds: [],
          focusedResponseId: undefined,
          hoveredClusterLabel: null,
        });
      },
      reset: () =>
        set({
          ...defaultState,
          jitterToken: null,
          isHistoryOpen: false,
          isGenerating: false,
          currentRunId: undefined,
          results: undefined,
          hoveredPointId: undefined,
          hoveredSegmentId: undefined,
          focusedResponseId: undefined,
          runHistory: [],
        }),
    }),
    {
      name: "semantic-landscape-run-store",
      partialize: ({
        prompt,
        n,
        temperature,
        topP,
        model,
        seed,
        maxTokens,
        viewMode,
        levelMode,
        pointSize,
        spreadFactor,
        showDensity,
        showEdges,
        showParentThreads,
      }) => ({
        prompt,
        n,
        temperature,
        topP,
        model,
        seed,
        maxTokens,
        viewMode,
        levelMode,
        pointSize,
        spreadFactor,
        showDensity,
        showEdges,
        showParentThreads,
      }),
    },
  ),
);

function buildClusterPalette(clusters: ClusterSummary[]): Record<string, string> {
  if (!clusters.length) {
    return {};
  }
  const hues = [210, 280, 340, 20, 100, 160];
  const palette: Record<string, string> = {};
  clusters.forEach((cluster, index) => {
    const hue = hues[index % hues.length] + index * 11;
    palette[String(cluster.label)] = `hsl(${hue % 360}deg 80% 62%)`;
  });
  palette["-1"] = "#94a3b8";
  return palette;
}

export function filterSegmentsByRole(
  segments: SegmentPoint[],
  roleVisibility: RoleVisibilityMap,
): SegmentPoint[] {
  const activeRoles = Object.entries(roleVisibility)
    .filter(([, visible]) => visible)
    .map(([role]) => role);
  if (!activeRoles.length) {
    return segments;
  }
  const allowAll = Object.values(roleVisibility).every((visible) => visible);
  if (allowAll) {
    return segments;
  }
  return segments.filter((segment) => {
    if (!segment.role) {
      return true;
    }
    const role = segment.role.toLowerCase();
    return roleVisibility[role] ?? true;
  });
}

export function filterEdgesByVisibility(
  edges: SegmentEdge[],
  visibleSegments: Set<string>,
): SegmentEdge[] {
  return edges.filter(
    (edge) => visibleSegments.has(edge.source_id) && visibleSegments.has(edge.target_id),
  );
}
