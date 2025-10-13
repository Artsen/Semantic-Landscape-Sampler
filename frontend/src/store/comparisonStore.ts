import { create } from "zustand";
import { persist } from "zustand/middleware";

import { compareRuns } from "@/services/api";
import type {
  CompareRunsRequest,
  CompareRunsResponse,
  ComparisonMode,
} from "@/types/run";

export type ComparisonViewMode = "side-by-side" | "overlay";

export interface ComparisonConfig {
  leftRunId: string | null;
  rightRunId: string | null;
  mode: ComparisonMode;
  view: ComparisonViewMode;
  minShared: number;
  histogramBins: number;
  maxLinks: number;
  saveResults: boolean;
  showSharedOnly: boolean;
  highlightThreshold: number;
  showMovementVectors: boolean;
  showDensity: boolean;
  showSimilarityEdges: boolean;
  showParentThreads: boolean;
  showResponseHulls: boolean;
  viewDimension: "2d" | "3d";
}

interface ComparisonState {
  config: ComparisonConfig;
  result?: CompareRunsResponse;
  isLoading: boolean;
  error?: string | null;
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
  resetResult: () => void;
  runComparison: () => Promise<CompareRunsResponse | undefined>;
}

const defaultConfig: ComparisonConfig = {
  leftRunId: null,
  rightRunId: null,
  mode: "segments",
  view: "side-by-side",
  minShared: 3,
  histogramBins: 10,
  maxLinks: 600,
  saveResults: true,
  showSharedOnly: false,
  highlightThreshold: 0.35,
  showMovementVectors: true,
  showDensity: true,
  showSimilarityEdges: true,
  showParentThreads: true,
  showResponseHulls: false,
  viewDimension: "3d",
};

export const useComparisonStore = create<ComparisonState>()(
  persist(
    (set, get) => ({
      config: defaultConfig,
      result: undefined,
      isLoading: false,
      error: null,
      setLeftRunId: (runId) =>
        set((state) => ({
          config: { ...state.config, leftRunId: runId },
          result: undefined,
        })),
      setRightRunId: (runId) =>
        set((state) => ({
          config: { ...state.config, rightRunId: runId },
          result: undefined,
        })),
      swapRuns: () =>
        set((state) => ({
          config: {
            ...state.config,
            leftRunId: state.config.rightRunId,
            rightRunId: state.config.leftRunId,
          },
          result: undefined,
        })),
      setMode: (mode) =>
        set((state) => ({ config: { ...state.config, mode } })),
      setView: (view) =>
        set((state) => ({ config: { ...state.config, view } })),
      setMinShared: (value) =>
        set((state) => ({ config: { ...state.config, minShared: Math.max(0, Math.round(value)) } })),
      setHistogramBins: (value) =>
        set((state) => ({
          config: { ...state.config, histogramBins: Math.max(4, Math.round(value)) },
        })),
      setMaxLinks: (value) =>
        set((state) => ({
          config: { ...state.config, maxLinks: Math.max(50, Math.round(value)) },
        })),
      setSaveResults: (value) =>
        set((state) => ({ config: { ...state.config, saveResults: value } })),
      setShowSharedOnly: (value) =>
        set((state) => ({ config: { ...state.config, showSharedOnly: value } })),
      setHighlightThreshold: (value) =>
        set((state) => ({ config: { ...state.config, highlightThreshold: Math.max(0, value) } })),
      setShowMovementVectors: (value) => {
        set((state) => ({ config: { ...state.config, showMovementVectors: value } }));
      },
      setShowDensity: (value) => {
        set((state) => ({ config: { ...state.config, showDensity: value } }));
      },
      setShowSimilarityEdges: (value) => {
        set((state) => ({ config: { ...state.config, showSimilarityEdges: value } }));
      },
      setShowParentThreads: (value) => {
        set((state) => ({ config: { ...state.config, showParentThreads: value } }));
      },
      setShowResponseHulls: (value) => {
        set((state) => ({ config: { ...state.config, showResponseHulls: value } }));
      },
      setViewDimension: (value) =>
        set((state) => ({ config: { ...state.config, viewDimension: value } })),
      resetResult: () => set({ result: undefined, error: null }),
      runComparison: async () => {
        const { config } = get();
        if (!config.leftRunId || !config.rightRunId) {
          set({ error: "Select both runs to compare." });
          return undefined;
        }
        set({ isLoading: true, error: null });
        try {
          const payload: CompareRunsRequest = {
            left_run_id: config.leftRunId,
            right_run_id: config.rightRunId,
            mode: config.mode,
            min_shared: config.minShared,
            histogram_bins: config.histogramBins,
            max_links: config.maxLinks,
            save: config.saveResults,
          };
          const response = await compareRuns(payload);
          set({ result: response, isLoading: false });
          return response;
        } catch (error) {
          const message = error instanceof Error ? error.message : "Failed to compare runs";
          set({ error: message, isLoading: false });
          return undefined;
        }
      },
    }),
    {
      name: "comparison-store",
      partialize: (state) => ({ config: state.config }),
    },
  ),
);
