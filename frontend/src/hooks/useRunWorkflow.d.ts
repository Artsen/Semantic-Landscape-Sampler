/**
 * Custom hook orchestrating run creation, sampling, fetching, and export interactions.
 *
 * Returns:
 *  - generate(): Create a run, trigger sampling, and hydrate the store with results.
 *  - exportDataset(options): Download the current run payload or fine-grained subset.
 *  - isLoading / error / results: Reactive state derived from the run store and mutation lifecycle.
 */
import type { RunResultsResponse, RunSummary, ExportFormat, ExportInclude, ExportMode, ExportScope, ExportRequestOptions } from "@/types/run";
export declare function useRunWorkflow(): {
    generate: () => void;
    exportDataset: ({ scope, format, mode, clusterId, selectionIds, viewport, include, }: {
        scope: ExportScope;
        format?: ExportFormat;
        mode?: ExportMode;
        clusterId?: number;
        selectionIds?: string[];
        viewport?: ExportRequestOptions["viewport"];
        include?: ExportInclude[];
    }) => Promise<void>;
    loadFromHistory: (summary: RunSummary) => void;
    duplicateRun: (summary: RunSummary) => void;
    isLoading: boolean;
    error: string | null;
    results: RunResultsResponse | undefined;
};
export type RunWorkflow = ReturnType<typeof useRunWorkflow>;
