/**
 * Drawer overlay showing recently sampled runs with metadata and quick loading actions.
 *
 * Components:
 *  - RunHistoryDrawer: Fetches persisted runs from the backend, groups them by day,
 *    and lets users load a prior run without re-sampling.
 */
import type { RunWorkflow } from "@/hooks/useRunWorkflow";
type RunHistoryDrawerProps = {
    workflow: RunWorkflow;
};
export declare const RunHistoryDrawer: import("react").NamedExoticComponent<RunHistoryDrawerProps>;
export {};
