/**
 * Side panel showing detailed metrics for selected or hovered responses and segments.
 *
 * Components:
 *  - PointDetailsPanel: Chooses between response or segment detail stacks.
 *  - ResponseDetails / SegmentDetails: Present per-item metrics, raw text, and contextual badges.
 */
import type { RunWorkflow } from "@/hooks/useRunWorkflow";
type PointDetailsPanelProps = {
    workflow: RunWorkflow;
};
export declare const PointDetailsPanel: import("react").NamedExoticComponent<PointDetailsPanelProps>;
export {};
