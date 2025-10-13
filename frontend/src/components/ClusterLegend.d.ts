/**
 * Legend for response and segment clusters displayed beside the point cloud.
 *
 * Components:
 *  - ClusterLegend: Lists clusters with keywords, counts, toggles, and export quick actions.
 */
import type { ClusterSummary, SegmentClusterSummary } from "@/types/run";
interface ClusterLegendProps {
    responseClusters: ClusterSummary[];
    segmentClusters: SegmentClusterSummary[];
    onExportCluster?: (payload: {
        label: number;
        mode: "responses" | "segments";
    }) => void;
    exportDisabled?: boolean;
    formatLabel?: string;
}
export declare const ClusterLegend: import("react").NamedExoticComponent<ClusterLegendProps>;
export {};
