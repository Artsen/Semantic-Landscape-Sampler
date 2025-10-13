/**
 * Three.js-based visualization of response and segment embeddings.
 *
 * Components:
 *  - PointCloudScene: Switches between response and segment views based on store state.
 *  - BaseCloud: Shared renderer that prepares geometry, tooltips, and lasso selection.
 *  - ResponseCloud / SegmentCloud: Mode-specific wrappers that apply palettes, filters, and overlays.
 *  - SceneContents: Low-level renderer that updates buffer attributes and manages pointer events.
 *  - SegmentEdgesMesh, ParentThreadsMesh, ResponseHullMesh: Optional overlays for connections and hulls.
 */
import type { ResponsePoint, ResponseHull, SegmentEdge, SegmentPoint } from "@/types/run";
interface PointCloudSceneProps {
    responses: ResponsePoint[];
    segments: SegmentPoint[];
    segmentEdges: SegmentEdge[];
    responseHulls: ResponseHull[];
}
export declare const PointCloudScene: import("react").NamedExoticComponent<PointCloudSceneProps>;
export {};
