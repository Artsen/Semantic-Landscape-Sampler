import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
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
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PointMaterial, Points, Stats } from "@react-three/drei";
import { memo, useCallback, useEffect, useMemo, useRef, useState, } from "react";
import * as THREE from "three";
import { filterEdgesByVisibility, filterSegmentsByRole, useRunStore, } from "@/store/runStore";
import { useSegmentContext } from "@/hooks/useSegmentContext";
function derivePositionScales(points, spread = 1) {
    if (!points.length) {
        return { scale3d: 1, scale2d: 1, center3d: [0, 0, 0], center2d: [0, 0] };
    }
    let sumX3 = 0;
    let sumY3 = 0;
    let sumZ3 = 0;
    let sumX2 = 0;
    let sumY2 = 0;
    points.forEach((point) => {
        sumX3 += point.coords_3d[0];
        sumY3 += point.coords_3d[1];
        sumZ3 += point.coords_3d[2];
        sumX2 += point.coords_2d[0];
        sumY2 += point.coords_2d[1];
    });
    const count = points.length;
    const center3d = [sumX3 / count, sumY3 / count, sumZ3 / count];
    const center2d = [sumX2 / count, sumY2 / count];
    let maxRadius3d = 0;
    let maxRadius2d = 0;
    points.forEach((point) => {
        const dx3 = point.coords_3d[0] - center3d[0];
        const dy3 = point.coords_3d[1] - center3d[1];
        const dz3 = point.coords_3d[2] - center3d[2];
        const radius3d = Math.sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);
        if (radius3d > maxRadius3d) {
            maxRadius3d = radius3d;
        }
        const dx2 = point.coords_2d[0] - center2d[0];
        const dy2 = point.coords_2d[1] - center2d[1];
        const radius2d = Math.sqrt(dx2 * dx2 + dy2 * dy2);
        if (radius2d > maxRadius2d) {
            maxRadius2d = radius2d;
        }
    });
    const scale3d = maxRadius3d > 0 ? spread / maxRadius3d : spread;
    const scale2d = maxRadius2d > 0 ? spread / maxRadius2d : spread;
    return {
        scale3d,
        scale2d,
        center3d,
        center2d,
    };
}
function BaseCloud({ points, selectedIds, hoveredId, onHoverChange, onSelectChange, viewMode, pointSize, clusterPalette, showDensity, renderTooltip, colorize, scales, overlay, focusPredicate, highlightedCluster, }) {
    const [tooltip, setTooltip] = useState(null);
    const projectedRef = useRef(null);
    const densityCanvasRef = useRef(null);
    const containerRef = useRef(null);
    const setViewportBounds = useRunStore((state) => state.setViewportBounds);
    const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
    const visibleMap = useMemo(() => new Map(points.map((point) => [point.id, point])), [points]);
    const prepared = useMemo(() => {
        const positions = new Float32Array(points.length * 3);
        const colors = new Float32Array(points.length * 3);
        const scaled3d = new Float32Array(points.length * 3);
        const scaled2d = new Float32Array(points.length * 3);
        const ids = [];
        const idToIndex = new Map();
        const color = new THREE.Color();
        const outlierAccent = new THREE.Color("#f97316");
        const lowSimilarityAccent = new THREE.Color("#94a3b8");
        const hoverAccent = new THREE.Color("#ffffff");
        const focusAccent = new THREE.Color("#f1f5f9");
        const dimColor = new THREE.Color("#0f172a");
        const { scale3d, scale2d, center3d, center2d } = scales;
        points.forEach((point, index) => {
            const [x3, y3, z3] = point.coords_3d;
            const [x2, y2] = point.coords_2d;
            const offset = index * 3;
            const isNoise = point.cluster === undefined || point.cluster === null || point.cluster === -1;
            const isFocused = focusPredicate ? focusPredicate(point) : true;
            const scaledX3 = (x3 - center3d[0]) * scale3d;
            const scaledY3 = (y3 - center3d[1]) * scale3d;
            const scaledZ3 = (z3 - center3d[2]) * scale3d;
            scaled3d[offset] = scaledX3;
            scaled3d[offset + 1] = scaledY3;
            scaled3d[offset + 2] = scaledZ3;
            const scaledX2 = (x2 - center2d[0]) * scale2d;
            const scaledY2 = (y2 - center2d[1]) * scale2d;
            scaled2d[offset] = scaledX2;
            scaled2d[offset + 1] = scaledY2;
            scaled2d[offset + 2] = 0;
            if (viewMode === "3d") {
                positions[offset] = scaledX3;
                positions[offset + 1] = scaledY3;
                positions[offset + 2] = scaledZ3;
            }
            else {
                positions[offset] = scaledX2;
                positions[offset + 1] = scaledY2;
                positions[offset + 2] = 0;
            }
            const key = String(isNoise ? -1 : point.cluster);
            color.set(clusterPalette[key] ?? "#38bdf8");
            const outlierScore = Math.min(1, Math.max(0, point.outlier_score ?? (isNoise ? 1 : 0)));
            if (outlierScore > 0) {
                const blend = 0.35 + outlierScore * 0.4;
                color.lerp(outlierAccent, Math.min(1, blend));
            }
            const similarity = point.similarity_to_centroid;
            if (similarity != null) {
                const normalised = Math.max(0, Math.min(1, (similarity + 1) * 0.5));
                const drift = 1 - normalised;
                if (drift > 0.1) {
                    color.lerp(lowSimilarityAccent, Math.min(0.35, drift));
                }
            }
            if (colorize) {
                colorize({ point, color });
            }
            if (highlightedCluster !== undefined && highlightedCluster !== null) {
                const clusterValue = point.cluster ?? -1;
                if (clusterValue === highlightedCluster) {
                    color.lerp(hoverAccent, 0.18);
                }
                else {
                    color.lerp(dimColor, 0.55);
                }
            }
            if (focusPredicate) {
                if (isFocused) {
                    color.lerp(focusAccent, 0.15);
                }
                else {
                    color.lerp(dimColor, 0.6);
                }
            }
            if (selectedSet.has(point.id)) {
                color.offsetHSL(0, 0, 0.18);
            }
            if (hoveredId === point.id) {
                color.lerp(hoverAccent, 0.45);
            }
            colors[offset] = color.r;
            colors[offset + 1] = color.g;
            colors[offset + 2] = color.b;
            ids.push(point.id);
            idToIndex.set(point.id, index);
        });
        return { positions, colors, ids, scaled3d, scaled2d, idToIndex };
    }, [points, clusterPalette, selectedSet, hoveredId, viewMode, scales, colorize, highlightedCluster]);
    const handleHover = useCallback((payload) => {
        if (!payload) {
            setTooltip(null);
            onHoverChange(undefined);
            return;
        }
        setTooltip(payload);
        onHoverChange(payload.id);
    }, [onHoverChange]);
    const handleSelect = useCallback((pointId, multi, toggle) => {
        onSelectChange((current) => {
            const currentSet = new Set(current);
            if (toggle) {
                if (currentSet.has(pointId)) {
                    currentSet.delete(pointId);
                }
                else {
                    currentSet.add(pointId);
                }
                return Array.from(currentSet);
            }
            if (multi) {
                currentSet.add(pointId);
                return Array.from(currentSet);
            }
            return [pointId];
        });
    }, [onSelectChange]);
    const handleProjectedUpdate = useCallback((projected, rect) => {
        projectedRef.current = { points: projected, rect };
        const margin = 12;
        const visible = [];
        for (const point of projected) {
            if (point.x < -margin || point.x > rect.width + margin) {
                continue;
            }
            if (point.y < -margin || point.y > rect.height + margin) {
                continue;
            }
            const datum = visibleMap.get(point.id);
            if (datum) {
                visible.push(datum);
            }
        }
        if (!visible.length) {
            setViewportBounds(null);
        }
        else if (viewMode === "3d") {
            let minX = Number.POSITIVE_INFINITY;
            let maxX = Number.NEGATIVE_INFINITY;
            let minY = Number.POSITIVE_INFINITY;
            let maxY = Number.NEGATIVE_INFINITY;
            let minZ = Number.POSITIVE_INFINITY;
            let maxZ = Number.NEGATIVE_INFINITY;
            visible.forEach((item) => {
                const [x, y, z] = item.coords_3d;
                if (x < minX)
                    minX = x;
                if (x > maxX)
                    maxX = x;
                if (y < minY)
                    minY = y;
                if (y > maxY)
                    maxY = y;
                if (z < minZ)
                    minZ = z;
                if (z > maxZ)
                    maxZ = z;
            });
            setViewportBounds({
                dimension: "3d",
                minX,
                maxX,
                minY,
                maxY,
                minZ,
                maxZ,
            });
        }
        else {
            let minX = Number.POSITIVE_INFINITY;
            let maxX = Number.NEGATIVE_INFINITY;
            let minY = Number.POSITIVE_INFINITY;
            let maxY = Number.NEGATIVE_INFINITY;
            let minZ = Number.POSITIVE_INFINITY;
            let maxZ = Number.NEGATIVE_INFINITY;
            visible.forEach((item) => {
                const [x2, y2] = item.coords_2d;
                if (x2 < minX)
                    minX = x2;
                if (x2 > maxX)
                    maxX = x2;
                if (y2 < minY)
                    minY = y2;
                if (y2 > maxY)
                    maxY = y2;
                const z3 = item.coords_3d[2];
                if (z3 < minZ)
                    minZ = z3;
                if (z3 > maxZ)
                    maxZ = z3;
            });
            const bounds2d = {
                dimension: "2d",
                minX,
                maxX,
                minY,
                maxY,
            };
            if (Number.isFinite(minZ) && Number.isFinite(maxZ)) {
                bounds2d.minZ = minZ;
                bounds2d.maxZ = maxZ;
            }
            setViewportBounds(bounds2d);
        }
        if (!showDensity) {
            return;
        }
        const canvas = densityCanvasRef.current;
        if (!canvas) {
            return;
        }
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            return;
        }
        const width = rect.width;
        const height = rect.height;
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "rgba(14, 165, 233, 0.08)";
        const radius = Math.max(12, Math.min(32, Math.sqrt(width * height) * 0.04));
        projected.forEach((point) => {
            const grad = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
            grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
            grad.addColorStop(1, "rgba(56, 189, 248, 0)");
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            ctx.fill();
        });
    }, [showDensity, setViewportBounds, visibleMap, viewMode]);
    useEffect(() => {
        if (!showDensity && densityCanvasRef.current) {
            const ctx = densityCanvasRef.current.getContext("2d");
            ctx?.clearRect(0, 0, densityCanvasRef.current.width, densityCanvasRef.current.height);
        }
    }, [showDensity]);
    const overlayElement = overlay ? overlay({ points, scales, viewMode, geometry: prepared }) : null;
    const hoveredBuffer = useMemo(() => {
        if (!hoveredId) {
            return null;
        }
        const index = prepared.idToIndex.get(hoveredId);
        if (index == null) {
            return null;
        }
        const offset = index * 3;
        const buffer = new Float32Array(3);
        buffer[0] = prepared.positions[offset];
        buffer[1] = prepared.positions[offset + 1];
        buffer[2] = prepared.positions[offset + 2];
        return buffer;
    }, [hoveredId, prepared]);
    return (_jsxs("div", { ref: containerRef, className: "relative h-full w-full", children: [_jsxs(Canvas, { orthographic: viewMode === "2d", camera: viewMode === "2d" ? { position: [0, 0, 10], zoom: 90 } : { position: [0, 0, 6], fov: 50 }, dpr: [1, 2], gl: { antialias: true, alpha: true }, children: [_jsx("color", { attach: "background", args: ["#050816"] }), _jsx(SceneContents, { prepared: prepared, pointSize: pointSize, viewMode: viewMode, onHover: handleHover, onSelect: handleSelect, onProjectedUpdate: handleProjectedUpdate }), hoveredBuffer ? (_jsx(Points, { positions: hoveredBuffer, children: _jsx(PointMaterial, { color: "#f8fafc", size: pointSize * 1.8, sizeAttenuation: true, transparent: true, depthWrite: false }) })) : null, overlayElement, _jsx(Stats, { className: "text-xs" })] }, viewMode), showDensity ? (_jsx("canvas", { ref: densityCanvasRef, className: "pointer-events-none absolute inset-0 opacity-75 mix-blend-screen", width: 0, height: 0 })) : null, _jsx(LassoOverlay, { projectedRef: projectedRef, currentSelection: selectedIds, onSelect: onSelectChange }), tooltip ? (() => {
                const point = visibleMap.get(tooltip.id);
                if (!point) {
                    return null;
                }
                const { title, body, footer } = renderTooltip(point);
                return (_jsxs("div", { className: "pointer-events-none absolute z-40 max-w-xs rounded-lg border border-slate-700/80 bg-slate-900/90 p-3 text-xs text-slate-200 shadow-lg", style: { left: tooltip.x + 16, top: tooltip.y + 16 }, children: [_jsx("div", { className: "font-semibold text-cyan-300", children: title }), body ? (_jsx("div", { className: "mt-2 space-y-1 text-[11px] text-slate-300", children: body })) : null, footer ? (_jsx("div", { className: "mt-2 text-[10px] text-slate-500", children: footer })) : null] }));
            })() : null] }));
}
function SceneContents({ prepared, pointSize, viewMode, onHover, onSelect, onProjectedUpdate }) {
    const { camera, gl, raycaster } = useThree();
    const projectionVector = useMemo(() => new THREE.Vector3(), []);
    const idsRef = useRef(prepared.ids);
    const pointsRef = useRef(null);
    const pointCount = Math.max(0, Math.floor(prepared.positions.length / 3));
    useEffect(() => {
        idsRef.current = prepared.ids;
    }, [prepared.ids]);
    useEffect(() => {
        const points = pointsRef.current;
        if (!points) {
            return;
        }
        const geometry = points.geometry;
        let positionAttr = geometry.getAttribute("position");
        let colorAttr = geometry.getAttribute("color");
        if (!positionAttr || positionAttr.count !== pointCount) {
            positionAttr = new THREE.BufferAttribute(prepared.positions, 3);
            geometry.setAttribute("position", positionAttr);
        }
        else {
            positionAttr.copyArray(prepared.positions);
        }
        positionAttr.needsUpdate = true;
        if (!colorAttr || colorAttr.count !== pointCount) {
            colorAttr = new THREE.BufferAttribute(prepared.colors, 3);
            geometry.setAttribute("color", colorAttr);
        }
        else {
            colorAttr.copyArray(prepared.colors);
        }
        colorAttr.needsUpdate = true;
        geometry.setDrawRange(0, pointCount);
        geometry.computeBoundingSphere();
    }, [prepared.positions, prepared.colors, pointCount]);
    useEffect(() => {
        if (!raycaster.params.Points) {
            raycaster.params.Points = { threshold: 0 };
        }
        raycaster.params.Points.threshold = Math.max(0.015, pointSize * (viewMode === "3d" ? 1.2 : 0.9));
    }, [raycaster, pointSize, viewMode]);
    const computeScreenPosition = useCallback((index, rectOverride) => {
        const rect = rectOverride ?? gl.domElement.getBoundingClientRect();
        const px = (prepared.positions[index * 3] ?? 0);
        const py = (prepared.positions[index * 3 + 1] ?? 0);
        const pz = (prepared.positions[index * 3 + 2] ?? 0);
        projectionVector.set(px, py, pz).project(camera);
        const screenX = (projectionVector.x + 1) * 0.5 * rect.width;
        const screenY = (-projectionVector.y + 1) * 0.5 * rect.height;
        return { x: screenX, y: screenY, rect };
    }, [camera, gl, prepared.positions, projectionVector]);
    const pixelThreshold = useCallback((rect) => Math.max(8, pointSize * rect.width * 0.02), [pointSize]);
    const handlePointerMove = useCallback((event) => {
        const index = event.index;
        if (index === undefined) {
            onHover(null);
            return;
        }
        const id = idsRef.current[index];
        if (!id) {
            onHover(null);
            return;
        }
        const { x, y, rect } = computeScreenPosition(index);
        const dx = event.clientX - rect.left - x;
        const dy = event.clientY - rect.top - y;
        if (Math.hypot(dx, dy) > pixelThreshold(rect)) {
            onHover(null);
            return;
        }
        onHover({ id, x, y });
    }, [computeScreenPosition, onHover, pixelThreshold]);
    const handlePointerLeave = useCallback(() => {
        onHover(null);
    }, [onHover]);
    const handlePointerDown = useCallback((event) => {
        if (event.button !== 0) {
            return;
        }
        const index = event.index;
        if (index === undefined) {
            return;
        }
        const id = idsRef.current[index];
        if (!id) {
            return;
        }
        const { x, y, rect } = computeScreenPosition(index);
        const dx = event.clientX - rect.left - x;
        const dy = event.clientY - rect.top - y;
        if (Math.hypot(dx, dy) > pixelThreshold(rect)) {
            return;
        }
        event.stopPropagation();
        onSelect(id, event.shiftKey || event.metaKey || event.ctrlKey, event.metaKey || event.ctrlKey);
    }, [computeScreenPosition, onSelect, pixelThreshold]);
    useFrame(() => {
        const rect = gl.domElement.getBoundingClientRect();
        const projected = [];
        for (let index = 0; index < prepared.positions.length / 3; index += 1) {
            const { x, y } = computeScreenPosition(index, rect);
            projected.push({ id: idsRef.current[index], x, y });
        }
        onProjectedUpdate(projected, rect);
    });
    const rotation = viewMode === "3d" ? [-0.12, 0.14, 0] : [0, 0, 0];
    return (_jsxs("group", { rotation: rotation, children: [_jsx("ambientLight", { intensity: 0.8 }), _jsx("directionalLight", { position: [5, 5, 10], intensity: 0.6 }), _jsxs(Points, { ref: pointsRef, positions: prepared.positions, colors: prepared.colors, stride: 3, frustumCulled: false, onPointerMove: handlePointerMove, onPointerLeave: handlePointerLeave, onPointerDown: handlePointerDown, children: [_jsx("bufferGeometry", { ref: pointsRef }), _jsx(PointMaterial, { transparent: true, vertexColors: true, size: pointSize, sizeAttenuation: true, depthWrite: false, opacity: 0.92 })] }, `cloud-${viewMode}-${pointCount}`), _jsx(OrbitControls, { enablePan: true, enableRotate: viewMode === "3d", enableZoom: true, minDistance: 1.5, maxDistance: 14, zoomSpeed: 0.8, dampingFactor: 0.08 })] }));
}
function LassoOverlay({ projectedRef, currentSelection, onSelect }) {
    const [drag, setDrag] = useState(null);
    const [shiftPressed, setShiftPressed] = useState(false);
    const toLocal = useCallback((event) => {
        const rect = projectedRef.current?.rect ?? event.currentTarget.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top,
        };
    }, [projectedRef]);
    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "Shift") {
                setShiftPressed(true);
            }
        };
        const handleKeyUp = (event) => {
            if (event.key === "Shift") {
                setShiftPressed(false);
                setDrag(null);
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        window.addEventListener("keyup", handleKeyUp);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            window.removeEventListener("keyup", handleKeyUp);
        };
    }, []);
    const handlePointerDown = (event) => {
        if (!shiftPressed || event.button !== 0) {
            return;
        }
        const local = toLocal(event);
        setDrag({ startX: local.x, startY: local.y, currentX: local.x, currentY: local.y });
    };
    const handlePointerMove = (event) => {
        if (!drag) {
            return;
        }
        const local = toLocal(event);
        setDrag((current) => (current ? { ...current, currentX: local.x, currentY: local.y } : null));
    };
    const handlePointerUp = () => {
        if (!drag) {
            return;
        }
        const data = projectedRef.current;
        if (data) {
            const minX = Math.min(drag.startX, drag.currentX);
            const maxX = Math.max(drag.startX, drag.currentX);
            const minY = Math.min(drag.startY, drag.currentY);
            const maxY = Math.max(drag.startY, drag.currentY);
            const additions = data.points
                .filter((point) => point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY)
                .map((point) => point.id);
            if (additions.length) {
                onSelect((current) => {
                    const base = Array.isArray(current) ? current : Array.from(currentSelection);
                    const merged = new Set(base);
                    additions.forEach((id) => merged.add(id));
                    return Array.from(merged);
                });
            }
        }
        setDrag(null);
    };
    return (_jsxs("div", { className: "absolute inset-0 z-30", style: { pointerEvents: shiftPressed ? "auto" : "none" }, onPointerDown: handlePointerDown, onPointerMove: handlePointerMove, onPointerUp: handlePointerUp, children: [drag ? (_jsx("div", { className: "pointer-events-none absolute border border-cyan-400/80 bg-cyan-400/10", style: {
                    left: Math.min(drag.startX, drag.currentX),
                    top: Math.min(drag.startY, drag.currentY),
                    width: Math.abs(drag.currentX - drag.startX),
                    height: Math.abs(drag.currentY - drag.startY),
                } })) : null, shiftPressed && !drag ? (_jsx("div", { className: "pointer-events-none absolute right-4 top-4 rounded-lg border border-slate-700/80 bg-slate-900/80 px-3 py-1 text-[11px] text-slate-300", children: "Shift drag for lasso select" })) : null] }));
}
export const PointCloudScene = memo(function PointCloudScene({ responses, segments, segmentEdges, responseHulls }) {
    const levelMode = useRunStore((state) => state.levelMode);
    if (levelMode === "segments") {
        return (_jsx(SegmentCloud, { segments: segments, edges: segmentEdges, hulls: responseHulls }));
    }
    return _jsx(ResponseCloud, { points: responses });
});
const ResponseCloud = memo(function ResponseCloud({ points }) {
    const { viewMode, pointSize, clusterVisibility, clusterPalette, selectedPointIds, hoveredPointId, setHoveredPoint, setSelectedPoints, showDensity, spreadFactor, focusedResponseId, hoveredClusterLabel, } = useRunStore((state) => ({
        viewMode: state.viewMode,
        pointSize: state.pointSize,
        clusterVisibility: state.clusterVisibility,
        clusterPalette: state.clusterPalette,
        selectedPointIds: state.selectedPointIds,
        hoveredPointId: state.hoveredPointId,
        setHoveredPoint: state.setHoveredPoint,
        setSelectedPoints: state.setSelectedPoints,
        showDensity: state.showDensity,
        spreadFactor: state.spreadFactor,
        focusedResponseId: state.focusedResponseId,
        hoveredClusterLabel: state.hoveredClusterLabel,
    }));
    const visiblePoints = useMemo(() => points.filter((point) => {
        const key = String(point.cluster ?? -1);
        const visible = clusterVisibility[key];
        return visible !== false && !point.hidden;
    }), [points, clusterVisibility]);
    const scales = useMemo(() => derivePositionScales(visiblePoints, spreadFactor), [visiblePoints, spreadFactor]);
    const renderTooltip = useCallback((point) => ({
        title: `Sample #${point.index}`,
        body: point.text_preview ?? "",
    }), []);
    return (_jsx(BaseCloud, { points: visiblePoints, selectedIds: selectedPointIds, hoveredId: hoveredPointId, onHoverChange: setHoveredPoint, onSelectChange: setSelectedPoints, viewMode: viewMode, pointSize: pointSize, clusterPalette: clusterPalette, focusPredicate: focusedResponseId ? (point) => point.id === focusedResponseId : undefined, showDensity: showDensity, renderTooltip: renderTooltip, scales: scales, highlightedCluster: hoveredClusterLabel ?? null }));
});
const SegmentCloud = memo(function SegmentCloud({ segments, edges, hulls }) {
    const { viewMode, pointSize, clusterVisibility, clusterPalette, roleVisibility, selectedSegmentIds, hoveredSegmentId, setHoveredSegment, setSelectedSegments, showDensity, showEdges, showParentThreads, spreadFactor, focusedResponseId, setFocusedResponse, hoveredClusterLabel, showNeighborSpokes, graphEdgeK, showDuplicatesOnly, } = useRunStore((state) => ({
        viewMode: state.viewMode,
        pointSize: state.pointSize,
        clusterVisibility: state.clusterVisibility,
        clusterPalette: state.clusterPalette,
        roleVisibility: state.roleVisibility,
        selectedSegmentIds: state.selectedSegmentIds,
        hoveredSegmentId: state.hoveredSegmentId,
        setHoveredSegment: state.setHoveredSegment,
        setSelectedSegments: state.setSelectedSegments,
        showDensity: state.showDensity,
        showEdges: state.showEdges,
        showParentThreads: state.showParentThreads,
        spreadFactor: state.spreadFactor,
        focusedResponseId: state.focusedResponseId,
        setFocusedResponse: state.setFocusedResponse,
        hoveredClusterLabel: state.hoveredClusterLabel,
        showNeighborSpokes: state.showNeighborSpokes,
        graphEdgeK: state.graphEdgeK,
        showDuplicatesOnly: state.showDuplicatesOnly,
    }));
    const [hoverProbeId, setHoverProbeId] = useState(undefined);
    useEffect(() => {
        if (!hoveredSegmentId) {
            const timeout = window.setTimeout(() => setHoverProbeId(undefined), 120);
            return () => window.clearTimeout(timeout);
        }
        const timeout = window.setTimeout(() => setHoverProbeId(hoveredSegmentId), 120);
        return () => window.clearTimeout(timeout);
    }, [hoveredSegmentId]);
    const neighborCount = Math.max(3, Math.min(12, graphEdgeK));
    const { data: hoveredContext, isFetching: contextLoading, } = useSegmentContext(hoverProbeId, {
        enabled: Boolean(hoverProbeId),
        k: neighborCount,
        staleTimeMs: 180_000,
    });
    const roleFiltered = useMemo(() => filterSegmentsByRole(segments, roleVisibility), [segments, roleVisibility]);
    const duplicatesFiltered = useMemo(() => (showDuplicatesOnly ? roleFiltered.filter((segment) => segment.is_duplicate) : roleFiltered), [roleFiltered, showDuplicatesOnly]);
    const sequenceRatios = useMemo(() => {
        const maxByResponse = new Map();
        segments.forEach((segment) => {
            const current = maxByResponse.get(segment.response_id);
            const nextValue = Math.max(current ?? Number.NEGATIVE_INFINITY, segment.position);
            maxByResponse.set(segment.response_id, nextValue);
        });
        const ratios = new Map();
        segments.forEach((segment) => {
            const maxPosition = maxByResponse.get(segment.response_id);
            if (maxPosition == null || maxPosition <= 0) {
                ratios.set(segment.id, 0);
                return;
            }
            ratios.set(segment.id, Math.max(0, Math.min(1, segment.position / maxPosition)));
        });
        return ratios;
    }, [segments]);
    const neighborPreview = useMemo(() => {
        if (!hoveredContext || hoveredContext.segment_id !== hoveredSegmentId) {
            return [];
        }
        return hoveredContext.neighbors.slice(0, Math.min(12, neighborCount));
    }, [hoveredContext, hoveredSegmentId, neighborCount]);
    const neighborSpokeTargets = showNeighborSpokes ? neighborPreview : [];
    const [startColor, midColor, endColor] = useMemo(() => [new THREE.Color("#16a34a"), new THREE.Color("#2563eb"), new THREE.Color("#dc2626")], []);
    const visibleSegments = useMemo(() => duplicatesFiltered.filter((segment) => {
        const key = String(segment.cluster ?? -1);
        const visible = clusterVisibility[key];
        return visible !== false;
    }), [duplicatesFiltered, clusterVisibility]);
    const scales = useMemo(() => derivePositionScales(visibleSegments, spreadFactor), [visibleSegments, spreadFactor]);
    const colorize = useCallback(({ point, color }) => {
        const ratio = sequenceRatios.get(point.id) ?? 0;
        const clampRatio = Math.max(0, Math.min(1, ratio));
        const midPoint = 0.5;
        if (clampRatio <= midPoint) {
            const t = clampRatio / midPoint;
            color.copy(startColor).lerp(midColor, t);
        }
        else {
            const t = (clampRatio - midPoint) / midPoint;
            color.copy(midColor).lerp(endColor, t);
        }
    }, [sequenceRatios, startColor, midColor, endColor]);
    const renderTooltip = useCallback((segment) => {
        const roleLabel = segment.role ? segment.role : "segment";
        const title = `${roleLabel} · ${segment.position + 1}`;
        const rawText = segment.text ?? "";
        const truncated = rawText.length > 220 ? `${rawText.slice(0, 220)}…` : rawText;
        const isHover = hoveredSegmentId === segment.id;
        const contextMatch = hoveredContext && hoveredContext.segment_id === segment.id;
        if (!contextMatch) {
            return {
                title,
                body: truncated,
                footer: isHover && contextLoading ? "Loading context…" : undefined,
            };
        }
        const topTerms = hoveredContext.top_terms.slice(0, 4);
        const neighbors = neighborPreview;
        const neighborSummary = neighbors.slice(0, 3);
        const exemplarPreview = hoveredContext.exemplar_preview || "";
        const why = hoveredContext.why_here ?? {};
        return {
            title,
            body: (_jsxs(_Fragment, { children: [_jsx("p", { className: "text-[11px] text-slate-200", children: truncated }), (segment.is_cached || segment.is_duplicate) && (_jsxs("div", { className: "mt-2 flex flex-wrap gap-2 text-[10px]", children: [segment.is_cached ? (_jsx("span", { className: "rounded-full border border-emerald-400/40 bg-emerald-500/10 px-2 py-[1px] text-emerald-200", children: "Cached" })) : null, segment.is_duplicate ? (_jsx("span", { className: "rounded-full border border-amber-400/40 bg-amber-500/10 px-2 py-[1px] text-amber-200", children: "Duplicate" })) : null] })), topTerms.length ? (_jsx("div", { className: "mt-2 flex flex-wrap gap-1", children: topTerms.map((term) => (_jsx("span", { className: "rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2 py-[1px] text-[10px] text-cyan-100", title: `TF-IDF weight ${term.weight.toFixed(2)}`, children: term.term }, `${segment.id}-${term.term}`))) })) : null, exemplarPreview ? (_jsxs("p", { className: "mt-2 text-[10px] text-slate-400", children: ["Closest exemplar \u00B7 ", exemplarPreview.slice(0, 80), exemplarPreview.length > 80 ? "…" : ""] })) : null, _jsxs("div", { className: "mt-2 space-y-1 text-[10px] text-slate-400", children: [why.sim_to_exemplar != null ? (_jsxs("p", { children: ["Sim to exemplar ", why.sim_to_exemplar.toFixed(2)] })) : null, why.sim_to_nn != null ? (_jsxs("p", { children: ["Sim to nearest ", why.sim_to_nn.toFixed(2)] })) : null] }), neighborSummary.length ? (_jsxs("div", { className: "mt-2 space-y-1 text-[10px] text-slate-400", children: [_jsx("p", { children: "Nearest neighbours" }), neighborSummary.map((neighbor) => (_jsxs("p", { className: "text-slate-300", children: [_jsx("span", { className: "text-cyan-200", children: neighbor.similarity.toFixed(2) }), " \u00B7 ", neighbor.text] }, neighbor.id)))] })) : null] })),
            footer: contextLoading ? "Refreshing context…" : undefined,
        };
    }, [contextLoading, hoveredContext, hoveredSegmentId, neighborPreview]);
    const overlay = useCallback(({ points, scales: overlayScales, viewMode: overlayView, geometry }) => {
        if (!points.length) {
            return null;
        }
        const segmentMap = new Map(points.map((segment) => [segment.id, segment]));
        return (_jsxs(_Fragment, { children: [showEdges ? (_jsx(SegmentEdgesMesh, { edges: edges, segments: segmentMap, viewMode: overlayView, geometry: geometry })) : null, showParentThreads ? (_jsx(ParentThreadsMesh, { segments: segmentMap, viewMode: overlayView, geometry: geometry, focusedResponseId: focusedResponseId })) : null, showParentThreads ? (_jsx(ResponseHullMesh, { hulls: hulls, viewMode: overlayView, scales: overlayScales })) : null, hoveredContext && neighborSpokeTargets.length
                    ? (_jsx(NeighborSpokesMesh, { rootId: hoveredContext.segment_id, neighbors: neighborSpokeTargets, segments: segmentMap, viewMode: overlayView, geometry: geometry }))
                    : null] }));
    }, [edges, hulls, showEdges, showParentThreads, scales, focusedResponseId, hoveredContext, neighborSpokeTargets]);
    return (_jsx(BaseCloud, { points: visibleSegments, selectedIds: selectedSegmentIds, hoveredId: hoveredSegmentId, onHoverChange: setHoveredSegment, onSelectChange: setSelectedSegments, viewMode: viewMode, pointSize: pointSize * 0.9, clusterPalette: clusterPalette, showDensity: showDensity, renderTooltip: renderTooltip, colorize: colorize, scales: scales, overlay: overlay, highlightedCluster: hoveredClusterLabel ?? null }));
});
function useBufferAttributeUpdate(array) {
    const attributeRef = useRef(null);
    useEffect(() => {
        if (!array || !attributeRef.current) {
            return;
        }
        attributeRef.current.needsUpdate = true;
    }, [array]);
    return attributeRef;
}
const LineSegmentsPrimitive = memo(function LineSegmentsPrimitive({ positions, color, opacity, depthWrite, }) {
    const attributeRef = useBufferAttributeUpdate(positions);
    return (_jsxs("lineSegments", { children: [_jsx("bufferGeometry", { children: _jsx("bufferAttribute", { ref: attributeRef, attach: "attributes-position", count: positions.length / 3, array: positions, itemSize: 3 }) }), _jsx("lineBasicMaterial", { color: color, opacity: opacity, transparent: true, depthWrite: depthWrite ?? undefined })] }));
});
const LineLoopPrimitive = memo(function LineLoopPrimitive({ positions, color, opacity }) {
    const attributeRef = useBufferAttributeUpdate(positions);
    return (_jsxs("lineLoop", { children: [_jsx("bufferGeometry", { children: _jsx("bufferAttribute", { ref: attributeRef, attach: "attributes-position", count: positions.length / 3, array: positions, itemSize: 3 }) }), _jsx("lineBasicMaterial", { color: color, opacity: opacity, transparent: true })] }));
});
const SegmentEdgesMesh = memo(function SegmentEdgesMesh({ edges, segments, viewMode, geometry }) {
    const positions = useMemo(() => {
        const filtered = filterEdgesByVisibility(edges, new Set(segments.keys()));
        if (!filtered.length) {
            return null;
        }
        const coords = viewMode === "3d" ? geometry.scaled3d : geometry.scaled2d;
        const indexLookup = geometry.idToIndex;
        const data = new Float32Array(filtered.length * 6);
        let writeOffset = 0;
        filtered.forEach((edge) => {
            const sourceIndex = indexLookup.get(edge.source_id);
            const targetIndex = indexLookup.get(edge.target_id);
            if (sourceIndex == null || targetIndex == null) {
                return;
            }
            const sourceOffset = sourceIndex * 3;
            const targetOffset = targetIndex * 3;
            data[writeOffset] = coords[sourceOffset] ?? 0;
            data[writeOffset + 1] = coords[sourceOffset + 1] ?? 0;
            data[writeOffset + 2] = coords[sourceOffset + 2] ?? 0;
            data[writeOffset + 3] = coords[targetOffset] ?? 0;
            data[writeOffset + 4] = coords[targetOffset + 1] ?? 0;
            data[writeOffset + 5] = coords[targetOffset + 2] ?? 0;
            writeOffset += 6;
        });
        return writeOffset === data.length ? data : data.subarray(0, writeOffset);
    }, [edges, segments, viewMode, geometry]);
    if (!positions) {
        return null;
    }
    return _jsx(LineSegmentsPrimitive, { positions: positions, color: "#38bdf8", opacity: 0.12 });
});
const NeighborSpokesMesh = memo(function NeighborSpokesMesh({ rootId, neighbors, segments, viewMode, geometry, }) {
    const positions = useMemo(() => {
        if (!neighbors.length || !segments.has(rootId)) {
            return null;
        }
        const coords = viewMode === "3d" ? geometry.scaled3d : geometry.scaled2d;
        const indexLookup = geometry.idToIndex;
        const rootIndex = indexLookup.get(rootId);
        if (rootIndex == null) {
            return null;
        }
        const rootOffset = rootIndex * 3;
        const baseX = coords[rootOffset] ?? 0;
        const baseY = coords[rootOffset + 1] ?? 0;
        const baseZ = coords[rootOffset + 2] ?? 0;
        const data = new Float32Array(neighbors.length * 6);
        let writeOffset = 0;
        neighbors.forEach((neighbor) => {
            const targetIndex = indexLookup.get(neighbor.id);
            if (targetIndex == null) {
                return;
            }
            const targetOffset = targetIndex * 3;
            data[writeOffset] = baseX;
            data[writeOffset + 1] = baseY;
            data[writeOffset + 2] = baseZ;
            data[writeOffset + 3] = coords[targetOffset] ?? 0;
            data[writeOffset + 4] = coords[targetOffset + 1] ?? 0;
            data[writeOffset + 5] = coords[targetOffset + 2] ?? 0;
            writeOffset += 6;
        });
        if (writeOffset === 0) {
            return null;
        }
        return writeOffset === data.length ? data : data.subarray(0, writeOffset);
    }, [geometry, neighbors, rootId, segments, viewMode]);
    if (!positions) {
        return null;
    }
    return _jsx(LineSegmentsPrimitive, { positions: positions, color: "#facc15", opacity: 0.45, depthWrite: false });
});
const ParentThreadsMesh = memo(function ParentThreadsMesh({ segments, viewMode, geometry, focusedResponseId, }) {
    const { focusedLines, fadedLines } = useMemo(() => {
        const byResponse = new Map();
        segments.forEach((segment) => {
            const list = byResponse.get(segment.response_id) ?? [];
            list.push(segment);
            byResponse.set(segment.response_id, list);
        });
        const focus = [];
        const faded = [];
        const coords = viewMode === "3d" ? geometry.scaled3d : geometry.scaled2d;
        const indexLookup = geometry.idToIndex;
        byResponse.forEach((list, responseId) => {
            if (list.length < 2) {
                return;
            }
            const sorted = [...list].sort((a, b) => a.position - b.position);
            const data = new Float32Array((sorted.length - 1) * 6);
            let writeOffset = 0;
            for (let i = 0; i < sorted.length - 1; i += 1) {
                const startIndex = indexLookup.get(sorted[i].id);
                const endIndex = indexLookup.get(sorted[i + 1].id);
                if (startIndex == null || endIndex == null) {
                    continue;
                }
                const startOffset = startIndex * 3;
                const endOffset = endIndex * 3;
                data[writeOffset] = coords[startOffset] ?? 0;
                data[writeOffset + 1] = coords[startOffset + 1] ?? 0;
                data[writeOffset + 2] = coords[startOffset + 2] ?? 0;
                data[writeOffset + 3] = coords[endOffset] ?? 0;
                data[writeOffset + 4] = coords[endOffset + 1] ?? 0;
                data[writeOffset + 5] = coords[endOffset + 2] ?? 0;
                writeOffset += 6;
            }
            if (writeOffset === 0) {
                return;
            }
            const slice = writeOffset === data.length ? data : data.subarray(0, writeOffset);
            if (!focusedResponseId || responseId === focusedResponseId) {
                focus.push({ id: responseId, data: slice });
            }
            else {
                faded.push({ id: responseId, data: slice });
            }
        });
        return { focusedLines: focus, fadedLines: faded };
    }, [segments, viewMode, geometry, focusedResponseId]);
    if (!focusedLines.length && !fadedLines.length) {
        return null;
    }
    return (_jsxs("group", { children: [fadedLines.map(({ data }, index) => (_jsx(LineSegmentsPrimitive, { positions: data, color: "#fcd34d", opacity: 0.06 }, `faded-thread-${index}`))), focusedLines.map(({ data, id }) => (_jsx(LineSegmentsPrimitive, { positions: data, color: "#fcd34d", opacity: 0.28 }, `focused-thread-${id}`)))] }));
});
const ResponseHullMesh = memo(function ResponseHullMesh({ hulls, viewMode, scales }) {
    const loops = useMemo(() => {
        if (!hulls.length) {
            return [];
        }
        const { scale3d, scale2d, center3d, center2d } = scales;
        const result = [];
        hulls.forEach((hull) => {
            const coords = viewMode === "3d" ? hull.coords_3d : hull.coords_2d;
            if (!coords.length) {
                return;
            }
            const data = new Float32Array(coords.length * 3);
            coords.forEach((point, index) => {
                const offset = index * 3;
                if (viewMode === "3d") {
                    data[offset] = (point[0] - center3d[0]) * scale3d;
                    data[offset + 1] = (point[1] - center3d[1]) * scale3d;
                    data[offset + 2] = (point[2] - center3d[2]) * scale3d;
                }
                else {
                    data[offset] = (point[0] - center2d[0]) * scale2d;
                    data[offset + 1] = (point[1] - center2d[1]) * scale2d;
                    data[offset + 2] = 0;
                }
            });
            result.push(data);
        });
        return result;
    }, [hulls, viewMode, scales]);
    if (!loops.length) {
        return null;
    }
    return (_jsx("group", { children: loops.map((positions, index) => (_jsx(LineLoopPrimitive, { positions: positions, color: "#22d3ee", opacity: 0.15 }, `response-hull-${index}`))) }));
});
