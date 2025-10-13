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

import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PointMaterial, Points, Stats } from "@react-three/drei";
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import * as THREE from "three";

import {
  filterEdgesByVisibility,
  filterSegmentsByRole,
  useRunStore,
  type SceneDimension,
} from "@/store/runStore";
import { useSegmentContext } from "@/hooks/useSegmentContext";
import type { ResponsePoint, ResponseHull, SegmentEdge, SegmentPoint, Viewport2D, Viewport3D } from "@/types/run";

interface PointCloudSceneProps {
  responses: ResponsePoint[];
  segments: SegmentPoint[];
  segmentEdges: SegmentEdge[];
  responseHulls: ResponseHull[];
}

interface PreparedGeometry {
  positions: Float32Array;
  colors: Float32Array;
  ids: string[];
  scaled3d: Float32Array;
  scaled2d: Float32Array;
  idToIndex: Map<string, number>;
}

type TooltipState = {
  id: string;
  x: number;
  y: number;
};

type TooltipContent = {
  title: ReactNode;
  body?: ReactNode;
  footer?: ReactNode;
};

type ProjectedPoint = {
  id: string;
  x: number;
  y: number;
};

type CloudPoint = ResponsePoint | SegmentPoint;

interface PositionScales {
  scale3d: number;
  scale2d: number;
  center3d: [number, number, number];
  center2d: [number, number];
}

interface OverlayContext<T extends CloudPoint> {
  points: readonly T[];
  scales: PositionScales;
  viewMode: SceneDimension;
  geometry: PreparedGeometry;
}

interface BaseCloudProps<T extends CloudPoint> {
  points: readonly T[];
  selectedIds: readonly string[];
  hoveredId?: string;
  onHoverChange: (id: string | undefined) => void;
  onSelectChange: (payload: string[] | ((current: string[]) => string[])) => void;
  viewMode: SceneDimension;
  pointSize: number;
  clusterPalette: Record<string, string>;
  showDensity: boolean;
  renderTooltip: (point: T) => TooltipContent;
  colorize?: (payload: { point: T; color: THREE.Color }) => void;
  scales: PositionScales;
  overlay?: (context: OverlayContext<T>) => ReactNode;
  onViewportChange?: (bounds: ViewportBounds | null) => void;
  showPerformanceStatsOverride?: boolean;
  onShowPerformanceStatsChange?: (value: boolean) => void;
  focusPredicate?: (point: T) => boolean;
  highlightedCluster?: number | null;
}

type ViewportBounds = (Viewport3D & { dimension: "3d" }) | (Viewport2D & { dimension: "2d"; minZ?: number; maxZ?: number });
function derivePositionScales<T extends CloudPoint>(points: readonly T[], spread: number = 1): PositionScales {
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
  const center3d: [number, number, number] = [sumX3 / count, sumY3 / count, sumZ3 / count];
  const center2d: [number, number] = [sumX2 / count, sumY2 / count];

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

function BaseCloud<T extends CloudPoint>({
  points,
  selectedIds,
  hoveredId,
  onHoverChange,
  onSelectChange,
  viewMode,
  pointSize,
  clusterPalette,
  showDensity,
  renderTooltip,
  colorize,
  scales,
  overlay,
  focusPredicate,
  highlightedCluster,
  onViewportChange,
  showPerformanceStatsOverride,
  onShowPerformanceStatsChange,
}: BaseCloudProps<T>) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const projectedRef = useRef<{ points: ProjectedPoint[]; rect: DOMRect } | null>(null);
  const densityCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const setViewportBounds = useRunStore((state) => state.setViewportBounds);

  const { showPerformanceStats, setShowPerformanceStats } = useRunStore((state) => ({
    showPerformanceStats: state.showPerformanceStats,
    setShowPerformanceStats: state.setShowPerformanceStats,
  }));

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const visibleMap = useMemo(() => new Map(points.map((point) => [point.id, point])), [points]);

  const prepared = useMemo<PreparedGeometry>(() => {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const scaled3d = new Float32Array(points.length * 3);
    const scaled2d = new Float32Array(points.length * 3);
    const ids: string[] = [];
    const idToIndex = new Map<string, number>();

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
      } else {
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
        } else {
          color.lerp(dimColor, 0.55);
        }
      }

      if (focusPredicate) {
        if (isFocused) {
          color.lerp(focusAccent, 0.15);
        } else {
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

  const handleHover = useCallback(
    (payload: TooltipState | null) => {
      if (!payload) {
        setTooltip(null);
        onHoverChange(undefined);
        return;
      }
      setTooltip(payload);
      onHoverChange(payload.id);
    },
    [onHoverChange],
  );

  const handleSelect = useCallback(
    (pointId: string, multi: boolean, toggle: boolean) => {
      onSelectChange((current) => {
        const currentSet = new Set(current);
        if (toggle) {
          if (currentSet.has(pointId)) {
            currentSet.delete(pointId);
          } else {
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
    },
    [onSelectChange],
  );

  const handleProjectedUpdate = useCallback(
    (projected: ProjectedPoint[], rect: DOMRect) => {
      projectedRef.current = { points: projected, rect };

      const margin = 12;
      const visible: T[] = [];
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
      } else if (viewMode === "3d") {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        let minZ = Number.POSITIVE_INFINITY;
        let maxZ = Number.NEGATIVE_INFINITY;
        visible.forEach((item) => {
          const [x, y, z] = item.coords_3d;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          if (z < minZ) minZ = z;
          if (z > maxZ) maxZ = z;
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
      } else {
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        let minZ = Number.POSITIVE_INFINITY;
        let maxZ = Number.NEGATIVE_INFINITY;
        visible.forEach((item) => {
          const [x2, y2] = item.coords_2d;
          if (x2 < minX) minX = x2;
          if (x2 > maxX) maxX = x2;
          if (y2 < minY) minY = y2;
          if (y2 > maxY) maxY = y2;
          const z3 = item.coords_3d[2];
          if (z3 < minZ) minZ = z3;
          if (z3 > maxZ) maxZ = z3;
        });
        const bounds2d: {
          dimension: "2d";
          minX: number;
          maxX: number;
          minY: number;
          maxY: number;
          minZ?: number;
          maxZ?: number;
        } = {
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
    },
    [showDensity, setViewportBounds, visibleMap, viewMode],
  );

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

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full"
      onDoubleClick={() => setShowPerformanceStats(!showPerformanceStats)}
    >
      <Canvas
        key={viewMode}
        orthographic={viewMode === "2d"}
        camera={viewMode === "2d" ? { position: [0, 0, 10], zoom: 90 } : { position: [0, 0, 6], fov: 50 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={["#050816"]} />
        <SceneContents
          prepared={prepared}
          pointSize={pointSize}
          viewMode={viewMode}
          onHover={handleHover}
          onSelect={handleSelect}
          onProjectedUpdate={handleProjectedUpdate}
        />
        {hoveredBuffer ? (
          <Points positions={hoveredBuffer}>
            <PointMaterial color="#f8fafc" size={pointSize * 1.8} sizeAttenuation transparent depthWrite={false} />
          </Points>
        ) : null}
        {overlayElement}
        {showPerformanceStats ? <Stats className="text-xs" /> : null}
      </Canvas>

      {showDensity ? (
        <canvas
          ref={densityCanvasRef}
          className="pointer-events-none absolute inset-0 opacity-75 mix-blend-screen"
          width={0}
          height={0}
        />
      ) : null}

      <LassoOverlay projectedRef={projectedRef} currentSelection={selectedIds} onSelect={onSelectChange} />

      {tooltip ? (() => {
        const point = visibleMap.get(tooltip.id) as T | undefined;
        if (!point) {
          return null;
        }
        const { title, body, footer } = renderTooltip(point);
        return (
          <div
            className="pointer-events-none absolute z-40 max-w-xs rounded-lg border border-slate-700/80 bg-slate-900/90 p-3 text-xs text-slate-200 shadow-lg"
            style={{ left: tooltip.x + 16, top: tooltip.y + 16 }}
          >
            <div className="font-semibold text-cyan-300">{title}</div>
            {body ? (
              <div className="mt-2 space-y-1 text-[11px] text-slate-300">{body}</div>
            ) : null}
            {footer ? (
              <div className="mt-2 text-[10px] text-slate-500">{footer}</div>
            ) : null}
          </div>
        );
      })() : null}
    </div>
  );
}

interface SceneContentsProps {
  prepared: PreparedGeometry;
  pointSize: number;
  viewMode: SceneDimension;
  onHover: (payload: TooltipState | null) => void;
  onSelect: (id: string, multi: boolean, toggle: boolean) => void;
  onProjectedUpdate: (points: ProjectedPoint[], rect: DOMRect) => void;
}

function SceneContents({ prepared, pointSize, viewMode, onHover, onSelect, onProjectedUpdate }: SceneContentsProps) {
  const { camera, gl, raycaster } = useThree();
  const projectionVector = useMemo(() => new THREE.Vector3(), []);
  const idsRef = useRef(prepared.ids);
  const pointsRef = useRef<THREE.Points | null>(null);
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
    let positionAttr = geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
    let colorAttr = geometry.getAttribute("color") as THREE.BufferAttribute | undefined;
    if (!positionAttr || positionAttr.count !== pointCount) {
      positionAttr = new THREE.BufferAttribute(prepared.positions, 3);
      geometry.setAttribute("position", positionAttr);
    } else {
      positionAttr.copyArray(prepared.positions);
    }
    positionAttr.needsUpdate = true;
    if (!colorAttr || colorAttr.count !== pointCount) {
      colorAttr = new THREE.BufferAttribute(prepared.colors, 3);
      geometry.setAttribute("color", colorAttr);
    } else {
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

  const computeScreenPosition = useCallback(
    (index: number, rectOverride?: DOMRect) => {
      const rect = rectOverride ?? gl.domElement.getBoundingClientRect();
      const px = (prepared.positions[index * 3] ?? 0);
      const py = (prepared.positions[index * 3 + 1] ?? 0);
      const pz = (prepared.positions[index * 3 + 2] ?? 0);
      projectionVector.set(px, py, pz).project(camera);
      const screenX = (projectionVector.x + 1) * 0.5 * rect.width;
      const screenY = (-projectionVector.y + 1) * 0.5 * rect.height;
      return { x: screenX, y: screenY, rect } as const;
    },
    [camera, gl, prepared.positions, projectionVector],
  );

  const pixelThreshold = useCallback(
    (rect: DOMRect) => Math.max(8, pointSize * rect.width * 0.02),
    [pointSize],
  );

  const handlePointerMove = useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      const index = event.index as number | undefined;
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
    },
    [computeScreenPosition, onHover, pixelThreshold],
  );

  const handlePointerLeave = useCallback(() => {
    onHover(null);
  }, [onHover]);

  const handlePointerDown = useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      if (event.button !== 0) {
        return;
      }
      const index = event.index as number | undefined;
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
    },
    [computeScreenPosition, onSelect, pixelThreshold],
  );

  useFrame(() => {
    const rect = gl.domElement.getBoundingClientRect();
    const projected: ProjectedPoint[] = [];
    for (let index = 0; index < prepared.positions.length / 3; index += 1) {
      const { x, y } = computeScreenPosition(index, rect);
      projected.push({ id: idsRef.current[index], x, y });
    }
    onProjectedUpdate(projected, rect);
  });

  const rotation: [number, number, number] = viewMode === "3d" ? [-0.12, 0.14, 0] : [0, 0, 0];

  return (
    <group rotation={rotation}>
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 5, 10]} intensity={0.6} />
      <Points
        key={`cloud-${viewMode}-${pointCount}`}
        ref={pointsRef}
        positions={prepared.positions}
        colors={prepared.colors}
        stride={3}
        frustumCulled={false}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
        onPointerDown={handlePointerDown}
      >
        <bufferGeometry ref={pointsRef} />
        <PointMaterial
          transparent
          vertexColors
          size={pointSize}
          sizeAttenuation
          depthWrite={false}
          opacity={0.92}
        />
      </Points>
      <OrbitControls
        enablePan
        enableRotate={viewMode === "3d"}
        enableZoom
        minDistance={1.5}
        maxDistance={14}
        zoomSpeed={0.8}
        dampingFactor={0.08}
      />
    </group>
  );
}

interface LassoOverlayProps {
  projectedRef: MutableRefObject<{ points: ProjectedPoint[]; rect: DOMRect } | null>;
  currentSelection: readonly string[];
  onSelect: (payload: string[] | ((current: string[]) => string[])) => void;
}

function LassoOverlay({ projectedRef, currentSelection, onSelect }: LassoOverlayProps) {
  const [drag, setDrag] = useState<{ startX: number; startY: number; currentX: number; currentY: number } | null>(null);
  const [shiftPressed, setShiftPressed] = useState(false);

  const toLocal = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const rect = projectedRef.current?.rect ?? event.currentTarget.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    },
    [projectedRef],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Shift") {
        setShiftPressed(true);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
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

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!shiftPressed || event.button !== 0) {
      return;
    }
    const local = toLocal(event);
    setDrag({ startX: local.x, startY: local.y, currentX: local.x, currentY: local.y });
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
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

  return (
    <div
      className="absolute inset-0 z-30"
      style={{ pointerEvents: shiftPressed ? "auto" : "none" }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {drag ? (
        <div
          className="pointer-events-none absolute border border-cyan-400/80 bg-cyan-400/10"
          style={{
            left: Math.min(drag.startX, drag.currentX),
            top: Math.min(drag.startY, drag.currentY),
            width: Math.abs(drag.currentX - drag.startX),
            height: Math.abs(drag.currentY - drag.startY),
          }}
        />
      ) : null}
      {shiftPressed && !drag ? (
        <div className="pointer-events-none absolute right-4 top-4 rounded-lg border border-slate-700/80 bg-slate-900/80 px-3 py-1 text-[11px] text-slate-300">
          Shift drag for lasso select
        </div>
      ) : null}
    </div>
  );
}

export const PointCloudScene = memo(function PointCloudScene({ responses, segments, segmentEdges, responseHulls }: PointCloudSceneProps) {
  const levelMode = useRunStore((state) => state.levelMode);
  if (levelMode === "segments") {
    return (
      <SegmentCloud segments={segments} edges={segmentEdges} hulls={responseHulls} />
    );
  }
  return <ResponseCloud points={responses} />;
});

const ResponseCloud = memo(function ResponseCloud({ points }: { points: ResponsePoint[] }) {
  const {
    viewMode,
    pointSize,
    clusterVisibility,
    clusterPalette,
    selectedPointIds,
    hoveredPointId,
    setHoveredPoint,
    setSelectedPoints,
    showDensity,
    spreadFactor,
    focusedResponseId,
    hoveredClusterLabel,
  } = useRunStore((state) => ({
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

  const visiblePoints = useMemo(
    () =>
      points.filter((point) => {
        const key = String(point.cluster ?? -1);
        const visible = clusterVisibility[key];
        return visible !== false && !point.hidden;
      }),
    [points, clusterVisibility],
  );

  const scales = useMemo(() => derivePositionScales(visiblePoints, spreadFactor), [visiblePoints, spreadFactor]);

  const renderTooltip = useCallback(
    (point: ResponsePoint): TooltipContent => ({
      title: `Sample #${point.index}`,
      body: point.text_preview ?? "",
    }),
    [],
  );

  return (
    <BaseCloud
      points={visiblePoints}
      selectedIds={selectedPointIds}
      hoveredId={hoveredPointId}
      onHoverChange={setHoveredPoint}
      onSelectChange={setSelectedPoints}
      viewMode={viewMode}
      pointSize={pointSize}
      clusterPalette={clusterPalette}
      focusPredicate={focusedResponseId ? (point) => point.id === focusedResponseId : undefined}
      showDensity={showDensity}
      renderTooltip={renderTooltip}
      scales={scales}
      highlightedCluster={hoveredClusterLabel ?? null}
    />
  );
});

interface SegmentCloudProps {
  segments: SegmentPoint[];
  edges: SegmentEdge[];
  hulls: ResponseHull[];
}

const SegmentCloud = memo(function SegmentCloud({ segments, edges, hulls }: SegmentCloudProps) {
  const {
    viewMode,
    pointSize,
    clusterVisibility,
    clusterPalette,
    roleVisibility,
    selectedSegmentIds,
    hoveredSegmentId,
    setHoveredSegment,
    setSelectedSegments,
    showDensity,
    showEdges,
    showParentThreads,
    spreadFactor,
    focusedResponseId,
    setFocusedResponse,
    hoveredClusterLabel,
    showNeighborSpokes,
    graphEdgeK,
    showDuplicatesOnly,
  } = useRunStore((state) => ({
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

  const [hoverProbeId, setHoverProbeId] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!hoveredSegmentId) {
      const timeout = window.setTimeout(() => setHoverProbeId(undefined), 120);
      return () => window.clearTimeout(timeout);
    }
    const timeout = window.setTimeout(
      () => setHoverProbeId(hoveredSegmentId),
      120,
    );
    return () => window.clearTimeout(timeout);
  }, [hoveredSegmentId]);

  const neighborCount = Math.max(3, Math.min(12, graphEdgeK));

  const {
    data: hoveredContext,
    isFetching: contextLoading,
  } = useSegmentContext(hoverProbeId, {
    enabled: Boolean(hoverProbeId),
    k: neighborCount,
    staleTimeMs: 180_000,
  });

  const roleFiltered = useMemo(
    () => filterSegmentsByRole(segments, roleVisibility),
    [segments, roleVisibility],
  );

  const duplicatesFiltered = useMemo(
    () => (showDuplicatesOnly ? roleFiltered.filter((segment) => segment.is_duplicate) : roleFiltered),
    [roleFiltered, showDuplicatesOnly],
  );

  const sequenceRatios = useMemo(() => {
    const maxByResponse = new Map<string, number>();
    segments.forEach((segment) => {
      const current = maxByResponse.get(segment.response_id);
      const nextValue = Math.max(current ?? Number.NEGATIVE_INFINITY, segment.position);
      maxByResponse.set(segment.response_id, nextValue);
    });
    const ratios = new Map<string, number>();
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

  const [startColor, midColor, endColor] = useMemo(
    () => [new THREE.Color("#16a34a"), new THREE.Color("#2563eb"), new THREE.Color("#dc2626")],
    [],
  );

  const visibleSegments = useMemo(
    () =>
      duplicatesFiltered.filter((segment) => {
        const key = String(segment.cluster ?? -1);
        const visible = clusterVisibility[key];
        return visible !== false;
      }),
    [duplicatesFiltered, clusterVisibility],
  );

  const scales = useMemo(() => derivePositionScales(visibleSegments, spreadFactor), [visibleSegments, spreadFactor]);

  const colorize = useCallback(
    ({ point, color }: { point: SegmentPoint; color: THREE.Color }) => {
      const ratio = sequenceRatios.get(point.id) ?? 0;
      const clampRatio = Math.max(0, Math.min(1, ratio));
      const midPoint = 0.5;
      if (clampRatio <= midPoint) {
        const t = clampRatio / midPoint;
        color.copy(startColor).lerp(midColor, t);
      } else {
        const t = (clampRatio - midPoint) / midPoint;
        color.copy(midColor).lerp(endColor, t);
      }
    },
    [sequenceRatios, startColor, midColor, endColor],
  );

  const renderTooltip = useCallback(
    (segment: SegmentPoint): TooltipContent => {
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
        body: (
          <>
            <p className="text-[11px] text-slate-200">{truncated}</p>
            {(segment.is_cached || segment.is_duplicate) && (
              <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
                {segment.is_cached ? (
                  <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-2 py-[1px] text-emerald-200">
                    Cached
                  </span>
                ) : null}
                {segment.is_duplicate ? (
                  <span className="rounded-full border border-amber-400/40 bg-amber-500/10 px-2 py-[1px] text-amber-200">
                    Duplicate
                  </span>
                ) : null}
              </div>
            )}
            {topTerms.length ? (
              <div className="mt-2 flex flex-wrap gap-1">
                {topTerms.map((term) => (
                  <span
                    key={`${segment.id}-${term.term}`}
                    className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-2 py-[1px] text-[10px] text-cyan-100"
                    title={`TF-IDF weight ${term.weight.toFixed(2)}`}
                  >
                    {term.term}
                  </span>
                ))}
              </div>
            ) : null}
            {exemplarPreview ? (
              <p className="mt-2 text-[10px] text-slate-400">
                Closest exemplar · {exemplarPreview.slice(0, 80)}
                {exemplarPreview.length > 80 ? "…" : ""}
              </p>
            ) : null}
            <div className="mt-2 space-y-1 text-[10px] text-slate-400">
              {why.sim_to_exemplar != null ? (
                <p>Sim to exemplar {why.sim_to_exemplar.toFixed(2)}</p>
              ) : null}
              {why.sim_to_nn != null ? (
                <p>Sim to nearest {why.sim_to_nn.toFixed(2)}</p>
              ) : null}
            </div>
            {neighborSummary.length ? (
              <div className="mt-2 space-y-1 text-[10px] text-slate-400">
                <p>Nearest neighbours</p>
                {neighborSummary.map((neighbor) => (
                  <p key={neighbor.id} className="text-slate-300">
                    <span className="text-cyan-200">{neighbor.similarity.toFixed(2)}</span> · {neighbor.text}
                  </p>
                ))}
              </div>
            ) : null}
          </>
        ),
        footer: contextLoading ? "Refreshing context…" : undefined,
      };
    },
    [contextLoading, hoveredContext, hoveredSegmentId, neighborPreview],
  );

  const overlay = useCallback((
    { points, scales: overlayScales, viewMode: overlayView, geometry }: OverlayContext<SegmentPoint>,
  ) => {
    if (!points.length) {
      return null;
    }
    const segmentMap = new Map(points.map((segment) => [segment.id, segment]));
    return (
      <>
        {showEdges ? (
          <SegmentEdgesMesh
            edges={edges}
            segments={segmentMap}
            viewMode={overlayView}
            geometry={geometry}
          />
        ) : null}
        {showParentThreads ? (
          <ParentThreadsMesh
            segments={segmentMap}
            viewMode={overlayView}
            geometry={geometry}
            focusedResponseId={focusedResponseId}
          />
        ) : null}
        {showParentThreads ? (
          <ResponseHullMesh hulls={hulls} viewMode={overlayView} scales={overlayScales} />
        ) : null}
        {hoveredContext && neighborSpokeTargets.length
          ? (
              <NeighborSpokesMesh
                rootId={hoveredContext.segment_id}
                neighbors={neighborSpokeTargets}
                segments={segmentMap}
                viewMode={overlayView}
                geometry={geometry}
              />
            )
          : null}
      </>
    );
  }, [edges, hulls, showEdges, showParentThreads, scales, focusedResponseId, hoveredContext, neighborSpokeTargets]);

  return (
    <BaseCloud
      points={visibleSegments}
      selectedIds={selectedSegmentIds}
      hoveredId={hoveredSegmentId}
      onHoverChange={setHoveredSegment}
      onSelectChange={setSelectedSegments}
      viewMode={viewMode}
      pointSize={pointSize * 0.9}
      clusterPalette={clusterPalette}
      showDensity={showDensity}
      renderTooltip={renderTooltip}
      colorize={colorize}
      scales={scales}
      overlay={overlay}
      highlightedCluster={hoveredClusterLabel ?? null}
    />
  );
});


interface SegmentEdgesMeshProps {
  edges: SegmentEdge[];
  segments: Map<string, SegmentPoint>;
  viewMode: SceneDimension;
  geometry: PreparedGeometry;
}



function useBufferAttributeUpdate(array: Float32Array | null) {
  const attributeRef = useRef<THREE.BufferAttribute | null>(null);

  useEffect(() => {
    if (!array || !attributeRef.current) {
      return;
    }
    attributeRef.current.needsUpdate = true;
  }, [array]);

  return attributeRef;
}

interface LineSegmentsPrimitiveProps {
  positions: Float32Array;
  color: string;
  opacity: number;
  depthWrite?: boolean;
}

const LineSegmentsPrimitive = memo(function LineSegmentsPrimitive({
  positions,
  color,
  opacity,
  depthWrite,
}: LineSegmentsPrimitiveProps) {
  const attributeRef = useBufferAttributeUpdate(positions);
  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute
          ref={attributeRef}
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} opacity={opacity} transparent depthWrite={depthWrite ?? undefined} />
    </lineSegments>
  );
});

interface LineLoopPrimitiveProps {
  positions: Float32Array;
  color: string;
  opacity: number;
}

const LineLoopPrimitive = memo(function LineLoopPrimitive({ positions, color, opacity }: LineLoopPrimitiveProps) {
  const attributeRef = useBufferAttributeUpdate(positions);
  return (
    <lineLoop>
      <bufferGeometry>
        <bufferAttribute
          ref={attributeRef}
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} opacity={opacity} transparent />
    </lineLoop>
  );
});

const SegmentEdgesMesh = memo(function SegmentEdgesMesh({ edges, segments, viewMode, geometry }: SegmentEdgesMeshProps) {
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

  return <LineSegmentsPrimitive positions={positions} color="#38bdf8" opacity={0.12} />;
});

interface NeighborSpokesMeshProps {
  rootId: string;
  neighbors: Array<{ id: string; similarity: number }>;
  segments: Map<string, SegmentPoint>;
  viewMode: SceneDimension;
  geometry: PreparedGeometry;
}

const NeighborSpokesMesh = memo(function NeighborSpokesMesh({
  rootId,
  neighbors,
  segments,
  viewMode,
  geometry,
}: NeighborSpokesMeshProps) {
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

  return <LineSegmentsPrimitive positions={positions} color="#facc15" opacity={0.45} depthWrite={false} />;
});

interface ParentThreadsMeshProps {
  segments: Map<string, SegmentPoint>;
  viewMode: SceneDimension;
  geometry: PreparedGeometry;
  focusedResponseId?: string;
}

const ParentThreadsMesh = memo(function ParentThreadsMesh({
  segments,
  viewMode,
  geometry,
  focusedResponseId,
}: ParentThreadsMeshProps) {
  const { focusedLines, fadedLines } = useMemo(() => {
    const byResponse = new Map<string, SegmentPoint[]>();
    segments.forEach((segment) => {
      const list = byResponse.get(segment.response_id) ?? [];
      list.push(segment);
      byResponse.set(segment.response_id, list);
    });

    const focus: Array<{ id: string; data: Float32Array }> = [];
    const faded: Array<{ id: string; data: Float32Array }> = [];
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
      } else {
        faded.push({ id: responseId, data: slice });
      }
    });

    return { focusedLines: focus, fadedLines: faded };
  }, [segments, viewMode, geometry, focusedResponseId]);

  if (!focusedLines.length && !fadedLines.length) {
    return null;
  }

  return (
    <group>
      {fadedLines.map(({ data }, index) => (
        <LineSegmentsPrimitive key={`faded-thread-${index}`} positions={data} color="#fcd34d" opacity={0.06} />
      ))}
      {focusedLines.map(({ data, id }) => (
        <LineSegmentsPrimitive key={`focused-thread-${id}`} positions={data} color="#fcd34d" opacity={0.28} />
      ))}
    </group>
  );
});


interface ResponseHullMeshProps {
  hulls: ResponseHull[];
  viewMode: SceneDimension;
  scales: PositionScales;
}

const ResponseHullMesh = memo(function ResponseHullMesh({ hulls, viewMode, scales }: ResponseHullMeshProps) {
  const loops = useMemo(() => {
    if (!hulls.length) {
      return [] as Float32Array[];
    }
    const { scale3d, scale2d, center3d, center2d } = scales;
    const result: Float32Array[] = [];
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
        } else {
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

  return (
    <group>
      {loops.map((positions, index) => (
        <LineLoopPrimitive key={`response-hull-${index}`} positions={positions} color="#22d3ee" opacity={0.15} />
      ))}
    </group>
  );
});










export {
  BaseCloud,
  derivePositionScales,
  LineSegmentsPrimitive,
  SegmentEdgesMesh,
  ParentThreadsMesh,
  ResponseHullMesh,
};

export type { CloudPoint, PreparedGeometry };
