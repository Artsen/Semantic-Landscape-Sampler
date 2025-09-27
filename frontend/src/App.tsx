/**
 * Semantic Landscape Sampler root layout.
 *
 * Components:
 *  - App: Hosts the control panel, point cloud scene, and detail panes while wiring run-store state.
 */

import { useMemo } from "react";

import { useRunWorkflow } from "@/hooks/useRunWorkflow";

import { ClusterLegend } from "@/components/ClusterLegend";
import { ControlsPanel } from "@/components/ControlsPanel";
import { RunHistoryDrawer } from "@/components/RunHistoryDrawer";
import { PointCloudScene } from "@/components/PointCloudScene";
import { RunMetadataBar } from "@/components/RunMetadataBar";
import { RunNotesEditor } from "@/components/RunNotesEditor";
import { PointDetailsPanel } from "@/components/PointDetailsPanel";
import { useRunStore } from "@/store/runStore";

export default function App() {
  const { results, isGenerating } = useRunStore((state) => ({
    results: state.results,
    isGenerating: state.isGenerating,
  }));
  const workflow = useRunWorkflow();

  const hasData = useMemo(
    () => (results?.points.length ?? 0) > 0 || (results?.segments.length ?? 0) > 0,
    [results],
  );

  const placeholder = useMemo(
    () => (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        <div className="max-w-sm text-center">
          <p className="font-semibold text-slate-200">Awaiting your first landscape</p>
          <p className="mt-2 text-slate-400">
            Enter a prompt and hit “Generate” to sample, embed, and visualize LLM responses in a semantic point cloud.
          </p>
        </div>
      </div>
    ),
    [],
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="relative shrink-0">
        <ControlsPanel workflow={workflow} />
        <RunHistoryDrawer workflow={workflow} />
      </div>
      <main className="flex flex-1 flex-col gap-4 p-4">
        <RunMetadataBar />
        <RunNotesEditor />
        <section className="flex h-full gap-4">
          <div className="relative flex-1">
            <div className="glass-panel h-full w-full overflow-hidden rounded-2xl border border-slate-800/50">
              {hasData && results ? (
                <PointCloudScene
                  responses={results.points}
                  segments={results.segments}
                  segmentEdges={results.segment_edges}
                  responseHulls={results.response_hulls}
                />
              ) : (
                placeholder
              )}
              {isGenerating ? (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-slate-950/40 backdrop-blur-sm">
                  <div className="animate-pulse rounded-full border border-cyan-400/60 px-6 py-2 text-sm text-cyan-200">
                    Sampling responses…
                  </div>
                </div>
              ) : null}
            </div>
          </div>
          <PointDetailsPanel />
        </section>
        <ClusterLegend
          responseClusters={results?.clusters ?? []}
          segmentClusters={results?.segment_clusters ?? []}
        />
      </main>
    </div>
  );
}

