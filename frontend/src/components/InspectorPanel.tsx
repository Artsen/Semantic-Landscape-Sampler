/**
 * Right-hand inspector that surfaces selection details, analytics, and run history.
 */

import clsx from "clsx";
import { memo } from "react";

import type { RunWorkflow } from "@/hooks/useRunWorkflow";

import { ClusterMetricsPanel } from "@/components/ClusterMetricsPanel";
import { PointDetailsPanel } from "@/components/PointDetailsPanel";
import { ProjectionQualityPanel } from "@/components/ProjectionQualityPanel";
import { ProcessingTimelinePanel } from "@/components/ProcessingTimelinePanel";
import { RunNotesEditor } from "@/components/RunNotesEditor";
import { RunProvenancePanel } from "@/components/RunProvenancePanel";

export type InspectorTab = "selection" | "analytics" | "history";

type InspectorPanelProps = {
  activeTab: InspectorTab;
  onSelectTab: (tab: InspectorTab) => void;
  workflow: RunWorkflow;
};

const TAB_METADATA: Array<{ id: InspectorTab; label: string; description: string; shortcut?: string }> = [
  { id: "selection", label: "Selection", description: "Focused responses, segments, and actions", shortcut: "1" },
  { id: "analytics", label: "Analytics", description: "Cluster quality, projection gauges, metrics", shortcut: "2" },
  { id: "history", label: "History", description: "Notes, provenance, and change log", shortcut: "3" },
];

export const InspectorPanel = memo(function InspectorPanel({ activeTab, onSelectTab, workflow }: InspectorPanelProps) {
  return (
    <aside className="flex w-[360px] min-w-[320px] max-w-[400px] flex-col gap-4 overflow-hidden border-l border-border bg-panel">
      <header className="flex flex-col gap-2 border-b border-border px-5 py-4">
        <h2 className="text-xs font-semibold uppercase tracking-[0.32rem] text-muted">Inspector</h2>
        <nav className="flex gap-2">
          {TAB_METADATA.map((tab) => {
            const isActive = tab.id === activeTab;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onSelectTab(tab.id)}
                className={clsx(
                  "flex flex-1 flex-col rounded-xl border px-3 py-2 text-left transition",
                  isActive
                    ? "border-accent bg-accent/10 text-text"
                    : "border-transparent bg-panel-elev text-muted hover:border-border hover:text-text",
                )}
              >
                <span className="flex items-center justify-between text-sm font-medium">
                  {tab.label}
                  {tab.shortcut ? <span className="text-[10px] text-muted">{tab.shortcut}</span> : null}
                </span>
                <span className="mt-1 text-[11px] text-text-dim">{tab.description}</span>
              </button>
            );
          })}
        </nav>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-6 pt-2">
        {activeTab === "selection" ? <PointDetailsPanel workflow={workflow} /> : null}
        {activeTab === "analytics" ? (
          <div className="flex flex-col gap-4">
            <ProjectionQualityPanel />
            <ProcessingTimelinePanel />
            <ClusterMetricsPanel />
          </div>
        ) : null}
        {activeTab === "history" ? (
          <div className="flex flex-col gap-4">
            <RunNotesEditor />
            <RunProvenancePanel />
          </div>
        ) : null}
      </div>
    </aside>
  );
});
