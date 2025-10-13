/**
 * Semantic Landscape Sampler root layout.
 *
 * Phase 2 shell: collapsible navigation, layered top bar, inspector tabs, and popover controls.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useRunWorkflow } from "@/hooks/useRunWorkflow";

import { ClusterLegend } from "@/components/ClusterLegend";
import { ClusterTuningPanel } from "@/components/ClusterTuningPanel";
import { ExportPanel } from "@/components/ExportPanel";
import { ControlsPanel } from "@/components/ControlsPanel";
import { FooterStatusBar } from "@/components/FooterStatusBar";
import { InspectorPanel, type InspectorTab } from "@/components/InspectorPanel";
import { LayersPopover } from "@/components/LayersPopover";
import { NavigationRail, type NavigationRailItem } from "@/components/NavigationRail";
import { CommandPalette, type CommandPaletteCommand } from "@/components/CommandPalette";
import { PointCloudScene } from "@/components/PointCloudScene";
import { ProgressToast } from "@/components/ProgressToast";
import { RunHistoryDrawer } from "@/components/RunHistoryDrawer";
import { SlideOver } from "@/components/SlideOver";
import { TopBar } from "@/components/TopBar";
import { CompareView } from "@/components/CompareView";
import { useRunStore } from "@/store/runStore";
import { useComparisonStore } from "@/store/comparisonStore";
import type { ExportInclude } from "@/types/run";

export default function App() {
  const {
    results,
    isGenerating,
    segmentEdges,
    exportFormat,
    exportIncludeProvenance,
    levelMode,
    viewportBounds,
    setHistoryOpen,
    showDensity,
    setShowDensity,
    showEdges,
    setShowEdges,
    showParentThreads,
    setShowParentThreads,
    showPerformanceStats,
    setShowPerformanceStats,
    currentRunId,
    runHistory,
  } = useRunStore((state) => ({
    results: state.results,
    isGenerating: state.isGenerating,
    segmentEdges: state.segmentEdges,
    exportFormat: state.exportFormat,
    exportIncludeProvenance: state.exportIncludeProvenance,
    levelMode: state.levelMode,
    viewportBounds: state.viewportBounds,
    setHistoryOpen: state.setHistoryOpen,
    showDensity: state.showDensity,
    setShowDensity: state.setShowDensity,
    showEdges: state.showEdges,
    setShowEdges: state.setShowEdges,
    showParentThreads: state.showParentThreads,
    setShowParentThreads: state.setShowParentThreads,
    showPerformanceStats: state.showPerformanceStats,
    setShowPerformanceStats: state.setShowPerformanceStats,
    currentRunId: state.currentRunId,
    runHistory: state.runHistory,
  }));

  const workflow = useRunWorkflow();
  const { loadRunById, isLoading, loadFromHistory } = workflow;
  const {
    config: comparisonConfig,
    setLeftRunId: setComparisonLeftRunId,
    setRightRunId: setComparisonRightRunId,
  } = useComparisonStore((state) => ({
    config: state.config,
    setLeftRunId: state.setLeftRunId,
    setRightRunId: state.setRightRunId,
  }));


  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("selection");
  const [isRunSetupOpen, setRunSetupOpen] = useState(false);
  const [isClusterTuningOpen, setClusterTuningOpen] = useState(false);
  const [isExportOpen, setExportOpen] = useState(false);
  const [isSidebarExpanded, setSidebarExpanded] = useState(false);
  const [isLayersOpen, setLayersOpen] = useState(false);
  const [activeNav, setActiveNav] = useState<string>("explore");
  const [isCommandPaletteOpen, setCommandPaletteOpen] = useState(false);

  const layersButtonRef = useRef<HTMLButtonElement | null>(null);
  const lastRunFromUrlRef = useRef<string | null>(null);
  const parsedCompareRef = useRef(false);

  const updateRunUrl = useCallback((runId: string | null) => {
    if (typeof window === "undefined") {
      return;
    }
    const url = new URL(window.location.href);
    if (runId) {
      url.searchParams.set("run", runId);
    } else {
      url.searchParams.delete("run");
    }
    window.history.replaceState(null, "", url.toString());
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || parsedCompareRef.current) {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    if (params.get("view") === "compare") {
      setActiveNav("compare");
      const leftParam = params.get("compare_left");
      const rightParam = params.get("compare_right");
      if (leftParam) {
        setComparisonLeftRunId(leftParam);
      }
      if (rightParam) {
        setComparisonRightRunId(rightParam);
      }
    }
    parsedCompareRef.current = true;
  }, [setComparisonLeftRunId, setComparisonRightRunId]);

  const navigationItems: NavigationRailItem[] = [
    {
      id: "explore",
      label: "Explore",
      hint: "Explore landscape",
      onSelect: () => {
        setActiveNav("explore");
        setInspectorTab("selection");
      },
      shortcut: "E",
    },
    {
      id: "run-setup",
      label: "Run Setup",
      hint: "Configure sampling",
      onSelect: () => {
        setActiveNav("run-setup");
        setRunSetupOpen(true);
      },
      shortcut: "R",
    },
    {
      id: "cluster-tuning",
      label: "Cluster Tuning",
      hint: "Adjust projection and clustering",
      onSelect: () => {
        setActiveNav("cluster-tuning");
        setClusterTuningOpen(true);
      },
      shortcut: "T",
    },
    {
      id: "compare",
      label: "Compare",
      hint: "Compare runs",
      onSelect: () => {
        setActiveNav("compare");
      },
      shortcut: "C",
    },
    {
      id: "layers",
      label: "Layers",
      hint: "Overlay visibility",
      onSelect: () => {
        setLayersOpen((value) => !value);
        setActiveNav((current) => (current === "layers" ? "explore" : "layers"));
      },
      shortcut: "L",
    },
    {
      id: "history",
      label: "History",
      hint: "Saved runs",
      onSelect: () => {
        setActiveNav("history");
        setHistoryOpen(true);
      },
      shortcut: "H",
    },
    {
      id: "export",
      label: "Export",
      hint: "Download data",
      onSelect: () => {
        setActiveNav("export");
        setExportOpen(true);
      },
      shortcut: "X",
    },
  ];
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const runIdParam = params.get("run");
    if (params.get("view") === "compare") {
      return;
    }
    if (!runIdParam) {
      lastRunFromUrlRef.current = null;
      return;
    }
    if (runIdParam === currentRunId) {
      lastRunFromUrlRef.current = runIdParam;
      return;
    }
    if (isLoading || lastRunFromUrlRef.current === runIdParam) {
      return;
    }
    if (typeof loadRunById !== "function") {
      return;
    }
    lastRunFromUrlRef.current = runIdParam;
    loadRunById(runIdParam).catch((error) => {
      console.error("Failed to load run from URL", error);
      lastRunFromUrlRef.current = null;
    });
  }, [currentRunId, isLoading, loadRunById]);
  useEffect(() => {
    updateRunUrl(currentRunId ?? null);
  }, [currentRunId, updateRunUrl]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const url = new URL(window.location.href);
    if (activeNav === "compare") {
      url.searchParams.set("view", "compare");
      if (comparisonConfig.leftRunId) {
        url.searchParams.set("compare_left", comparisonConfig.leftRunId);
      } else {
        url.searchParams.delete("compare_left");
      }
      if (comparisonConfig.rightRunId) {
        url.searchParams.set("compare_right", comparisonConfig.rightRunId);
      } else {
        url.searchParams.delete("compare_right");
      }
      url.searchParams.delete("run");
    } else {
      if (url.searchParams.get("view") === "compare") {
        url.searchParams.delete("view");
        url.searchParams.delete("compare_left");
        url.searchParams.delete("compare_right");
      }
    }
    window.history.replaceState(null, "", url.toString());
  }, [activeNav, comparisonConfig.leftRunId, comparisonConfig.rightRunId]);

  useEffect(() => {
    if (activeNav !== "compare") {
      return;
    }
    if (!Array.isArray(runHistory) || runHistory.length > 0) {
      return;
    }
    if (typeof loadFromHistory !== "function") {
      return;
    }
    loadFromHistory().catch((error) => {
      console.error("Failed to load run history for compare view", error);
    });
  }, [activeNav, loadFromHistory, runHistory]);



  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const editable = target?.getAttribute("contenteditable");
      const isTyping = Boolean(tag && (tag === "input" || tag === "textarea" || editable === "true"));
      const key = event.key.toLowerCase();

      if ((event.metaKey || event.ctrlKey) && key === "k") {
        event.preventDefault();
        setCommandPaletteOpen((value) => !value);
        return;
      }

      if (isTyping) {
        return;
      }

      if (event.shiftKey && !event.metaKey && !event.ctrlKey) {
        if (key === "d") {
          event.preventDefault();
          setShowDensity(!showDensity);
        } else if (key === "e") {
          event.preventDefault();
          setShowEdges(!showEdges);
        } else if (key === "p") {
          event.preventDefault();
          setShowParentThreads(!showParentThreads);
        } else if (key === "f") {
          event.preventDefault();
          setShowPerformanceStats(!showPerformanceStats);
        }
      }
    };

    document.addEventListener("keydown", handler);
    return () => {
      document.removeEventListener("keydown", handler);
    };
  }, [showDensity, setShowDensity, showEdges, setShowEdges, showParentThreads, setShowParentThreads, showPerformanceStats, setShowPerformanceStats]);



  const commands = useMemo((): CommandPaletteCommand[] => {
    const mode = levelMode === "segments" ? "segments" : "responses";
    const include: ExportInclude[] | undefined = exportIncludeProvenance ? ["provenance"] : undefined;

    return [
      {
        id: "generate",
        label: "Generate landscape",
        description: "Create a fresh semantic landscape with current parameters",
        onTrigger: () => {
          workflow.generate();
        },
      },
      {
        id: "run-setup",
        label: "Open run setup",
        description: "Adjust sampling, segmentation, and embeddings",
        onTrigger: () => {
          setRunSetupOpen(true);
          setActiveNav("run-setup");
        },
      },
      {
        id: "cluster-tuning",
        label: "Open cluster tuning",
        description: "Retune clustering without re-sampling",
        onTrigger: () => {
          setClusterTuningOpen(true);
          setActiveNav("cluster-tuning");
        },
      },
      {
        id: "toggle-density",
        label: showDensity ? "Hide density overlay" : "Show density overlay",
        description: "Toggle the kernel density heatmap",
        onTrigger: () => {
          setShowDensity(!showDensity);
        },
      },
      {
        id: "toggle-edges",
        label: showEdges ? "Hide similarity edges" : "Show similarity edges",
        description: "Toggle similarity graph edges",
        onTrigger: () => {
          setShowEdges(!showEdges);
        },
      },
      {
        id: "toggle-parents",
        label: showParentThreads ? "Hide parent threads" : "Show parent threads",
        description: "Toggle response lineage spokes",
        onTrigger: () => {
          setShowParentThreads(!showParentThreads);
        },
      },
      {
        id: "toggle-stats",
        label: showPerformanceStats ? "Hide performance stats" : "Show performance stats",
        description: "Toggle the Three.js performance overlay",
        hint: "Shift+F",
        onTrigger: () => {
          setShowPerformanceStats(!showPerformanceStats);
        },
      },
      {
        id: "inspector-selection",
        label: "Focus inspector on Selection",
        description: "Switch inspector to the selection tab",
        onTrigger: () => {
          setInspectorTab("selection");
        },
      },
      {
        id: "inspector-analytics",
        label: "Focus inspector on Analytics",
        description: "Review projection and cluster diagnostics",
        onTrigger: () => {
          setInspectorTab("analytics");
        },
      },
      {
        id: "inspector-history",
        label: "Focus inspector on History",
        description: "Open notes and provenance",
        onTrigger: () => {
          setInspectorTab("history");
        },
      },
      {
        id: "open-history",
        label: "Open saved runs drawer",
        description: "Browse previously sampled landscapes",
        onTrigger: () => {
          setHistoryOpen(true);
          setActiveNav("history");
        },
      },
      {
        id: "open-export",
        label: "Open export panel",
        description: "Configure scope, format, and includes",
        onTrigger: () => {
          setExportOpen(true);
          setActiveNav("export");
        },
      },
      {
        id: "export-run",
        label: "Export run (" + exportFormat.toUpperCase() + ")",
        description: "Download the current landscape payload",
        onTrigger: () =>
          workflow.exportDataset({
            scope: "run",
            format: exportFormat,
            mode,
            include,
          }),
      },
    ];
  }, [
    exportFormat,
    exportIncludeProvenance,
    levelMode,
    setActiveNav,
    setClusterTuningOpen,
    setExportOpen,
    setHistoryOpen,
    setInspectorTab,
    setRunSetupOpen,
    setShowDensity,
    setShowEdges,
    setShowParentThreads,
    showDensity,
    showEdges,
    showParentThreads,
    workflow,
  ]);

  const hasData = useMemo(
    () => (results?.points.length ?? 0) > 0 || (results?.segments.length ?? 0) > 0,
    [results],
  );

  const placeholder = useMemo(
    () => (
      <div className="flex h-full items-center justify-center text-sm text-muted">
        <div className="max-w-sm text-center">
          <p className="font-semibold text-text">Awaiting your first landscape</p>
          <p className="mt-2 text-text-dim">
            Enter a prompt and hit "Generate" to sample, embed, and visualise LLM responses in an interactive point cloud.
          </p>
        </div>
      </div>
    ),
    [],
  );

  const isCompareView = activeNav === "compare";

  const handleCloseRunSetup = () => {
    setRunSetupOpen(false);
    setActiveNav("explore");
  };

  const handleCloseClusterTuning = () => {
    setClusterTuningOpen(false);
    setActiveNav("explore");
  };

  const handleCloseExport = () => {
    setExportOpen(false);
    setActiveNav("explore");
  };

  const handleCloseLayers = () => {
    setLayersOpen(false);
    setActiveNav((current) => (current === "layers" ? "explore" : current));
  };

  const handleOpenNotes = () => {
    setInspectorTab("history");
  };

  return (
    <div className="flex h-screen flex-col bg-bg text-text">
      <TopBar
        onOpenSavedRuns={() => setHistoryOpen(true)}
        onOpenRunSetup={() => {
          setRunSetupOpen(true);
          setActiveNav("run-setup");
        }}
        onOpenExport={() => {
          setExportOpen(true);
          setActiveNav("export");
        }}
        onOpenNotes={handleOpenNotes}
        onRequestShare={() => console.info("Share coming soon")}
        onRequestSave={() => console.info("Save workflow pending")}
        onToggleLayers={() => {
          setLayersOpen((value) => !value);
          setActiveNav((current) => (current === "layers" ? "explore" : "layers"));
        }}
        layersButtonRef={layersButtonRef}
        onOpenCommandPalette={() => setCommandPaletteOpen(true)}
      />

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <NavigationRail
          items={navigationItems}
          activeId={activeNav}
          expanded={isSidebarExpanded}
          onExpandedChange={setSidebarExpanded}
        />

        <div className="flex min-h-0 flex-1 overflow-hidden">
          <main className="flex min-h-0 flex-1 flex-col overflow-hidden px-6 py-6">
            <div className="flex min-h-0 flex-1 gap-6 overflow-hidden">
              {isCompareView ? (
                <CompareView />
              ) : (
                <>
                  <section className="flex min-h-0 flex-1 flex-col gap-4">
                    <div className="relative flex min-h-0 flex-1 overflow-hidden rounded-2xl border border-border bg-panel shadow-panel">
                      {hasData && results ? (
                        <PointCloudScene
                          responses={results.points}
                          segments={results.segments}
                          segmentEdges={segmentEdges}
                          responseHulls={results.response_hulls}
                        />
                      ) : (
                        placeholder
                      )}
                      {isGenerating ? (
                        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-bg/60">
                          <div className="animate-pulse rounded-full border border-accent/40 px-6 py-2 text-sm text-accent">
                            Sampling responses...
                          </div>
                        </div>
                      ) : null}
                    </div>
                    <ClusterLegend
                      responseClusters={results?.clusters ?? []}
                      segmentClusters={results?.segment_clusters ?? []}
                      onExportCluster={async ({ label, mode }) => {
                        try {
                          await workflow.exportDataset({ scope: "cluster", clusterId: label, mode });
                        } catch (error) {
                          console.error(error);
                        }
                      }}
                      exportDisabled={workflow.isLoading || isGenerating}
                      formatLabel={exportFormat.toUpperCase()}
                    />
                  </section>

                  <InspectorPanel activeTab={inspectorTab} onSelectTab={setInspectorTab} workflow={workflow} />
                </>
              )}
            </div>
          </main>
        </div>
      </div>

      <FooterStatusBar />

      <LayersPopover anchorRef={layersButtonRef} open={isLayersOpen} onClose={handleCloseLayers} />

      <SlideOver
        open={isRunSetupOpen}
        onClose={handleCloseRunSetup}
        title="Run setup"
        description="Configure sampling, segmentation, and embeddings"
      >
        <ControlsPanel />
      </SlideOver>

      <SlideOver
        open={isClusterTuningOpen}
        onClose={handleCloseClusterTuning}
        title="Cluster tuning"
        description="Adjust projection and clustering parameters"
        widthClassName="w-[420px]"
      >
        <ClusterTuningPanel />
      </SlideOver>

      <SlideOver
        open={isExportOpen}
        onClose={handleCloseExport}
        title="Export"
        description="Select scope and format"
        widthClassName="w-[420px]"
      >
        <ExportPanel workflow={workflow} onComplete={handleCloseExport} />
      </SlideOver>

      <CommandPalette
        open={isCommandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        commands={commands}
      />
      <RunHistoryDrawer workflow={workflow} />
      {isGenerating ? (
        <div className="pointer-events-none fixed inset-x-0 top-14 bottom-0 z-40 bg-bg/80 backdrop-blur-sm transition-opacity" />
      ) : null}
      <ProgressToast />
    </div>
  );
}
