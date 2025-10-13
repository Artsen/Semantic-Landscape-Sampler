import { useRunStore } from "@/store/runStore";

export function LayersPanel() {
  const {
    showDensity,
    setShowDensity,
    showEdges,
    setShowEdges,
    showParentThreads,
    setShowParentThreads,
    showNeighborSpokes,
    setShowNeighborSpokes,
    showDuplicatesOnly,
    setShowDuplicatesOnly,
    graphEdgeThreshold,
    setGraphEdgeThreshold,
  } = useRunStore((state) => ({
    showDensity: state.showDensity,
    setShowDensity: state.setShowDensity,
    showEdges: state.showEdges,
    setShowEdges: state.setShowEdges,
    showParentThreads: state.showParentThreads,
    setShowParentThreads: state.setShowParentThreads,
    showNeighborSpokes: state.showNeighborSpokes,
    setShowNeighborSpokes: state.setShowNeighborSpokes,
    showDuplicatesOnly: state.showDuplicatesOnly,
    setShowDuplicatesOnly: state.setShowDuplicatesOnly,
    graphEdgeThreshold: state.graphEdgeThreshold,
    setGraphEdgeThreshold: state.setGraphEdgeThreshold,
  }));

  return (
    <div className="space-y-4">
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-dim">Overlays</h3>
        <ToggleRow label="Density" value={showDensity} onChange={setShowDensity} />
        <ToggleRow label="Similarity edges" value={showEdges} onChange={setShowEdges} />
        <ToggleRow label="Parent threads" value={showParentThreads} onChange={setShowParentThreads} />
        <ToggleRow label="Neighbor spokes" value={showNeighborSpokes} onChange={setShowNeighborSpokes} />
        <ToggleRow label="Show duplicates only" value={showDuplicatesOnly} onChange={setShowDuplicatesOnly} />
      </section>
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-dim">Edge threshold</h3>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(graphEdgeThreshold * 100)}
          onChange={(event) => setGraphEdgeThreshold(Number(event.target.value) / 100)}
          className="w-full"
        />
        <p className="text-[11px] text-muted">{(graphEdgeThreshold * 100).toFixed(0)}% cosine</p>
      </section>
    </div>
  );
}

type ToggleRowProps = {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
};

function ToggleRow({ label, value, onChange }: ToggleRowProps) {
  return (
    <label className="flex items-center justify-between gap-3 text-sm text-text">
      <span>{label}</span>
      <input
        type="checkbox"
        className="h-4 w-4 cursor-pointer accent-accent"
        checked={value}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}
