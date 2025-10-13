/**
 * Displays provenance metadata for the current run with copy-to-clipboard helpers.
 */

import { memo, useMemo, useState } from "react";

import { useRunStore } from "@/store/runStore";

export const RunProvenancePanel = memo(function RunProvenancePanel() {
  const { results } = useRunStore((state) => ({ results: state.results }));
  const provenance = results?.provenance;

  const [copyMessage, setCopyMessage] = useState<string | null>(null);

  const jsonString = useMemo(() => {
    if (!provenance) {
      return "";
    }
    try {
      return JSON.stringify(provenance, null, 2);
    } catch (err) {
      console.error("Failed to serialise provenance", err);
      return "";
    }
  }, [provenance]);

  const handleCopy = async () => {
    if (!jsonString) {
      return;
    }
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopyMessage("Copied");
      setTimeout(() => setCopyMessage(null), 1500);
    } catch (err) {
      console.error("Clipboard write failed", err);
      setCopyMessage("Copy failed");
      setTimeout(() => setCopyMessage(null), 1500);
    }
  };

  if (!results) {
    return null;
  }

  return (
    <section className="glass-panel flex flex-col gap-3 rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4 text-xs text-slate-300">
      <header className="flex items-center justify-between">
        <h3 className="text-sm font-semibold tracking-wide text-slate-200">Provenance</h3>
        <button
          type="button"
          onClick={handleCopy}
          disabled={!jsonString}
          className="rounded-full border border-slate-700/60 px-3 py-1 text-[11px] uppercase tracking-wide text-slate-200 transition hover:border-cyan-400 hover:text-cyan-200 disabled:cursor-not-allowed disabled:border-slate-700/40 disabled:text-slate-500"
        >
          {copyMessage ?? "Copy JSON"}
        </button>
      </header>
      {jsonString ? (
        <pre className="max-h-64 overflow-auto rounded-lg bg-slate-900/70 p-3 text-[11px] leading-5 text-cyan-100 scrollbar-thin">
          {jsonString}
        </pre>
      ) : (
        <p className="text-[11px] text-slate-500">
          Provenance metadata will appear after a run completes.
        </p>
      )}
    </section>
  );
});

