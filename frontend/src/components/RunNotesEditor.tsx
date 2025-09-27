/**
 * Compact editor for annotating the currently loaded run with free-form notes.
 */

import { memo, useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";

import { updateRun } from "@/services/api";
import { useRunStore } from "@/store/runStore";

export const RunNotesEditor = memo(function RunNotesEditor() {
  const { results, currentRunId, setRunNotes, isGenerating } = useRunStore((state) => ({
    results: state.results,
    currentRunId: state.currentRunId,
    setRunNotes: state.setRunNotes,
    isGenerating: state.isGenerating,
  }));

  const run = results?.run;
  const runId = run?.id ?? currentRunId;
  const [draft, setDraft] = useState(run?.notes ?? "");
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);

  useEffect(() => {
    setDraft(run?.notes ?? "");
    setLastSavedAt(null);
  }, [run?.id, run?.notes]);

  const mutation = useMutation({
    mutationKey: ["update-run-notes"],
    mutationFn: async (note: string) => {
      if (!runId) {
        throw new Error("No run selected");
      }
      const normalised = note.replace(/\r\n/g, "\n").trim();
      const payload = { notes: normalised.length ? normalised : null };
      return updateRun(runId, payload);
    },
    onSuccess: (resource) => {
      setRunNotes(resource.id, resource.notes ?? null);
      setDraft(resource.notes ?? "");
      setLastSavedAt(Date.now());
    },
  });

  if (!run || !runId) {
    return null;
  }

  const normalisedDraft = draft.replace(/\r\n/g, "\n");
  const cleanedDraft = normalisedDraft.trim();
  const baseline = run.notes ?? "";
  const cleanedBaseline = baseline.replace(/\r\n/g, "\n").trim();
  const hasChanges = cleanedDraft !== cleanedBaseline;
  const isBusy = mutation.isPending;
  const disableActions = isBusy || isGenerating;

  const handleSave = () => {
    if (!hasChanges || disableActions) {
      return;
    }
    mutation.mutate(draft);
  };

  const handleClear = () => {
    if (!baseline && !draft) {
      return;
    }
    setDraft("");
    mutation.mutate("");
  };

  const statusMessage = mutation.isError
    ? mutation.error instanceof Error
      ? mutation.error.message
      : "Failed to save notes"
    : mutation.isPending
    ? "Saving notes..."
    : lastSavedAt
    ? "Notes saved"
    : hasChanges
    ? "Unsaved changes"
    : "";

  return (
    <section className="glass-panel rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-4 text-sm text-slate-200">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.22rem] text-slate-500">Run notes</p>
          <h3 className="text-base font-semibold text-slate-100">Personal annotations</h3>
        </div>
        <span className="text-[11px] uppercase tracking-wide text-slate-500">
          {timeFormatter.format(new Date(run.updated_at))}
        </span>
      </header>

      <textarea
        value={draft}
        onChange={(event) => {
          setDraft(event.target.value);
          setLastSavedAt(null);
        }}
        rows={4}
        className="mt-3 w-full resize-none rounded-xl border border-slate-800/60 bg-slate-900/50 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
        placeholder="Jot down why this run matters, anomalies to revisit, or follow-up ideas."
        disabled={disableActions}
      />

      <div className="mt-3 flex items-center justify-between text-[11px]">
        <span className="text-slate-400">{statusMessage}</span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleClear}
            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-rose-400 hover:text-rose-200 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={disableActions || (!baseline && !draft)}
          >
            Clear
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="rounded-full border border-slate-700/60 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-100 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={disableActions || !hasChanges}
          >
            Save notes
          </button>
        </div>
      </div>
    </section>
  );
});

const timeFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

