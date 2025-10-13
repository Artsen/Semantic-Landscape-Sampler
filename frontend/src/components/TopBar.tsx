import { useEffect, useMemo, useState, type RefObject } from "react";

import { useRunStore } from "@/store/runStore";

type TopBarProps = {
  onOpenSavedRuns: () => void;
  onOpenRunSetup: () => void;
  onOpenExport: () => void;
  onOpenNotes: () => void;
  onRequestSave: () => void;
  onRequestShare: () => void;
  onToggleLayers: () => void;
  layersButtonRef: RefObject<HTMLButtonElement>;
  onOpenCommandPalette?: () => void;
};

function useThemePreference() {
  const [theme, setTheme] = useState<string>(() => {
    if (typeof document === "undefined") {
      return "dark";
    }
    return document.documentElement.dataset.theme || "dark";
  });

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    localStorage.setItem("sls-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const stored = localStorage.getItem("sls-theme");
    if (stored) {
      setTheme(stored);
    }
  }, []);

  return { theme, setTheme };
}

const truncate = (value: string, length = 72) => {
  if (value.length <= length) {
    return value;
  }
  return value.slice(0, length - 3) + "...";
};

export function TopBar({
  onOpenSavedRuns,
  onOpenRunSetup,
  onOpenExport,
  onOpenNotes,
  onRequestSave,
  onRequestShare,
  onToggleLayers,
  layersButtonRef,
  onOpenCommandPalette,
}: TopBarProps) {
  const { results, prompt } = useRunStore((state) => ({
    results: state.results,
    prompt: state.prompt,
  }));
  const { theme, setTheme } = useThemePreference();

  const runTitle = useMemo(() => {
    if (results?.run?.prompt) {
      return truncate(results.run.prompt, 96);
    }
    if (results?.prompt) {
      return truncate(results.prompt, 96);
    }
    if (prompt) {
      return truncate(prompt, 96);
    }
    return "Untitled landscape";
  }, [prompt, results?.prompt, results?.run?.prompt]);

  const handleCommandClick = () => {
    if (onOpenCommandPalette) {
      onOpenCommandPalette();
      return;
    }
    console.info("Command palette coming soon");
  };

  const handleThemeToggle = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <header className="flex h-[56px] items-center gap-4 border-b border-border bg-panel px-5 text-sm text-text">
      <div className="flex items-center gap-3">
        <span className="rounded-md bg-accent/10 px-2 py-1 text-xs font-semibold tracking-wide text-accent">SLS</span>
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <p className="text-sm font-semibold text-text">{runTitle}</p>
          </div>
          <p className="text-xs text-text-dim">Semantic Landscape Sampler</p>
        </div>
        <button
          type="button"
          className="rounded-lg border border-border px-3 py-1 text-xs font-medium text-text transition hover:border-accent hover:text-accent"
          onClick={onOpenSavedRuns}
        >
          Saved runs
        </button>
      </div>

      <div className="flex flex-1 items-center gap-3">
        <button
          type="button"
          onClick={handleCommandClick}
          className="flex h-10 flex-1 items-center gap-3 rounded-xl border border-border bg-panel-elev px-3 text-left text-sm text-text transition hover:border-accent"
        >
          <span className="text-xs uppercase tracking-wide text-muted">Cmd+K</span>
          <span className="flex-1 truncate text-text-dim">Search commands, runs, or segments</span>
        </button>
        <div className="flex items-center gap-2 text-xs">
          <button
            type="button"
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent"
            onClick={onOpenRunSetup}
          >
            Run setup
          </button>
          <button
            type="button"
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:border-border disabled:text-muted"
            onClick={onRequestSave}
            disabled
            title="Save coming soon"
          >
            Save
          </button>
          <button
            type="button"
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent"
            onClick={onOpenExport}
          >
            Export
          </button>
          <button
            type="button"
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent"
            onClick={onRequestShare}
          >
            Share
          </button>
          <button
            type="button"
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent"
            onClick={onOpenNotes}
          >
            Notes
          </button>
          <button
            type="button"
            ref={layersButtonRef}
            className="rounded-lg border border-border px-3 py-1 font-medium text-text transition hover:border-accent hover:text-accent"
            onClick={onToggleLayers}
          >
            Layers
          </button>
          <button
            type="button"
            className="rounded-lg border border-border px-2 py-1 text-sm transition hover:border-accent hover:text-accent"
            onClick={handleThemeToggle}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? "☀" : "🌙"}
          </button>
        </div>
      </div>
    </header>
  );
}
