/**
 * Command palette with fuzzy filtering and keyboard navigation.
 */

import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

type CommandPaletteCommand = {
  id: string;
  label: string;
  description?: string;
  hint?: string;
  onTrigger: () => void | Promise<void>;
};

type CommandPaletteProps = {
  open: boolean;
  commands: CommandPaletteCommand[];
  onClose: () => void;
};

const normalize = (value: string) => value.trim().toLowerCase();

const matchesQuery = (value: string, query: string) => {
  if (!query) {
    return true;
  }
  return normalize(value).includes(query);
};

export function CommandPalette({ open, commands, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const normalisedQuery = normalize(query);

  const filtered = useMemo(() => {
    if (!normalisedQuery) {
      return commands;
    }
    return commands.filter((command) =>
      matchesQuery(command.label, normalisedQuery) || matchesQuery(command.description ?? "", normalisedQuery),
    );
  }, [commands, normalisedQuery]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setQuery("");
    setHighlightedIndex(0);
    const node = inputRef.current;
    const timeout = window.setTimeout(() => {
      node?.focus();
    }, 16);
    return () => {
      window.clearTimeout(timeout);
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (!filtered.length) {
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setHighlightedIndex((index) => (index + 1) % filtered.length);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setHighlightedIndex((index) => (index - 1 + filtered.length) % filtered.length);
      } else if (event.key === "Enter") {
        event.preventDefault();
        const command = filtered[highlightedIndex];
        if (command) {
          Promise.resolve(command.onTrigger()).catch((error) => {
            console.error("Command execution failed", error);
          });
          onClose();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [filtered, highlightedIndex, onClose, open]);

  useEffect(() => {
    setHighlightedIndex(0);
  }, [normalisedQuery]);

  if (!open) {
    return null;
  }

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 py-20 backdrop-blur-sm">
      <div className="w-full max-w-xl overflow-hidden rounded-2xl border border-border bg-panel shadow-panel">
        <div className="border-b border-border px-4 py-3">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="What do you need?"
            className="w-full border-none bg-transparent text-sm text-text placeholder:text-muted focus:outline-none"
            autoFocus
          />
        </div>
        <ul className="max-h-80 overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <li className="px-4 py-3 text-sm text-muted">No matching commands.</li>
          ) : (
            filtered.map((command, index) => {
              const isHighlighted = index === highlightedIndex;
              return (
                <li key={command.id}>
                  <button
                    type="button"
                    className={clsx(
                      "flex w-full flex-col items-start gap-1 px-4 py-3 text-left text-sm transition",
                      isHighlighted ? "bg-accent/10 text-text" : "text-text",
                    )}
                    onMouseEnter={() => setHighlightedIndex(index)}
                    onClick={() => {
                      Promise.resolve(command.onTrigger()).catch((error) => {
                        console.error("Command execution failed", error);
                      });
                      onClose();
                    }}
                  >
                    <span className="font-medium">{command.label}</span>
                    {command.description ? (
                      <span className="text-xs text-muted">{command.description}</span>
                    ) : null}
                    {command.hint ? (
                      <span className="text-[11px] text-muted">{command.hint}</span>
                    ) : null}
                  </button>
                </li>
              );
            })
          )}
        </ul>
      </div>
    </div>,
    document.body,
  );
}

export type { CommandPaletteCommand };
