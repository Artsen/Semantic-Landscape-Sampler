import clsx from "clsx";
import type { ReactNode } from "react";

type NavigationRailItem = {
  id: string;
  label: string;
  hint?: string;
  onSelect: () => void;
  icon?: ReactNode;
  shortcut?: string;
};

type NavigationRailProps = {
  items: NavigationRailItem[];
  activeId?: string;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
};

export type { NavigationRailItem };

export function NavigationRail({ items, activeId, expanded = false, onExpandedChange }: NavigationRailProps) {
  const handleMouseEnter = () => {
    onExpandedChange?.(true);
  };

  const handleMouseLeave = () => {
    onExpandedChange?.(false);
  };

  return (
    <nav
      className={clsx(
        "flex h-full flex-col border-r border-border bg-panel py-4 text-xs text-muted transition-all duration-200",
        expanded ? "w-64" : "w-20",
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {items.map((item) => {
        const isActive = item.id === activeId;
        return (
          <button
            key={item.id}
            type="button"
            className={clsx(
              "mx-2 mt-1 flex items-center gap-3 rounded-xl border px-3 py-2 text-left transition",
              isActive
                ? "border-accent bg-accent/10 text-accent"
                : "border-transparent hover:border-border hover:text-text",
            )}
            onClick={item.onSelect}
            title={item.hint ?? item.label}
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-panel-elev text-sm font-semibold">
              {item.icon ?? item.label.slice(0, 2)}
            </span>
            {expanded ? (
              <span className="flex-1 text-sm font-medium text-text">{item.label}</span>
            ) : null}
            {expanded && item.shortcut ? (
              <span className="text-[11px] text-muted">{item.shortcut}</span>
            ) : null}
          </button>
        );
      })}
    </nav>
  );
}
