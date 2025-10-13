/**
 * Lightweight popover that anchors the layers panel to a trigger button.
 */

import { useEffect, useLayoutEffect, useState } from "react";
import { createPortal } from "react-dom";

import { LayersPanel } from "@/components/LayersPanel";

type LayersPopoverProps = {
  anchorRef: React.RefObject<HTMLElement>;
  open: boolean;
  onClose: () => void;
};

const PANEL_WIDTH = 320;

export function LayersPopover({ anchorRef, open, onClose }: LayersPopoverProps) {
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    const node = anchorRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    const top = rect.bottom + 8;
    const viewportWidth = window.innerWidth;
    const left = Math.min(rect.left, viewportWidth - PANEL_WIDTH - 16);
    setPosition({ top, left: Math.max(16, left) });
  }, [anchorRef, open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const handlePointer = (event: MouseEvent) => {
      if (!anchorRef.current) {
        return;
      }
      const panel = document.getElementById("layers-popover-panel");
      if (!panel) {
        return;
      }
      if (panel.contains(event.target as Node)) {
        return;
      }
      if (anchorRef.current.contains(event.target as Node)) {
        return;
      }
      onClose();
    };

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    document.addEventListener("mousedown", handlePointer);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handlePointer);
      document.removeEventListener("keydown", handleKey);
    };
  }, [anchorRef, onClose, open]);

  if (!open || !position) {
    return null;
  }

  return createPortal(
    <div className="fixed inset-0 z-40 flex items-start justify-start" style={{ pointerEvents: "none" }}>
      <div
        id="layers-popover-panel"
        className="pointer-events-auto mt-0.5 w-[320px] rounded-2xl border border-border bg-panel shadow-panel"
        style={{
          position: "absolute",
          top: position.top,
          left: position.left,
        }}
      >
        <div className="border-b border-border px-4 py-3 text-sm font-semibold text-text">Layers</div>
        <div className="max-h-[420px] overflow-y-auto px-4 py-3 text-sm text-text">
          <LayersPanel />
        </div>
      </div>
    </div>,
    document.body,
  );
}
