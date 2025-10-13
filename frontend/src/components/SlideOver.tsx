import { ComponentProps, ReactNode, useEffect } from "react";
import { createPortal } from "react-dom";

type BaseDivProps = Pick<ComponentProps<"div">, "className">;

export type SlideOverProps = {
  open: boolean;
  title?: ReactNode;
  description?: ReactNode;
  side?: "right" | "left";
  onClose: () => void;
  children: ReactNode;
  widthClassName?: string;
} & BaseDivProps;

const portalRoot = typeof document !== "undefined" ? document.body : null;

export function SlideOver({
  open,
  title,
  description,
  side = "right",
  onClose,
  children,
  widthClassName = "w-[420px]",
  className,
}: SlideOverProps) {
  useEffect(() => {
    if (!open) {
      return;
    }
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open || !portalRoot) {
    return null;
  }

  const sideClass = side === "right" ? "translate-x-0" : "-translate-x-0";

  return createPortal(
    <div className="fixed inset-0 z-40 flex">
      <button
        type="button"
        className="h-full flex-1 cursor-default bg-black/40 backdrop-blur-sm"
        aria-label="Close panel"
        onClick={onClose}
      />
      <section
        className={`relative h-full bg-panel shadow-panel transition-transform duration-200 ease-out ${widthClassName} ${sideClass}`}
      >
        <div className={`flex h-full flex-col border-l border-border ${className ?? ""}`}>
          <header className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
            <div className="space-y-1">
              {typeof title === "string" ? (
                <h2 className="text-sm font-semibold text-text">{title}</h2>
              ) : (
                title
              )}
              {description ? (
                <p className="text-xs text-text-dim">{description}</p>
              ) : null}
            </div>
            <button
              type="button"
              className="rounded-md border border-border px-2 py-1 text-xs text-muted transition hover:text-text"
              onClick={onClose}
            >
              Close
            </button>
          </header>
          <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        </div>
      </section>
    </div>,
    portalRoot,
  );
}
