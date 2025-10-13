import { memo } from "react";

interface InfoTooltipProps {
  text: string;
  className?: string;
}

/**
 * Lightweight tooltip badge that exposes explanatory copy on hover/focus.
 */
export const InfoTooltip = memo(function InfoTooltip({ text, className }: InfoTooltipProps) {
  return (
    <span
      className={`inline-flex h-4 w-4 items-center justify-center rounded-full bg-slate-800/80 text-[10px] font-semibold text-slate-200 shadow-sm ${className ?? ""}`}
      role="img"
      aria-label={text}
      title={text}
    >
      ?
    </span>
  );
});
