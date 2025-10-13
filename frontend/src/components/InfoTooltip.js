import { jsx as _jsx } from "react/jsx-runtime";
import { memo } from "react";
/**
 * Lightweight tooltip badge that exposes explanatory copy on hover/focus.
 */
export const InfoTooltip = memo(function InfoTooltip({ text, className }) {
    return (_jsx("span", { className: `inline-flex h-4 w-4 items-center justify-center rounded-full bg-slate-800/80 text-[10px] font-semibold text-slate-200 shadow-sm ${className ?? ""}`, role: "img", "aria-label": text, title: text, children: "?" }));
});
