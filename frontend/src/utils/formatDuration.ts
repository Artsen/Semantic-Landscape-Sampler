export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) {
    return "--";
  }
  if (ms >= 1000) {
    const seconds = ms / 1000;
    return seconds >= 10 ? `${seconds.toFixed(1)}s` : `${seconds.toFixed(2)}s`;
  }
  if (ms >= 100) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${ms.toFixed(1)}ms`;
}
