/**
 * React Query hook for fetching cached segment context (top terms, neighbors, metrics).
 */
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchSegmentContext } from "@/services/api";
export function useSegmentContext(segmentId, { enabled = true, k = 8, staleTimeMs = 5 * 60 * 1000 } = {}) {
    const queryClient = useQueryClient();
    const query = useQuery({
        queryKey: ["segment-context", segmentId, k],
        queryFn: async () => {
            if (!segmentId) {
                throw new Error("segmentId is required");
            }
            return fetchSegmentContext(segmentId, k);
        },
        enabled: enabled && Boolean(segmentId),
        staleTime: staleTimeMs,
        gcTime: staleTimeMs,
    });
    const prefetch = (id, neighbors = k) => queryClient.prefetchQuery({
        queryKey: ["segment-context", id, neighbors],
        queryFn: () => fetchSegmentContext(id, neighbors),
        staleTime: staleTimeMs,
        gcTime: staleTimeMs,
    });
    return { ...query, prefetch };
}
