/**
      chunk_overlap: 1,
 * Regression coverage for run store reducers and selectors.
 *
 * Verifies cluster palette generation, visibility toggles, selection helpers, and role filtering.
 */
import { afterEach, expect, test } from "vitest";
import { useRunStore } from "@/store/runStore";
function buildMockResults() {
    return {
        run: {
            id: "run-1",
            prompt: "Test prompt",
            n: 3,
            model: "gpt-4.1-mini",
            system_prompt: "Return a concise answer",
            embedding_model: "text-embedding-3-large",
            preproc_version: "norm-nfkc-v1",
            use_cache: true,
            cluster_algo: "hdbscan",
            hdbscan_min_cluster_size: 3,
            hdbscan_min_samples: 1,
            chunk_size: 3,
            chunk_overlap: 1,
            umap: {
                n_neighbors: 30,
                min_dist: 0.3,
                metric: "cosine",
                seed: 42,
                seed_source: "ui",
            },
            temperature: 0.8,
            top_p: 1,
            seed: 42,
            max_tokens: 800,
            status: "completed",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            error_message: null,
            notes: null,
        },
        prompt: "Test prompt",
        model: "gpt-4.1-mini",
        system_prompt: "Return a concise answer",
        preproc_version: "norm-nfkc-v1",
        chunk_overlap: 1,
        embedding_model: "text-embedding-3-large",
        n: 3,
        chunk_size: 3,
        chunk_overlap: 1,
        umap: {
            n_neighbors: 30,
            min_dist: 0.3,
            metric: "cosine",
            seed: 42,
            seed_source: "ui",
        },
        quality: {
            trustworthiness_2d: 0.78,
            trustworthiness_3d: 0.81,
            continuity_2d: 0.73,
            continuity_3d: 0.75,
        },
        projection_quality: {
            umap: {
                trustworthiness_2d: 0.78,
                trustworthiness_3d: 0.81,
                continuity_2d: 0.73,
                continuity_3d: 0.75,
            },
        },
        clusters: [
            {
                label: 0,
                size: 2,
                centroid_xyz: [0.1, -0.2, 0.3],
                exemplar_ids: ["resp-1", "resp-2"],
                average_similarity: 0.92,
                method: "hdbscan",
                keywords: [],
                noise: false,
            },
            {
                label: -1,
                size: 1,
                centroid_xyz: [0.0, 0.0, 0.0],
                exemplar_ids: ["resp-3"],
                average_similarity: null,
                method: "hdbscan",
                keywords: [],
                noise: true,
            },
        ],
        points: [
            {
                id: "resp-1",
                index: 0,
                text_preview: "Preview 1",
                full_text: "Full response 1",
                tokens: 120,
                finish_reason: "stop",
                usage: { completion_tokens: 120, prompt_tokens: 20, total_tokens: 140 },
                cluster: 0,
                probability: 0.95,
                similarity_to_centroid: 0.9,
                coords_3d: [0.1, 0.2, 0.3],
                coords_2d: [0.1, 0.2],
            },
            {
                id: "resp-2",
                index: 1,
                text_preview: "Preview 2",
                full_text: "Full response 2",
                tokens: 130,
                finish_reason: "stop",
                usage: { completion_tokens: 130, prompt_tokens: 20, total_tokens: 150 },
                cluster: 0,
                probability: 0.87,
                similarity_to_centroid: 0.85,
                coords_3d: [0.2, 0.1, -0.1],
                coords_2d: [0.2, 0.1],
            },
            {
                id: "resp-3",
                index: 2,
                text_preview: "Preview 3",
                full_text: "Full response 3",
                tokens: 110,
                finish_reason: "stop",
                usage: { completion_tokens: 110, prompt_tokens: 20, total_tokens: 130 },
                cluster: -1,
                probability: 0.45,
                similarity_to_centroid: null,
                coords_3d: [-0.1, -0.3, 0.2],
                coords_2d: [-0.1, -0.3],
            },
        ],
        segments: [],
        segment_clusters: [],
        segment_edges: [],
        response_hulls: [],
        cluster_metrics: {
            algo: "hdbscan",
            params: { min_cluster_size: 3, min_samples: 1 },
            silhouette_embed: 0.42,
            silhouette_feature: 0.56,
            davies_bouldin: 0.9,
            calinski_harabasz: 128.4,
            n_clusters: 1,
            n_noise: 1,
            stability: {
                bootstrap: {
                    mode: "bootstrap",
                    fraction: 0.85,
                    iterations: 10,
                    clusters: {
                        "0": { mean: 0.78, std: 0.05, samples: 10 },
                    },
                },
            },
            sweep: {
                baseline: { min_cluster_size: 3, min_samples: 1 },
                points: [
                    { min_cluster_size: 2, min_samples: 1, algo: "hdbscan", silhouette_feature: 0.5, silhouette_embed: 0.45, davies_bouldin: 0.95, calinski_harabasz: 110 },
                ],
            },
            created_at: new Date().toISOString(),
        },
        costs: {
            model: "gpt-4.1-mini",
            embedding_model: "text-embedding-3-large",
            completion_input_tokens: 0,
            completion_output_tokens: 0,
            completion_cost: 0,
            embedding_tokens: 0,
            embedding_cost: 0,
            total_cost: 0,
        },
        quality: {
            trustworthiness_2d: 0.91,
            trustworthiness_3d: 0.89,
            continuity_2d: 0.88,
            continuity_3d: 0.9,
        },
        provenance: null,
    };
}
afterEach(() => {
    useRunStore.getState().reset();
});
test("setResults hydrates palette and visibility", () => {
    const mockResults = buildMockResults();
    ;
    useRunStore.getState().setResults(mockResults);
    const state = useRunStore.getState();
    expect(state.clusterPalette).toMatchSnapshot();
    expect(Object.keys(state.clusterVisibility).length).toBe(2);
    expect(state.clusterVisibility["0"]).toBe(true);
    expect(state.clusterVisibility["-1"]).toBe(true);
    expect(state.projectionMethod).toBe("umap");
    expect(state.projectionVariants.umap).toBeDefined();
    expect(state.projectionWarnings).toEqual([]);
    expect(state.isProjectionLoading).toBe(false);
    expect(state.projectionError).toBeNull();
    expect(state.results?.points.every((point) => point.hidden === false)).toBe(true);
    expect(state.results?.projection_quality?.umap?.trustworthiness_2d).toBeCloseTo(0.78, 2);
    expect(state.chunkSize).toBe(3);
    expect(state.chunkOverlap).toBe(1);
    expect(state.clusterAlgo).toBe("hdbscan");
    expect(state.hdbscanMinClusterSize).toBe(3);
    expect(state.clusterMetrics?.algo).toBe("hdbscan");
    expect(state.segmentEdges).toEqual([]);
    expect(state.segmentGraphMode).toBe("full");
    expect(state.segmentGraphAutoSimplified).toBe(false);
    expect(state.segmentGraphLoading).toBe(false);
    expect(state.segmentGraphError).toBeNull();
});
test("setProjectionMethod uses cached variant when available", async () => {
    const store = useRunStore.getState();
    const mockResults = buildMockResults();
    store.setResults(mockResults);
    const variantState = {
        coords: {
            "resp-1": { coords2d: [0.5, 0.4], coords3d: [0.5, 0.4, 0.3] },
            "resp-2": { coords2d: [0.4, 0.2], coords3d: [0.4, 0.2, 0.1] },
            "resp-3": { coords2d: [-0.2, -0.3], coords3d: [-0.2, -0.3, 0.0] },
        },
        subsetIds: null,
        metadata: {
            fromCache: true,
            cachedAt: new Date().toISOString(),
            warnings: ["tsne-preview:1000"],
            requestedParams: { perplexity: 30 },
            resolvedParams: { perplexity: 30 },
            isSubsample: false,
            pointCount: 3,
            totalCount: 3,
            subsampleStrategy: null,
        },
        quality: {
            trustworthiness_2d: 0.71,
            trustworthiness_3d: 0.74,
            continuity_2d: 0.69,
            continuity_3d: 0.7,
        },
    };
    useRunStore.setState((state) => ({
        projectionVariants: { ...state.projectionVariants, tsne: variantState },
    }));
    await store.setProjectionMethod("tsne");
    const state = useRunStore.getState();
    expect(state.projectionMethod).toBe("tsne");
    expect(state.projectionWarnings).toEqual(variantState.metadata.warnings);
    expect(state.isProjectionLoading).toBe(false);
    expect(state.projectionError).toBeNull();
    expect(state.results?.projection_quality?.tsne).toEqual(variantState.quality);
    expect(state.results?.points[0].coords_2d).toEqual([0.5, 0.4]);
    expect(state.results?.points[0].coords_3d).toEqual([0.5, 0.4, 0.3]);
});
test("setRunNotes updates results and history", () => {
    const runId = "run-notes";
    const runResource = {
        id: runId,
        prompt: "Test prompt",
        n: 1,
        model: "gpt-4.1-mini",
        system_prompt: "Return a concise answer",
        chunk_overlap: 1,
        embedding_model: "text-embedding-3-large",
        preproc_version: "norm-nfkc-v1",
        use_cache: true,
        cluster_algo: "hdbscan",
        hdbscan_min_cluster_size: 3,
        hdbscan_min_samples: 1,
        chunk_size: 3,
        umap: {
            n_neighbors: 30,
            min_dist: 0.3,
            metric: "cosine",
            seed: 42,
            seed_source: "ui",
        },
        temperature: 0.8,
        top_p: 1,
        seed: 42,
        max_tokens: 400,
        status: "completed",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        error_message: null,
        notes: null,
    };
    useRunStore.setState({
        results: {
            run: runResource,
            points: [],
            clusters: [],
            segments: [],
            segment_clusters: [],
            segment_edges: [],
            response_hulls: [],
            prompt: runResource.prompt,
            model: runResource.model,
            system_prompt: runResource.system_prompt,
            embedding_model: runResource.embedding_model,
            preproc_version: runResource.preproc_version,
            n: runResource.n,
            quality: {
                trustworthiness_2d: null,
                trustworthiness_3d: null,
                continuity_2d: null,
                continuity_3d: null,
            },
            projection_quality: {},
            costs: {
                model: runResource.model,
                embedding_model: runResource.embedding_model,
                completion_input_tokens: 0,
                completion_output_tokens: 0,
                completion_cost: 0,
                embedding_tokens: 0,
                embedding_cost: 0,
                total_cost: 0,
            },
        },
        runHistory: [
            {
                id: runResource.id,
                prompt: runResource.prompt,
                n: runResource.n,
                model: runResource.model,
                system_prompt: runResource.system_prompt,
                embedding_model: runResource.embedding_model,
                preproc_version: runResource.preproc_version,
                use_cache: true,
                umap: runResource.umap,
                temperature: runResource.temperature,
                chunk_size: runResource.chunk_size,
                chunk_overlap: runResource.chunk_overlap,
                top_p: runResource.top_p,
                seed: runResource.seed,
                max_tokens: runResource.max_tokens,
                status: runResource.status,
                created_at: runResource.created_at,
                updated_at: runResource.updated_at,
                response_count: 0,
                segment_count: 0,
                notes: null,
            },
        ],
    });
    useRunStore.getState().setRunNotes(runId, "Review later");
    expect(useRunStore.getState().results?.run.notes).toBe("Review later");
    expect(useRunStore.getState().runHistory[0].notes).toBe("Review later");
    useRunStore.getState().setRunNotes(runId, null);
    expect(useRunStore.getState().results?.run.notes).toBeNull();
    expect(useRunStore.getState().runHistory[0].notes).toBeNull();
});
test("graph controls update state", async () => {
    const store = useRunStore.getState();
    store.setSegmentEdges([
        {
            source_id: "seg-a",
            target_id: "seg-b",
            score: 0.82,
        },
    ], {
        mode: "simplified",
        autoSimplified: true,
        k: 12,
        threshold: 0.4,
    });
    let state = useRunStore.getState();
    expect(state.segmentEdges).toHaveLength(1);
    expect(state.segmentGraphMode).toBe("simplified");
    expect(state.segmentGraphAutoSimplified).toBe(true);
    expect(state.graphEdgeK).toBe(12);
    expect(state.graphEdgeThreshold).toBe(0.4);
    expect(state.segmentGraphLoading).toBe(false);
    expect(state.segmentGraphError).toBeNull();
    await store.setSimplifyEdges(true);
    state = useRunStore.getState();
    expect(state.simplifyEdges).toBe(true);
    await store.setGraphEdgeK(8);
    state = useRunStore.getState();
    expect(state.graphEdgeK).toBe(8);
    await store.setGraphEdgeThreshold(0.55);
    state = useRunStore.getState();
    expect(state.graphEdgeThreshold).toBeCloseTo(0.55, 5);
});
