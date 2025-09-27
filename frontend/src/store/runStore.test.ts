/**
 * Regression coverage for run store reducers and selectors.
 *
 * Verifies cluster palette generation, visibility toggles, selection helpers, and role filtering.
 */

import { afterEach, expect, test } from "vitest";

import { useRunStore } from "@/store/runStore";
import type { RunResultsResponse } from "@/types/run";

afterEach(() => {
  useRunStore.getState().reset();
});

test("setResults hydrates palette and visibility", () => {
  const mockResults: RunResultsResponse = {
    run: {
      id: "run-1",
      prompt: "Test prompt",
      n: 3,
      model: "gpt-4.1-mini",
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
    n: 3,
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
  };

  useRunStore.getState().setResults(mockResults);
  const state = useRunStore.getState();

  expect(state.clusterPalette).toMatchSnapshot();
  expect(Object.keys(state.clusterVisibility).length).toBe(2);
  expect(state.clusterVisibility["0"]).toBe(true);
  expect(state.clusterVisibility["-1"]).toBe(true);
});


test("setRunNotes updates results and history", () => {
  const runId = "run-notes";
  const runResource = {
    id: runId,
    prompt: "Test prompt",
    n: 1,
    model: "gpt-4.1-mini",
    temperature: 0.8,
    top_p: 1,
    seed: 42,
    max_tokens: 400,
    status: "completed" as const,
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
      n: runResource.n,
    },
    runHistory: [
      {
        id: runResource.id,
        prompt: runResource.prompt,
        n: runResource.n,
        model: runResource.model,
        temperature: runResource.temperature,
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
