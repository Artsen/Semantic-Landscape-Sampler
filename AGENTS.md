# Semantic Landscape Sampler - Agent Playbook

## Quick Snapshot
- Mission: turn a single prompt into a reusable semantic landscape (responses, segments, embeddings, UMAP layouts, clusters, ANN graphs, overlays, exports).
- Backend (FastAPI + SQLModel) and frontend (React + Vite + Zustand + react-three-fiber) both run locally; SQLite + on-disk ANN indexes live under `backend/data/`.
- Embedding cache, provenance capture, ANN graph generation, and fine-grained exports are already implemented and should be preserved in any change.
- Processing telemetry captures total runtime plus per-stage durations (LLM sampling, segmentation, embeddings, UMAP, clustering, ANN, persistence) with UI and API exposure.
- Latest state (2025-10-07): backend pytest passes; frontend build emits known react-three-fiber typing warnings; README, CHANGELOG, notices all refreshed.

## Live Constraints & Environment
- Windows PowerShell environment; use `python` (not `python3`) for virtualenvs and scripts.
- Python 3.11+, Node 20+ with Corepack to manage pnpm. Ruff/Black for Python; ESLint/Prettier for TS.
- Requires `OPENAI_API_KEY`; mock fixtures cover CI (`backend/tests`).
- Sandbox writes limited to repo root; ANN indexes serialize to `backend/data/indexes/` – ensure the directory exists before sampling.
- Respect env defaults in `backend/app/core/config.py` (UMAP seeds, cache on by default, HDBSCAN as default clustering).

## Architecture at a Glance
- **Backend**: `RunService` orchestrates sampling -> segmentation -> embedding cache lookup -> feature blend -> UMAP (2D/3D) -> clustering -> ANN -> overlays/insights -> persistence. Exposes REST endpoints for run CRUD, results, metrics, graph, provenance, exports, and segment context.
- **Frontend**: `runStore` (Zustand) drives controls, 2D/3D scene, metadata panels, history drawer, exports. `useRunWorkflow` wraps API calls; react-three-fiber scene keeps overlays aligned via shared spread/centering.
- **Data**: SQLite schema includes runs, responses, response_segments, embeddings, projections, clusters, cluster_metrics, segment_edges, response_hulls, segment_insights, ann_index, embedding_cache, run_provenance.

## Key Files & Entry Points
- Backend orchestration: `backend/app/services/runs.py`
- Feature blending / UMAP / clustering / ANN helpers: `backend/app/services/projection.py`
- Segment insights & metrics: `backend/app/services/cluster_metrics.py`, `segment_insight.py`
- API surface: `backend/app/api/routes/run.py`
- Frontend store + workflow: `frontend/src/store/runStore.ts`, `frontend/src/hooks/useRunWorkflow.ts`
- 3D scene + overlays: `frontend/src/components/PointCloudScene.tsx`
- Controls + metadata UI: `frontend/src/components/ControlsPanel.tsx`, `RunMetadataBar.tsx`, `RunHistoryDrawer.tsx`
- Processing timeline UI: `frontend/src/components/ProcessingTimelinePanel.tsx` couples stage timings with the metadata bar timeline.
- Docs/notes: `README.md` (feature overview), `.github/agent-notes.md` (file index), `.github/plans.md` (roadmap)

## Operational Checklist
- Prefer the existing `backend/.venv`; avoid rebuilding the root `.venv` unless required.
1. Copy `.env.example` -> `.env`, set `OPENAI_API_KEY` (or ensure backend tests use mocks).
2. `cd backend && python -m venv .venv && pip install -r requirements.txt` -> run `uvicorn app.main:app --reload --port 8000` when manual testing.
3. `cd frontend && corepack enable && pnpm install && pnpm dev` for UI validation (`http://localhost:5173` proxies `http://localhost:8000`).
4. Before large changes: run `pytest` and `pnpm test -- --run`; expect known TS warnings but no failures.
5. When sampling manually: verify `backend/data/indexes/` is writable; ANN build will fail otherwise.
6. Large assets (`2025-09-27_17-46-43.mp4`, `.venv/`) live at repo root; scope `rg`/`ls` to relevant paths to avoid slow scans.

## Known Pitfalls & Lessons Learned
- **Embedding cache**: requires normalized text hash (NFKC + whitespace collapse + casefold). If schema/PRAGMAs are out of date, call `init_db()` or run the backend once to apply migrations.
- **Duplicates flag**: only set when cache is disabled; keep that behaviour if touching segmentation or cache flows.
- **ANN indexes**: stored per-run; loading requires matching method and dimensionality. Keep PCA/normalisation consistent with the builder before persisting.
- **Provenance**: captured at run creation. Any new runtime dependency should update `RunProvenance` and serializers or exports drift.
- **Frontend spread alignment**: overlays (edges, hulls, threads) share spread/centering maths with the point cloud; adjust all together or visuals desync.
- **Mermaid diagrams**: subgraph names must be unique (e.g., `SceneLayer`); otherwise GitHub renderers break.
- **Processing telemetry**: Stage identifiers (`prepare-run`, `request-completions`, etc.) are mapped directly in the UI; keep names stable when altering sampling flow.
- **TypeScript build warnings**: existing react-three-fiber typing gaps are known; avoid introducing new warnings without justification.
- **CI without pnpm**: some shells lack pnpm; rely on `corepack enable` before running scripts.

## Testing & Verification
- Backend: `cd backend && pytest` (OpenAI mocked). Run after changes to cache, clustering, ANN, exports, or schemas.
- Frontend: `cd frontend && pnpm lint` + `pnpm test -- --run`. Store tests cover presets/derived selectors; add snapshots for new UI states.
- Manual sanity: create run via UI or curl, ensure `/run/{id}/metrics`, `/run/{id}/graph?mode=simplified`, `/segments/{id}/context`, and scoped exports return expected payloads.
- Manual timeline: confirm metadata badge and Processing Timeline panel show total runtime and per-stage breakdown matching `/run/{id}/metrics`.

## Data & Fixtures
- Quick seed (with backend running):
  ```bash
  curl -X POST http://localhost:8000/run \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"How will climate change reshape coastal cities?","n":25,"model":"gpt-4.1-mini","temperature":0.9,"top_p":1.0,"seed":123,"max_tokens":800,"use_cache":true}'
  curl -X POST http://localhost:8000/run/<run_id>/sample
  curl http://localhost:8000/run/<run_id>/results | jq
  ```
- Export variants to verify: `scope=cluster`, `scope=selection`, `scope=viewport`, `include=provenance`.

## Invariants & Decisions to Preserve
- Cache default `use_cache=true`; cache hits should avoid re-embedding but still record `is_cached`/`is_duplicate` correctly.
- UMAP must produce stable layouts for a given seed; tests rely on deterministic output.
- HDBSCAN is primary clustering; ensure fallback to KMeans only when necessary and metrics continue to populate.
- ANN edges default to full graph, with simplified mutual-k option exposed via API + UI.
- Provenance + exports must always include schema/version info for reproducibility.

## Next Experiments / Open Threads
- Add pnpm tooling to shared CLI image so Vitest can run in CI shells without manual setup.
- Document OpenAI mocking fixtures usage (README + here).
- Resolve or formally suppress react-three-fiber type warnings.
- Future roadmap: model comparison overlays, automated topic labelling once pipeline is stable.

Keep this playbook concise; update only when new lessons or constraints emerge. Detailed plans belong in `.github/plans.md`; day-to-day reasoning should reference this file first.

