# Semantic Landscape Sampler

Semantic Landscape Sampler fans a single question out across many LLM completions, stores every artifact, and turns the results into an explorable semantic landscape. The backend handles sampling, embeddings, dimensionality reduction, clustering, and persistence; the frontend delivers an interactive 2D/3D point cloud with overlays for segments, similarity edges, and response hulls.

## Table of Contents
- [Semantic Landscape Sampler](#semantic-landscape-sampler)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Capabilities](#key-capabilities)
    - [Backend](#backend)
    - [Frontend](#frontend)
  - [Architecture](#architecture)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Running the Stack](#running-the-stack)
  - [Using the Visualiser](#using-the-visualiser)
    - [Controls Reference](#controls-reference)
    - [Exploration Tips](#exploration-tips)
  - [Data Model and Persistence](#data-model-and-persistence)
  - [API Endpoints](#api-endpoints)
  - [Testing and Quality Gates](#testing-and-quality-gates)
    - [Backend](#backend-1)
    - [Frontend](#frontend-1)
  - [Seed Sample Data](#seed-sample-data)
  - [Troubleshooting](#troubleshooting)
  - [Roadmap and Next Steps](#roadmap-and-next-steps)
  - [Contributing](#contributing)

## Overview

Semantic Landscape Sampler is built for rapid sense-making of large language model responses. Instead of skimming dozens of plain text completions, you can project them into a shared embedding space, inspect clusters, surface outliers, and drill into individual sentences. The project is intentionally split into a stateless API layer and a rich browser client so that analytics workloads, automation, or other frontends can reuse the same contracts.

Recent UX improvement: the segment overlays (thread mesh, similarity edges, and response hulls) now share the exact same scaling pipeline as the rendered point cloud. Adjusting the point spread slider keeps every mesh aligned with the visible points, which makes it much easier to reason about relative distances while exploring segments.

## Key Capabilities

### Backend
- FastAPI + SQLModel service layer with async persistence to SQLite during local development.
- OpenAI Chat Completions for generation (gpt-5-codex, gpt-4.1, or user-supplied models).
- Embedding blend: OpenAI 	ext-embedding-3-large (small fallback), TF-IDF, prompt similarity, and lightweight statistics.
- Dimensionality reduction via deterministic UMAP for 3D and 2D projections (shared random state for reproducibility).
- Clustering with HDBSCAN plus a KMeans fallback. Each cluster tracks membership probability, centroid similarity, keyword themes, silhouette, and outlier scores.
- Response segmentation, optional discourse tagging, similarity graph construction, response hull computation, and parent thread stitching.
- Clean JSON schemas via Pydantic for run configuration, results payloads, and exports.

### Frontend
- React + Vite + Tailwind + Zustand application scaffolded for fast iteration.
- 
eact-three-fiber + drei rendering of the point cloud, complete with hover tooltips, lasso selection, camera controls, Stats overlay, and density heatmaps.
- Response vs segment modes, cluster legends, role filters, similarity edges, parent thread meshes, and convex hull overlays.
- Detail drawer for metrics, raw text, embeddings, and parent response context.
- Run history drawer, run notes editor, quick outlier selectors, and exports that mirror backend payloads (JSON and CSV).

## Architecture

```
backend/
  app/
    api/            # FastAPI routers (runs, sampling, exports)
    core/           # Settings, environment, feature flags
    db/             # SQLModel session helpers
    models/         # SQLite table declarations
    schemas/        # Pydantic response/input models
    services/       # Sampling, embedding, projection, clustering
  tests/            # Pytest suites with OpenAI fixtures and projection goldens
frontend/
  src/
    components/     # Controls, scene, drawers, legends, overlays
    hooks/          # Run workflow orchestration
    services/       # REST API client
    store/          # Zustand store + tests
    types/          # Shared API types
  public/           # Static assets (if any)
.github/
  workflows/ci.yml  # Format, lint, and test gates for both stacks
```

## Prerequisites

- Python 3.11 or 3.12
- Node.js 20.x with Corepack (pnpm) enabled
- An OpenAI API key with access to the desired chat and embedding models
- (Optional) uv if you prefer its virtualenv workflow

## Environment Configuration

1. Copy the sample environment file and fill in secrets:
   ```bash
   cp .env.example .env
   ```
2. Set at minimum OPENAI_API_KEY. You can also tune sampling defaults (DEFAULT_MODEL, DEFAULT_TEMPERATURE, etc.) and SQLite paths.
3. Backend and frontend both read from .env at the project root; keep secrets out of version control.

## Backend Setup

```bash
cd backend
python -m venv .venv  # or: uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launching the API during development:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The backend automatically initialises the SQLite schema on first run. If you upgrade from an older database image and encounter missing columns (for example, notes), delete backend/data/semantic_sampler.db or run the migration snippet in AGENTS.md.

## Frontend Setup

```bash
cd frontend
corepack enable
pnpm install
pnpm dev -- --open
```

By default Vite serves on http://localhost:5173 and proxies API calls to http://localhost:8000.

## Running the Stack

1. Start the FastAPI server (uvicorn app.main:app --reload).
2. Start the frontend (pnpm dev).
3. Visit http://localhost:5173 and enter a prompt.
4. Set sampling parameters (N, temperature, top_p, seed, max tokens, model).
5. Click **Generate Landscape** to trigger the /run + /run/{id}/sample workflow.
6. The scene auto-refreshes once the backend persists results; cluster palettes, legends, hulls, and overlays update immediately.

## Using the Visualiser

### Controls Reference
- **Point Spread**: rescales the projection while keeping segment meshes, similarity edges, and hulls perfectly aligned with visible points.
- **Point Size**: adjusts rendered particle size (affects hover thresholds).
- **Density Overlay**: toggles a heatmap computed in screen space.
- **View Mode**: switch between 3D orbit controls and an orthographic 2D camera.
- **Level Mode**: switch between response-level and segment-level clouds.
- **Cluster Legend**: hover to spotlight a cluster, click to toggle visibility.
- **Role Filters** (segment mode): filter by assistant/system/user roles when discourse tagging is enabled.
- **Similarity Edges**: visualise high-similarity pairs generated from segment embeddings.
- **Parent Threads**: show the conversational mesh plus response hulls for context (uses the same scaling as the point cloud).
- **Run Notes**: capture annotations and hypotheses for future reference.

### Exploration Tips
- Hover points to preview summary text and metadata.
- Shift + drag to lasso select; selections drive the details panel.
- Use the run history drawer to reopen prior experiments with stored parameters and notes.
- Export JSON/CSV from the controls panel to analyse in notebooks or BI tools.

## Data Model and Persistence

SQLite schema (simplified):

| Table | Purpose |
| --- | --- |
| runs | Prompt, model, sampling configuration, status, notes. |
| responses | Raw chat completions and metadata (cluster label, centroid similarity, outlier score). |
| response_segments | Sentence/discourse segments tied to parent responses. |
| embeddings | Embedding vectors (responses and segments) for reproducibility. |
| projections | UMAP/t-SNE coordinates in 2D and 3D. |
| clusters | Cluster assignments and stats for responses and segments. |
| segment_edges | High-similarity edges between segments. |
| response_hulls | Convex hull coordinates for each response (2D and 3D). |

Results are returned as a single JSON payload so the frontend can hydrate the scene offline if needed.

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| POST | /run | Create a run configuration (prompt, sampling params, optional notes). |
| POST | /run/{id}/sample | Execute sampling, segmentation, embedding, projection, clustering, persistence. |
| GET | /run/{id}/results | Fetch the full visualisation payload (responses, segments, clusters, edges, hulls). |
| GET | /run/{id}/export.json | Download results as canonical JSON. |
| GET | /run/{id}/export.csv | Flattened CSV for response-level analytics. |
| PATCH | /run/{id} | Update metadata such as run notes. |

All endpoints return JSON and expect/produce Pydantic schemas located in backend/app/schemas/run.py.

## Testing and Quality Gates

### Backend
```bash
cd backend
source .venv/bin/activate
pytest
ruff check app tests
black app tests --check
```

backend/tests ships fixtures that mock OpenAI calls so the suite runs offline. Golden files cover projection determinism (fixed random state).

### Frontend
```bash
cd frontend
pnpm lint
pnpm test -- --run
```

Vitest is configured for DOM testing (frontend/src/setupTests.ts). When pnpm is unavailable in your shell, install Node 20 and enable Corepack first.

GitHub Actions (.github/workflows/ci.yml) runs formatters, linters, and unit tests for both stacks.

## Seed Sample Data

With both services running:

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"How will climate change reshape coastal cities?","n":25,"model":"gpt-4.1-mini","temperature":0.9,"top_p":1.0,"seed":123,"max_tokens":800}'
```

Note the run_id from the response, then:

```bash
curl -X POST http://localhost:8000/run/<run_id>/sample
curl http://localhost:8000/run/<run_id>/results | jq
```

These payloads can be imported directly into the frontend store for demos or regression testing.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| AttributeError: coverage.types.Tracer when running UMAP | Stick to coverage==7.5.3; the shim in app/services/projection.py patches numba coverage hooks. |
| table runs has no column named notes | Run the migration snippet in AGENTS.md or delete the SQLite file to recreate it. |
| Empty scene after sampling | Ensure the backend logs show completed sampling; the frontend now refreshes draw ranges immediately after buffer updates. |
| Segment mesh does not move with spread slider | Update to this revision; overlays now reuse the same scaled geometry pipeline as the point cloud. |
| Frontend tests complain about missing pnpm | Install Node 20, run corepack enable, then pnpm install. |

## Roadmap and Next Steps

- Integrate pnpm into the CLI image so Vitest can run locally and in CI without manual setup.
- Publish OpenAI mocking recipes for CI in the onboarding docs.
- Layer in model comparison overlays and richer topic labelling (see AGENTS.md).
- Add camera bookmarks and saved viewpoints per run.

## Contributing

Issues and pull requests are welcome. Please run the backend and frontend test suites (or note why they could not be executed) before submitting changes. Review the mission details in AGENTS.md for broader context and follow the style guides enforced by Ruff, Black, ESLint, and Prettier.

Happy sampling!
