# Contributing to Semantic Landscape Sampler

Thanks for your interest in improving Semantic Landscape Sampler! This guide explains how to set up your environment, follow our coding conventions, and submit a pull request that can be merged quickly.

## Getting Started

1. **Fork and clone** the repository from GitHub.
2. **Create a Python virtual environment** and install backend dependencies:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Install frontend tooling**:
   ```bash
   cd frontend
   corepack enable
   pnpm install
   ```
4. Copy `.env.example` to `.env` and fill in the required values (`OPENAI_API_KEY` at minimum).

## Branching & Commits

- Create feature branches from `main` using a descriptive name (e.g., `feature/segment-mesh-sync`).
- Keep commits focused and write meaningful messages ("Fix segment hull scaling" beats "wip").
- Rebase on `main` before opening a pull request to keep history tidy.

## Code Style

- **Python**: format with Black (`black app tests`), lint with Ruff (`ruff check app tests`).
- **TypeScript/React**: use Prettier via `pnpm lint` and follow the existing Tailwind conventions.
- Prefer type-safe schemas (Pydantic in the backend, Zod in the frontend) and avoid untyped JSON.
- Document non-obvious logic with concise comments and keep docstrings in Google style.

## Testing

Run the relevant tests before pushing:

```bash
# Backend
cd backend
source .venv/bin/activate
pytest

# Frontend
cd frontend
pnpm lint
pnpm test -- --run
```

The backend test suite stubs OpenAI APIs; if you add new API calls, extend the fixtures under `backend/tests` accordingly. For deterministic embeddings or projections, update the golden files intentionally and mention it in your PR description.

## Development Workflow

- Use `uvicorn app.main:app --reload` for backend hot reloading.
- Use `pnpm dev` for the frontend dev server (defaults to `http://localhost:5173`).
- When iterating on projection visuals, capture screenshots or GIFs to help reviewers understand the change.
- If you touch the data model, document migrations or schema updates in the README and `CHANGELOG.md`.

## Pull Request Checklist

- [ ] Tests pass locally for the areas you touched.
- [ ] Linting/formatting checks pass.
- [ ] README or other docs updated when behaviour changes.
- [ ] Screenshots/GIFs attached when the UI changes.
- [ ] Linked issues referenced with `Fixes #123` when applicable.

Once ready, open a pull request describing the change, motivation, testing performed, and any follow-up work. Reviewers will aim to respond within a few business days.

Thanks again for contributing!
