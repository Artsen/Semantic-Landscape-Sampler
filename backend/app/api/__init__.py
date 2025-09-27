"""API router composition for the backend.

The module assembles individual route groups into a single `api_router` that can be mounted on the app.
"""

from fastapi import APIRouter

from app.api.routes.run import router as run_router

api_router = APIRouter()
api_router.include_router(run_router)

__all__ = ["api_router"]

