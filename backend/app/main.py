"""Application bootstrap for the Semantic Landscape Sampler API.

This module wires the FastAPI application, attaches middleware, and exposes small lifecycle utilities.

Functions:
    lifespan(app: FastAPI): Initialise database state on startup and yield control back to FastAPI.
    health_check(): Lightweight readiness probe used by monitoring and local smoke tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_settings
from app.db.session import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

