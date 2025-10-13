"""Compare runs endpoint exposing alignment and diff analytics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.session import get_session
from app.schemas import CompareRunsRequest, CompareRunsResponse
from app.services.compare import CompareService

router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=CompareRunsResponse)
async def compare_runs(
    payload: CompareRunsRequest,
    session: AsyncSession = Depends(get_session),
) -> CompareRunsResponse:
    service = CompareService(session)
    try:
        return await service.compare_runs(payload)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message) from exc

