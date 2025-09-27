"""Response hull geometry model.

Classes:
    ResponseHull: Stores convex hull coordinates for segment meshes in either 2D or 3D space.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class ResponseHull(SQLModel, table=True):
    __tablename__ = "response_hulls"
    __table_args__ = (
        UniqueConstraint("response_id", "dim", name="uq_response_hull_dim"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    response_id: UUID = Field(foreign_key="responses.id")
    dim: int
    points_json: str

