"""Text segmentation helpers for response analysis.

Classes:
    SegmentDraft: Intermediate representation used while carving responses into discourse windows.

Functions:
    split_into_segments(text): Break responses into sentences or bullet-driven fragments.
    estimate_role(text): Heuristic role classification for a segment.
    make_segment_drafts(...): Build windowed segment drafts with optional overlaps.
    flatten_drafts(collections): Flatten nested segment draft collections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from uuid import UUID, uuid4

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")
_BULLET_BOUNDARY = re.compile(r"\n[-*+]\s+")
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

from app.utils.tokenization import count_tokens as count_model_tokens

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "problem": ("challenge", "problem", "issue", "risk", "threat"),
    "forecast": ("will", "likely", "project", "expected", "scenario"),
    "solution": ("should", "recommend", "strategy", "plan", "invest"),
    "risk": ("vulnerable", "hazard", "exposure", "danger", "mitigate"),
    "opportunity": ("opportunity", "benefit", "advantage", "unlock"),
    "action": ("implement", "deploy", "build", "launch", "enact"),
}


@dataclass(slots=True)
class SegmentDraft:
    response_id: UUID
    response_index: int
    position: int
    text: str
    id: UUID = field(default_factory=uuid4)
    role: str | None = None
    tokens: int = 0
    prompt_similarity: float | None = None
    embedding: list[float] | None = None
    coords3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coords2d: tuple[float, float] = (0.0, 0.0)
    cluster_label: int | None = None
    cluster_probability: float | None = None
    cluster_similarity: float | None = None
    outlier_score: float | None = None
    silhouette_score: float | None = None
    text_hash: str | None = None
    is_cached: bool = False
    is_duplicate: bool = False
    simhash64: int | None = None


def split_into_segments(text: str) -> list[str]:
    clean = text.strip()
    if not clean:
        return []

    bullet_parts = _BULLET_BOUNDARY.split(clean)
    segments: list[str] = []
    for part in bullet_parts:
        part = part.strip()
        if not part:
            continue
        pieces = _SENTENCE_BOUNDARY.split(part)
        for piece in pieces:
            piece = piece.strip()
            if len(piece) < 2:
                continue
            segments.append(piece)

    if not segments:
        segments.append(clean)
    return segments


def estimate_role(text: str) -> str | None:
    lowered = text.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return role
    if lowered.startswith("in conclusion"):
        return "conclusion"
    if lowered.startswith("first") or lowered.startswith("second"):
        return "structure"
    return None


def make_segment_drafts(
    response_id: UUID,
    response_index: int,
    text: str,
    *,
    word_window: int = 0,
    word_overlap: int = 1,
) -> list[SegmentDraft]:
    pieces = split_into_segments(text)
    drafts: list[SegmentDraft] = []
    position = 0
    for piece in pieces:
        windows = _slice_word_windows(piece, word_window, word_overlap)
        for window_text in windows:
            role = estimate_role(window_text)
            drafts.append(
                SegmentDraft(
                    response_id=response_id,
                    response_index=response_index,
                    position=position,
                    text=window_text,
                    role=role,
                    tokens=count_model_tokens(window_text, None),
                )
            )
            position += 1
    return drafts


def _slice_word_windows(text: str, word_window: int, word_overlap: int) -> list[str]:
    if word_window <= 0:
        return [text]

    tokens = _WORD_PATTERN.findall(text)
    if not tokens:
        return [text]

    stride = max(1, word_window - max(0, word_overlap))
    windows: list[str] = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + word_window]
        if len(chunk) < word_window:
            if chunk:
                windows.append(" ".join(chunk))
            break
        windows.append(" ".join(chunk))
    if not windows and tokens:
        windows.append(" ".join(tokens))
    return windows


def flatten_drafts(collections: Iterable[list[SegmentDraft]]) -> list[SegmentDraft]:
    flattened: list[SegmentDraft] = []
    for items in collections:
        flattened.extend(items)
    return flattened
