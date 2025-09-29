"""Utility helpers for counting tokens across models."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for environments without tiktoken
    tiktoken = None  # type: ignore


_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=64)
def _encoding_for_model(model: Optional[str]) -> Optional["tiktoken.Encoding"]:  # type: ignore[name-defined]
    if tiktoken is None:  # pragma: no cover
        return None
    if not model:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:  # pragma: no cover - unknown model IDs
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str | None, model: Optional[str] = None) -> int:
    """Return an estimated token count for *text* under the given *model*.

    Falls back to a simple whitespace split when `tiktoken` is unavailable.
    """

    if not text:
        return 0

    encoding = _encoding_for_model(model)
    if encoding is None:
        return len(text.strip().split())
    return len(encoding.encode(text))

