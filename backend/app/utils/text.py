"""Text normalisation and similarity hashing helpers."""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Iterable

_WHITESPACE = tuple("\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000")


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into single spaces."""

    if not text:
        return ""

    parts: list[str] = []
    current: list[str] = []
    for char in text:
        if char in _WHITESPACE:
            if current:
                parts.append("".join(current))
                current.clear()
        else:
            if current or parts:
                pass
            current.append(char)
    if current:
        parts.append("".join(current))
    collapsed = " ".join(part for part in parts if part)
    return collapsed


def normalise_for_embedding(text: str) -> tuple[str, str]:
    """Return collapsed text alongside the hashable normalised form."""

    stripped = text.strip()
    if not stripped:
        return "", ""

    collapsed = collapse_whitespace(stripped)
    nfkc = unicodedata.normalize("NFKC", collapsed)
    normalised = nfkc.casefold()
    return collapsed, normalised


def compute_simhash64(tokens: Iterable[str]) -> int | None:
    """Compute a 64-bit SimHash for the provided token stream."""

    weights = [0] * 64
    token_seen = False
    for token in tokens:
        if not token:
            continue
        token_seen = True
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big", signed=False)
        for bit in range(64):
            if value & (1 << bit):
                weights[bit] += 1
            else:
                weights[bit] -= 1
    if not token_seen:
        return None
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    if fingerprint >= 2**63:
        fingerprint -= 2**64
    return fingerprint
