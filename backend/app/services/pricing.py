"""Pricing tables for OpenAI completion and embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompletionPricing:
    model: str
    input_cost: float
    cached_input_cost: float
    output_cost: float


@dataclass(frozen=True)
class EmbeddingPricing:
    model: str
    input_cost: float


_COMPLETION_PRICING: dict[str, CompletionPricing] = {
    "gpt-5": CompletionPricing("gpt-5", 0.00000125, 0.000000125, 0.00001),
    "gpt-5-mini": CompletionPricing("gpt-5-mini", 0.00000025, 0.000000025, 0.000002),
    "gpt-5-nano": CompletionPricing("gpt-5-nano", 0.00000005, 0.000000005, 0.0000004),
    "gpt-5-chat-latest": CompletionPricing("gpt-5-chat-latest", 0.00000125, 0.000000125, 0.00001),
    "gpt-5-codex": CompletionPricing("gpt-5-codex", 0.00000125, 0.000000125, 0.00001),
    "gpt-4.1": CompletionPricing("gpt-4.1", 0.000002, 0.0000005, 0.000008),
    "gpt-4.1-mini": CompletionPricing("gpt-4.1-mini", 0.0000004, 0.0000001, 0.0000016),
    "gpt-4.1-nano": CompletionPricing("gpt-4.1-nano", 0.0000001, 0.000000025, 0.0000004),
    "gpt-4o": CompletionPricing("gpt-4o", 0.0000025, 0.00000125, 0.00001),
    "gpt-4o-2024-05-13": CompletionPricing("gpt-4o-2024-05-13", 0.000005, 0.0, 0.000015),
    "gpt-4o-mini": CompletionPricing("gpt-4o-mini", 0.00000015, 0.000000075, 0.0000006),
    "o1": CompletionPricing("o1", 0.000015, 0.0000075, 0.00006),
    "o1-pro": CompletionPricing("o1-pro", 0.00015, 0.0, 0.0006),
    "o3-pro": CompletionPricing("o3-pro", 0.00002, 0.0, 0.00008),
    "o3": CompletionPricing("o3", 0.000002, 0.0000005, 0.000008),
    "o3-deep-research": CompletionPricing("o3-deep-research", 0.00001, 0.0000025, 0.00004),
    "o4-mini": CompletionPricing("o4-mini", 0.0000011, 0.000000275, 0.0000044),
    "o4-mini-deep-research": CompletionPricing("o4-mini-deep-research", 0.000002, 0.0000005, 0.000008),
    "o3-mini": CompletionPricing("o3-mini", 0.0000011, 0.00000055, 0.0000044),
    "o1-mini": CompletionPricing("o1-mini", 0.0000011, 0.00000055, 0.0000044),
    "codex-mini-latest": CompletionPricing("codex-mini-latest", 0.0000015, 0.000000375, 0.000006),
}

_EMBEDDING_PRICING: dict[str, EmbeddingPricing] = {
    "text-embedding-3-small": EmbeddingPricing("text-embedding-3-small", 0.00000002),
    "text-embedding-3-large": EmbeddingPricing("text-embedding-3-large", 0.00000013),
    "text-embedding-ada-002": EmbeddingPricing("text-embedding-ada-002", 0.00000010),
}


def get_completion_pricing(model: str) -> Optional[CompletionPricing]:
    key = model.lower()
    return _COMPLETION_PRICING.get(key)


def get_embedding_pricing(model: str) -> Optional[EmbeddingPricing]:
    key = model.lower()
    return _EMBEDDING_PRICING.get(key)

