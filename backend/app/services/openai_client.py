"""Async OpenAI client wrapper and related value objects.

Classes:
    ChatSample: Lightweight container for an individual chat completion.
    EmbeddingBatch: Collected embedding vectors plus metadata returned from the embeddings API.
    OpenAIService: Handles chat sampling, embeddings, and optional discourse tagging with retry semantics.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from openai import AsyncOpenAI
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings

SYSTEM_PROMPT = "Return a concise answer with reasoning steps suppressed; vary framing and examples."
SEGMENT_TAG_PROMPT = (
    "You will receive numbered sentences derived from a single user prompt. "
    "Assign each sentence a discourse role chosen from: problem, forecast, solution, risk, opportunity, action, background, conclusion. "
    "Respond with valid JSON: [{\"index\": <number>, \"role\": <lowercase role>, \"confidence\": <0-1 float>}]."
)

_EMBED_BATCH_MAX = 256


@dataclass(slots=True)
class ChatSample:
    index: int
    text: str
    tokens: Optional[int]
    finish_reason: Optional[str]
    usage: Optional[dict[str, Any]]


@dataclass(slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model: str
    dim: int
    model_revision: str | None = None
    provider: str = "openai"


class OpenAIService:
    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        settings = get_settings()
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        if client is not None:
            self._client = client
        elif api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            self._client = None
        self._settings = settings

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def sample_chat(
        self,
        *,
        prompt: str,
        n: int,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        jitter_token: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> list[ChatSample]:
        if self._client is None:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")

        chosen_model = model or self._settings.openai_chat_model
        temp = temperature if temperature is not None else self._settings.default_temperature
        nucleus = top_p if top_p is not None else self._settings.default_top_p

        sys_prompt = system_prompt.strip() if system_prompt else SYSTEM_PROMPT

        async def _call(index: int) -> ChatSample:
            call_seed = seed + index if seed is not None else None
            jitter = jitter_token or secrets.token_hex(4)
            user_prompt = prompt if jitter_token is None else f"[{jitter}] {prompt}"

            payload = dict(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temp,
                top_p=nucleus,
                n=1,
            )
            if call_seed is not None:
                payload["seed"] = call_seed
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens

            response = await _retry_chat(self._client, payload)
            choice = response.choices[0]
            message_content = getattr(choice.message, "content", "") or ""
            usage_dict = (
                response.usage.model_dump()  # type: ignore[attr-defined]
                if getattr(response.usage, "model_dump", None)
                else dict(response.usage or {})
            )
            completion_tokens = usage_dict.get("completion_tokens") if usage_dict else None

            return ChatSample(
                index=index,
                text=message_content.strip(),
                tokens=completion_tokens,
                finish_reason=getattr(choice, "finish_reason", None),
                usage=usage_dict,
            )

        tasks = [_call(i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model: Optional[str] = None,
    ) -> EmbeddingBatch:
        docs = list(texts)
        if self._client is None:
            raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")

        if not docs:
            return EmbeddingBatch(vectors=[], model=model or self._settings.openai_embedding_model, dim=0)

        chosen_model = model or self._settings.openai_embedding_model
        vectors: list[list[float]] = []
        dim = 0
        model_revision: str | None = None

        for start in range(0, len(docs), _EMBED_BATCH_MAX):
            chunk = docs[start : start + _EMBED_BATCH_MAX]
            payload = dict(model=chosen_model, input=chunk)
            try:
                response = await _retry_embeddings(self._client, payload)
            except RetryError as exc:  # pragma: no cover - surfaces original error message
                raise exc.last_attempt.result()  # type: ignore[misc]

            chunk_vectors = [item.embedding for item in response.data]
            vectors.extend(chunk_vectors)
            if not dim and chunk_vectors:
                dim = len(chunk_vectors[0])
            response_model = getattr(response, "model", None)
            if response_model:
                model_revision = response_model

        return EmbeddingBatch(
            vectors=vectors,
            model=chosen_model,
            dim=dim,
            model_revision=model_revision,
            provider="openai",
        )

    async def discourse_tag_segments(
        self,
        segments: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> list[str | None]:
        if not segments:
            return []

        if self._client is None or not self._settings.enable_discourse_tagging:
            return [None for _ in segments]

        chosen_model = model or self._settings.discourse_tagging_model or self._settings.openai_chat_model
        enumerated = [f"{idx}. {text}" for idx, text in enumerate(segments)]
        user_prompt = "\n".join(enumerated)

        payload = dict(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SEGMENT_TAG_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            top_p=0.9,
            n=1,
        )

        try:
            response = await _retry_chat(self._client, payload)
        except Exception:
            return [None for _ in segments]

        content = getattr(response.choices[0].message, "content", "") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return [None for _ in segments]

        roles: list[str | None] = [None for _ in segments]
        for item in data:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(roles):
                role = item.get("role")
                if isinstance(role, str):
                    roles[idx] = role.lower()
        return roles


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
async def _retry_chat(client: AsyncOpenAI, payload: dict[str, Any]):
    return await client.chat.completions.create(**payload)


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
async def _retry_embeddings(client: AsyncOpenAI, payload: dict[str, Any]):
    fallback_model = get_settings().openai_embedding_fallback_model
    try:
        return await client.embeddings.create(**payload)
    except Exception:  # pragma: no cover - fallback executed infrequently
        payload["model"] = fallback_model
        return await client.embeddings.create(**payload)

