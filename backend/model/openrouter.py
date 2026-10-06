"""OpenRouter adapter for structured, ephemeral model generation."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from .protocol import (
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
    UsageMetadata,
    parse_candidate,
)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_PRIMARY_MODEL = "google/gemma-2-27b-it"
OPENROUTER_FALLBACK_MODEL = "nvidia/nemotron-3.5-lightning:free"


class OpenRouterAdapter(ModelAdapter):
    """Call OpenRouter while keeping provider details behind ModelAdapter."""

    def __init__(
        self,
        api_key: str,
        *,
        primary_model: str = OPENROUTER_PRIMARY_MODEL,
        fallback_model: str = OPENROUTER_FALLBACK_MODEL,
        endpoint: str = OPENROUTER_ENDPOINT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY must not be empty.")
        self._api_key = api_key
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._endpoint = endpoint
        self._transport = transport

    @classmethod
    def from_environment(cls) -> "OpenRouterAdapter":
        """Build the adapter from runtime-only configuration."""

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return cls(
            api_key,
            primary_model=os.environ.get("OPENROUTER_PRIMARY_MODEL", OPENROUTER_PRIMARY_MODEL),
            fallback_model=os.environ.get("OPENROUTER_FALLBACK_MODEL", OPENROUTER_FALLBACK_MODEL),
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate with primary, then retry eligible model failures once."""

        try:
            async with asyncio.timeout(request.timeout.total_ms / 1000):
                try:
                    return await self._generate_model(request, self._primary_model)
                except GenerationError as error:
                    if error.code not in {
                        GenerationErrorCode.TIMEOUT,
                        GenerationErrorCode.UNAVAILABLE,
                        GenerationErrorCode.RATE_LIMITED,
                    }:
                        raise
                    return await self._generate_model(request, self._fallback_model)
        except TimeoutError as exc:
            raise GenerationError(
                GenerationErrorCode.TIMEOUT,
                "Model generation exceeded its timeout budget.",
                retryable=True,
            ) from exc

    async def _generate_model(
        self, request: GenerationRequest, model: str
    ) -> GenerationResult:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only one JSON object matching this schema: "
                        '{"outcome":"answer|qualified|refusal",'
                        '"answer_text":string|null,"synthesis":boolean,'
                        '"citations":[{"passage_id":string,"page":integer,'
                        '"section":string,"excerpt":string}]}.'
                    ),
                },
                {"role": "user", "content": self._user_content(request)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dmbok-compass.web.app",
            "X-Title": "DMBOK Compass",
        }

        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=request.timeout.total_ms / 1000,
            ) as client:
                response = await client.post(self._endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise GenerationError(
                GenerationErrorCode.TIMEOUT,
                "Model generation exceeded its timeout budget.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationError(
                GenerationErrorCode.UNAVAILABLE,
                "The model provider is unavailable.",
                retryable=True,
            ) from exc

        if response.status_code in {401, 403}:
            raise GenerationError(
                GenerationErrorCode.AUTHENTICATION_FAILED,
                "Model provider authentication failed.",
            )
        if response.status_code == 429:
            raise GenerationError(
                GenerationErrorCode.RATE_LIMITED,
                "The model provider rate limit was reached.",
                retryable=True,
            )
        if response.status_code >= 500:
            raise GenerationError(
                GenerationErrorCode.UNAVAILABLE,
                "The model provider is unavailable.",
                retryable=True,
            )
        if response.status_code >= 400:
            raise GenerationError(
                GenerationErrorCode.REQUEST_REJECTED,
                "The model provider rejected the request.",
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            candidate = parse_candidate(json.loads(self._strip_json_fence(content)))
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model returned an invalid structured candidate.",
            ) from exc

        input_tokens = self._token_value(usage, "prompt_tokens")
        output_tokens = self._token_value(usage, "completion_tokens")
        return GenerationResult(
            candidate=candidate,
            usage=UsageMetadata(
                provider="openrouter",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=self._token_value(usage, "total_tokens"),
            ),
        )

    @staticmethod
    def _user_content(request: GenerationRequest) -> str:
        evidence = "\n\n".join(
            f"[{passage.passage_id}] page {passage.page}, {passage.section}:\n{passage.text}"
            for passage in request.passages
        )
        return f"{request.prompt}\n\nSupplied evidence:\n{evidence}"

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        value = content.strip()
        if value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
            if value.startswith("json"):
                value = value[4:].strip()
        return value

    @staticmethod
    def _token_value(usage: dict[str, Any], key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and value >= 0 else None
