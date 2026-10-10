"""OpenRouter adapter for structured, ephemeral model generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any

import httpx
from app.contracts.api import ModelAttempt

from .langchain import LangChainGenerationRunnable, as_langchain_runnable
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
OPENROUTER_PRIMARY_MODEL = "nvidia/nemotron-3.5-lightning:free"
OPENROUTER_FALLBACK_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
logger = logging.getLogger("dmbok_compass.model")


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

    def as_langchain_runnable(self) -> LangChainGenerationRunnable:
        """Expose this adapter without changing provider routing behavior."""
        return as_langchain_runnable(self)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate with primary, then retry eligible model failures once."""

        attempts: list[ModelAttempt] = []
        active_model = self._primary_model
        deadline = asyncio.get_running_loop().time() + request.timeout.total_ms / 1000
        for index, model in enumerate((self._primary_model, self._fallback_model)):
            active_model = model
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            # Reserve time for the fallback instead of letting the primary
            # consume the entire request budget.
            attempt_budget = remaining / (2 - index)
            attempt_request = request.model_copy(update={
                "timeout": request.timeout.model_copy(update={
                    "total_ms": max(1, int(attempt_budget * 1000)),
                })
            })
            try:
                async with asyncio.timeout(attempt_budget):
                    result = await self._generate_model(attempt_request, model)
            except TimeoutError:
                attempts.append(ModelAttempt(model=model, outcome="timeout"))
                if index == 1:
                    break
                continue
            except GenerationError as error:
                attempts.append(ModelAttempt(
                    model=model, outcome=error.code.value, status_code=error.status_code
                ))
                error.attempts = tuple(attempts)
                if index == 1 or error.code not in {
                    GenerationErrorCode.TIMEOUT,
                    GenerationErrorCode.UNAVAILABLE,
                    GenerationErrorCode.RATE_LIMITED,
                }:
                    raise
            else:
                attempts.append(ModelAttempt(model=model, outcome="success", status_code=200))
                return result.model_copy(update={
                    "usage": result.usage.model_copy(update={"attempts": attempts})
                })

        raise GenerationError(
            GenerationErrorCode.TIMEOUT,
            "Model generation exceeded its timeout budget.",
            retryable=True,
            attempts=tuple(attempts or [ModelAttempt(model=active_model, outcome="timeout")]),
        )

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
                        '"citations":[{"citation_id":string,"page":integer,'
                        '"section":string,"excerpt":string}]}.'
                    ),
                },
                {"role": "user", "content": self._user_content(request)},
            ],
            "temperature": 0,
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
                status_code=response.status_code,
            )
        if response.status_code == 429:
            raise GenerationError(
                GenerationErrorCode.RATE_LIMITED,
                "The model provider rate limit was reached.",
                retryable=True,
                status_code=429,
            )
        # OpenRouter uses 404 when a configured model/route is no longer
        # available. Treat that as a provider outage so the fallback model can
        # still serve grounded answers.
        if response.status_code == 404 or response.status_code >= 500:
            raise GenerationError(
                GenerationErrorCode.UNAVAILABLE,
                "The model provider is unavailable.",
                retryable=True,
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise GenerationError(
                GenerationErrorCode.REQUEST_REJECTED,
                "The model provider rejected the request.",
                status_code=response.status_code,
            )

        try:
            body = response.json()
            raw_content = body["choices"][0]["message"]["content"]
            content = self._content_text(raw_content)
            extracted = self._extract_json(content)
            logger.info("openrouter_response %s", json.dumps({
                "model": model,
                "content_type": type(raw_content).__name__,
                "content_length": len(content),
                "json_length": len(extracted),
                "json_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
            }, sort_keys=True))
            candidate = parse_candidate(json.loads(extracted))
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.info("openrouter_response_invalid %s", json.dumps({
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
            }, sort_keys=True))
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
            raw_output=content,
        )

    @staticmethod
    def _user_content(request: GenerationRequest) -> str:
        evidence = "\n\n".join(
            f"[{passage.passage_id}] page {passage.page}, {passage.section}:\n{passage.text}"
            for passage in request.passages
        )
        return f"{request.prompt}\n\nSupplied evidence:\n{evidence}"

    @staticmethod
    def _content_text(content: object) -> str:
        """Normalize OpenRouter string or content-block responses."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            if parts:
                return "".join(parts)
        raise TypeError("OpenRouter returned no text content")

    @staticmethod
    def _extract_json(content: str) -> str:
        """Extract the structured object when reasoning surrounds JSON."""
        value = content.strip()
        if value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
            if value.startswith("json"):
                value = value[4:].strip()
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            raise ValueError("OpenRouter response contained no JSON object")
        return value[start:end + 1]

    @staticmethod
    def _token_value(usage: dict[str, Any], key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and value >= 0 else None
