"""Native LangChain OpenRouter generation with one bounded fallback."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from typing import Any

import openrouter
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter

from app.contracts.api import ModelAttempt

from .langchain import (
    LangChainGenerationRunnable,
    as_langchain_runnable,
    private_langchain_execution,
    private_runnable_config,
)
from .protocol import (
    GenerationCandidate,
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
    UsageMetadata,
    parse_candidate,
)

OPENROUTER_PRIMARY_MODEL = "mistralai/mistral-nemo"
OPENROUTER_FALLBACK_MODEL = "meta-llama/llama-3.1-8b-instruct"
PROVIDER = "openrouter"
_SYSTEM_PROMPT = (
    "Return only the final grounded answer in the required schema. "
    "Do not include reasoning, analysis, or hidden thoughts in any field."
)
_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", "{prompt}\n\nSupplied evidence:\n{evidence}")]
)
ChainFactory = Callable[[str, int, bool], Runnable[dict[str, str], Any]]


def parse_json_schema_supported(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("OPENROUTER_JSON_SCHEMA_SUPPORTED must be a boolean")


class OpenRouterAdapter(ModelAdapter):
    """Generate through native LangChain chat models and structured output."""

    def __init__(
        self,
        api_key: str,
        *,
        primary_model: str = OPENROUTER_PRIMARY_MODEL,
        fallback_model: str | None = OPENROUTER_FALLBACK_MODEL,
        json_schema_supported: bool = True,
        chain_factory: ChainFactory | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY must not be empty.")
        if not primary_model.strip() or (fallback_model is not None and not fallback_model.strip()):
            raise ValueError("OpenRouter model IDs must not be empty.")
        self._api_key = api_key.strip()
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._json_schema_supported = json_schema_supported
        self._chain_factory = chain_factory or self._build_chain
        self._transport = transport

    @classmethod
    def from_environment(cls) -> "OpenRouterAdapter":
        return cls(
            os.environ.get("OPENROUTER_API_KEY", ""),
            primary_model=os.environ.get(
                "OPENROUTER_PRIMARY_MODEL", OPENROUTER_PRIMARY_MODEL
            ),
            fallback_model=os.environ.get(
                "OPENROUTER_FALLBACK_MODEL", OPENROUTER_FALLBACK_MODEL
            ),
            json_schema_supported=parse_json_schema_supported(
                os.environ.get("OPENROUTER_JSON_SCHEMA_SUPPORTED", "true")
            ),
        )

    @property
    def primary_model(self) -> str:
        return self._primary_model

    def as_langchain_runnable(self) -> LangChainGenerationRunnable:
        return as_langchain_runnable(self)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        attempts: list[ModelAttempt] = []
        models = tuple(
            model for model in (self._primary_model, self._fallback_model) if model is not None
        )
        deadline = asyncio.get_running_loop().time() + request.timeout.total_ms / 1000
        for index, model in enumerate(models):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            attempt_seconds = remaining / (len(models) - index)
            try:
                async with asyncio.timeout(attempt_seconds):
                    result = await self._generate_model(
                        request,
                        model,
                        timeout_ms=max(1, int(attempt_seconds * 1000)),
                    )
            except TimeoutError:
                error = GenerationError(
                    GenerationErrorCode.TIMEOUT,
                    "The answer provider timed out.",
                    retryable=True,
                )
            except GenerationError as exc:
                error = exc
            except Exception as exc:
                error = _normalize_openrouter_error(exc)
            else:
                attempts.append(ModelAttempt(model=model, outcome="success", status_code=200))
                return result.model_copy(
                    update={"usage": result.usage.model_copy(update={"attempts": attempts})}
                )

            attempts.append(
                ModelAttempt(
                    model=model,
                    outcome=error.code.value,
                    status_code=error.status_code,
                )
            )
            if index == len(models) - 1 or error.code not in {
                GenerationErrorCode.TIMEOUT,
                GenerationErrorCode.UNAVAILABLE,
                GenerationErrorCode.RATE_LIMITED,
            }:
                error.attempts = tuple(attempts)
                raise error

        raise GenerationError(
            GenerationErrorCode.TIMEOUT,
            "Model generation exceeded its timeout budget.",
            retryable=True,
            attempts=tuple(attempts),
        )

    async def _generate_model(
        self, request: GenerationRequest, model: str, *, timeout_ms: int
    ) -> GenerationResult:
        chain = self._chain_factory(model, timeout_ms, self._json_schema_supported)
        values = {"prompt": request.prompt, "evidence": _render_evidence(request)}
        with private_langchain_execution():
            response = await chain.ainvoke(values, config=private_runnable_config())

        if self._json_schema_supported:
            if not isinstance(response, dict):
                raise GenerationError(
                    GenerationErrorCode.INVALID_OUTPUT,
                    "The model returned an invalid structured candidate.",
                )
            parsing_error = response.get("parsing_error")
            if parsing_error is not None:
                raise GenerationError(
                    GenerationErrorCode.INVALID_OUTPUT,
                    "The model returned an invalid structured candidate.",
                ) from parsing_error
            raw = response.get("raw")
            _message_payload(raw)
            candidate = parse_candidate(response.get("parsed"))
        else:
            raw = response
            candidate = parse_candidate(_message_payload(raw))

        candidate = _sanitize_candidate(candidate)
        known_passages = {passage.passage_id for passage in request.passages}
        if any(citation.citation_id not in known_passages for citation in candidate.citations):
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model cited evidence that was not supplied.",
            )

        return GenerationResult(
            candidate=candidate,
            usage=_usage_metadata(raw, model),
            # Provider messages can contain the full response body. Keep the
            # trace contract, but never copy provider content into it.
            raw_output=None,
        )

    def _build_chain(
        self, model: str, timeout_ms: int, json_schema_supported: bool
    ) -> Runnable[dict[str, str], Any]:
        # max_retries=0 alone preserves the SDK's default retry policy, so the
        # explicitly injected SDK client must also disable retry_config.
        async_client = (
            httpx.AsyncClient(transport=self._transport)
            if self._transport is not None
            else None
        )
        sdk = openrouter.OpenRouter(
            api_key=self._api_key,
            async_client=async_client,
            retry_config=None,
            timeout_ms=timeout_ms,
        )
        chat = ChatOpenRouter(
            model=model,
            api_key=self._api_key,
            client=sdk,
            max_retries=0,
            timeout=timeout_ms,
            temperature=0,
            cache=False,
        )
        if json_schema_supported:
            output = chat.with_structured_output(
                GenerationCandidate,
                method="json_schema",
                include_raw=True,
                strict=True,
            )
        else:
            output = chat.bind(response_format={"type": "json_object"})
        return _PROMPT | output


def _render_evidence(request: GenerationRequest) -> str:
    return "\n\n".join(
        f"[{passage.passage_id}] page {passage.page}, {passage.section}:\n{passage.text}"
        for passage in request.passages
    )


def _message_payload(message: Any) -> object:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model returned an invalid structured candidate.",
            ) from exc
    if isinstance(content, list):
        text = "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model returned an invalid structured candidate.",
            ) from exc
    return content


def _usage_metadata(message: Any, model: str) -> UsageMetadata:
    usage = getattr(message, "usage_metadata", None) or {}
    return UsageMetadata(
        provider=PROVIDER,
        model=model,
        input_tokens=_non_negative_int(usage.get("input_tokens")),
        output_tokens=_non_negative_int(usage.get("output_tokens")),
        total_tokens=_non_negative_int(usage.get("total_tokens")),
    )


def _non_negative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _sanitize_candidate(candidate: GenerationCandidate) -> GenerationCandidate:
    if not candidate.answer_text:
        return candidate
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        candidate.answer_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return candidate.model_copy(update={"answer_text": answer})


def _normalize_openrouter_error(error: BaseException) -> GenerationError:
    if isinstance(error, GenerationError):
        return error
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    text = str(error).lower()
    if status in {401, 403} or any(
        marker in text for marker in ("authentication", "unauthorized", "api key")
    ):
        return GenerationError(
            GenerationErrorCode.AUTHENTICATION_FAILED,
            "The answer provider rejected authentication.",
            status_code=status,
        )
    if status == 429 or "rate limit" in text:
        return GenerationError(
            GenerationErrorCode.RATE_LIMITED,
            "The answer provider rate limit was reached.",
            retryable=True,
            status_code=status,
        )
    if status in {404, 408, 500, 502, 503, 504} or any(
        marker in text for marker in ("timeout", "unavailable", "overloaded", "capacity")
    ):
        return GenerationError(
            GenerationErrorCode.UNAVAILABLE,
            "The answer provider is temporarily unavailable.",
            retryable=True,
            status_code=status,
        )
    if status in {400, 409, 422}:
        return GenerationError(
            GenerationErrorCode.REQUEST_REJECTED,
            "The answer provider rejected the request.",
            status_code=status,
        )
    return GenerationError(GenerationErrorCode.UNKNOWN, "Model generation failed.")
