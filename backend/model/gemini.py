"""Native LangChain Google generation for the existing explicit Vertex mode."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import tracing_context

from app.contracts.api import ModelAttempt

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

PRIMARY_MODEL = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"
PROVIDER = "google-gemini"
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return only the final grounded answer in the required schema. "
            "Do not include reasoning, analysis, or hidden thoughts.",
        ),
        ("human", "{prompt}\n\nSupplied evidence:\n{evidence}"),
    ]
)
ChainFactory = Callable[[str, float], Runnable[dict[str, str], Any]]


class GeminiAdapter(ModelAdapter):
    """Generate through native LangChain Vertex chat models."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str = "europe-west1",
        api_key: str | None = None,
        vertexai: bool = False,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str | None = FALLBACK_MODEL,
        chain_factory: ChainFactory | None = None,
    ) -> None:
        if vertexai and not project:
            raise ValueError("A Google Cloud project is required for Vertex AI.")
        self._project = project
        self._location = location
        self._api_key = api_key
        self._vertexai = vertexai
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._chain_factory = chain_factory or self._build_chain

    @property
    def primary_model(self) -> str:
        return self._primary_model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        models = tuple(
            model for model in (self._primary_model, self._fallback_model) if model is not None
        )
        attempts: list[ModelAttempt] = []
        deadline = asyncio.get_running_loop().time() + request.timeout.total_ms / 1000
        for index, model in enumerate(models):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            attempt_seconds = remaining / (len(models) - index)
            try:
                async with asyncio.timeout(attempt_seconds):
                    result = await self._generate_model(request, model, attempt_seconds)
            except TimeoutError:
                error = GenerationError(
                    GenerationErrorCode.TIMEOUT,
                    "The answer provider timed out.",
                    retryable=True,
                )
            except GenerationError as exc:
                error = exc
            except Exception as exc:
                error = _normalize_gemini_error(exc)
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
        self, request: GenerationRequest, model: str, timeout_seconds: float
    ) -> GenerationResult:
        chain = self._chain_factory(model, timeout_seconds)
        evidence = "\n\n".join(
            f"[{p.passage_id}] page {p.page}, {p.section}:\n{p.text}"
            for p in request.passages
        )
        with tracing_context(enabled=False):
            response = await chain.ainvoke(
                {"prompt": request.prompt, "evidence": evidence},
                config={"callbacks": []},
            )
        if not isinstance(response, dict) or response.get("parsing_error") is not None:
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model returned an invalid structured candidate.",
            )
        raw = response.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        return GenerationResult(
            candidate=parse_candidate(response.get("parsed")),
            usage=UsageMetadata(
                provider=PROVIDER,
                model=model,
                input_tokens=_usage_value(usage, "input_tokens"),
                output_tokens=_usage_value(usage, "output_tokens"),
                total_tokens=_usage_value(usage, "total_tokens"),
            ),
            raw_output=_raw_output(raw),
        )

    def _build_chain(
        self, model: str, timeout_seconds: float
    ) -> Runnable[dict[str, str], Any]:
        chat = ChatGoogleGenerativeAI(
            model=model,
            api_key=self._api_key,
            vertexai=self._vertexai,
            project=self._project,
            location=self._location,
            max_retries=0,
            timeout=timeout_seconds,
            temperature=0,
        )
        return _PROMPT | chat.with_structured_output(
            GenerationCandidate, method="json_schema", include_raw=True
        )


def create_gemini_adapter(
    *,
    api_key: str | None = None,
    project: str | None = None,
    location: str = "europe-west1",
    vertexai: bool = False,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str | None = FALLBACK_MODEL,
) -> GeminiAdapter:
    return GeminiAdapter(
        api_key=api_key,
        project=project,
        location=location,
        vertexai=vertexai,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )


def _usage_value(usage: dict[str, Any], name: str) -> int | None:
    value = usage.get(name)
    return value if isinstance(value, int) and value >= 0 else None


def _raw_output(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content[:20_000]
    return json.dumps(content, default=str)[:20_000] if content is not None else None


def _normalize_gemini_error(error: BaseException) -> GenerationError:
    if isinstance(error, GenerationError):
        return error
    if getattr(error, "provider_wide", False):
        return GenerationError(
            GenerationErrorCode.UNKNOWN,
            "The answer provider is unavailable.",
        )
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    text = str(error).lower()
    if status in {401, 403} or any(
        marker in text for marker in ("unauthenticated", "permission denied", "api key")
    ):
        return GenerationError(
            GenerationErrorCode.AUTHENTICATION_FAILED,
            "The answer provider rejected authentication.",
            status_code=status,
        )
    if status == 429 or any(
        marker in text for marker in ("rate limit", "resource exhausted")
    ):
        return GenerationError(
            GenerationErrorCode.RATE_LIMITED,
            "The answer provider rate limit was reached.",
            retryable=True,
            status_code=status,
        )
    if status in {408, 500, 502, 503, 504} or any(
        marker in text for marker in ("timeout", "unavailable", "overloaded")
    ):
        return GenerationError(
            GenerationErrorCode.UNAVAILABLE,
            "The answer provider is temporarily unavailable.",
            retryable=True,
            status_code=status,
        )
    if status in {400, 422} or any(
        marker in text for marker in ("invalid argument", "safety", "blocked")
    ):
        return GenerationError(
            GenerationErrorCode.REQUEST_REJECTED,
            "The answer provider rejected the request.",
            status_code=status,
        )
    return GenerationError(GenerationErrorCode.UNKNOWN, "Model generation failed.")
