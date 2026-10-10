"""Google Gemini implementation of the provider-neutral model adapter.

The adapter accepts only request-scoped data and returns only validated
candidate data plus content-free usage metadata.  Credentials are supplied by
the runtime (for example, from Secret Manager); this module never logs them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Protocol

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


class _AsyncModels(Protocol):
    async def generate_content(
        self, *, model: str, contents: str, config: Mapping[str, object]
    ) -> object:
        """Generate content using the subset of the SDK used by this adapter."""


class _GeminiClient(Protocol):
    @property
    def aio(self) -> Any:
        """Return the SDK's asynchronous client facade."""


class GeminiAdapter(ModelAdapter):
    """Call Gemini's primary model and use one same-provider fallback."""

    def __init__(
        self,
        client: _GeminiClient,
        *,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str = FALLBACK_MODEL,
    ) -> None:
        if not primary_model.strip() or not fallback_model.strip():
            raise ValueError("Gemini model identifiers must be non-empty")
        self._client = client
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate from the primary model, falling back once when eligible."""
        try:
            return await self._generate_with_model(request, self.primary_model)
        except GenerationError as primary_error:
            if primary_error.code not in {
                GenerationErrorCode.TIMEOUT,
                GenerationErrorCode.UNAVAILABLE,
                GenerationErrorCode.RATE_LIMITED,
            }:
                raise
            try:
                return await self._generate_with_model(request, self.fallback_model)
            except GenerationError as fallback_error:
                raise fallback_error from primary_error

    async def _generate_with_model(
        self, request: GenerationRequest, model: str
    ) -> GenerationResult:
        try:
            response = await asyncio.wait_for(
                self._models.generate_content(
                    model=model,
                    contents=_render_contents(request),
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": GenerationCandidate.model_json_schema(),
                    },
                ),
                timeout=request.timeout.total_ms / 1000,
            )
        except TimeoutError as exc:
            raise GenerationError(
                GenerationErrorCode.TIMEOUT,
                "Model generation exceeded its timeout budget.",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise _normalize_gemini_error(exc) from exc

        try:
            candidate = parse_candidate(_response_payload(response))
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                GenerationErrorCode.INVALID_OUTPUT,
                "The model returned an invalid structured candidate.",
            ) from exc

        return GenerationResult(
            candidate=candidate,
            usage=_usage_metadata(response, model),
            raw_output=getattr(response, "text", None),
        )

    @property
    def _models(self) -> _AsyncModels:
        try:
            return self._client.aio.models
        except AttributeError as exc:
            raise RuntimeError("Gemini client does not expose an async models API") from exc


def create_gemini_adapter(
    *,
    api_key: str | None = None,
    project: str | None = None,
    location: str = "europe-west1",
    vertexai: bool = False,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
) -> GeminiAdapter:
    """Create the production adapter using runtime-injected credentials.

    The caller should source ``api_key`` from Secret Manager.  When omitted,
    the Google SDK uses its normal environment or workload-identity settings.
    """
    from google import genai

    if vertexai:
        if not project:
            raise ValueError("A Google Cloud project is required for Vertex AI.")
        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
    return GeminiAdapter(
        client,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )


def _render_contents(request: GenerationRequest) -> str:
    """Build an ephemeral provider request without adding persistent fields."""
    evidence = "\n\n".join(
        f"[{passage.passage_id}] page {passage.page}, {passage.section}:\n{passage.text}"
        for passage in request.passages
    )
    return f"{request.prompt}\n\nSupplied evidence:\n{evidence}"


def _response_payload(response: object) -> object:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed.model_dump() if isinstance(parsed, GenerationCandidate) else parsed
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Gemini returned no structured response")
    return json.loads(text)


def _usage_metadata(response: object, model: str) -> UsageMetadata:
    usage = getattr(response, "usage_metadata", None)
    input_tokens = _usage_value(usage, "prompt_token_count")
    output_tokens = _usage_value(usage, "candidates_token_count")
    total_tokens = _usage_value(usage, "total_token_count")
    return UsageMetadata(
        provider=PROVIDER,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _usage_value(usage: object, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return value if isinstance(value, int) and value >= 0 else None


def _normalize_gemini_error(error: BaseException) -> GenerationError:
    if getattr(error, "provider_wide", False):
        return GenerationError(
            GenerationErrorCode.UNKNOWN,
            "The answer provider is unavailable.",
        )
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    text = str(error).lower()
    if isinstance(status_code, str) and status_code.isdigit():
        status_code = int(status_code)

    if status_code in {401, 403} or any(
        marker in text for marker in ("unauthenticated", "permission denied", "api key")
    ):
        return GenerationError(
            GenerationErrorCode.AUTHENTICATION_FAILED,
            "The answer provider rejected authentication.",
        )
    if status_code == 429 or any(marker in text for marker in ("rate limit", "resource exhausted")):
        return GenerationError(
            GenerationErrorCode.RATE_LIMITED,
            "The answer provider rate limit was reached.",
            retryable=True,
        )
    if status_code in {408, 500, 502, 503, 504} or any(
        marker in text for marker in ("timeout", "temporarily unavailable", "overloaded")
    ):
        return GenerationError(
            GenerationErrorCode.UNAVAILABLE,
            "The answer provider is temporarily unavailable.",
            retryable=True,
        )
    if status_code in {400, 422} or any(
        marker in text for marker in ("invalid argument", "safety", "blocked")
    ):
        return GenerationError(
            GenerationErrorCode.REQUEST_REJECTED,
            "The answer provider rejected the request.",
        )
    return GenerationError(GenerationErrorCode.UNKNOWN, "Model generation failed.")
