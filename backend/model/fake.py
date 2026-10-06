"""Deterministic in-process adapter for contract and integration tests."""

from __future__ import annotations

import asyncio

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


class FakeAdapter(ModelAdapter):
    """Return a configured candidate, or reproduce a normalized failure."""

    def __init__(
        self,
        candidate: GenerationCandidate | dict[str, object],
        *,
        provider: str = "fake",
        model: str = "fake-model",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        delay_ms: int = 0,
        failure: GenerationError | None = None,
    ) -> None:
        self._candidate = candidate
        self._usage = UsageMetadata(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(input_tokens or 0) + (output_tokens or 0)
            if input_tokens is not None and output_tokens is not None
            else None,
        )
        self._delay_ms = delay_ms
        self._failure = failure

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._delay_ms < 0:
            raise GenerationError(GenerationErrorCode.UNKNOWN, "Invalid fake adapter delay.")
        try:
            async with asyncio.timeout(request.timeout.total_ms / 1000):
                if self._delay_ms:
                    await asyncio.sleep(self._delay_ms / 1000)
                if self._failure is not None:
                    raise self._failure
                return GenerationResult(
                    candidate=parse_candidate(self._candidate),
                    usage=self._usage,
                )
        except TimeoutError as exc:
            raise GenerationError(
                GenerationErrorCode.TIMEOUT,
                "Model generation exceeded its timeout budget.",
                retryable=True,
            ) from exc
