"""Provider fallback with one shared request deadline."""

from __future__ import annotations

import asyncio

from app.contracts.api import ModelAttempt

from .protocol import (
    GenerationError,
    GenerationErrorCode,
    GenerationRequest,
    GenerationResult,
    ModelAdapter,
)


class FallbackAdapter(ModelAdapter):
    """Try a primary provider, then a secondary provider within one budget."""

    def __init__(self, primary: ModelAdapter, secondary: ModelAdapter) -> None:
        self._primary = primary
        self._secondary = secondary

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        deadline = asyncio.get_running_loop().time() + request.timeout.total_ms / 1000
        attempts: list[ModelAttempt] = []
        adapters = (self._primary, self._secondary)
        for index, adapter in enumerate(adapters):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            budget = remaining / (len(adapters) - index)
            attempt_request = request.model_copy(update={
                "timeout": request.timeout.model_copy(update={
                    "total_ms": max(1, int(budget * 1000)),
                })
            })
            try:
                async with asyncio.timeout(budget):
                    result = await adapter.generate(attempt_request)
            except TimeoutError:
                attempts.append(ModelAttempt(model=_adapter_name(adapter), outcome="timeout"))
                continue
            except GenerationError as error:
                attempts.extend(error.attempts or (
                    ModelAttempt(model=_adapter_name(adapter), outcome=error.code.value),
                ))
                if index == len(adapters) - 1 or error.code not in {
                    GenerationErrorCode.TIMEOUT,
                    GenerationErrorCode.UNAVAILABLE,
                    GenerationErrorCode.RATE_LIMITED,
                }:
                    error.attempts = tuple(attempts)
                    raise
                continue
            attempts.extend(result.usage.attempts or (
                ModelAttempt(model=result.usage.model, outcome="success", status_code=200),
            ))
            return result.model_copy(update={
                "usage": result.usage.model_copy(update={"attempts": attempts})
            })

        raise GenerationError(
            GenerationErrorCode.TIMEOUT,
            "Model generation exceeded its timeout budget.",
            retryable=True,
            attempts=tuple(attempts),
        )


def _adapter_name(adapter: ModelAdapter) -> str:
    return (
        getattr(adapter, "primary_model", None)
        or getattr(adapter, "_primary_model", None)
        or getattr(getattr(adapter, "_usage", None), "model", None)
        or adapter.__class__.__name__
    )
