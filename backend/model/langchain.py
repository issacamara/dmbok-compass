"""LangChain runnable boundary for the ephemeral generation contract.

The bridge exposes the existing provider-neutral request and result types. It
does not create callbacks, tracing handlers, or telemetry; callers cannot
provide callbacks because those can contain interaction content.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig

from .protocol import GenerationRequest, GenerationResult, ModelAdapter


class LangChainGenerationRunnable(Runnable[GenerationRequest, GenerationResult]):
    """Bounded LangChain-compatible runnable backed by a ``ModelAdapter``."""

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter

    def invoke(
        self,
        input: GenerationRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Synchronously invoke the adapter without accepting callbacks."""
        self._reject_callbacks(config)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, config, **kwargs))
        raise RuntimeError("Use ainvoke() when already running an event loop.")

    async def ainvoke(
        self,
        input: GenerationRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Asynchronously invoke the adapter with a validated bounded request."""
        self._reject_callbacks(config)
        request = input if isinstance(input, GenerationRequest) else GenerationRequest.model_validate(input)
        return await self._adapter.generate(request)

    @staticmethod
    def _reject_callbacks(config: RunnableConfig | None) -> None:
        if config and config.get("callbacks"):
            raise ValueError("Content-bearing LangChain callbacks are not supported.")


def as_langchain_runnable(adapter: ModelAdapter) -> LangChainGenerationRunnable:
    """Expose an adapter through the stable bounded runnable contract."""
    return LangChainGenerationRunnable(adapter)
