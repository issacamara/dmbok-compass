"""LangChain runnable boundary for the ephemeral generation contract.

The bridge exposes the existing provider-neutral request and result types. It
does not create callbacks, tracing handlers, or telemetry; callers cannot
provide callbacks because those can contain interaction content.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langchain_core.globals import set_debug, set_verbose
from langchain_core.runnables import Runnable, RunnableConfig
from langsmith import tracing_context

from .protocol import GenerationRequest, GenerationResult, ModelAdapter


_CONTENT_OBSERVABILITY_ENVIRONMENT = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_DEBUG",
    "LANGCHAIN_VERBOSE",
)


def enforce_private_langchain_runtime() -> None:
    """Disable process and LangChain switches that can export interaction data.

    These values are deliberately set rather than trusted from the deployment
    environment: a request's prompt, evidence, or output must never become a
    trace or verbose diagnostic merely because an operator enabled a library
    switch for a different workload in the same process.
    """

    for name in _CONTENT_OBSERVABILITY_ENVIRONMENT:
        os.environ[name] = "false"
    set_debug(False)
    set_verbose(False)


@contextmanager
def private_langchain_execution() -> Iterator[None]:
    """Run one native call with content tracing disabled regardless of env."""

    enforce_private_langchain_runtime()
    with tracing_context(enabled=False):
        yield


def private_runnable_config() -> RunnableConfig:
    """Return the only configuration passed to a native LangChain runnable."""

    return {"callbacks": []}


def reject_caller_runnable_configuration(
    config: RunnableConfig | None, kwargs: dict[str, Any]
) -> None:
    """Prevent callers from adding callbacks or tracing configuration."""

    if (config and config.get("callbacks")) or kwargs.get("callbacks"):
        raise ValueError("Content-bearing LangChain callbacks are not supported.")
    if config:
        raise ValueError("Caller-supplied LangChain configuration is not supported.")
    if kwargs:
        raise ValueError("Caller-supplied LangChain keyword configuration is not supported.")


class LangChainGenerationRunnable(Runnable[GenerationRequest, GenerationResult]):
    """Bounded LangChain-compatible runnable backed by a ``ModelAdapter``."""

    def __init__(self, adapter: ModelAdapter) -> None:
        enforce_private_langchain_runtime()
        self._adapter = adapter

    def invoke(
        self,
        input: GenerationRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Synchronously invoke the adapter without accepting callbacks."""
        reject_caller_runnable_configuration(config, kwargs)
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
        reject_caller_runnable_configuration(config, kwargs)
        request = input if isinstance(input, GenerationRequest) else GenerationRequest.model_validate(input)
        return await self._adapter.generate(request)


def as_langchain_runnable(adapter: ModelAdapter) -> LangChainGenerationRunnable:
    """Expose an adapter through the stable bounded runnable contract."""
    return LangChainGenerationRunnable(adapter)
