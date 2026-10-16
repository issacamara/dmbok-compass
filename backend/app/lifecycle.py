"""Process lifecycle coordination for the API runtime."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request
from fastapi.responses import JSONResponse


class ShutdownCoordinator:
    """Stop admission and cancel in-flight requests during process shutdown."""

    def __init__(self, timeout_seconds: float = 25.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.stopping = False
        self._tasks: set[asyncio.Task[object]] = set()

    def enter(self) -> asyncio.Task[object] | None:
        if self.stopping:
            return None
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        return task

    def leave(self, task: asyncio.Task[object] | None) -> None:
        if task is not None:
            self._tasks.discard(task)

    async def shutdown(self) -> None:
        self.stopping = True
        current = asyncio.current_task()
        pending = tuple(task for task in self._tasks if task is not current)
        for task in pending:
            task.cancel()
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), self.timeout_seconds
                )
            except TimeoutError:
                # The server's own graceful timeout remains the final
                # disposal boundary for code that ignores cancellation.
                pass


def configured_shutdown_timeout() -> float:
    raw = os.environ.get("API_SHUTDOWN_TIMEOUT_SECONDS", "25")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError("API_SHUTDOWN_TIMEOUT_SECONDS must be a number") from exc
    if value <= 0:
        raise RuntimeError("API_SHUTDOWN_TIMEOUT_SECONDS must be positive")
    return value


async def lifecycle_middleware(request: Request, call_next):
    coordinator: ShutdownCoordinator = request.app.state.shutdown_coordinator
    task = coordinator.enter()
    if task is None:
        return JSONResponse(
            status_code=503,
            content={"detail": {"code": "shutting_down", "message": "The service is shutting down."}},
            headers={"Connection": "close"},
        )
    try:
        return await call_next(request)
    finally:
        coordinator.leave(task)


@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    yield
    await app.state.shutdown_coordinator.shutdown()
