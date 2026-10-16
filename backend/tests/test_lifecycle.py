import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.lifecycle import ShutdownCoordinator, lifecycle_middleware


def test_shutdown_cancels_in_flight_request_task_within_budget() -> None:
    async def scenario() -> None:
        coordinator = ShutdownCoordinator(timeout_seconds=0.1)
        cancelled = asyncio.Event()

        async def request() -> None:
            task = coordinator.enter()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                coordinator.leave(task)

        request_task = asyncio.create_task(request())
        await asyncio.sleep(0)
        await coordinator.shutdown()
        assert cancelled.is_set()
        assert request_task.cancelled()

    asyncio.run(scenario())


def test_api_rejects_new_requests_after_shutdown_begins() -> None:
    app = FastAPI()
    coordinator = ShutdownCoordinator()
    app.state.shutdown_coordinator = coordinator
    app.middleware("http")(lifecycle_middleware)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    previous = coordinator.stopping
    coordinator.stopping = True
    try:
        response = TestClient(app).get("/health")
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "shutting_down"
    finally:
        coordinator.stopping = previous


def test_shutdown_timeout_is_configured_as_a_positive_number(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.lifecycle import configured_shutdown_timeout

    monkeypatch.setenv("API_SHUTDOWN_TIMEOUT_SECONDS", "0")
    with pytest.raises(RuntimeError, match="positive"):
        configured_shutdown_timeout()
