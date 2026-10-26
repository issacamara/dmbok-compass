import asyncio
from dataclasses import replace

from app.contracts import RuntimeSettings
from app.health import ReadinessChecker, SelectedRelease
from app.settings import RuntimeConfiguration


def configuration() -> RuntimeConfiguration:
    return RuntimeConfiguration(
        runtime=RuntimeSettings(
            project="test-project",
            location="test-location",
            provider="openrouter",
            primary_model="primary",
            fallback_model="fallback",
            secret_version="secret-ref",
        ),
        vector_collection="chunks",
        vector_schema_version="2",
        embedding_model="embedding",
        embedding_dimensions=768,
        retrieval_strategy="dense",
        request_timeout_ms=14000,
        generation_timeout_ms=8000,
        output_token_cap=1024,
        max_model_attempts=2,
        json_schema_supported=True,
    )


def release() -> SelectedRelease:
    return SelectedRelease("release-1", "config-1", "corpus-1", "chunks", "2")


class Resolver:
    def __init__(self, selected: SelectedRelease | None) -> None:
        self.selected = selected
        self.calls = 0

    def selected_release(self) -> SelectedRelease | None:
        self.calls += 1
        return self.selected


class Services:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def check(self, selected: SelectedRelease) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("store unavailable")


def test_liveness_does_not_depend_on_readiness() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    app.state.readiness_checker = None
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthy_readiness_is_cached_and_compatible() -> None:
    resolver = Resolver(release())
    services = Services()
    checker = ReadinessChecker(configuration(), resolver, services, cache_seconds=30)

    first = asyncio.run(checker.probe())
    second = asyncio.run(checker.probe())

    assert first.ready and second.ready
    assert resolver.calls == 1
    assert services.calls == 1


def test_readiness_fails_closed_for_missing_release_or_store() -> None:
    missing = ReadinessChecker(configuration(), Resolver(None), Services())
    failed_store = ReadinessChecker(configuration(), Resolver(release()), Services(fail=True))

    assert not asyncio.run(missing.probe()).ready
    assert not asyncio.run(failed_store.probe()).ready


def test_readiness_rejects_incompatible_collection() -> None:
    resolver = Resolver(replace(release(), vector_collection="other"))
    checker = ReadinessChecker(configuration(), resolver, Services())

    result = asyncio.run(checker.probe())
    assert not result.ready
    assert result.checks == {"release_compatibility": {"status": "failed"}}

