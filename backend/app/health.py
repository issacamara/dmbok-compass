"""Content-free liveness and bounded release-readiness checks."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from app.settings import RuntimeConfiguration

READINESS_TIMEOUT_SECONDS = 2.0
READINESS_CACHE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class SelectedRelease:
    """The non-secret serving identity required by a readiness decision."""

    release_id: str
    configuration_version_id: str
    corpus_version_id: str
    vector_collection: str
    vector_schema_version: str
    artifact_id: str | None = None


class ReleaseResolver(Protocol):
    def selected_release(self) -> SelectedRelease | Mapping[str, Any] | None:
        """Return the selected release, or ``None`` when no release is selected."""


class ServiceChecker(Protocol):
    def check(self, release: SelectedRelease) -> None:
        """Verify required backing-service access for the selected release."""


class FirestoreSelectedReleaseResolver:
    """Resolve and validate the approved release against its active corpus."""

    def __init__(self, decisions: Any, manifests: Any, corpus: Any) -> None:
        self.decisions = decisions
        self.manifests = manifests
        self.corpus = corpus

    def selected_release(self) -> SelectedRelease | None:
        decision = self.decisions.current_approved()
        if decision is None:
            return None
        manifest = self.manifests.get(decision.release_id)
        active = self.corpus.active()
        if manifest is None or active is None or active.version_id != decision.corpus_version_id:
            return None
        if (
            manifest.configuration_version_id != decision.configuration_version_id
            or manifest.corpus_version_id != decision.corpus_version_id
        ):
            return None
        binding = active.serving_binding
        if binding is None:
            return None
        return SelectedRelease(
            release_id=decision.release_id,
            configuration_version_id=decision.configuration_version_id,
            corpus_version_id=decision.corpus_version_id,
            vector_collection=binding.vector_collection,
            vector_schema_version=binding.vector_schema_version,
            artifact_id=manifest.candidate_digest,
        )


class NoopServiceChecker:
    """The Firestore resolver itself performs the required bounded reads."""

    def check(self, release: SelectedRelease) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    ready: bool
    checks: Mapping[str, Mapping[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": {name: dict(value) for name, value in self.checks.items()},
        }


class ReadinessChecker:
    """Run one bounded readiness check and share its result with concurrent callers."""

    def __init__(
        self,
        configuration: RuntimeConfiguration,
        release_resolver: ReleaseResolver,
        service_checker: ServiceChecker,
        *,
        timeout_seconds: float = READINESS_TIMEOUT_SECONDS,
        cache_seconds: float = READINESS_CACHE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > READINESS_TIMEOUT_SECONDS:
            raise ValueError("readiness timeout must be between zero and two seconds")
        if cache_seconds < 0:
            raise ValueError("readiness cache duration must not be negative")
        self.configuration = configuration
        self.release_resolver = release_resolver
        self.service_checker = service_checker
        self.timeout_seconds = timeout_seconds
        self.cache_seconds = cache_seconds
        self._clock = clock
        self._cached: tuple[float, ReadinessResult] | None = None
        self._in_flight: asyncio.Task[ReadinessResult] | None = None
        self._lock = asyncio.Lock()

    async def probe(self) -> ReadinessResult:
        now = self._clock()
        if self._cached is not None and self._cached[0] > now:
            return self._cached[1]

        async with self._lock:
            now = self._clock()
            if self._cached is not None and self._cached[0] > now:
                return self._cached[1]
            if self._in_flight is None:
                self._in_flight = asyncio.create_task(self._run())
            task = self._in_flight
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._in_flight is task:
                        self._in_flight = None

    async def _run(self) -> ReadinessResult:
        try:
            result = await asyncio.wait_for(self._check(), timeout=self.timeout_seconds)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            result = ReadinessResult(False, {"deadline": {"status": "failed"}})
        except Exception:
            # Readiness must fail closed and must never expose exception text or
            # provider/storage details to an unauthenticated probe.
            result = ReadinessResult(False, {"dependencies": {"status": "failed"}})
        self._cached = (self._clock() + self.cache_seconds, result)
        return result

    async def _check(self) -> ReadinessResult:
        release = await _resolve(self.release_resolver.selected_release)
        if release is None:
            return ReadinessResult(False, {"selected_release": {"status": "failed"}})
        try:
            selected = _normalise_release(release)
        except (TypeError, ValueError):
            return ReadinessResult(False, {"selected_release": {"status": "failed"}})

        compatibility = _compatible(selected, self.configuration)
        if not compatibility:
            return ReadinessResult(False, {"release_compatibility": {"status": "failed"}})

        await _resolve(self.service_checker.check, selected)
        return ReadinessResult(
            True,
            {
                "selected_release": {"status": "ok"},
                "release_compatibility": {"status": "ok"},
                "dependencies": {"status": "ok"},
            },
        )


def _normalise_release(value: SelectedRelease | Mapping[str, Any]) -> SelectedRelease:
    if isinstance(value, SelectedRelease):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("release must be a mapping")
    return SelectedRelease(
        release_id=_required(value, "release_id"),
        configuration_version_id=_required(value, "configuration_version_id"),
        corpus_version_id=_required(value, "corpus_version_id"),
        vector_collection=_required(value, "vector_collection"),
        vector_schema_version=_required(value, "vector_schema_version"),
        artifact_id=value.get("artifact_id"),
    )


def _required(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"missing release field: {key}")
    return item


def _compatible(release: SelectedRelease, configuration: RuntimeConfiguration) -> bool:
    expected_collection = configuration.vector_collection
    expected_schema = configuration.vector_schema_version
    if release.vector_collection != expected_collection or release.vector_schema_version != expected_schema:
        return False
    configured_corpus = getattr(configuration, "corpus_version_id", None)
    configured_configuration = getattr(configuration, "configuration_version_id", None)
    return not (
        configured_corpus is not None and release.corpus_version_id != configured_corpus
    ) and not (
        configured_configuration is not None
        and release.configuration_version_id != configured_configuration
    )


async def _resolve(function: Callable[..., Any], *args: Any) -> Any:
    result = function(*args)
    return await result if inspect.isawaitable(result) else result

