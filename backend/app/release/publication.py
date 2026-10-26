"""Approved release publication and serving-corpus resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Protocol

from app.contracts import CorpusVersion, ReleaseDecision
from app.release.firestore import ReleaseManifest


class PublicationError(ValueError):
    """Raised when an approved release cannot safely become serving."""


class ConcurrentPublicationError(PublicationError):
    """Raised when the serving release changed during compare-and-set."""


@dataclass(frozen=True, slots=True)
class PublishedRelease:
    release_id: str
    artifact_id: str
    configuration_version_id: str
    corpus_version_id: str
    vector_collection: str
    vector_schema_version: str
    source_uri: str
    published_at: datetime


class ServingReleaseStore(Protocol):
    def current(self) -> PublishedRelease | None: ...
    def previous(self) -> PublishedRelease | None: ...
    def publish(self, release: PublishedRelease, *, expected_previous_release_id: str | None) -> PublishedRelease: ...
    def rollback(self, *, expected_current_release_id: str) -> PublishedRelease: ...


class InMemoryServingReleaseStore:
    """Thread-safe reference store used by tests and offline workflows."""

    def __init__(self) -> None:
        self._current: PublishedRelease | None = None
        self._previous: PublishedRelease | None = None
        self._lock = RLock()

    def current(self) -> PublishedRelease | None:
        with self._lock:
            return deepcopy(self._current)

    def previous(self) -> PublishedRelease | None:
        with self._lock:
            return deepcopy(self._previous)

    def publish(self, release: PublishedRelease, *, expected_previous_release_id: str | None) -> PublishedRelease:
        with self._lock:
            current_id = self._current.release_id if self._current else None
            if current_id != expected_previous_release_id:
                raise ConcurrentPublicationError("serving release changed before publication could commit")
            self._previous = self._current
            self._current = release
            return deepcopy(release)

    def rollback(self, *, expected_current_release_id: str) -> PublishedRelease:
        with self._lock:
            if self._current is None or self._current.release_id != expected_current_release_id:
                raise ConcurrentPublicationError("serving release changed before rollback could commit")
            if self._previous is None:
                raise PublicationError("no previous serving release is available for rollback")
            restored = self._previous
            self._previous = self._current
            self._current = restored
            return deepcopy(restored)


class ReleasePublisher:
    """Publish only a fully approved, validated and ready candidate."""

    def __init__(self, store: ServingReleaseStore, *, clock: Any | None = None) -> None:
        self.store = store
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def publish(
        self, decision: ReleaseDecision, manifest: ReleaseManifest, corpus: CorpusVersion, *,
        expected_previous_release_id: str | None,
    ) -> PublishedRelease:
        if decision.decision != "approved":
            raise PublicationError("only an approved release decision can be published")
        if decision.release_id != manifest.release_id or manifest.corpus_version_id != corpus.version_id:
            raise PublicationError("release decision, manifest and corpus do not match")
        if corpus.status not in {"validated", "active"}:
            raise PublicationError("only a validated corpus can be published")
        binding = corpus.serving_binding
        evidence = corpus.validation_evidence
        if binding is None or evidence is None or not evidence.index_ready or not evidence.filter_ready:
            raise PublicationError("corpus is not ready for serving")
        return self.store.publish(
            PublishedRelease(
                release_id=manifest.release_id,
                artifact_id=manifest.candidate_digest,
                configuration_version_id=manifest.configuration_version_id,
                corpus_version_id=manifest.corpus_version_id,
                vector_collection=binding.vector_collection,
                vector_schema_version=binding.vector_schema_version,
                source_uri=corpus.source_uri,
                published_at=self.clock(),
            ),
            expected_previous_release_id=expected_previous_release_id,
        )


class FirestoreServingReleaseStore:
    """Firestore compare-and-set store for the serving release only."""

    def __init__(self, client: Any) -> None:
        self.reference = client.collection("release_state").document("serving")
        self.client = client

    def current(self) -> PublishedRelease | None:
        snapshot = self.reference.get()
        return _decode(snapshot.to_dict()) if snapshot.exists else None

    def previous(self) -> PublishedRelease | None:
        snapshot = self.reference.get()
        if not snapshot.exists or not snapshot.to_dict().get("previous"):
            return None
        return _decode(snapshot.to_dict()["previous"])

    def publish(self, release: PublishedRelease, *, expected_previous_release_id: str | None) -> PublishedRelease:
        from google.cloud import firestore

        @firestore.transactional
        def apply(transaction: Any) -> PublishedRelease:
            snapshot = self.reference.get(transaction=transaction)
            current = snapshot.to_dict().get("release_id") if snapshot.exists else None
            if current != expected_previous_release_id:
                raise ConcurrentPublicationError("serving release changed before publication could commit")
            current_data = snapshot.to_dict() if snapshot.exists else {}
            transaction.set(self.reference, {
                **asdict(release),
                "previous": current_data or None,
            })
            return release

        return apply(self.client.transaction())

    def rollback(self, *, expected_current_release_id: str) -> PublishedRelease:
        from google.cloud import firestore

        @firestore.transactional
        def apply(transaction: Any) -> PublishedRelease:
            snapshot = self.reference.get(transaction=transaction)
            data = snapshot.to_dict() if snapshot.exists else {}
            if data.get("release_id") != expected_current_release_id:
                raise ConcurrentPublicationError("serving release changed before rollback could commit")
            previous = data.get("previous")
            if not previous:
                raise PublicationError("no previous serving release is available for rollback")
            restored = _decode(previous)
            transaction.set(self.reference, {**asdict(restored), "previous": data})
            return restored

        return apply(self.client.transaction())


def _decode(data: dict[str, Any]) -> PublishedRelease:
    value = dict(data)
    value.pop("previous", None)
    if isinstance(value.get("published_at"), str):
        value["published_at"] = datetime.fromisoformat(value["published_at"])
    return PublishedRelease(**value)


class PublishedCorpusResolver:
    """Resolve the corpus named by the serving publication, never by a pointer."""

    def __init__(self, releases: ServingReleaseStore, corpora: Any, *, configuration_version_id: str,
                 vector_collection: str, vector_schema_version: str, artifact_id: str | None = None) -> None:
        self.releases = releases
        self.corpora = corpora
        self.configuration_version_id = configuration_version_id
        self.vector_collection = vector_collection
        self.vector_schema_version = vector_schema_version
        self.artifact_id = artifact_id

    def active(self) -> CorpusVersion | None:
        published = self.releases.current()
        if published is None:
            return None
        corpus = self.corpora.get(published.corpus_version_id)
        if corpus is None:
            return None
        validate_runtime_binding(
            published,
            configuration_version_id=self.configuration_version_id,
            corpus=corpus,
            vector_collection=self.vector_collection,
            vector_schema_version=self.vector_schema_version,
            artifact_id=self.artifact_id,
        )
        return corpus


def validate_runtime_binding(
    published: PublishedRelease, *, configuration_version_id: str, corpus: CorpusVersion,
    vector_collection: str, vector_schema_version: str, artifact_id: str | None = None,
) -> None:
    """Fail closed when the running artifact cannot serve the publication."""

    binding = corpus.serving_binding
    if (
        published.configuration_version_id != configuration_version_id
        or published.corpus_version_id != corpus.version_id
        or published.vector_collection != vector_collection
        or published.vector_schema_version != vector_schema_version
        or binding is None
        or binding.vector_collection != vector_collection
        or (artifact_id is not None and published.artifact_id != artifact_id)
    ):
        raise PublicationError("running artifact does not match the approved serving release")
