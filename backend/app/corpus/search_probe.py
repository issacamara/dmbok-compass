"""Candidate-filtered vector-search readiness evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from time import monotonic, sleep
from typing import Any, Callable, Protocol

from google.cloud.firestore_v1.base_query import FieldFilter


@dataclass(frozen=True, slots=True)
class ReadinessSignal:
    """One independently timestamped readiness signal."""

    status: str
    observed_at: datetime
    collection_id: str
    index_id: str
    candidate_chunk_id: str
    corpus_version_id: str


@dataclass(frozen=True, slots=True)
class SearchReadinessEvidence:
    """Completeness and vector-search evidence kept as separate signals."""

    complete: ReadinessSignal
    search_ready: ReadinessSignal | None
    attempts: int

    def model_dump(self) -> dict[str, Any]:
        def dump(signal: ReadinessSignal | None) -> dict[str, Any] | None:
            if signal is None:
                return None
            return {
                "status": signal.status,
                "observed_at": signal.observed_at.isoformat(),
                "collection_id": signal.collection_id,
                "index_id": signal.index_id,
                "candidate_chunk_id": signal.candidate_chunk_id,
                "corpus_version_id": signal.corpus_version_id,
            }

        return {
            "complete": dump(self.complete),
            "search_ready": dump(self.search_ready),
            "attempts": self.attempts,
        }


class ReadinessStore(Protocol):
    def persist_search_readiness(
        self, version_id: str, evidence: SearchReadinessEvidence
    ) -> None: ...


class CandidateFilteredVectorProbe:
    """Prove the same filtered, score-returning query used for retrieval."""

    def __init__(
        self,
        vector_store: Any,
        *,
        collection_id: str,
        schema_version: str = "2",
        index_id: str = "firestore-native-vector-index",
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.0,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")
        self.vector_store = vector_store
        self.collection_id = collection_id
        self.schema_version = schema_version
        self.index_id = index_id
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sleeper = sleeper

    def run(self, *, version_id: str, candidate: Any) -> SearchReadinessEvidence:
        complete = self.complete_signal(version_id=version_id, candidate=candidate)
        last_error: Exception | None = None
        started = monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                results = self.vector_store.similarity_search_with_score(
                    candidate.text,
                    k=5,
                    filters=FieldFilter(
                        "metadata.corpus_version_id", "==", version_id
                    ),
                )
                self._validate_result(
                    results,
                    version_id=version_id,
                    candidate=candidate,
                    schema_version=self.schema_version,
                )
                search_ready = ReadinessSignal(
                    status="search_ready",
                    observed_at=datetime.now(timezone.utc),
                    collection_id=self.collection_id,
                    index_id=self.index_id,
                    candidate_chunk_id=candidate.chunk_id,
                    corpus_version_id=version_id,
                )
                return SearchReadinessEvidence(complete, search_ready, attempt)
            except Exception as error:  # bounded propagation/index readiness retry
                last_error = error
                if attempt < self.max_attempts and monotonic() - started < 30:
                    self.sleeper(self.retry_delay_seconds)
        raise RuntimeError("candidate-filtered vector readiness probe failed") from last_error

    def complete_signal(self, *, version_id: str, candidate: Any) -> ReadinessSignal:
        """Return completeness evidence before probing index propagation."""

        return ReadinessSignal(
            status="complete",
            observed_at=datetime.now(timezone.utc),
            collection_id=self.collection_id,
            index_id=self.index_id,
            candidate_chunk_id=candidate.chunk_id,
            corpus_version_id=version_id,
        )

    @staticmethod
    def _validate_result(
        results: Any, *, version_id: str, candidate: Any, schema_version: str
    ) -> None:
        if not results:
            raise ValueError("candidate-filtered vector probe returned no results")
        for document, score in results:
            if not isfinite(float(score)):
                continue
            metadata = document.metadata.get("metadata", document.metadata)
            if not isinstance(metadata, dict):
                continue
            if (
                metadata.get("corpus_version_id") == version_id
                and metadata.get("chunk_id") == candidate.chunk_id
                and metadata.get("schema_version") == schema_version
                and metadata.get("page") == candidate.page
                and metadata.get("section") == candidate.section
            ):
                return
        raise ValueError("candidate-filtered vector probe returned invalid provenance")
