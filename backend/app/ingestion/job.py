"""Idempotent, staged corpus ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from app.contracts import CorpusVersion
from app.corpus.indexing import CorpusEmbeddingWriter
from app.corpus.search_probe import (
    CandidateFilteredVectorProbe,
    SearchReadinessEvidence,
)
from app.corpus.versions import CorpusVersionStore
from app.corpus.chunking import TextChunk


@dataclass(frozen=True, slots=True)
class IngestionResult:
    version: CorpusVersion
    chunks: tuple[TextChunk, ...]
    activated: bool
    ready_for_evaluation: bool = False


class IngestionInterrupted(RuntimeError):
    """Raised only after a fully persisted batch makes a safe stop possible."""


class CorpusIngestionJob:
    """Stage, write and validate one immutable corpus version.

    Publication is deliberately separate: a successful run never changes the
    serving release or the legacy active pointer.
    """

    def __init__(self, versions: CorpusVersionStore, writer: CorpusEmbeddingWriter) -> None:
        self.versions = versions
        self.writer = writer

    def run(
        self,
        version: CorpusVersion,
        chunks: list[TextChunk] | tuple[TextChunk, ...],
        *,
        termination_requested: Callable[[], bool] | None = None,
    ) -> IngestionResult:
        existing = self.versions.get(version.version_id)
        if existing is not None and existing.status in {"validated", "active"}:
            return IngestionResult(existing, (), False, existing.status == "validated")

        staged = self.versions.stage(version)
        requested = termination_requested or (lambda: False)
        begin = getattr(self.versions, "begin_staging", None)
        finish = getattr(self.versions, "finish_staging", None)
        if callable(begin):
            begin(staged.version_id)

        try:
            stored: list[TextChunk] = []
            for start in range(0, len(chunks), self.writer.max_batch_size):
                if requested():
                    raise IngestionInterrupted("ingestion stopped before the next persisted batch")
                batch = chunks[start : start + self.writer.max_batch_size]
                stored.extend(self.writer.write_batch(batch))
                if requested():
                    raise IngestionInterrupted("ingestion stopped after a persisted batch")

            validated = self.versions.validate(staged.version_id, stored)
            persist_probe = getattr(self.versions, "persist_search_readiness", None)
            if callable(persist_probe) and hasattr(self.writer.vector_store, "similarity_search_with_score"):
                binding = validated.serving_binding
                collection_id = binding.vector_collection if binding is not None else "document_chunks_v2"
                probe = CandidateFilteredVectorProbe(
                    self.writer.vector_store,
                    collection_id=collection_id,
                )
                persist_probe(
                    validated.version_id,
                    SearchReadinessEvidence(
                        complete=probe.complete_signal(
                            version_id=validated.version_id, candidate=stored[0]
                        ),
                        search_ready=None,
                        attempts=0,
                    ),
                )
                evidence = probe.run(version_id=validated.version_id, candidate=stored[0])
                persist_probe(validated.version_id, evidence)
        except IngestionInterrupted:
            if callable(finish):
                finish(staged.version_id, status="interrupted")
            raise
        except BaseException:
            if callable(finish):
                finish(staged.version_id, status="failed")
            raise
        else:
            if callable(finish):
                finish(staged.version_id, status="completed")
            return IngestionResult(validated, tuple(stored), False, True)
