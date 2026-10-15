"""Idempotent, staged corpus ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from app.contracts import CorpusVersion
from app.corpus.indexing import CorpusEmbeddingWriter
from app.corpus.versions import CorpusVersionStore
from app.corpus.chunking import TextChunk


@dataclass(frozen=True, slots=True)
class IngestionResult:
    version: CorpusVersion
    chunks: tuple[TextChunk, ...]
    activated: bool


class IngestionInterrupted(RuntimeError):
    """Raised only after a fully persisted batch makes a safe stop possible."""


class CorpusIngestionJob:
    """Stage, write, validate, and promote one immutable corpus version.

    Version and chunk identifiers are deterministic inputs.  Re-running a
    completed job therefore returns the existing active version without
    creating another version or activating a second pointer.
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
        current = self.versions.active()
        existing = self.versions.get(version.version_id)
        if existing is not None and existing.status == "active":
            return IngestionResult(existing, (), False)

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
            active = self.versions.activate(
                validated.version_id,
                expected_active_version_id=current.version_id if current is not None else None,
            )
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
            return IngestionResult(active, tuple(stored), True)
