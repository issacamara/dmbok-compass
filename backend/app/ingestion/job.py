"""Idempotent, staged corpus ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass

from app.contracts import CorpusVersion, DocumentChunk
from app.corpus.indexing import CorpusEmbeddingWriter
from app.corpus.versions import CorpusVersionStore
from app.corpus.chunking import TextChunk


@dataclass(frozen=True, slots=True)
class IngestionResult:
    version: CorpusVersion
    chunks: tuple[DocumentChunk, ...]
    activated: bool


class CorpusIngestionJob:
    """Stage, write, validate, and promote one immutable corpus version.

    Version and chunk identifiers are deterministic inputs.  Re-running a
    completed job therefore returns the existing active version without
    creating another version or activating a second pointer.
    """

    def __init__(self, versions: CorpusVersionStore, writer: CorpusEmbeddingWriter) -> None:
        self.versions = versions
        self.writer = writer

    def run(self, version: CorpusVersion, chunks: list[TextChunk] | tuple[TextChunk, ...]) -> IngestionResult:
        existing = self.versions.get(version.version_id)
        if existing is not None and existing.status == "active":
            return IngestionResult(existing, (), False)

        staged = self.versions.stage(version)
        stored = self.writer.write(chunks)
        validated = self.versions.validate(staged.version_id, stored)
        active = self.versions.activate(validated.version_id)
        return IngestionResult(active, stored, True)

