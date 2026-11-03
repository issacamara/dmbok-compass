"""Idempotent, staged corpus ingestion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

from app.contracts import ChildDocumentChunk, CorpusVersion, ParentChunk
from app.corpus.indexing import CorpusEmbeddingWriter, HierarchicalCorpusWriter
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


@dataclass(frozen=True, slots=True)
class HierarchicalIngestionResult:
    version: CorpusVersion
    parents: tuple[ParentChunk, ...]
    children: tuple[ChildDocumentChunk, ...]


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

    def run_hierarchical(
        self,
        version: CorpusVersion,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildDocumentChunk],
        *,
        termination_requested: Callable[[], bool] | None = None,
    ) -> HierarchicalIngestionResult:
        """Persist a complete parent/child candidate without validating or activating it."""

        if version.status != "staged":
            raise ValueError("hierarchical corpus writes require a staged version")
        if not isinstance(self.writer, HierarchicalCorpusWriter):
            raise TypeError("hierarchical ingestion requires a HierarchicalCorpusWriter")
        _validate_hierarchy(version.version_id, parents, children)
        existing = self.versions.get(version.version_id)
        if existing is not None and existing.status in {"validated", "active"}:
            return HierarchicalIngestionResult(existing, tuple(parents), tuple(children))

        staged = self.versions.stage(version)
        requested = termination_requested or (lambda: False)
        begin = getattr(self.versions, "begin_staging", None)
        finish = getattr(self.versions, "finish_staging", None)
        if callable(begin):
            begin(staged.version_id)

        try:
            for start in range(0, len(parents), self.writer.max_batch_size):
                if requested():
                    raise IngestionInterrupted("ingestion stopped before the next parent batch")
                self.writer.write_parent_batch(parents[start : start + self.writer.max_batch_size])
                if requested():
                    raise IngestionInterrupted("ingestion stopped after a persisted parent batch")

            for start in range(0, len(children), self.writer.max_batch_size):
                if requested():
                    raise IngestionInterrupted("ingestion stopped before the next child batch")
                self.writer.write_child_batch(children[start : start + self.writer.max_batch_size])
                if requested():
                    raise IngestionInterrupted("ingestion stopped after a persisted child batch")
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
            return HierarchicalIngestionResult(staged, tuple(parents), tuple(children))


def _validate_hierarchy(
    version_id: str,
    parents: Sequence[ParentChunk],
    children: Sequence[ChildDocumentChunk],
) -> None:
    if not parents or not children:
        raise ValueError("hierarchical corpus requires parent and child records")
    parent_by_id: dict[str, ParentChunk] = {}
    for parent in parents:
        if parent.corpus_version_id != version_id:
            raise ValueError("parent belongs to another corpus version")
        if parent.parent_id in parent_by_id:
            raise ValueError("hierarchical corpus contains duplicate parent IDs")
        parent_by_id[parent.parent_id] = parent

    child_ids: set[str] = set()
    referenced_parent_ids: set[str] = set()
    for child in children:
        if child.corpus_version_id != version_id:
            raise ValueError("child belongs to another corpus version")
        if child.chunk_id in child_ids:
            raise ValueError("hierarchical corpus contains duplicate child IDs")
        child_ids.add(child.chunk_id)
        parent = parent_by_id.get(child.parent_id)
        if parent is None:
            raise ValueError("child references a missing or cross-corpus parent")
        if (
            child.section_path != parent.section_path
            or child.page_start < parent.page_start
            or child.page_end > parent.page_end
        ):
            raise ValueError("child provenance does not match its parent")
        for child_span in child.source_spans:
            if not any(
                parent_span.page == child_span.page
                and parent_span.start_offset <= child_span.start_offset
                and child_span.end_offset <= parent_span.end_offset
                for parent_span in parent.source_spans
            ):
                raise ValueError("child source span is outside its parent")
        referenced_parent_ids.add(parent.parent_id)
    if referenced_parent_ids != set(parent_by_id):
        raise ValueError("each persisted parent must have at least one linked child")
