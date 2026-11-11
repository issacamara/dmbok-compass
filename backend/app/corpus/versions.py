"""Corpus-version lifecycle and active-pointer coordination.

The store in this module is deliberately backend-neutral.  Production callers
can map the same operations to Firestore transactions, while tests and the
offline ingestion job use :class:`InMemoryCorpusVersionStore`.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from threading import RLock
from collections.abc import Sequence
from typing import Any, Iterable, Protocol

from app.contracts import (
    ChildDocumentChunk,
    CorpusServingBinding,
    CorpusValidationEvidence,
    CorpusVersion,
    ParentChunk,
)
from app.corpus.indexing import VECTOR_SCHEMA_VERSION
from app.corpus.search_probe import SearchReadinessEvidence

_UNSET = object()


class CorpusVersionError(ValueError):
    """Base error for invalid corpus-version operations."""


class CorpusCompletenessError(CorpusVersionError):
    """Raised when a staged corpus is incomplete or internally inconsistent."""


class ConcurrentCorpusActivationError(CorpusVersionError):
    """Raised when the active corpus changed while a candidate was ingesting."""


def legacy_serving_binding(
    version: CorpusVersion,
    *,
    vector_collection: str,
) -> CorpusServingBinding:
    """Adapt a pre-binding record without changing its persisted shape.

    Earlier records did not persist a serving binding. They remain readable
    only through the already-selected factory configuration; newly staged
    records must persist a binding and never rely on this compatibility path.
    """

    import hashlib

    fingerprint = hashlib.sha256(
        f"legacy:{version.version_id}:{vector_collection}".encode()
    ).hexdigest()
    return CorpusServingBinding(
        corpus_version_id=version.version_id,
        vector_collection=vector_collection,
        vector_schema_version=VECTOR_SCHEMA_VERSION,
        embedding_model=version.embedding_model,
        embedding_dimensions=version.embedding_dimensions,
        ingestion_fingerprint=fingerprint,
        configuration_fingerprint=fingerprint,
    )


class CorpusVersionStore(Protocol):
    def stage(self, version: CorpusVersion) -> CorpusVersion: ...

    def validate(self, version_id: str, chunks: Iterable[Any]) -> CorpusVersion: ...

    def validate_hierarchical(
        self,
        version_id: str,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildDocumentChunk],
    ) -> CorpusVersion: ...

    def activate(
        self,
        version_id: str,
        *,
        expected_active_version_id: str | None,
    ) -> CorpusVersion: ...

    def rollback(self, version_id: str | None = None) -> CorpusVersion: ...

    def get(self, version_id: str) -> CorpusVersion | None: ...

    def active(self) -> CorpusVersion | None: ...


class InMemoryCorpusVersionStore:
    """Reference implementation of the transactional version contract.

    A Firestore adapter can implement :class:`CorpusVersionStore` with one
    transaction around ``active_version`` and the affected version records.
    """

    def __init__(self) -> None:
        self._versions: dict[str, CorpusVersion] = {}
        self._active_id: str | None = None
        self._previous_active_id: str | None = None
        self._search_readiness: dict[str, SearchReadinessEvidence] = {}
        self._hierarchical_child_ids: dict[str, set[str]] = {}
        self._lock = RLock()

    def persist_search_readiness(
        self, version_id: str, evidence: SearchReadinessEvidence
    ) -> None:
        if evidence.complete.corpus_version_id != version_id:
            raise CorpusVersionError("search readiness evidence is bound to another corpus")
        version = self.get(version_id)
        if version is None:
            raise CorpusVersionError(f"unknown corpus version: {version_id}")
        expected_collection = (
            version.serving_binding.vector_collection
            if version.serving_binding is not None
            else "document_chunks_v2"
        )
        if (
            evidence.complete.collection_id != expected_collection
            or evidence.complete.status not in {"complete", "incomplete"}
            or not evidence.complete.candidate_chunk_id
            or not evidence.complete.index_id
            or evidence.complete.observed_at.tzinfo is None
        ):
            raise CorpusVersionError("search readiness evidence has invalid identity or timestamp")
        if evidence.search_ready is not None and (
            evidence.search_ready.corpus_version_id != version_id
            or evidence.search_ready.candidate_chunk_id != evidence.complete.candidate_chunk_id
            or evidence.search_ready.collection_id != evidence.complete.collection_id
            or evidence.search_ready.index_id != evidence.complete.index_id
            or evidence.search_ready.status != "search_ready"
            or evidence.complete.status != "complete"
            or evidence.search_ready.observed_at.tzinfo is None
            or evidence.search_ready.observed_at < evidence.complete.observed_at
            or not 1 <= evidence.attempts <= 3
        ):
            raise CorpusVersionError("search readiness evidence is inconsistent")
        if evidence.search_ready is None and evidence.attempts != 0:
            raise CorpusVersionError("search attempts require a search readiness signal")
        with self._lock:
            version = self._require(version_id)
            known_child_ids = self._hierarchical_child_ids.get(version_id)
            if (
                evidence.search_ready is not None
                and known_child_ids is not None
                and evidence.complete.candidate_chunk_id not in known_child_ids
            ):
                raise CorpusVersionError("search readiness candidate is not in the validated hierarchy")
            self._search_readiness[version_id] = deepcopy(evidence)
            validation = version.validation_evidence
            if (
                version.status == "staged"
                and validation is not None
                and not validation.index_ready
                and not validation.filter_ready
                and evidence.complete.status == "complete"
                and evidence.search_ready is not None
            ):
                ready_evidence = validation.model_copy(
                    update={"index_ready": True, "filter_ready": True}
                )
                self._versions[version_id] = version.model_copy(
                    update={"status": "validated", "validation_evidence": ready_evidence},
                    deep=True,
                )

    def get(self, version_id: str) -> CorpusVersion | None:
        with self._lock:
            version = self._versions.get(version_id)
            return deepcopy(version) if version else None

    def active(self) -> CorpusVersion | None:
        with self._lock:
            return self.get(self._active_id) if self._active_id else None

    def stage(self, version: CorpusVersion) -> CorpusVersion:
        if version.status != "staged":
            raise CorpusVersionError("new corpus versions must start in staged state")
        with self._lock:
            existing = self._versions.get(version.version_id)
            if existing is not None:
                if existing.source_uri != version.source_uri:
                    raise CorpusVersionError("version_id is already bound to another source")
                return deepcopy(existing)
            self._versions[version.version_id] = version.model_copy(deep=True)
            return deepcopy(version)

    def validate_hierarchical(
        self,
        version_id: str,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildDocumentChunk],
    ) -> CorpusVersion:
        """Validate the complete hierarchy without marking search as ready."""

        validate_hierarchical_records(version_id, parents, children)
        with self._lock:
            version = self._require(version_id)
            if version.status in {"validated", "active"}:
                return deepcopy(version)
            if version.status != "staged":
                raise CorpusVersionError(f"cannot validate a {version.status} corpus version")
            evidence = hierarchical_validation_evidence(
                version, parents, children, index_ready=False, filter_ready=False
            )
            updated = version.model_copy(update={"validation_evidence": evidence}, deep=True)
            self._versions[version_id] = updated
            self._hierarchical_child_ids[version_id] = {child.chunk_id for child in children}
            return deepcopy(updated)

    def validate(self, version_id: str, chunks: Iterable[Any]) -> CorpusVersion:
        chunk_list = tuple(chunks)
        with self._lock:
            version = self._require(version_id)
            if version.status == "validated" or version.status == "active":
                return deepcopy(version)
            if version.status != "staged":
                raise CorpusVersionError(f"cannot validate a {version.status} corpus version")
            if not chunk_list:
                raise CorpusCompletenessError("corpus must contain at least one chunk")
            chunk_ids = [chunk.chunk_id for chunk in chunk_list]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise CorpusCompletenessError("corpus contains duplicate chunk IDs")
            if any(chunk.corpus_version_id != version_id for chunk in chunk_list):
                raise CorpusCompletenessError("every chunk must belong to the staged corpus version")
            if any(not chunk.text.strip() for chunk in chunk_list):
                raise CorpusCompletenessError("every chunk must contain text")
            if any(
                hasattr(chunk, "embedding") and len(chunk.embedding) != 768
                for chunk in chunk_list
            ):
                raise CorpusCompletenessError("every persisted embedding must have 768 dimensions")
            updated = version.model_copy(update={"status": "validated"}, deep=True)
            self._versions[version_id] = updated
            return deepcopy(updated)

    def activate(
        self,
        version_id: str,
        *,
        expected_active_version_id: str | None | object = _UNSET,
    ) -> CorpusVersion:
        with self._lock:
            version = self._require(version_id)
            if version.status == "active" and self._active_id == version_id:
                return deepcopy(version)
            if version.status != "validated":
                raise CorpusVersionError("only a validated corpus version can become active")
            if (
                expected_active_version_id is not _UNSET
                and self._active_id != expected_active_version_id
            ):
                raise ConcurrentCorpusActivationError(
                    "active corpus changed before activation could commit"
                )
            previous_id = self._active_id
            if previous_id is not None:
                previous = self._require(previous_id)
                self._versions[previous_id] = previous.model_copy(
                    update={"status": "retired", "created_at": previous.created_at}, deep=True
                )
            active = version.model_copy(update={"status": "active"}, deep=True)
            self._versions[version_id] = active
            self._active_id = version_id
            self._previous_active_id = previous_id
            return deepcopy(active)

    def rollback(self, version_id: str | None = None) -> CorpusVersion:
        with self._lock:
            target_id = version_id or self._previous_active_id
            if target_id is None:
                raise CorpusVersionError("no previous valid corpus version is available for rollback")
            target = self._require(target_id)
            if target.status not in {"retired", "active"}:
                raise CorpusVersionError("rollback target must be a previously valid corpus version")
            current_id = self._active_id
            if current_id == target_id:
                return deepcopy(target)
            if current_id is not None:
                current = self._require(current_id)
                self._versions[current_id] = current.model_copy(update={"status": "retired"}, deep=True)
            restored = target.model_copy(update={"status": "active"}, deep=True)
            self._versions[target_id] = restored
            self._active_id = target_id
            self._previous_active_id = current_id
            return deepcopy(restored)

    def _require(self, version_id: str) -> CorpusVersion:
        version = self._versions.get(version_id)
        if version is None:
            raise CorpusVersionError(f"unknown corpus version: {version_id}")
        return version


def validate_hierarchical_records(
    version_id: str,
    parents: Sequence[ParentChunk],
    children: Sequence[ChildDocumentChunk],
) -> None:
    """Reject malformed, incomplete, oversized, or incorrectly linked candidates."""

    from app.corpus.chunking import CHILD_MAX_TOKENS, PARENT_MAX_TOKENS, _token_count

    if not parents or not children:
        raise CorpusCompletenessError("hierarchical corpus requires parent and child records")
    parent_by_id: dict[str, ParentChunk] = {}
    for parent in parents:
        if parent.corpus_version_id != version_id:
            raise CorpusCompletenessError("parent belongs to another corpus version")
        if parent.parent_id in parent_by_id:
            raise CorpusCompletenessError("hierarchical corpus contains duplicate parent IDs")
        _validate_record_text(
            parent.text, parent.source_text, parent.source_spans, parent.content_hash, parent.token_count,
            PARENT_MAX_TOKENS, _token_count, "parent",
        )
        parent_by_id[parent.parent_id] = parent

    child_ids: set[str] = set()
    children_by_parent: dict[str, list[ChildDocumentChunk]] = {}
    for child in children:
        if child.corpus_version_id != version_id:
            raise CorpusCompletenessError("child belongs to another corpus version")
        if child.chunk_id in child_ids:
            raise CorpusCompletenessError("hierarchical corpus contains duplicate child IDs")
        child_ids.add(child.chunk_id)
        parent = parent_by_id.get(child.parent_id)
        if parent is None:
            raise CorpusCompletenessError("child references a missing or cross-corpus parent")
        if (
            child.section_path != parent.section_path
            or child.page_start < parent.page_start
            or child.page_end > parent.page_end
        ):
            raise CorpusCompletenessError("child provenance does not match its parent")
        _validate_record_text(
            child.text, child.source_text, child.source_spans, child.content_hash, child.token_count,
            CHILD_MAX_TOKENS, _token_count, "child",
        )
        children_by_parent.setdefault(parent.parent_id, []).append(child)

    if set(children_by_parent) != set(parent_by_id):
        raise CorpusCompletenessError("each persisted parent must have at least one linked child")
    for parent_id, parent in parent_by_id.items():
        linked = sorted(children_by_parent[parent_id], key=lambda child: child.child_ordinal)
        if [child.child_ordinal for child in linked] != list(range(len(linked))):
            raise CorpusCompletenessError("child ordinals must be contiguous within each parent")
        child_span_order = [
            (span.page, span.block_order if span.block_order is not None else -1, span.start_offset, span.end_offset)
            for child in linked
            for span in child.source_spans
        ]
        if child_span_order != sorted(child_span_order):
            raise CorpusCompletenessError("child ordinals do not preserve source span order")
        covered: set[int] = set()
        for parent_span in parent.source_spans:
            source_segments = sorted(
                (
                    span
                    for child in linked
                    for span in child.source_spans
                    if span.page == parent_span.page
                    and span.block_id == parent_span.block_id
                    and parent_span.start_offset <= span.start_offset
                    and span.end_offset <= parent_span.end_offset
                ),
                key=lambda span: (span.block_order if span.block_order is not None else -1, span.start_offset),
            )
            cursor = parent_span.start_offset
            text_parts: list[str] = []
            for span in source_segments:
                if span.start_offset != cursor:
                    raise CorpusCompletenessError("child spans do not provide contiguous parent coverage")
                cursor = span.end_offset
                text_parts.append(span.text)
            if cursor != parent_span.end_offset or "".join(text_parts) != parent_span.text:
                raise CorpusCompletenessError("child spans do not fully cover their parent source span")
            covered.update(id(span) for span in source_segments)
        child_span_ids = {id(span) for child in linked for span in child.source_spans}
        if covered != child_span_ids:
            raise CorpusCompletenessError("child source spans fall outside their parent")


def _validate_record_text(
    text: str,
    source_text: str | None,
    spans: Sequence[Any],
    content_hash: str,
    token_count: int,
    maximum_tokens: int,
    count_tokens: Any,
    kind: str,
) -> None:
    expected_source_text = "\n".join(span.text for span in spans)
    if source_text is not None and source_text != expected_source_text:
        raise CorpusCompletenessError(f"{kind} source text does not reconstruct from its source spans")
    if source_text is None and text != expected_source_text:
        raise CorpusCompletenessError(f"{kind} text does not match its source spans")
    if sha256(text.encode("utf-8")).hexdigest() != content_hash:
        raise CorpusCompletenessError(f"{kind} content hash does not match its text")
    actual_tokens = count_tokens(text)
    if token_count != actual_tokens or actual_tokens > maximum_tokens:
        raise CorpusCompletenessError(f"{kind} token count is invalid or exceeds its limit")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def hierarchical_provenance_sha256(
    parents: Sequence[ParentChunk], children: Sequence[ChildDocumentChunk]
) -> str:
    """Return the source-metadata digest shared by readiness and serving checks."""

    provenance = [
        {
            "record_type": "parent",
            "record_id": parent.parent_id,
            "section_path": list(parent.section_path),
            "page_start": parent.page_start,
            "page_end": parent.page_end,
            "source_text": parent.source_text,
            "table_header": parent.table_header,
            "source_spans": [span.model_dump(mode="json", exclude_none=True) for span in parent.source_spans],
        }
        for parent in sorted(parents, key=lambda item: item.parent_id)
    ] + [
        {
            "record_type": "child",
            "record_id": child.chunk_id,
            "parent_id": child.parent_id,
            "section_path": list(child.section_path),
            "page_start": child.page_start,
            "page_end": child.page_end,
            "child_ordinal": child.child_ordinal,
            "source_text": child.source_text,
            "source_spans": [span.model_dump(mode="json", exclude_none=True) for span in child.source_spans],
        }
        for child in sorted(children, key=lambda item: item.chunk_id)
    ]
    return _stable_hash(provenance)


def hierarchical_validation_evidence(
    version: CorpusVersion,
    parents: Sequence[ParentChunk],
    children: Sequence[ChildDocumentChunk],
    *,
    index_ready: bool,
    filter_ready: bool,
) -> CorpusValidationEvidence:
    """Hash all persisted hierarchy records into deterministic readiness evidence."""

    content = [
        {"record_type": "parent", "record_id": parent.parent_id, "content_hash": parent.content_hash}
        for parent in sorted(parents, key=lambda item: item.parent_id)
    ] + [
        {"record_type": "child", "record_id": child.chunk_id, "content_hash": child.content_hash}
        for child in sorted(children, key=lambda item: item.chunk_id)
    ]
    return CorpusValidationEvidence(
        chunk_count=len(children),
        content_hashes_sha256=_stable_hash(content),
        provenance_sha256=hierarchical_provenance_sha256(parents, children),
        embedding_dimensions=version.embedding_dimensions,
        index_ready=index_ready,
        filter_ready=filter_ready,
    )
