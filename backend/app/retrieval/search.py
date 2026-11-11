"""Score-preserving native LangChain retrieval for one pinned corpus."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
import re
from time import perf_counter
from typing import Any, Callable, Protocol, TypeVar

from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from pydantic import ValidationError

from app.contracts import (
    ChildDocumentChunk, GenerationContext, GenerationContextParent, ParentChunk,
    RawSourceSpan, RetrievedPassage, SelectedParent,
)
from app.corpus.indexing import PARENT_COLLECTION
from app.corpus.tokenization import count_tokens

DEFAULT_LIMIT = 5
MAX_PARENT_CONTEXT_TOKENS = 6_000
# Reserve for the answer prompt, provider framing, and serialized request fields.
GENERATION_PROVIDER_OVERHEAD_TOKENS = 512

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """One request's child-ranked evidence and bounded parent context."""

    passages: tuple[RetrievedPassage, ...]
    parent_context: tuple[ParentChunk, ...]
    selected_parents: tuple[SelectedParent, ...]
    child_sources: tuple[ChildDocumentChunk, ...] = ()
    source_parents: tuple[ParentChunk, ...] = ()
    generation_context: GenerationContext | None = None


class RetrievalProvenanceError(RuntimeError):
    """Raised when a retrieved child cannot be bound to valid parent sources."""


class ParentChunkReader(Protocol):
    def get_many(
        self, parent_ids: tuple[str, ...], *, corpus_version_id: str
    ) -> tuple[ParentChunk, ...]: ...


class FirestoreParentChunkReader:
    """Read only the parents named by one retrieval request."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.collection = client.collection(PARENT_COLLECTION)

    def get_many(
        self, parent_ids: tuple[str, ...], *, corpus_version_id: str
    ) -> tuple[ParentChunk, ...]:
        if not parent_ids:
            return ()
        references = tuple(self.collection.document(parent_id) for parent_id in parent_ids)
        snapshots = self.client.get_all(references)
        requested = set(parent_ids)
        parents: dict[str, ParentChunk] = {}
        for snapshot in snapshots:
            if not snapshot.exists:
                continue
            values = snapshot.to_dict()
            if not isinstance(values, dict):
                continue
            values.pop("metadata", None)
            try:
                parent = ParentChunk.model_validate(values)
            except ValidationError:
                continue
            if (
                parent.parent_id in requested
                and parent.corpus_version_id == corpus_version_id
            ):
                parents[parent.parent_id] = parent
        return tuple(parents[parent_id] for parent_id in parent_ids if parent_id in parents)

# Firestore's LangChain vector-store integration exposes only synchronous search.
# Keep that work off request event loops and cap concurrent in-process RPCs.
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="dmbok-retrieval",
)


async def retrieve_off_event_loop(
    retrieve: Callable[..., T],
    question: str,
    *,
    active_corpus_version_id: str,
    limit: int,
) -> tuple[T, float]:
    """Run one request-local synchronous retrieval in the bounded pool."""

    def execute() -> tuple[T, float]:
        started = perf_counter()
        passages = retrieve(
            question,
            active_corpus_version_id=active_corpus_version_id,
            limit=limit,
        )
        return passages, (perf_counter() - started) * 1000

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _RETRIEVAL_EXECUTOR,
        execute,
    )


class FirestorePassageRetriever:
    """Map native LangChain search results to the stable application contract."""

    def __init__(
        self,
        vector_store: FirestoreVectorStore,
        parent_reader: ParentChunkReader | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.parent_reader = parent_reader

    def retrieve(
        self,
        question: str,
        *,
        active_corpus_version_id: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[RetrievedPassage, ...]:
        if not question or not question.strip():
            raise ValueError("question must be non-empty")

        return tuple(passage for passage, _, _ in self._ranked_children(
            question,
            active_corpus_version_id=active_corpus_version_id,
            limit=limit,
        ))

    def retrieve_with_parent_context(
        self,
        question: str,
        *,
        active_corpus_version_id: str,
        limit: int = DEFAULT_LIMIT,
    ) -> RetrievalEvidence:
        """Return unchanged dense child ranking with request-local parent context."""

        if self.parent_reader is None:
            ranked = self._ranked_children(
                question,
                active_corpus_version_id=active_corpus_version_id,
                limit=limit,
            )
            children = tuple(child for _, _, child in ranked if child is not None)
            if children:
                raise RetrievalProvenanceError("hierarchical retrieval requires a parent reader")
            return RetrievalEvidence(
                tuple(passage for passage, _, _ in ranked),
                (),
                (),
                children,
            )
        ranked = self._ranked_children(
            question,
            active_corpus_version_id=active_corpus_version_id,
            limit=limit,
        )
        parent_ids = tuple(dict.fromkeys(
            parent_id for _, parent_id, _ in ranked if parent_id
        ))
        parents = self.parent_reader.get_many(
            parent_ids, corpus_version_id=active_corpus_version_id
        )
        parents = tuple(
            parent
            for parent in parents
            if parent.corpus_version_id == active_corpus_version_id
            and parent.parent_id in parent_ids
        )
        if len({parent.parent_id for parent in parents}) != len(parent_ids):
            missing = sorted(set(parent_ids) - {parent.parent_id for parent in parents})
            raise RetrievalProvenanceError(
                "retrieved children have missing or cross-corpus parents: " + ", ".join(missing)
            )
        child_scores: dict[str, float] = {}
        for passage, parent_id, _ in ranked:
            if parent_id and passage.relevance_score is not None:
                child_scores[parent_id] = max(
                    child_scores.get(parent_id, float("-inf")), passage.relevance_score
                )
        prompt_and_provider_reserve = (
            count_tokens(question)
            + GENERATION_PROVIDER_OVERHEAD_TOKENS
        )
        child_context_tokens = sum(
            count_tokens(passage.excerpt) + count_tokens(passage.section)
            for passage, _, _ in ranked
        )
        parent_budget = MAX_PARENT_CONTEXT_TOKENS - prompt_and_provider_reserve - child_context_tokens
        if parent_budget <= 0:
            raise RetrievalProvenanceError(
                "question and ranked child evidence leave no budget for parent context"
            )
        context, selected = select_parent_context(
            question,
            tuple((passage, parent_id) for passage, parent_id, _ in ranked),
            parents,
            child_scores=child_scores,
            max_tokens=parent_budget,
        )
        if not context:
            raise RetrievalProvenanceError(
                "no ranked parent fits the bounded generation context"
            )
        children = tuple(child for _, _, child in ranked if child is not None)
        generation_context = _generation_context(
            active_corpus_version_id, context, children
        )
        return RetrievalEvidence(
            tuple(passage for passage, _, _ in ranked),
            context,
            selected,
            children,
            parents,
            generation_context,
        )

    def _ranked_children(
        self,
        question: str,
        *,
        active_corpus_version_id: str,
        limit: int,
    ) -> tuple[tuple[RetrievedPassage, str | None, ChildDocumentChunk | None], ...]:
        if not question or not question.strip():
            raise ValueError("question must be non-empty")
        if not active_corpus_version_id or not active_corpus_version_id.strip():
            raise ValueError("active_corpus_version_id must be non-empty")
        if limit < 1 or limit > DEFAULT_LIMIT:
            raise ValueError(f"limit must be between 1 and {DEFAULT_LIMIT}")

        results = self.vector_store.similarity_search_with_score(
            question,
            k=limit,
            filters=FieldFilter(
                "metadata.corpus_version_id", "==", active_corpus_version_id
            ),
        )
        passages = []
        for document, score in results:
            if not isfinite(score):
                continue
            child = _child_source_from_document(document)
            parent_id = child.parent_id if child is not None else _parent_id_from_document(document)
            passages.append((_passage_from_result(document, score, child), parent_id, child))
        passages.sort(
            key=lambda item: (
                -(
                    item[0].relevance_score
                    if item[0].relevance_score is not None
                    else float("-inf")
                ),
                item[0].chunk_id,
            )
        )
        return tuple(passages[:limit])


def select_parent_context(
    question: str,
    ranked_children: tuple[tuple[RetrievedPassage, str | None], ...],
    parents: tuple[ParentChunk, ...],
    *,
    child_scores: dict[str, float] | None = None,
    max_tokens: int = MAX_PARENT_CONTEXT_TOKENS,
) -> tuple[tuple[ParentChunk, ...], tuple[SelectedParent, ...]]:
    """Deduplicate and select relevant parents without exceeding request budget."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    parent_by_id = {parent.parent_id: parent for parent in parents}
    ranked_parent_ids = tuple(dict.fromkeys(
        parent_id
        for _, parent_id in ranked_children
        if parent_id in parent_by_id
    ))
    if not ranked_parent_ids:
        return (), ()

    scores = child_scores or {}
    ordered_ids = tuple(sorted(
        ranked_parent_ids,
        key=lambda parent_id: (-scores.get(parent_id, 0.0), parent_id),
    ))
    token_counts = {
        parent_id: _parent_passage_tokens(parent_by_id[parent_id])
        for parent_id in ordered_ids
    }
    subjects = _comparison_subjects(question)
    selected: set[str] = set()
    used_tokens = 0

    # Reserve one relevant parent for each comparison subject before filling
    # the remaining budget by the strongest child score.
    for subject in subjects:
        for parent_id in ordered_ids:
            parent = parent_by_id[parent_id]
            if not _matches_subject(subject, parent, ranked_children):
                continue
            needed = token_counts[parent_id]
            if parent_id in selected:
                break
            if used_tokens + needed <= max_tokens:
                selected.add(parent_id)
                used_tokens += needed
                break

    for parent_id in ordered_ids:
        if len(selected) >= DEFAULT_LIMIT or parent_id in selected:
            continue
        needed = token_counts[parent_id]
        if used_tokens + needed <= max_tokens:
            selected.add(parent_id)
            used_tokens += needed

    final_ids = tuple(parent_id for parent_id in ordered_ids if parent_id in selected)
    context = tuple(parent_by_id[parent_id] for parent_id in final_ids)
    selected_records = tuple(
        SelectedParent(
            parent_id=parent_id,
            token_count=token_counts[parent_id],
            rank=rank,
        )
        for rank, parent_id in enumerate(final_ids)
    )
    return context, selected_records


def bound_parent_context(
    parents: tuple[ParentChunk, ...], *, max_tokens: int = MAX_PARENT_CONTEXT_TOKENS
) -> tuple[ParentChunk, ...]:
    """Apply a final request boundary to already-ranked parent context."""

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    selected: list[ParentChunk] = []
    seen: set[str] = set()
    used_tokens = 0
    for parent in parents:
        if parent.parent_id in seen or len(selected) >= DEFAULT_LIMIT:
            continue
        seen.add(parent.parent_id)
        needed = _parent_passage_tokens(parent)
        if used_tokens + needed > max_tokens:
            continue
        selected.append(parent)
        used_tokens += needed
    return tuple(selected)


def _parent_passage_tokens(parent: ParentChunk) -> int:
    """Count both text fields that `_parent_context_passage` sends upstream."""

    section = " / ".join(parent.section_path)[:500]
    return count_tokens(parent.text) + count_tokens(section)


def _comparison_subjects(question: str) -> tuple[str, ...]:
    patterns = (
        r"\bbetween\s+(.+?)\s+(?:and|versus|vs\.?)\s+(.+?)(?=[?.!,]|$)",
        r"\bcompare\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)(?=[?.!,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            subjects = []
            for value in match.groups():
                value = value.strip().strip(" \t\"'")
                value = re.split(
                    r"\s+(?:in terms of|with respect to|with regard to|regarding|for|within|across|during|when)\b",
                    value,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip()
                value = re.sub(r"^the\s+", "", value, flags=re.IGNORECASE)
                if value:
                    subjects.append(value)
            if subjects:
                return tuple(subjects)
    return ()


def _matches_subject(
    subject: str,
    parent: ParentChunk,
    ranked_children: tuple[tuple[RetrievedPassage, str | None], ...],
) -> bool:
    evidence = " ".join(
        passage.excerpt
        for passage, parent_id in ranked_children
        if parent_id == parent.parent_id
    )
    text = " ".join(f"{parent.text} {evidence}".casefold().split())
    normalized_subject = " ".join(subject.casefold().split())
    return bool(normalized_subject) and re.search(
        rf"(?<!\w){re.escape(normalized_subject)}(?!\w)", text
    ) is not None


def _parent_id_from_document(document: Document) -> str | None:
    metadata = document.metadata.get("metadata", document.metadata)
    if not isinstance(metadata, dict):
        return None
    parent_id = metadata.get("parent_id")
    return str(parent_id) if isinstance(parent_id, str) and parent_id else None


def _generation_context(
    corpus_version_id: str,
    parents: tuple[ParentChunk, ...],
    children: tuple[ChildDocumentChunk, ...],
) -> GenerationContext:
    """Bind exactly the selected parent context to page-addressable sources."""

    child_ids_by_parent: dict[str, list[str]] = {}
    for child in children:
        child_ids_by_parent.setdefault(child.parent_id, []).append(child.chunk_id)
    records = []
    for parent in parents:
        children_for_parent = tuple(sorted(set(child_ids_by_parent.get(parent.parent_id, ()))))
        if not children_for_parent:
            raise RetrievalProvenanceError(
                f"selected parent has no ranked backing child: {parent.parent_id}"
            )
        spans = []
        for span in parent.source_spans:
            if not span.span_id or not span.block_id:
                raise RetrievalProvenanceError(
                    f"selected parent has incomplete source span identity: {parent.parent_id}"
                )
            spans.append(RawSourceSpan(
                span_id=span.span_id,
                block_id=span.block_id,
                physical_page=span.page,
                printed_page_label=span.printed_page_label,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                raw_text=span.text,
            ))
        try:
            records.append(GenerationContextParent(
                parent_id=parent.parent_id,
                spans=tuple(spans),
                backing_child_ids=children_for_parent,
                embedding_text=parent.text,
            ))
        except ValueError as error:
            raise RetrievalProvenanceError(
                f"selected parent provenance is invalid: {parent.parent_id}"
            ) from error
    try:
        return GenerationContext(corpus_version_id=corpus_version_id, parents=tuple(records))
    except ValueError as error:
        raise RetrievalProvenanceError("generation context provenance is invalid") from error


def _child_source_from_document(document: Document) -> ChildDocumentChunk | None:
    metadata = document.metadata.get("metadata", document.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("retrieved document metadata is invalid")
    if metadata.get("schema_version") != "child-document-chunk-v1":
        return None
    fields = (
        "schema_version", "chunk_id", "corpus_version_id", "parent_id",
        "section_path", "page_start", "page_end", "source_spans",
        "source_text", "child_ordinal", "content_hash", "token_count",
    )
    try:
        return ChildDocumentChunk.model_validate({
            **{field: metadata[field] for field in fields if field in metadata},
            "text": document.page_content,
        })
    except ValidationError:
        raise RetrievalProvenanceError("retrieved child has invalid source provenance") from None


def _passage_from_result(
    document: Document, score: float, child: ChildDocumentChunk | None = None
) -> RetrievedPassage:
    if child is not None:
        return RetrievedPassage(
            chunk_id=child.chunk_id,
            page=child.page_start,
            section=" / ".join(child.section_path)[:500],
            excerpt=document.page_content,
            relevance_score=float(score),
        )
    metadata = document.metadata.get("metadata", document.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("retrieved document metadata is invalid")
    reference = document.metadata.get("reference", {})
    reference_path = reference.get("path", "") if isinstance(reference, dict) else ""
    chunk_id = metadata.get("chunk_id") or reference_path.rsplit("/", 1)[-1]
    if isinstance(metadata.get("section_path"), (list, tuple)):
        section = " / ".join(str(item) for item in metadata["section_path"])[:500]
        page = metadata.get("page_start")
    else:
        section = metadata.get("section")
        page = metadata.get("page")
    if not chunk_id or not section or not isinstance(page, int) or page < 1:
        raise ValueError("retrieved document is missing chunk provenance")
    if not document.page_content:
        raise ValueError("retrieved document is missing text")
    return RetrievedPassage(
        chunk_id=str(chunk_id),
        page=page,
        section=str(section),
        excerpt=document.page_content,
        relevance_score=float(score),
    )
