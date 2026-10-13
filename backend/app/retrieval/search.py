"""Score-preserving native LangChain retrieval for one pinned corpus."""

from __future__ import annotations

from math import isfinite

from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore

from app.contracts import RetrievedPassage

DEFAULT_LIMIT = 5


class FirestorePassageRetriever:
    """Map native LangChain search results to the stable application contract."""

    def __init__(self, vector_store: FirestoreVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        active_corpus_version_id: str,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[RetrievedPassage, ...]:
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
        passages = [
            _passage_from_result(document, score)
            for document, score in results
            if isfinite(score)
        ]
        passages.sort(
            key=lambda passage: (
                -(
                    passage.relevance_score
                    if passage.relevance_score is not None
                    else float("-inf")
                ),
                passage.chunk_id,
            )
        )
        return tuple(passages[:limit])


def _passage_from_result(document: Document, score: float) -> RetrievedPassage:
    metadata = document.metadata.get("metadata", document.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("retrieved document metadata is invalid")
    reference = document.metadata.get("reference", {})
    reference_path = reference.get("path", "") if isinstance(reference, dict) else ""
    chunk_id = metadata.get("chunk_id") or reference_path.rsplit("/", 1)[-1]
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
