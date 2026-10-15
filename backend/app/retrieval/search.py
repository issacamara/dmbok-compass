"""Score-preserving native LangChain retrieval for one pinned corpus."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from time import perf_counter
from typing import Callable

from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore

from app.contracts import RetrievedPassage

DEFAULT_LIMIT = 5

# Firestore's LangChain vector-store integration exposes only synchronous search.
# Keep that work off request event loops and cap concurrent in-process RPCs.
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="dmbok-retrieval",
)


async def retrieve_off_event_loop(
    retrieve: Callable[..., tuple[RetrievedPassage, ...]],
    question: str,
    *,
    active_corpus_version_id: str,
    limit: int,
) -> tuple[tuple[RetrievedPassage, ...], float]:
    """Run one request-local synchronous retrieval in the bounded pool."""

    def execute() -> tuple[tuple[RetrievedPassage, ...], float]:
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
