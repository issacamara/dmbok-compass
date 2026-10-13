from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrieval.search import FirestorePassageRetriever


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, question, *, k, filters):
        self.calls.append((question, k, filters))
        return self.results


def test_retriever_filters_active_version_and_returns_ranked_top_five() -> None:
    results = [
        (
            Document(
                page_content=f"Passage {index}",
                metadata={
                    "metadata": {
                        "chunk_id": f"v1:{index}:0",
                        "page": index + 1,
                        "section": "Governance",
                    }
                },
            ),
            score,
        )
        for index, score in enumerate([0.6, 0.9, 0.7, 0.8, 0.5, float("nan")])
    ]
    store = FakeVectorStore(results)
    retriever = FirestorePassageRetriever(store)

    passages = retriever.retrieve("What is governance?", active_corpus_version_id="v1")

    assert len(passages) == 5
    assert [passage.chunk_id for passage in passages] == [
        "v1:1:0",
        "v1:3:0",
        "v1:2:0",
        "v1:0:0",
        "v1:4:0",
    ]
    _, limit, corpus_filter = store.calls[0]
    assert limit == 5
    assert corpus_filter.field_path == "metadata.corpus_version_id"
    assert corpus_filter.value == "v1"


def test_retriever_rejects_invalid_limits_and_empty_active_version() -> None:
    retriever = FirestorePassageRetriever(FakeVectorStore([]))

    with pytest.raises(ValueError, match="active_corpus_version_id"):
        retriever.retrieve("question", active_corpus_version_id=" ")
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve("question", active_corpus_version_id="v1", limit=6)
