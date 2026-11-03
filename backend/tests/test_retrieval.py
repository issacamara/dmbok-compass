from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document

from app.contracts import ParentChunk, RetrievedPassage, SourceSpan
from app.retrieval.search import (
    FirestorePassageRetriever,
    FirestoreParentChunkReader,
    MAX_PARENT_CONTEXT_TOKENS,
    RetrievalEvidence,
    select_parent_context,
)


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, question, *, k, filters):
        self.calls.append((question, k, filters))
        return self.results


class FakeParentReader:
    def __init__(self, parents):
        self.parents = parents
        self.calls = []

    def get_many(self, parent_ids, *, corpus_version_id):
        self.calls.append((parent_ids, corpus_version_id))
        return self.parents


def parent(parent_id: str, corpus: str, text: str) -> ParentChunk:
    return ParentChunk(
        parent_id=parent_id,
        corpus_version_id=corpus,
        section_path=("Governance",),
        page_start=1,
        page_end=1,
        source_spans=[SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)],
        content_hash="a" * 64,
        token_count=max(1, len(text.split())),
        text=text,
    )


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
    assert [passage.relevance_score for passage in passages] == [0.9, 0.8, 0.7, 0.6, 0.5]
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


def test_parent_context_deduplicates_parents_of_top_five_children() -> None:
    results = [
        (
            Document(
                page_content=f"Child {index}",
                metadata={
                    "metadata": {
                        "chunk_id": f"v1:child:{index}",
                        "parent_id": "v1:parent:0",
                        "corpus_version_id": "v1",
                        "page": index + 1,
                        "section": "Governance",
                    }
                },
            ),
            0.9 - index / 100,
        )
        for index in range(5)
    ]
    reader = FakeParentReader([parent("v1:parent:0", "v1", "Governance parent context")])
    retriever = FirestorePassageRetriever(FakeVectorStore(results), reader)

    evidence = retriever.retrieve_with_parent_context(
        "What is governance?", active_corpus_version_id="v1"
    )

    assert isinstance(evidence, RetrievalEvidence)
    assert [passage.chunk_id for passage in evidence.passages] == [
        f"v1:child:{index}" for index in range(5)
    ]
    assert [context.parent_id for context in evidence.parent_context] == ["v1:parent:0"]
    assert reader.calls == [(("v1:parent:0",), "v1")]
    assert [item.parent_id for item in evidence.selected_parents] == ["v1:parent:0"]


def test_parent_context_ignores_missing_and_wrong_corpus_parents() -> None:
    results = [
        (
            Document(
                page_content="Child excerpt",
                metadata={
                    "metadata": {
                        "chunk_id": f"v1:child:{index}",
                        "parent_id": parent_id,
                        "corpus_version_id": "v1",
                        "page": index + 1,
                        "section": "Governance",
                    }
                },
            ),
            .9 - index / 10,
        )
        for index, parent_id in enumerate(("v1:parent:wrong", "v1:parent:missing"))
    ]
    reader = FakeParentReader([parent("v1:parent:wrong", "v2", "Other corpus context")])
    retriever = FirestorePassageRetriever(FakeVectorStore(results), reader)

    evidence = retriever.retrieve_with_parent_context(
        "What is governance?", active_corpus_version_id="v1"
    )

    assert len(evidence.passages) == 2
    assert evidence.parent_context == ()
    assert evidence.selected_parents == ()


def test_firestore_parent_reader_gets_only_requested_documents_and_filters_corpus() -> None:
    matching = parent("v1:parent:0", "v1", "Matching context").model_dump(mode="json")
    matching["metadata"] = {"source_sha256": "a" * 64}
    wrong_corpus = parent("v1:parent:wrong", "v2", "Wrong context").model_dump(mode="json")

    class FakeReference:
        def __init__(self, document_id):
            self.id = document_id

    class FakeCollection:
        def __init__(self):
            self.ids = []

        def document(self, document_id):
            self.ids.append(document_id)
            return FakeReference(document_id)

    class FakeSnapshot:
        def __init__(self, document_id, data):
            self.id = document_id
            self._data = data
            self.exists = data is not None

        def to_dict(self):
            return dict(self._data)

    class FakeClient:
        def __init__(self):
            self.parent_collection = FakeCollection()
            self.requested = ()

        def collection(self, collection_id):
            assert collection_id == "corpus_parent_chunks"
            return self.parent_collection

        def get_all(self, references):
            self.requested = tuple(reference.id for reference in references)
            return iter((
                FakeSnapshot("v1:parent:wrong", wrong_corpus),
                FakeSnapshot("v1:parent:0", matching),
                FakeSnapshot("v1:parent:missing", None),
            ))

    client = FakeClient()
    reader = FirestoreParentChunkReader(client)

    parents = reader.get_many(
        ("v1:parent:0", "v1:parent:wrong", "v1:parent:missing"),
        corpus_version_id="v1",
    )

    assert client.requested == (
        "v1:parent:0", "v1:parent:wrong", "v1:parent:missing"
    )
    assert [item.parent_id for item in parents] == ["v1:parent:0"]


def test_parent_context_never_exceeds_token_budget() -> None:
    children = tuple(
        (
            RetrievedPassage(
                chunk_id=f"v1:child:{index}",
                page=index + 1,
                section="Governance",
                excerpt=f"Child {index}",
                relevance_score=1 - index / 10,
            ),
            f"v1:parent:{index}",
        )
        for index in range(5)
    )
    parents = tuple(
        parent(f"v1:parent:{index}", "v1", "one two three four five six seven eight")
        for index in range(5)
    )
    budget = 20

    context, selected = select_parent_context(
        "Explain governance.", children, parents, max_tokens=budget
    )

    assert len(context) < 5
    assert sum(item.token_count for item in selected) <= budget
    assert all(item.token_count <= MAX_PARENT_CONTEXT_TOKENS for item in selected)


def test_comparison_subjects_are_reserved_before_higher_ranked_other_context() -> None:
    children = (
        (RetrievedPassage(chunk_id="v1:other", page=1, section="S", excerpt="Other", relevance_score=.99), "v1:other"),
        (RetrievedPassage(chunk_id="v1:a", page=1, section="S", excerpt="A subject", relevance_score=.8), "v1:a"),
        (RetrievedPassage(chunk_id="v1:b", page=1, section="S", excerpt="B subject", relevance_score=.7), "v1:b"),
    )
    parents = (
        parent("v1:other", "v1", "General material with no comparison topic"),
        parent("v1:a", "v1", "A subject detail"),
        parent("v1:b", "v1", "B subject detail"),
    )
    budget = 6

    context, selected = select_parent_context(
        "Compare A subject and B subject in terms of accountability.",
        children,
        parents,
        max_tokens=budget,
        child_scores={"v1:other": .99, "v1:a": .8, "v1:b": .7},
    )

    selected_ids = {parent_chunk.parent_id for parent_chunk in context}
    assert {"v1:a", "v1:b"} <= selected_ids
    assert "v1:other" not in selected_ids
    assert sum(item.token_count for item in selected) <= budget


def test_concurrent_parent_context_reads_use_request_local_corpus_ids() -> None:
    class VersionedVectorStore:
        def similarity_search_with_score(self, question, *, k, filters):
            version = filters.value
            return [
                (
                    Document(
                        page_content=f"{version} child",
                        metadata={
                            "metadata": {
                                "chunk_id": f"{version}:child:0",
                                "parent_id": f"{version}:parent:0",
                                "corpus_version_id": version,
                                "page": 1,
                                "section": "Governance",
                            }
                        },
                    ),
                    .9,
                )
            ]

    class VersionedParentReader:
        def get_many(self, parent_ids, *, corpus_version_id):
            return (parent(parent_ids[0], corpus_version_id, f"{corpus_version_id} context"),)

    retriever = FirestorePassageRetriever(VersionedVectorStore(), VersionedParentReader())

    async def retrieve_all():
        return await asyncio.gather(*(
            asyncio.to_thread(
                retriever.retrieve_with_parent_context,
                f"question {version}",
                active_corpus_version_id=version,
            )
            for version in ("v1", "v2", "v3")
        ))

    results = asyncio.run(retrieve_all())

    assert [result.passages[0].chunk_id for result in results] == [
        "v1:child:0", "v2:child:0", "v3:child:0"
    ]
    assert [result.parent_context[0].corpus_version_id for result in results] == [
        "v1", "v2", "v3"
    ]
