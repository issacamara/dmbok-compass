from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.retrieval.search import (
    DISTANCE_RESULT_FIELD,
    QUERY_TASK_TYPE,
    FirestorePassageRetriever,
    VertexQueryEmbedder,
)


class FakeEmbeddingModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * 768)])


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.models = FakeEmbeddingModels()


class FakeDocument:
    def __init__(self, data: dict) -> None:
        self.data = data

    def to_dict(self) -> dict:
        return self.data


class FakeNearestQuery:
    def __init__(self, documents: list[FakeDocument]) -> None:
        self.documents = documents

    def stream(self):
        return iter(self.documents)


class FakeFilteredQuery:
    def __init__(self, parent: "FakeCollection") -> None:
        self.parent = parent

    def find_nearest(self, **kwargs):
        self.parent.nearest_kwargs = kwargs
        return FakeNearestQuery(self.parent.documents)


class FakeCollection:
    def __init__(self, documents: list[FakeDocument]) -> None:
        self.documents = documents
        self.filter = None
        self.nearest_kwargs = None

    def where(self, **kwargs):
        self.filter = kwargs["filter"]
        return FakeFilteredQuery(self)


class FakeFirestore:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection_value = collection

    def collection(self, collection_id: str) -> FakeCollection:
        assert collection_id == "document_chunks"
        return self.collection_value


def test_query_embedder_uses_query_task_and_document_dimensions() -> None:
    client = FakeEmbeddingClient()

    vector = VertexQueryEmbedder(client).embed("What is data governance?")

    assert len(vector) == 768
    assert client.models.calls[0]["config"] == {
        "task_type": QUERY_TASK_TYPE,
        "output_dimensionality": 768,
    }


def test_retriever_filters_active_version_and_returns_ranked_top_five() -> None:
    documents = [
        FakeDocument(
            {
                "chunk_id": f"v1:{index}:0",
                "page": index + 1,
                "section": "Governance",
                "text": f"Passage {index}",
                DISTANCE_RESULT_FIELD: distance,
            }
        )
        for index, distance in enumerate([0.4, 0.1, 0.3, 0.2, 0.5, 0.6])
    ]
    collection = FakeCollection(documents)
    retriever = FirestorePassageRetriever(
        FakeFirestore(collection), VertexQueryEmbedder(FakeEmbeddingClient())
    )

    passages = retriever.retrieve("What is governance?", active_corpus_version_id="v1")

    assert len(passages) == 5
    assert [passage.chunk_id for passage in passages] == [
        "v1:1:0",
        "v1:3:0",
        "v1:2:0",
        "v1:0:0",
        "v1:4:0",
    ]
    assert collection.filter.field_path == "corpus_version_id"
    assert collection.filter.value == "v1"
    assert collection.nearest_kwargs["limit"] == 5
    assert collection.nearest_kwargs["vector_field"] == "embedding"


def test_retriever_rejects_invalid_limits_and_empty_active_version() -> None:
    retriever = FirestorePassageRetriever(
        FakeFirestore(FakeCollection([])), VertexQueryEmbedder(FakeEmbeddingClient())
    )

    with pytest.raises(ValueError, match="active_corpus_version_id"):
        retriever.retrieve("question", active_corpus_version_id=" ")
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve("question", active_corpus_version_id="v1", limit=6)
