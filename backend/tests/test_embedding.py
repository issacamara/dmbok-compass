from types import SimpleNamespace

import pytest

from app.corpus import VertexDocumentEmbedder


class FakeModels:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls: list[dict[str, object]] = []

    def embed_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=self.values)])


class FakeClient:
    def __init__(self, values: list[float]) -> None:
        self.models = FakeModels(values)


def test_document_embedding_uses_fixed_model_task_and_dimension() -> None:
    client = FakeClient([0.0] * 768)

    vector = VertexDocumentEmbedder(client).embed("A corpus passage.")

    assert len(vector) == 768
    assert client.models.calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "A corpus passage.",
            "config": {
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": 768,
            },
        }
    ]


def test_document_embedding_rejects_wrong_vertex_dimension() -> None:
    with pytest.raises(ValueError, match="expected 768"):
        VertexDocumentEmbedder(FakeClient([0.0] * 767)).embed("passage")
