from types import SimpleNamespace

import pytest

from app.corpus import create_vertex_embeddings


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

    embeddings = create_vertex_embeddings(
        project="test-project", location="europe-west1", client=client
    )
    vector = embeddings.embed_documents(["A corpus passage."])[0]

    assert len(vector) == 768
    assert client.models.calls[0]["model"] == "gemini-embedding-001"
    assert client.models.calls[0]["contents"][0]["parts"][0]["text"] == "A corpus passage."
    assert client.models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert client.models.calls[0]["config"].output_dimensionality == 768


def test_query_embedding_uses_query_task_and_document_dimensions() -> None:
    client = FakeClient([0.25] * 768)
    embeddings = create_vertex_embeddings(
        project="test-project", location="europe-west1", client=client
    )

    vector = embeddings.embed_query("What is data governance?")

    assert len(vector) == 768
    assert client.models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert client.models.calls[0]["config"].output_dimensionality == 768


def test_document_embedding_rejects_wrong_vertex_dimension() -> None:
    with pytest.raises(ValueError, match="expected 768"):
        create_vertex_embeddings(
            project="test-project",
            location="europe-west1",
            client=FakeClient([0.0] * 767),
        ).embed_documents(["passage"])
