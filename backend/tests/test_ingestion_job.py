from datetime import datetime, timezone

from app.contracts import CorpusVersion
from app.corpus import (
    CorpusEmbeddingWriter,
    FirestoreChunkRepository,
    InMemoryCorpusVersionStore,
    SourcePage,
    VertexDocumentEmbedder,
    chunk_pages,
)
from app.ingestion import CorpusIngestionJob


class FakeDocument:
    def __init__(self) -> None:
        self.data: dict[str, object] | None = None

    def set(self, data: dict[str, object]) -> None:
        self.data = data


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, FakeDocument] = {}

    def document(self, document_id: str) -> FakeDocument:
        return self.documents.setdefault(document_id, FakeDocument())


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def collection(self, collection_id: str) -> FakeCollection:
        return self.collections.setdefault(collection_id, FakeCollection())


class FakeEmbeddingModels:
    def embed_content(self, **kwargs: object) -> object:
        return type("Response", (), {"embeddings": [type("Embedding", (), {"values": [0.0] * 768})()]})()


class FakeEmbeddingClient:
    models = FakeEmbeddingModels()


def test_retry_does_not_create_duplicate_chunks_or_versions() -> None:
    client = FakeClient()
    job = CorpusIngestionJob(
        InMemoryCorpusVersionStore(),
        CorpusEmbeddingWriter(VertexDocumentEmbedder(FakeEmbeddingClient()), FirestoreChunkRepository(client)),
    )
    version = CorpusVersion(
        version_id="v1",
        source_uri="gs://corpus/source.pdf#1",
        embedding_model="gemini-embedding-001",
        status="staged",
        created_at=datetime.now(timezone.utc),
    )
    chunks = chunk_pages(
        [SourcePage(page=1, section="Body", text="A passage")],
        corpus_version_id="v1",
        target_tokens=10,
        overlap_tokens=0,
    )

    first = job.run(version, chunks)
    second = job.run(version, chunks)

    assert first.activated is True
    assert second.activated is False
    assert len(client.collections["document_chunks"].documents) == 1
