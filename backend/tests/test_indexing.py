from app.contracts import DocumentChunk
from app.corpus import (
    CorpusEmbeddingWriter,
    FirestoreChunkRepository,
    SourcePage,
    VertexDocumentEmbedder,
    chunk_pages,
)


class FakeDocument:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []

    def set(self, data: dict[str, object]) -> None:
        self.writes.append(data)


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
        return type(
            "Response",
            (),
            {"embeddings": [type("Embedding", (), {"values": [0.0] * 768})()]},
        )()


class FakeEmbeddingClient:
    models = FakeEmbeddingModels()


def test_chunk_upsert_is_deterministic_and_corpus_version_scoped() -> None:
    client = FakeClient()
    repository = FirestoreChunkRepository(client)
    chunk = DocumentChunk(
        chunk_id="v1:3:0",
        corpus_version_id="v1",
        page=3,
        section="Body",
        chunk_ordinal=0,
        content_hash="a" * 64,
        text="A passage",
        embedding=[0.0] * 768,
    )

    repository.upsert(chunk)
    repository.upsert(chunk)

    document = client.collections["document_chunks"].documents[chunk.chunk_id]
    assert len(document.writes) == 2
    assert document.writes[0] == document.writes[1]
    assert document.writes[0]["corpus_version_id"] == "v1"


def test_embedding_writer_persists_the_embedding_with_the_chunk() -> None:
    client = FakeClient()
    repository = FirestoreChunkRepository(client)
    embedder = VertexDocumentEmbedder(FakeEmbeddingClient())
    chunks = chunk_pages(
        [SourcePage(page=1, section="Body", text="A passage")],
        corpus_version_id="v1",
        target_tokens=10,
        overlap_tokens=0,
    )

    stored = CorpusEmbeddingWriter(embedder, repository).write(chunks)

    assert stored[0].embedding == [0.0] * 768
    assert (
        client.collections["document_chunks"].documents["v1:1:0"].writes[0]["corpus_version_id"]
        == "v1"
    )
