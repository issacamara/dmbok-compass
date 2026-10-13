from app.corpus import (
    CorpusEmbeddingWriter,
    SourcePage,
    chunk_pages,
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], list[str]]] = []

    def add_documents(self, documents, *, ids):
        self.calls.append((documents, ids))
        return ids


def test_chunk_upsert_is_deterministic_and_corpus_version_scoped() -> None:
    store = FakeVectorStore()
    chunks = chunk_pages(
        [SourcePage(page=3, section="Body", text="A passage")],
        corpus_version_id="v1",
        target_tokens=10,
        overlap_tokens=0,
    )
    writer = CorpusEmbeddingWriter(store, source_metadata={"source_uri": "gs://corpus.pdf#1"})

    writer.write(chunks)
    writer.write(chunks)

    first_documents, first_ids = store.calls[0]
    assert first_ids == ["v1:3:0"]
    assert store.calls[1][1] == first_ids
    assert first_documents[0].metadata["corpus_version_id"] == "v1"
    assert first_documents[0].metadata["source_uri"] == "gs://corpus.pdf#1"


def test_embedding_writer_persists_the_embedding_with_the_chunk() -> None:
    store = FakeVectorStore()
    chunks = chunk_pages(
        [SourcePage(page=1, section="Body", text="A passage")],
        corpus_version_id="v1",
        target_tokens=10,
        overlap_tokens=0,
    )

    stored = CorpusEmbeddingWriter(store).write(chunks)

    assert stored == chunks
    document = store.calls[0][0][0]
    assert document.page_content == "A passage"
    assert document.metadata["chunk_id"] == "v1:1:0"
    assert document.metadata["schema_version"] == "2"
