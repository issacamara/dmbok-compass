from datetime import datetime, timezone

from app.contracts import CorpusVersion
from app.corpus import (
    CorpusEmbeddingWriter,
    InMemoryCorpusVersionStore,
    SourcePage,
    chunk_pages,
)
from app.ingestion import CorpusIngestionJob


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, object] = {}

    def add_documents(self, documents, *, ids):
        self.documents.update(zip(ids, documents))
        return ids


def test_retry_does_not_create_duplicate_chunks_or_versions() -> None:
    store = FakeVectorStore()
    job = CorpusIngestionJob(
        InMemoryCorpusVersionStore(),
        CorpusEmbeddingWriter(store),
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
    assert len(store.documents) == 1
