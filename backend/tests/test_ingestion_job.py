from datetime import datetime, timezone

from app.contracts import CorpusVersion
from app.corpus import (
    CorpusEmbeddingWriter,
    InMemoryCorpusVersionStore,
    SourcePage,
    chunk_pages,
    create_source_manifest,
    default_derivation,
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


def test_changed_derivation_for_the_same_pdf_stages_a_new_corpus_without_overwriting_legacy_ids() -> None:
    vector_store = FakeVectorStore()
    versions = InMemoryCorpusVersionStore()
    job = CorpusIngestionJob(versions, CorpusEmbeddingWriter(vector_store))
    source = {
        "bucket": "corpus",
        "object_name": "source.pdf",
        "generation": 1,
        "content_sha256": "a" * 64,
    }
    original_manifest = create_source_manifest(**source)
    changed_derivation = default_derivation()
    changed_derivation["normalization"]["version"] = "whitespace-v2"
    changed_manifest = create_source_manifest(**source, derivation=changed_derivation)

    def ingest(manifest):
        version = CorpusVersion(
            version_id=manifest.corpus_version_id,
            source_uri=manifest.object_uri,
            embedding_model="gemini-embedding-001",
            status="staged",
            created_at=datetime.now(timezone.utc),
        )
        chunks = chunk_pages(
            [SourcePage(page=1, section="Body", text="A passage")],
            corpus_version_id=manifest.corpus_version_id,
            target_tokens=10,
            overlap_tokens=0,
        )
        return job.run(version, chunks), chunks

    first, first_chunks = ingest(original_manifest)
    second, second_chunks = ingest(changed_manifest)

    assert first.activated is True
    assert second.activated is True
    assert original_manifest.corpus_version_id != changed_manifest.corpus_version_id
    assert first_chunks[0].chunk_id in vector_store.documents
    assert second_chunks[0].chunk_id in vector_store.documents
    assert versions.get(original_manifest.corpus_version_id).status == "retired"  # type: ignore[union-attr]
