from datetime import datetime, timezone
from hashlib import sha256

import pytest

from app.contracts import CorpusVersion
from app.corpus import (
    CorpusEmbeddingWriter,
    InMemoryCorpusVersionStore,
    SourcePage,
    TextChunk,
    chunk_pages,
    create_source_manifest,
    default_derivation,
)
from app.ingestion import CorpusIngestionJob
from app.ingestion.job import IngestionInterrupted


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, object] = {}
        self.batch_sizes: list[int] = []

    def add_documents(self, documents, *, ids):
        self.batch_sizes.append(len(ids))
        self.documents.update(zip(ids, documents))
        return ids


def staged_version(version_id: str = "v1") -> CorpusVersion:
    return CorpusVersion(
        version_id=version_id,
        source_uri="gs://corpus/source.pdf#1",
        embedding_model="gemini-embedding-001",
        status="staged",
        created_at=datetime.now(timezone.utc),
    )


def many_chunks(count: int, *, version_id: str = "v1") -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id=f"{version_id}:1:{ordinal}",
            corpus_version_id=version_id,
            page=1,
            section="Body",
            chunk_ordinal=ordinal,
            content_hash=sha256(f"chunk {ordinal}".encode()).hexdigest(),
            text=f"chunk {ordinal}",
        )
        for ordinal in range(count)
    ]


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

    assert first.activated is False
    assert first.ready_for_evaluation is True
    assert second.ready_for_evaluation is True
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

    assert first.ready_for_evaluation is True
    assert second.ready_for_evaluation is True
    assert original_manifest.corpus_version_id != changed_manifest.corpus_version_id
    assert first_chunks[0].chunk_id in vector_store.documents
    assert second_chunks[0].chunk_id in vector_store.documents
    assert versions.get(original_manifest.corpus_version_id).status == "validated"  # type: ignore[union-attr]


def test_interruption_after_a_500_chunk_batch_keeps_resumable_staging_and_retry_activates() -> None:
    vector_store = FakeVectorStore()
    versions = InMemoryCorpusVersionStore()
    job = CorpusIngestionJob(versions, CorpusEmbeddingWriter(vector_store))
    chunks = many_chunks(501)

    with pytest.raises(IngestionInterrupted, match="persisted batch"):
        job.run(
            staged_version(),
            chunks,
            termination_requested=lambda: len(vector_store.batch_sizes) >= 1,
        )

    assert versions.get("v1").status == "staged"  # type: ignore[union-attr]
    assert len(vector_store.documents) == 500
    assert vector_store.batch_sizes == [500]
    first_document = vector_store.documents["v1:1:0"]
    assert first_document.metadata["chunk_id"] == "v1:1:0"  # type: ignore[union-attr]

    retried = job.run(staged_version(), chunks)

    assert retried.ready_for_evaluation is True
    assert versions.get("v1").status == "validated"  # type: ignore[union-attr]
    assert len(vector_store.documents) == 501
    assert vector_store.batch_sizes == [500, 500, 1]


def test_entrypoint_exits_nonzero_and_emits_content_free_interruption_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.ingestion import entrypoint

    monkeypatch.setattr(
        entrypoint,
        "_run",
        lambda termination_requested: (_ for _ in ()).throw(IngestionInterrupted("stop")),
    )
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *_: entrypoint.signal.SIG_DFL)

    with pytest.raises(SystemExit) as exited:
        entrypoint.main()

    assert exited.value.code == 1
    assert '"event": "dmbok_ingestion_interrupted"' in capsys.readouterr().out
