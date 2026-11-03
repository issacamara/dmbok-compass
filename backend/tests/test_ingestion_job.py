from datetime import datetime, timezone
from hashlib import sha256

import pytest

from app.contracts import ChildDocumentChunk, CorpusVersion, ParentChunk, SourceSpan
from app.corpus import (
    CorpusEmbeddingWriter,
    InMemoryCorpusVersionStore,
    SourcePage,
    TextChunk,
    chunk_pages,
    create_source_manifest,
    default_derivation,
)
from app.corpus.indexing import HierarchicalCorpusWriter
from app.ingestion import CorpusIngestionJob
from app.ingestion.job import IngestionInterrupted


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, object] = {}
        self.batch_sizes: list[int] = []
        self.fail_next = False

    def add_documents(self, documents, *, ids):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("vector write failed")
        self.batch_sizes.append(len(ids))
        self.documents.update(zip(ids, documents))
        return ids


class FakeParentStore:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, object]] = {}
        self.batch_sizes: list[int] = []

    def write_batch(self, parents, *, metadata=None):
        self.batch_sizes.append(len(parents))
        for parent in parents:
            self.documents[parent.parent_id] = {
                **parent.model_dump(mode="json"),
                "metadata": dict(metadata or {}),
            }
        return tuple(parents)


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


def many_hierarchical_records(count: int, *, version_id: str = "v1"):
    text = "x" * count
    parent = ParentChunk(
        parent_id=f"{version_id}:parent:chapter:0",
        corpus_version_id=version_id,
        section_path=("Chapter",),
        page_start=1,
        page_end=1,
        source_spans=[SourceSpan(page=1, start_offset=0, end_offset=count, text=text)],
        content_hash=sha256(text.encode()).hexdigest(),
        token_count=count,
        text=text,
    )
    children = tuple(
        ChildDocumentChunk(
            chunk_id=f"{version_id}:child:{ordinal}",
            corpus_version_id=version_id,
            parent_id=parent.parent_id,
            section_path=parent.section_path,
            page_start=1,
            page_end=1,
            source_spans=[
                SourceSpan(page=1, start_offset=ordinal, end_offset=ordinal + 1, text="x")
            ],
            child_ordinal=ordinal,
            content_hash=sha256(b"x").hexdigest(),
            token_count=1,
            text="x",
        )
        for ordinal in range(count)
    )
    return (parent,), children


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


def test_hierarchical_ingestion_batches_501_children_and_retries_by_explicit_identity() -> None:
    vector_store = FakeVectorStore()
    parent_store = FakeParentStore()
    versions = InMemoryCorpusVersionStore()
    writer = HierarchicalCorpusWriter(
        vector_store,
        parent_store,
        source_metadata={"source_uri": "gs://corpus/source.pdf#1"},
    )
    job = CorpusIngestionJob(versions, writer)
    parents, children = many_hierarchical_records(501)

    first = job.run_hierarchical(staged_version(), parents, children)
    retry = job.run_hierarchical(staged_version(), parents, children)

    assert first.version.status == retry.version.status == "staged"
    assert vector_store.batch_sizes == [500, 1, 500, 1]
    assert parent_store.batch_sizes == [1, 1]
    assert set(vector_store.documents) == {child.chunk_id for child in children}
    assert len(vector_store.documents) == len(children)
    assert all(
        document.metadata["parent_id"] == parents[0].parent_id
        and document.metadata["chunk_id"] == chunk_id
        and document.metadata["source_uri"] == "gs://corpus/source.pdf#1"
        and document.metadata["content_hash"] == sha256(b"x").hexdigest()
        for chunk_id, document in vector_store.documents.items()
    )
    assert len(parent_store.documents) == 1
    assert versions.active() is None


def test_hierarchical_write_failure_leaves_active_corpus_untouched() -> None:
    vector_store = FakeVectorStore()
    parent_store = FakeParentStore()
    versions = InMemoryCorpusVersionStore()
    legacy = staged_version("legacy")
    versions.stage(legacy)
    versions.validate("legacy", many_chunks(1, version_id="legacy"))
    versions.activate("legacy")
    active_before = versions.active()
    parents, children = many_hierarchical_records(1, version_id="candidate")
    vector_store.fail_next = True
    job = CorpusIngestionJob(
        versions,
        HierarchicalCorpusWriter(vector_store, parent_store),
    )
    candidate = staged_version("candidate")

    with pytest.raises(RuntimeError, match="vector write failed"):
        job.run_hierarchical(candidate, parents, children)

    assert versions.active() == active_before
    assert versions.get("candidate").status == "staged"  # type: ignore[union-attr]
    assert len(parent_store.documents) == 1
    assert vector_store.documents == {}
    assert vector_store.batch_sizes == []


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
