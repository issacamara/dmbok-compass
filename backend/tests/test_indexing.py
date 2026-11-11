from hashlib import sha256

from app.contracts import ChildDocumentChunk, ParentChunk, SourceSpan
from app.corpus import CorpusEmbeddingWriter, SourcePage, chunk_pages, create_source_manifest
from app.corpus.indexing import HierarchicalCorpusWriter


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], list[str]]] = []
        self.documents: dict[str, object] = {}

    def add_documents(self, documents, *, ids):
        self.calls.append((documents, ids))
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


def hierarchical_records(count: int = 1):
    text = "x" * count
    parent = ParentChunk(
        parent_id="v1:parent:chapter:0",
        corpus_version_id="v1",
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
            chunk_id=f"v1:child:{ordinal}",
            corpus_version_id="v1",
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


def test_embedding_writer_persists_the_canonical_corpus_manifest_metadata() -> None:
    store = FakeVectorStore()
    manifest = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256="a" * 64,
    )
    chunks = chunk_pages(
        [SourcePage(page=1, section="Body", text="A passage")],
        corpus_version_id=manifest.corpus_version_id,
        target_tokens=10,
        overlap_tokens=0,
    )

    CorpusEmbeddingWriter(store, source_metadata=manifest.corpus_metadata()).write(chunks)

    metadata = store.calls[0][0][0].metadata
    assert metadata["corpus_manifest_sha256"] == manifest.manifest_sha256
    assert metadata["corpus_manifest"]["derivation"]["splitter"]["chunk_size"] == 700


def test_hierarchical_writer_preserves_explicit_ids_and_nested_metadata() -> None:
    vector_store = FakeVectorStore()
    parent_store = FakeParentStore()
    parents, children = hierarchical_records(2)
    manifest = create_source_manifest(
        bucket="dmbok-corpus",
        object_name="source/dmbok.pdf",
        generation=7,
        content_sha256="a" * 64,
    )
    writer = HierarchicalCorpusWriter(
        vector_store,
        parent_store,
        source_metadata=manifest.corpus_metadata(),
    )

    writer.write_parent_batch(parents)
    writer.write_child_batch(children)

    parent_record = parent_store.documents[parents[0].parent_id]
    child_document = vector_store.documents[children[0].chunk_id]
    assert "embedding" not in parent_record
    assert parent_record["source_spans"][0]["page"] == 1
    assert parent_record["metadata"]["corpus_manifest_sha256"] == manifest.manifest_sha256
    assert vector_store.calls[0][1] == [child.chunk_id for child in children]
    assert child_document.metadata["parent_id"] == parents[0].parent_id
    assert child_document.metadata["source_spans"][0] == children[0].source_spans[0].model_dump(
        mode="json", exclude_none=True
    )
    assert child_document.metadata["corpus_manifest_sha256"] == manifest.manifest_sha256
