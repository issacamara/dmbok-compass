from dataclasses import dataclass

import pytest

from app.contracts import ChildDocumentChunk, SourceSpan
from app.corpus.search_probe import CandidateFilteredVectorProbe


@dataclass
class Document:
    metadata: dict[str, object]


@dataclass
class FakeVectorStore:
    results: list[tuple[Document, float]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def similarity_search_with_score(self, query, *, k, filters):
        self.calls.append({"query": query, "k": k, "filters": filters})
        return self.results


def candidate():
    return type(
        "Candidate",
        (),
        {
            "chunk_id": "v1:1:0",
            "corpus_version_id": "v1",
            "text": "A known corpus passage.",
            "page": 1,
            "section": "Introduction",
        },
    )()


def document(*, version_id: str = "v1", chunk_id: str = "v1:1:0") -> Document:
    return Document(
        {
            "metadata": {
                "corpus_version_id": version_id,
                "chunk_id": chunk_id,
                "schema_version": "2",
                "page": 1,
                "section": "Introduction",
            }
        }
    )


def test_probe_uses_filtered_score_path_and_records_independent_readiness() -> None:
    store = FakeVectorStore([(document(), 0.91)])

    evidence = CandidateFilteredVectorProbe(
        store, collection_id="document_chunks_v2", index_id="idx-1"
    ).run(version_id="v1", candidate=candidate())

    assert evidence.complete.status == "complete"
    assert evidence.search_ready is not None
    assert evidence.search_ready.status == "search_ready"
    assert evidence.search_ready.index_id == "idx-1"
    assert evidence.search_ready.candidate_chunk_id == "v1:1:0"
    assert store.calls[0]["k"] == 5
    assert store.calls[0]["filters"].field_path == "metadata.corpus_version_id"
    assert store.calls[0]["filters"].value == "v1"


def test_probe_rejects_a_stale_or_non_finite_result() -> None:
    store = FakeVectorStore([(document(version_id="old"), float("nan"))])

    with pytest.raises(RuntimeError, match="readiness probe failed"):
        CandidateFilteredVectorProbe(store, collection_id="document_chunks_v2").run(
            version_id="v1", candidate=candidate()
        )


def test_probe_retries_bounded_index_propagation_failures() -> None:
    class FailingStore:
        def __init__(self) -> None:
            self.calls = 0

        def similarity_search_with_score(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("index building")

    store = FailingStore()
    with pytest.raises(RuntimeError):
        CandidateFilteredVectorProbe(
            store,
            collection_id="document_chunks_v2",
            max_attempts=2,
            sleeper=lambda _: None,
        ).run(version_id="v1", candidate=candidate())
    assert store.calls == 2


def test_probe_validates_hierarchical_child_provenance_and_schema() -> None:
    span = SourceSpan(page=2, start_offset=4, end_offset=9, text="child")
    child = ChildDocumentChunk(
        chunk_id="v1:child:abc",
        corpus_version_id="v1",
        parent_id="v1:parent:1",
        section_path=("Chapter 1", "Section A"),
        page_start=2,
        page_end=2,
        source_spans=[span],
        child_ordinal=0,
        content_hash="a" * 64,
        token_count=1,
        text="child",
    )
    metadata = {
        **child.model_dump(mode="json"),
        "chunk_id": child.chunk_id,
    }
    store = FakeVectorStore([(Document(metadata), 0.91)])

    evidence = CandidateFilteredVectorProbe(
        store, collection_id="document_chunks_v2", index_id="idx-1"
    ).run(version_id="v1", candidate=child)

    assert evidence.search_ready is not None
    assert evidence.search_ready.candidate_chunk_id == child.chunk_id
    assert store.calls[0]["filters"].field_path == "metadata.corpus_version_id"
    assert store.calls[0]["filters"].value == "v1"

    stale = {**metadata, "parent_id": "v0:parent:1"}
    stale_store = FakeVectorStore([(Document(stale), 0.91)])
    with pytest.raises(RuntimeError, match="readiness probe failed"):
        CandidateFilteredVectorProbe(
            stale_store,
            collection_id="document_chunks_v2",
            max_attempts=2,
            sleeper=lambda _: None,
        ).run(version_id="v1", candidate=child)
    assert len(stale_store.calls) == 2
