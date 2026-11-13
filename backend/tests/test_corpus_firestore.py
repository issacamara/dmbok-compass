from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256

import pytest
from google.cloud.firestore_v1.vector import Vector

from app.contracts import ChildDocumentChunk, CorpusServingBinding, CorpusValidationEvidence, CorpusVersion, ParentChunk, SourceSpan
from app.corpus.chunking import TextChunk
from app.corpus.firestore import FirestoreCorpusVersionStore, FirestoreParentChunkStore
from app.corpus.search_probe import ReadinessSignal, SearchReadinessEvidence
from app.corpus.chunking import _token_count
from app.corpus.versions import ConcurrentCorpusActivationError, CorpusVersionError


@dataclass
class FakeSnapshot:
    reference: "FakeDocument"
    data: dict[str, object] | None

    @property
    def exists(self) -> bool:
        return self.data is not None

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.data) if self.data is not None else {}


@dataclass
class FakeDocument:
    client: "FakeFirestore"
    path: str

    def get(self, *, transaction: "FakeTransaction | None" = None) -> FakeSnapshot:
        source = transaction.data if transaction is not None else self.client.data
        return FakeSnapshot(self, source.get(self.path))

    def set(self, data: dict[str, object]) -> None:
        self.client.data[self.path] = deepcopy(data)


@dataclass
class FakeBatch:
    client: "FakeFirestore"
    deletions: list[FakeDocument] = field(default_factory=list)
    writes: list[tuple[FakeDocument, dict[str, object]]] = field(default_factory=list)

    def delete(self, reference: FakeDocument) -> None:
        self.deletions.append(reference)

    def set(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.writes.append((reference, deepcopy(data)))

    def commit(self) -> None:
        for reference, data in self.writes:
            self.client.data[reference.path] = data
        for reference in self.deletions:
            self.client.data.pop(reference.path, None)


@dataclass
class FakeQuery:
    collection: "FakeCollection"
    field: str
    value: object
    maximum: int | None = None

    def limit(self, maximum: int) -> "FakeQuery":
        self.maximum = maximum
        return self

    def stream(self) -> list[FakeSnapshot]:
        snapshots = [
            snapshot
            for snapshot in self.collection.stream()
            if _value_at_path(snapshot.to_dict(), self.field) == self.value
        ]
        return snapshots[: self.maximum]


def _value_at_path(data: dict[str, object], path: str) -> object | None:
    value: object = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


@dataclass
class FakeCollection:
    client: "FakeFirestore"
    path: str

    def document(self, document_id: str) -> FakeDocument:
        return FakeDocument(self.client, f"{self.path}/{document_id}")

    def where(
        self,
        field: str | None = None,
        operator: str | None = None,
        value: object | None = None,
        *,
        filter: object | None = None,
    ) -> FakeQuery:
        if filter is not None:
            return FakeQuery(self, filter.field_path, filter.value)  # type: ignore[attr-defined]
        assert field is not None and operator == "=="
        return FakeQuery(self, field, value)

    def stream(self) -> list[FakeSnapshot]:
        depth = self.path.count("/") + 1
        return [
            FakeSnapshot(FakeDocument(self.client, path), data)
            for path, data in self.client.data.items()
            if path.startswith(f"{self.path}/") and path.count("/") == depth
        ]


@dataclass
class FakeTransaction:
    client: "FakeFirestore"
    data: dict[str, dict[str, object]]

    def update(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.data[reference.path] = {
            **self.data.get(reference.path, {}),
            **deepcopy(data),
        }

    def set(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.data[reference.path] = deepcopy(data)


@dataclass
class FakeFirestore:
    data: dict[str, dict[str, object]] = field(default_factory=dict)

    def collection(self, collection_id: str) -> FakeCollection:
        return FakeCollection(self, collection_id)

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self, deepcopy(self.data))

    def batch(self) -> FakeBatch:
        return FakeBatch(self)


def run_transaction(transaction: FakeTransaction, callback):  # type: ignore[no-untyped-def]
    result = callback(transaction)
    transaction.client.data = transaction.data
    return result


def binding(version_id: str, collection: str = "document_chunks_v2") -> CorpusServingBinding:
    return CorpusServingBinding(
        corpus_version_id=version_id,
        vector_collection=collection,
        vector_schema_version="2",
        embedding_model="gemini-embedding-001",
        ingestion_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )


def evidence() -> CorpusValidationEvidence:
    return CorpusValidationEvidence(
        chunk_count=1,
        content_hashes_sha256="c" * 64,
        provenance_sha256="d" * 64,
        index_ready=True,
        filter_ready=True,
    )


def version(
    version_id: str,
    status: str,
    *,
    serving_binding: CorpusServingBinding | None = None,
    validation_evidence: CorpusValidationEvidence | None = None,
) -> CorpusVersion:
    return CorpusVersion(
        version_id=version_id,
        source_uri=f"gs://corpus/source.pdf#{version_id}",
        embedding_model="gemini-embedding-001",
        status=status,
        created_at=datetime.now(timezone.utc),
        serving_binding=serving_binding,
        validation_evidence=validation_evidence,
    )


def store(client: FakeFirestore) -> FirestoreCorpusVersionStore:
    return FirestoreCorpusVersionStore(client, transaction_runner=run_transaction)


def persist(client: FakeFirestore, item: CorpusVersion) -> None:
    client.data[f"corpus_versions/{item.version_id}"] = item.model_dump(mode="json")


def chunk(version_id: str) -> TextChunk:
    text = "Stored provenance."
    return TextChunk(
        chunk_id=f"{version_id}:1:0",
        corpus_version_id=version_id,
        page=1,
        section="Introduction",
        chunk_ordinal=0,
        content_hash=sha256(text.encode()).hexdigest(),
        text=text,
    )


def parent_chunk(version_id: str = "v1") -> ParentChunk:
    spans = [
        SourceSpan(page=1, start_offset=2, end_offset=5, text="one"),
        SourceSpan(page=2, start_offset=0, end_offset=3, text="two"),
    ]
    return ParentChunk(
        parent_id=f"{version_id}:parent:chapter:0",
        corpus_version_id=version_id,
        section_path=("Chapter", "Section"),
        page_start=1,
        page_end=2,
        source_spans=spans,
        content_hash=sha256(b"one\ntwo").hexdigest(),
        token_count=_token_count("one\ntwo"),
        text="one\ntwo",
    )


def test_firestore_parent_store_round_trips_source_spans_and_nested_manifest_metadata() -> None:
    client = FakeFirestore()
    original = parent_chunk("v2")
    spans = [
        span.model_copy(update={
            "span_id": f"span-{index}",
            "block_id": f"block-{index}",
            "block_order": index,
            "printed_page_label": f"label-{span.page}",
        })
        for index, span in enumerate(original.source_spans)
    ]
    parent_text = "Section: Chapter > Section\n" + "\n".join(span.text for span in spans)
    parent = original.model_copy(update={
        "source_spans": spans,
        "source_text": "\n".join(span.text for span in spans),
        "table_header": "| Name | Value |\n| --- | --- |",
        "text": parent_text,
        "content_hash": sha256(parent_text.encode()).hexdigest(),
        "token_count": _token_count(parent_text),
    })
    metadata = {
        "source_uri": "gs://corpus/source.pdf#7",
        "corpus_manifest": {"derivation": {"version": "hierarchical-corpus-v1"}},
    }

    FirestoreParentChunkStore(client).write_batch([parent], metadata=metadata)

    record = client.data[f"corpus_parent_chunks/{parent.parent_id}"]
    restored = ParentChunk.model_validate(
        {key: value for key, value in record.items() if key != "metadata"}
    )
    assert restored == parent
    assert [span.page for span in restored.source_spans] == [1, 2]
    assert [span.block_id for span in restored.source_spans] == ["block-0", "block-1"]
    assert restored.source_text == "one\ntwo"
    assert restored.table_header == "| Name | Value |\n| --- | --- |"
    assert record["metadata"] == metadata
    assert "embedding" not in record


def test_firestore_parent_store_rejects_batches_over_firestore_limit() -> None:
    with pytest.raises(ValueError, match="at most 500"):
        FirestoreParentChunkStore(FakeFirestore()).write_batch([parent_chunk()] * 501)


def test_hierarchical_validation_reads_back_parents_children_and_gates_activation() -> None:
    client = FakeFirestore()
    candidate_binding = binding("v2")
    candidate = version("v2", "staged", serving_binding=candidate_binding)
    persist(client, candidate)
    text = "child passage"
    span = SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)
    parent = ParentChunk(
        parent_id="v2:parent:1",
        corpus_version_id="v2",
        section_path=("Chapter 1",),
        page_start=1,
        page_end=1,
        source_spans=[span],
        content_hash=sha256(text.encode()).hexdigest(),
        token_count=_token_count(text),
        text=text,
    )
    child = ChildDocumentChunk(
        chunk_id="v2:child:1",
        corpus_version_id="v2",
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=1,
        page_end=1,
        source_spans=[span],
        child_ordinal=0,
        content_hash=parent.content_hash,
        token_count=parent.token_count,
        text=text,
    )
    FirestoreParentChunkStore(client).write_batch([parent])
    metadata = {
        key: value
        for key, value in child.model_dump(mode="json").items()
        if key != "text"
    }
    metadata.update(
        corpus_manifest_sha256=candidate_binding.ingestion_fingerprint,
        derivation_sha256=candidate_binding.configuration_fingerprint,
    )
    client.data[f"{candidate_binding.vector_collection}/{child.chunk_id}"] = {
        "text": child.text,
        "embedding": Vector([0.0] * candidate.embedding_dimensions),
        "metadata": metadata,
    }

    repository = store(client)
    checked = repository.validate_hierarchical("v2", [parent], [child])

    assert checked.status == "staged"
    assert checked.validation_evidence is not None
    assert checked.validation_evidence.index_ready is False
    assert checked.validation_evidence.filter_ready is False
    with pytest.raises(CorpusVersionError, match="only a validated"):
        repository.activate("v2", expected_active_version_id=None)

    complete = ReadinessSignal(
        "complete", datetime.now(timezone.utc), candidate_binding.vector_collection,
        "index-v2", child.chunk_id, "v2",
    )
    repository.persist_search_readiness(
        "v2", SearchReadinessEvidence(complete, None, 0)
    )
    assert repository.get("v2").status == "staged"  # type: ignore[union-attr]
    ready = ReadinessSignal(
        "search_ready", datetime.now(timezone.utc), candidate_binding.vector_collection,
        "index-v2", child.chunk_id, "v2",
    )
    repository.persist_search_readiness(
        "v2", SearchReadinessEvidence(complete, ready, 1)
    )
    assert repository.get("v2").status == "validated"  # type: ignore[union-attr]
    assert client.data["corpus_search_readiness/v2"]["complete"]["observed_at"]
    assert client.data["corpus_search_readiness/v2"]["search_ready"]["index_id"] == "index-v2"


def test_complete_search_signal_is_rejected_before_persisted_coverage_validation() -> None:
    client = FakeFirestore()
    candidate_binding = binding("v2")
    persist(client, version("v2", "staged", serving_binding=candidate_binding))
    complete = ReadinessSignal(
        "complete", datetime.now(timezone.utc), candidate_binding.vector_collection,
        "index-v2", "v2:child:1", "v2",
    )

    with pytest.raises(CorpusVersionError, match="requires persisted corpus completeness validation"):
        store(client).persist_search_readiness(
            "v2", SearchReadinessEvidence(complete, None, 0)
        )

    assert "corpus_search_readiness/v2" not in client.data


def test_hierarchical_validation_rejects_missing_or_tampered_persisted_child() -> None:
    client = FakeFirestore()
    candidate = version("v2", "staged", serving_binding=binding("v2"))
    persist(client, candidate)
    text = "child"
    span = SourceSpan(page=1, start_offset=0, end_offset=len(text), text=text)
    parent = ParentChunk(
        parent_id="v2:parent:1", corpus_version_id="v2", section_path=("Chapter",),
        page_start=1, page_end=1, source_spans=[span],
        content_hash=sha256(text.encode()).hexdigest(), token_count=_token_count(text), text=text,
    )
    child = ChildDocumentChunk(
        chunk_id="v2:child:1", corpus_version_id="v2", parent_id=parent.parent_id,
        section_path=parent.section_path, page_start=1, page_end=1, source_spans=[span],
        child_ordinal=0, content_hash=parent.content_hash, token_count=parent.token_count, text=text,
    )
    FirestoreParentChunkStore(client).write_batch([parent])

    with pytest.raises(ValueError, match="persisted children do not match"):
        store(client).validate_hierarchical("v2", [parent], [child])

    assert store(client).get("v2").status == "staged"  # type: ignore[union-attr]
    assert store(client).active() is None


def test_hierarchical_validation_rejects_provider_dimension_mismatch() -> None:
    client = FakeFirestore()
    candidate = version("v2", "staged", serving_binding=binding("v2"))
    persist(client, candidate)
    parent = parent_chunk("v2")
    child = ChildDocumentChunk(
        chunk_id="v2:child:1",
        corpus_version_id="v2",
        parent_id=parent.parent_id,
        section_path=parent.section_path,
        page_start=parent.page_start,
        page_end=parent.page_end,
        source_spans=parent.source_spans,
        child_ordinal=0,
        content_hash=parent.content_hash,
        token_count=_token_count(parent.text),
        text=parent.text,
    )
    FirestoreParentChunkStore(client).write_batch([parent])
    metadata = {
        key: value
        for key, value in child.model_dump(mode="json").items()
        if key != "text"
    }
    item_binding = candidate.serving_binding
    assert item_binding is not None
    metadata.update(
        corpus_manifest_sha256=item_binding.ingestion_fingerprint,
        derivation_sha256=item_binding.configuration_fingerprint,
    )
    client.data[f"{item_binding.vector_collection}/{child.chunk_id}"] = {
        "text": child.text,
        "embedding": [0.0] * (candidate.embedding_dimensions - 1),
        "metadata": metadata,
    }

    with pytest.raises(ValueError, match="embedding has an invalid provider dimension"):
        store(client).validate_hierarchical("v2", [parent], [child])

    assert store(client).get("v2").status == "staged"  # type: ignore[union-attr]


def test_search_readiness_does_not_change_an_active_version_status() -> None:
    client = FakeFirestore()
    candidate_binding = binding("v2")
    partial = CorpusValidationEvidence(
        chunk_count=1,
        content_hashes_sha256="c" * 64,
        provenance_sha256="d" * 64,
        index_ready=False,
        filter_ready=False,
    )
    active = version(
        "v2", "active", serving_binding=candidate_binding, validation_evidence=partial
    )
    persist(client, active)
    client.data[f"{candidate_binding.vector_collection}/v2:child:1"] = {
        "text": "child",
        "embedding": [0.0] * 768,
        "metadata": {"corpus_version_id": "v2", "chunk_id": "v2:child:1"},
    }
    complete = ReadinessSignal(
        "complete", datetime.now(timezone.utc), candidate_binding.vector_collection,
        "index-v2", "v2:child:1", "v2",
    )
    ready = ReadinessSignal(
        "search_ready", datetime.now(timezone.utc), candidate_binding.vector_collection,
        "index-v2", "v2:child:1", "v2",
    )

    store(client).persist_search_readiness(
        "v2", SearchReadinessEvidence(complete, ready, 1)
    )

    assert store(client).get("v2").status == "active"  # type: ignore[union-attr]


def persist_vector(
    client: FakeFirestore,
    item: TextChunk,
    item_binding: CorpusServingBinding,
) -> None:
    client.data[f"{item_binding.vector_collection}/{item.chunk_id}"] = {
        "text": item.text,
        "embedding": [0.0] * 768,
        "metadata": {
            "corpus_version_id": item.corpus_version_id,
            "chunk_id": item.chunk_id,
            "page": item.page,
            "section": item.section,
            "chunk_ordinal": item.chunk_ordinal,
            "content_hash": item.content_hash,
            "schema_version": item_binding.vector_schema_version,
            "corpus_manifest_sha256": item_binding.ingestion_fingerprint,
            "derivation_sha256": item_binding.configuration_fingerprint,
        },
    }


def test_activation_uses_compare_and_set_and_rollback_restores_previous() -> None:
    client = FakeFirestore()
    persist(client, version("v1", "active"))
    persist(client, version("v2", "validated"))
    repository = store(client)

    activated = repository.activate("v2", expected_active_version_id="v1")
    restored = repository.rollback()

    assert activated.version_id == "v2"
    assert restored.version_id == "v1"
    assert repository.active() == restored
    assert client.data["corpus_versions/v2"]["status"] == "retired"
    assert client.data["corpus_state/active"]["previous_version_id"] == "v2"


def test_stale_activation_cannot_replace_a_newer_active_version() -> None:
    client = FakeFirestore()
    persist(client, version("v1", "active"))
    persist(client, version("v2", "validated"))
    persist(client, version("v3", "validated"))
    repository = store(client)
    repository.activate("v2", expected_active_version_id="v1")

    with pytest.raises(ConcurrentCorpusActivationError, match="active corpus changed"):
        repository.activate("v3", expected_active_version_id="v1")

    assert repository.active().version_id == "v2"  # type: ignore[union-attr]
    assert client.data["corpus_versions/v3"]["status"] == "validated"


def test_activation_persists_complete_binding_and_rollback_restores_it() -> None:
    client = FakeFirestore()
    first = binding("v1", "legacy_chunks")
    second = binding("v2", "document_chunks_v2")
    persist(
        client,
        version("v1", "active", serving_binding=first, validation_evidence=evidence()),
    )
    persist(
        client,
        version("v2", "validated", serving_binding=second, validation_evidence=evidence()),
    )
    repository = store(client)

    repository.activate("v2", expected_active_version_id="v1")
    assert client.data["corpus_state/active"]["binding"] == second.model_dump(mode="json")

    repository.rollback()
    assert client.data["corpus_state/active"]["binding"] == first.model_dump(mode="json")
    assert repository.active().serving_binding == first  # type: ignore[union-attr]


def test_missing_or_tampered_binding_evidence_prevents_activation() -> None:
    client = FakeFirestore()
    candidate = version("v2", "validated", serving_binding=binding("v2"))
    persist(client, candidate)
    repository = store(client)

    with pytest.raises(CorpusVersionError, match="readiness evidence"):
        repository.activate("v2", expected_active_version_id=None)

    candidate = candidate.model_copy(update={"validation_evidence": evidence()})
    persist(client, candidate)
    repository.activate("v2", expected_active_version_id=None)
    client.data["corpus_versions/v2"]["serving_binding"]["vector_collection"] = "tampered"

    with pytest.raises(CorpusVersionError, match="binding is inconsistent"):
        repository.active()


def test_validation_requires_persisted_bound_vector_evidence() -> None:
    client = FakeFirestore()
    candidate_binding = binding("v2")
    candidate = version("v2", "staged", serving_binding=candidate_binding)
    persisted_chunk = chunk("v2")
    persist(client, candidate)
    persist_vector(client, persisted_chunk, candidate_binding)

    validated = store(client).validate("v2", [persisted_chunk])

    assert validated.validation_evidence is not None
    assert validated.validation_evidence.chunk_count == 1
    assert validated.validation_evidence.filter_ready is True


def test_cleanup_is_dry_run_first_and_removes_only_the_exact_abandoned_version() -> None:
    client = FakeFirestore()
    active = version("active", "active", serving_binding=binding("active"))
    rollback = version("rollback", "retired", serving_binding=binding("rollback"))
    abandoned_binding = binding("abandoned")
    abandoned = version("abandoned", "staged", serving_binding=abandoned_binding)
    persist(client, active)
    persist(client, rollback)
    persist(client, abandoned)
    abandoned_chunk = chunk("abandoned")
    persist_vector(client, abandoned_chunk, abandoned_binding)
    client.data["corpus_state/active"] = {
        "version_id": "active",
        "previous_version_id": "rollback",
    }
    repository = store(client)

    preview = repository.cleanup_staging("abandoned")

    assert preview.dry_run is True
    assert "corpus_versions/abandoned" in preview.targets
    assert "document_chunks_v2/abandoned:1:0" in preview.targets
    assert "corpus_versions/abandoned" in client.data
    assert "corpus_versions/active" in client.data
    assert "corpus_versions/rollback" in client.data

    removed = repository.cleanup_staging("abandoned", dry_run=False)

    assert removed.dry_run is False
    assert "corpus_versions/abandoned" not in client.data
    assert "document_chunks_v2/abandoned:1:0" not in client.data
    assert "corpus_versions/active" in client.data
    assert "corpus_versions/rollback" in client.data


def test_cleanup_rejects_running_active_and_rollback_retained_versions() -> None:
    client = FakeFirestore()
    active = version("active", "active", serving_binding=binding("active"))
    rollback = version("rollback", "retired", serving_binding=binding("rollback"))
    running = version("running", "staged", serving_binding=binding("running"))
    persist(client, active)
    persist(client, rollback)
    persist(client, running)
    client.data["corpus_state/active"] = {
        "version_id": "active",
        "previous_version_id": "rollback",
    }
    repository = store(client)
    repository.begin_staging("running")

    with pytest.raises(CorpusVersionError, match="running"):
        repository.cleanup_staging("running")
    with pytest.raises(CorpusVersionError):
        repository.cleanup_staging("active")
    with pytest.raises(CorpusVersionError):
        repository.cleanup_staging("rollback")
