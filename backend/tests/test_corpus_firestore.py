from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256

import pytest

from app.contracts import CorpusServingBinding, CorpusValidationEvidence, CorpusVersion
from app.corpus.chunking import TextChunk
from app.corpus.firestore import FirestoreCorpusVersionStore
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

    def delete(self, reference: FakeDocument) -> None:
        self.deletions.append(reference)

    def commit(self) -> None:
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
