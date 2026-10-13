from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from app.contracts import CorpusVersion
from app.corpus.firestore import FirestoreCorpusVersionStore
from app.corpus.versions import ConcurrentCorpusActivationError


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
            if snapshot.to_dict().get(self.field) == self.value
        ]
        return snapshots[: self.maximum]


@dataclass
class FakeCollection:
    client: "FakeFirestore"
    path: str

    def document(self, document_id: str) -> FakeDocument:
        return FakeDocument(self.client, f"{self.path}/{document_id}")

    def where(self, field: str, operator: str, value: object) -> FakeQuery:
        assert operator == "=="
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


def run_transaction(transaction: FakeTransaction, callback):  # type: ignore[no-untyped-def]
    result = callback(transaction)
    transaction.client.data = transaction.data
    return result


def version(version_id: str, status: str) -> CorpusVersion:
    return CorpusVersion(
        version_id=version_id,
        source_uri=f"gs://corpus/source.pdf#{version_id}",
        embedding_model="gemini-embedding-001",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def store(client: FakeFirestore) -> FirestoreCorpusVersionStore:
    return FirestoreCorpusVersionStore(client, transaction_runner=run_transaction)


def persist(client: FakeFirestore, item: CorpusVersion) -> None:
    client.data[f"corpus_versions/{item.version_id}"] = item.model_dump(mode="json")


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
