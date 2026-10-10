from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from google.api_core.exceptions import AlreadyExists

from app.contracts import EvaluationDataset, EvaluationItem, EvaluationRun, GoldAnnotation
from app.evaluation.generations import (
    ConcurrentActivationError,
    FirestoreEvaluationGenerationStore,
    GenerationIntegrityError,
)
from app.evaluation.job import EvaluationRunError, FirestoreEvaluationRunStore


@dataclass
class FakeSnapshot:
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
        return FakeSnapshot(source.get(self.path))

    def set(self, data: dict[str, object]) -> None:
        self.client.data[self.path] = deepcopy(data)

    def create(self, data: dict[str, object]) -> None:
        if self.path in self.client.data:
            raise AlreadyExists("already exists")
        self.set(data)

    def update(self, data: dict[str, object]) -> None:
        self.client.data[self.path] = {**self.client.data[self.path], **deepcopy(data)}

    def collection(self, collection_id: str) -> "FakeCollection":
        return FakeCollection(self.client, f"{self.path}/{collection_id}")


@dataclass
class FakeCollection:
    client: "FakeFirestore"
    path: str

    def document(self, document_id: str) -> FakeDocument:
        return FakeDocument(self.client, f"{self.path}/{document_id}")

    def stream(self) -> list[FakeSnapshot]:
        depth = self.path.count("/") + 1
        return [
            FakeSnapshot(data)
            for path, data in self.client.data.items()
            if path.startswith(f"{self.path}/") and path.count("/") == depth
        ]


@dataclass
class FakeBatch:
    client: "FakeFirestore"
    writes: list[tuple[FakeDocument, dict[str, object]]] = field(default_factory=list)

    def set(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.writes.append((reference, deepcopy(data)))

    def commit(self) -> None:
        if self.client.fail_next_batch:
            self.client.fail_next_batch = False
            raise RuntimeError("simulated staging failure")
        for reference, data in self.writes:
            reference.set(data)


@dataclass
class FakeTransaction:
    client: "FakeFirestore"
    data: dict[str, dict[str, object]]

    def update(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.data[reference.path] = {**self.data.get(reference.path, {}), **deepcopy(data)}

    def set(self, reference: FakeDocument, data: dict[str, object]) -> None:
        self.data[reference.path] = deepcopy(data)


@dataclass
class FakeFirestore:
    data: dict[str, dict[str, object]] = field(default_factory=dict)
    fail_next_batch: bool = False

    def collection(self, collection_id: str) -> FakeCollection:
        return FakeCollection(self, collection_id)

    def batch(self) -> FakeBatch:
        return FakeBatch(self)

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self, deepcopy(self.data))


def run_fake_transaction(transaction: FakeTransaction, callback):  # type: ignore[no-untyped-def]
    result = callback(transaction)
    transaction.client.data = transaction.data
    return result


def dataset(generation_id: str, *, approved: bool = False) -> EvaluationDataset:
    return EvaluationDataset(
        dataset_version_id=generation_id,
        items=[
            EvaluationItem(
                item_id="item-1", question="What is metadata?", dataset_version_id=generation_id, category="definitions"
            )
        ],
        gold_annotations=[
            GoldAnnotation(
                item_id="item-1",
                expected_answer_or_rubric="A definition.",
                relevant_chunk_ids=["chunk-1"],
                review_status="approved" if approved else "candidate",
            )
        ],
    )


def store(client: FakeFirestore) -> FirestoreEvaluationGenerationStore:
    return FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)


def test_staging_writes_immutable_items_and_annotations_before_activation() -> None:
    client = FakeFirestore()
    repository = store(client)

    staged = repository.stage(dataset("generation-1", approved=True))

    assert staged == "generation-1"
    assert repository.active() is None
    assert client.data["evaluation_generations/generation-1/items/item-1"]["dataset_version_id"] == "generation-1"
    assert client.data["evaluation_generations/generation-1/annotations/item-1"]["review_status"] == "approved"


def test_failed_staging_never_replaces_the_active_generation() -> None:
    client = FakeFirestore()
    repository = store(client)
    repository.stage(dataset("generation-1"))
    repository.activate("generation-1", expected_active_generation_id=None)
    client.fail_next_batch = True

    with pytest.raises(RuntimeError, match="staging failure"):
        repository.stage(dataset("generation-2"))

    assert repository.active().generation_id == "generation-1"  # type: ignore[union-attr]
    assert repository.generation_status("generation-2") == "staging"


def test_activation_compare_and_set_allows_only_one_concurrent_winner() -> None:
    client = FakeFirestore()
    repository = store(client)
    repository.stage(dataset("generation-1"))
    repository.activate("generation-1", expected_active_generation_id=None)
    repository.stage(dataset("generation-2"))
    repository.stage(dataset("generation-3"))

    repository.activate("generation-2", expected_active_generation_id="generation-1")
    with pytest.raises(ConcurrentActivationError, match="active generation changed"):
        repository.activate("generation-3", expected_active_generation_id="generation-1")

    assert repository.active().generation_id == "generation-2"  # type: ignore[union-attr]
    assert repository.generation_status("generation-1") == "superseded"


def test_active_generation_survives_a_new_repository_instance() -> None:
    client = FakeFirestore()
    first = store(client)
    first.stage(dataset("generation-1"))
    first.activate("generation-1", expected_active_generation_id=None)

    restarted = store(client)

    assert restarted.active().generation_id == "generation-1"  # type: ignore[union-attr]


def test_active_dataset_load_and_eligibility_use_persisted_generation_state() -> None:
    client = FakeFirestore()
    repository = store(client)
    repository.stage(dataset("generation-1", approved=True))
    repository.activate("generation-1", expected_active_generation_id=None)

    active_dataset = repository.active_dataset()
    eligibility = repository.report_eligibility("generation-1")

    assert active_dataset is not None
    assert [item.item_id for item in active_dataset.items] == ["item-1"]
    assert eligibility.eligibility == "exploratory"
    assert eligibility.is_current_generation is True


def test_generation_ids_are_immutable_and_retrying_identical_stage_is_safe() -> None:
    client = FakeFirestore()
    repository = store(client)
    original = dataset("generation-1")
    assert repository.stage(original) == "generation-1"
    assert repository.stage(original) == "generation-1"

    changed = original.model_copy(update={"items": [original.items[0].model_copy(update={"question": "Changed"})]})
    with pytest.raises(GenerationIntegrityError, match="already bound"):
        repository.stage(changed)


def test_run_records_survive_a_new_firestore_store_and_reject_hash_collisions() -> None:
    client = FakeFirestore()
    run = EvaluationRun(
        run_id="run-1",
        dataset_version_id="generation-1",
        corpus_version_id="corpus-1",
        status="queued",
        selected_item_ids=["item-1"],
        configuration_version_id="config-1",
        model_version_id="model-1",
    )

    first = FirestoreEvaluationRunStore(client)
    assert first.create_or_get(run) == run
    restarted = FirestoreEvaluationRunStore(client)
    assert restarted.get("run-1") == run
    assert restarted.create_or_get(run) == run

    with pytest.raises(EvaluationRunError, match="already bound"):
        restarted.create_or_get(run.model_copy(update={"model_version_id": "model-2"}))
