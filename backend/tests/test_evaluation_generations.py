from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pytest
from google.api_core.exceptions import AlreadyExists

from app.contracts import DocumentChunk, EvaluationDataset, EvaluationItem, EvaluationRun, GoldAnnotation
from app.evaluation.generations import (
    ConcurrentActivationError,
    FirestoreEvaluationGenerationStore,
    GenerationIntegrityError,
)
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
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


def test_review_is_atomic_current_generation_only_and_updates_gold_count() -> None:
    client = FakeFirestore()
    repository = store(client)
    repository.stage(dataset("generation-1"))
    repository.activate("generation-1", expected_active_generation_id=None)
    repository.stage(dataset("generation-2"))

    approved = repository.review("generation-1", "item-1", "approved")

    assert approved.review_status == "approved"
    assert repository.active().approved_gold_count == 1  # type: ignore[union-attr]
    with pytest.raises(Exception, match="already final"):
        repository.review("generation-1", "item-1", "rejected")
    with pytest.raises(Exception, match="active evaluation generation"):
        repository.review("generation-2", "item-1", "approved")


def test_generation_ids_are_immutable_and_retrying_identical_stage_is_safe() -> None:
    client = FakeFirestore()
    repository = store(client)
    original = dataset("generation-1")
    assert repository.stage(original) == "generation-1"
    assert repository.stage(original) == "generation-1"

    changed = original.model_copy(update={"items": [original.items[0].model_copy(update={"question": "Changed"})]})
    with pytest.raises(GenerationIntegrityError, match="already bound"):
        repository.stage(changed)


def _candidate_review(dataset: EvaluationDataset, *, support: str = "Candidate support"):
    candidate = DocumentChunk(
        chunk_id="candidate-chunk",
        corpus_version_id="candidate-corpus",
        page=1,
        section="Candidate",
        chunk_ordinal=0,
        content_hash="a" * 64,
        text=support,
        embedding=[0.0] * 768,
    )
    return remap_gold_annotations(
        dataset,
        [candidate],
        CandidateGoldBinding(
            corpus_version_id="candidate-corpus",
            candidate_snapshot_sha256="b" * 64,
            source_provenance_sha256="c" * 64,
        ),
    )


def test_candidate_review_is_persisted_with_its_binding_and_requires_matching_support() -> None:
    client = FakeFirestore()
    repository = store(client)
    candidate_dataset = dataset("generation-1").model_copy(
        update={
            "gold_annotations": [
                GoldAnnotation(
                    item_id="item-1",
                    expected_answer_or_rubric="A definition.",
                    relevant_chunk_ids=["legacy-chunk"],
                    review_status="candidate",
                    supporting_passage="Candidate support",
                )
            ]
        }
    )
    review = _candidate_review(candidate_dataset)

    repository.stage(candidate_dataset, candidate_review=review)
    generation = client.data["evaluation_generations/generation-1"]
    annotation = client.data["evaluation_generations/generation-1/annotations/item-1"]
    assert generation["candidate_review"]["binding"]["corpus_version_id"] == "candidate-corpus"  # type: ignore[index]
    assert annotation["candidate_review"]["review_status"] == "candidate"  # type: ignore[index]

    repository.activate("generation-1", expected_active_generation_id=None)
    assert repository.review("generation-1", "item-1", "approved").review_status == "approved"


def test_candidate_review_clears_legacy_approval_and_unresolved_support() -> None:
    client = FakeFirestore()
    repository = store(client)
    approved = dataset("generation-approved", approved=True)
    approved = approved.model_copy(
        update={
            "gold_annotations": [
                approved.gold_annotations[0].model_copy(
                    update={
                        "semantic_assessment": {
                            "response_sha256": "d" * 64,
                            "scorer_version": "v1",
                            "claim_support": True,
                            "claim_citation_support": True,
                            "rubric_passed": True,
                        }
                    }
                )
            ]
        }
    )
    repository.stage(approved, candidate_review=_candidate_review(approved))
    staged_annotation = client.data["evaluation_generations/generation-approved/annotations/item-1"]
    assert staged_annotation["review_status"] == "candidate"
    assert staged_annotation["relevant_chunk_ids"] == []
    assert staged_annotation["semantic_assessment"] is None

    unresolved = dataset("generation-unresolved").model_copy(
        update={
            "gold_annotations": [
                GoldAnnotation(
                    item_id="item-1",
                    expected_answer_or_rubric="A definition.",
                    relevant_chunk_ids=["legacy-chunk"],
                    review_status="candidate",
                    supporting_passage="No candidate support",
                )
            ]
        }
    )
    repository.stage(unresolved, candidate_review=_candidate_review(unresolved))
    repository.activate("generation-unresolved", expected_active_generation_id=None)
    with pytest.raises(Exception, match="unresolved candidate support"):
        repository.review("generation-unresolved", "item-1", "approved")


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


def test_v2_revision_is_corpus_scoped_and_survives_store_restart() -> None:
    from app.contracts import EvaluationRowV2

    client = FakeFirestore()
    repository = store(client)
    row = EvaluationRowV2(item_id="one", origin="generated", question="Out of scope?", category="study",
                          expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                          expected_refusal=True, scope_rationale="Outside corpus")
    head = repository.stage_revision("candidate", 1, (row,))
    assert repository.corpus_head("candidate") is None
    repository.activate_revision(head, expected_revision=None)
    assert store(client).corpus_head("candidate") == head
    assert store(client).get_revision(head) == (row,)
    assert repository.corpus_head("serving") is None
    assert repository.active() is None


def test_v2_revision_failed_or_conflicting_publication_preserves_head() -> None:
    from app.contracts import EvaluationRowV2
    from app.evaluation.generations import EvaluationGenerationError

    client = FakeFirestore()
    repository = store(client)
    row = EvaluationRowV2(item_id="one", origin="generated", question="Out of scope?", category="study",
                          expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                          expected_refusal=True, scope_rationale="Outside corpus")
    head = repository.stage_revision("corpus", 1, (row,))
    repository.activate_revision(head, expected_revision=None)
    with pytest.raises(EvaluationGenerationError):
        repository.stage_revision("corpus", 2, ())
    client.fail_next_batch = True
    with pytest.raises(RuntimeError):
        repository.stage_revision("corpus", 2, (row,))
    assert repository.corpus_head("corpus") == head
    second = repository.stage_revision("corpus", 2, (row,))
    with pytest.raises(ConcurrentActivationError):
        repository.activate_revision(second, expected_revision=None)
    assert repository.corpus_head("corpus") == head
    repository.activate_revision(second, expected_revision=1)
    assert repository.get_revision(head) == (row,)


def test_v2_run_snapshot_survives_revision_change_and_restart() -> None:
    from app.contracts import CandidateDatasetBinding, CorpusServingBinding, EvaluationCandidateSnapshot, EvaluationRowV2, ServingDatasetBinding

    client = FakeFirestore()
    repository = store(client)
    row = EvaluationRowV2(item_id="one", origin="generated", question="Out of scope?", category="study",
                          expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                          expected_refusal=True, scope_rationale="Outside corpus")
    first = repository.stage_revision("corpus", 1, (row,))
    repository.activate_revision(first, expected_revision=None)
    binding = CandidateDatasetBinding(attempt_id="attempt", corpus_version_id="corpus", dataset_revision=1)
    configuration = EvaluationCandidateSnapshot(
        corpus_binding=CorpusServingBinding(corpus_version_id="corpus", vector_collection="v", vector_schema_version="2",
                                            embedding_model="model", ingestion_fingerprint="a" * 64,
                                            configuration_fingerprint="b" * 64),
        project="project", location="region", answer_provider="openrouter", primary_model="model",
        json_schema_supported=True, request_timeout_ms=1000, generation_timeout_ms=500,
        prompt_version="prompt", retrieval_policy_version="retrieval", evidence_policy_version="evidence",
        scorer_version="scorer", dependency_lock_sha256="c" * 64, integration_artifact="artifact", runtime_version="runtime",
    )
    snapshot = repository.create_run_snapshot(run_id="run", binding=binding, configuration_version_id="config",
                                               configuration=configuration, item_ids=("one",))
    second = repository.stage_revision("corpus", 2, (row.model_copy(update={"question": "Changed?"}),))
    with pytest.raises(ValueError, match="published corpus revision"):
        repository.create_run_snapshot(
            run_id="unpublished", binding=ServingDatasetBinding(release_id="release", corpus_version_id="corpus", dataset_revision=2),
            configuration_version_id="config", configuration=configuration,
        )
    repository.activate_revision(second, expected_revision=1)
    restored, restored_configuration = store(client).get_run_snapshot("run")
    assert restored == snapshot
    assert restored.items[0].question == "Out of scope?"
    assert restored.selection == "subset"
    assert restored_configuration == configuration
    serving = repository.create_run_snapshot(
        run_id="serving", binding=ServingDatasetBinding(release_id="release", corpus_version_id="corpus", dataset_revision=1),
        configuration_version_id="config", configuration=configuration,
    )
    assert serving.items[0].question == "Out of scope?"
    assert store(client).create_run_snapshot(run_id="run", binding=binding, configuration_version_id="config",
                                            configuration=configuration, item_ids=("one",)) == snapshot
    with pytest.raises(GenerationIntegrityError):
        store(client).create_run_snapshot(run_id="run", binding=binding, configuration_version_id="different",
                                          configuration=configuration, item_ids=("one",))


def test_v2_revision_rejects_invalid_shape_and_corrupted_staged_items() -> None:
    from app.contracts import EvaluationRowV2
    from pydantic import ValidationError

    client = FakeFirestore()
    repository = store(client)
    row = EvaluationRowV2(item_id="one", origin="generated", question="Outside?", category="study",
                          expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                          expected_refusal=True, scope_rationale="Outside corpus")
    with pytest.raises(ValidationError):
        repository.stage_revision("corpus", 1, (row.model_copy(update={"relevant_child_ids": ("fake",)}),))
    head = repository.stage_revision("corpus", 1, (row,))
    path = next(path for path in client.data if path.startswith("evaluation_dataset_revisions_v2/") and "/items/" in path)
    client.data[path]["row"]["question"] = "Altered"
    with pytest.raises(GenerationIntegrityError):
        repository.activate_revision(head, expected_revision=None)
    assert repository.corpus_head("corpus") is None


def test_v2_revision_never_admits_legacy_placeholder_bindings() -> None:
    from app.contracts import EvaluationRowV2, RawSourceSpan
    from app.evaluation.generations import EvaluationGenerationError

    span = RawSourceSpan(span_id="s", block_id="b", physical_page=1, start_offset=0, end_offset=4, raw_text="text")
    row = EvaluationRowV2(item_id="one", origin="imported", question="What?", category="definitions",
                          expected_outcome="answer", expected_answer_or_rubric="Text", expected_refusal=False,
                          source_spans=(span,), relevant_child_ids=("imported-passage-one",), parent_ids=("parent",))
    with pytest.raises(EvaluationGenerationError, match="placeholder"):
        store(FakeFirestore()).stage_revision("corpus", 1, (row,))
