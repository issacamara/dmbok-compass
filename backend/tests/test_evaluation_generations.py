from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
from google.api_core.exceptions import AlreadyExists

from app.contracts import (
    CorpusServingBinding, CorpusVersion, DocumentChunk, EvaluationDataset,
    EvaluationItem, EvaluationRun, GoldAnnotation,
)
from app.corpus.versions import hierarchical_validation_evidence
from app.evaluation.generations import (
    ConcurrentActivationError,
    FirestoreEvaluationGenerationStore,
    GenerationIntegrityError,
)
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
from app.evaluation.job import EvaluationRunError, FirestoreEvaluationRunStore
from test_candidate_generation import source_records


@dataclass
class FakeSnapshot:
    data: dict[str, object] | None
    id: str | None = None

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
    filters: tuple[tuple[str, object], ...] = ()

    def document(self, document_id: str) -> FakeDocument:
        return FakeDocument(self.client, f"{self.path}/{document_id}")

    def stream(self) -> list[FakeSnapshot]:
        depth = self.path.count("/") + 1
        rows = []
        for path, data in self.client.data.items():
            if not (path.startswith(f"{self.path}/") and path.count("/") == depth):
                continue
            if all(_field_value(data, field_path) == expected for field_path, expected in self.filters):
                rows.append(FakeSnapshot(data, id=path.rsplit("/", 1)[-1]))
        return rows

    def where(self, *, filter):  # type: ignore[no-untyped-def]
        return FakeCollection(
            self.client,
            self.path,
            (*self.filters, (filter.field_path, filter.value)),
        )


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


def _field_value(data: dict[str, object], field_path: str) -> object:
    value: object = data
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def test_parse_derived_record_lookup_rechecks_corpus_binding_and_isolation() -> None:
    children, parents = source_records()
    corpus_id = parents[0].corpus_version_id
    binding = CorpusServingBinding(
        corpus_version_id=corpus_id,
        vector_collection="document_chunks_v2",
        vector_schema_version="2",
        embedding_model="embedding-model",
        embedding_dimensions=768,
        ingestion_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
    )
    staged = CorpusVersion(
        version_id=corpus_id,
        source_uri="gs://corpus/source.pdf",
        embedding_model="embedding-model",
        created_at=datetime.now(timezone.utc),
        status="validated",
        serving_binding=binding,
    )
    ready = hierarchical_validation_evidence(staged, parents, children, index_ready=True, filter_ready=True)
    version = staged.model_copy(update={"validation_evidence": ready})
    firestore = FakeFirestore()
    firestore.data[f"corpus_versions/{corpus_id}"] = version.model_dump(mode="json")
    for parent in parents:
        firestore.data[f"corpus_parent_chunks/{parent.parent_id}"] = parent.model_dump(mode="json")
    for child in children:
        firestore.data[f"document_chunks_v2/{child.chunk_id}"] = {
            "text": child.text,
            "embedding": [0.0] * 768,
            "metadata": {
                **child.model_dump(mode="json", exclude={"text"}),
                "corpus_manifest_sha256": binding.ingestion_fingerprint,
                "derivation_sha256": binding.configuration_fingerprint,
            },
        }
    foreign = children[0].model_copy(update={"chunk_id": "foreign", "corpus_version_id": "other-corpus"})
    firestore.data["document_chunks_v2/foreign"] = {
        "text": foreign.text,
        "embedding": [0.0] * 768,
        "metadata": {
            **foreign.model_dump(mode="json", exclude={"text"}),
            "corpus_manifest_sha256": binding.ingestion_fingerprint,
            "derivation_sha256": binding.configuration_fingerprint,
        },
    }

    parents_found, children_found = FirestoreEvaluationGenerationStore(
        firestore, transaction_runner=run_fake_transaction
    ).get_parse_derived_records(corpus_version_id=corpus_id)

    assert parents_found == parents
    assert children_found == children
    assert all(child.corpus_version_id == corpus_id for child in children_found)


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
    changed = row.model_copy(update={"question": "Different retry?"})
    second = repository.stage_revision("corpus", 2, (changed,))
    with pytest.raises(ConcurrentActivationError):
        repository.activate_revision(second, expected_revision=None)
    assert repository.corpus_head("corpus") == head
    repository.activate_revision(second, expected_revision=1)
    assert repository.get_revision(head) == (row,)
    assert repository.get_revision(second) == (changed,)


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
    with pytest.raises(ValueError, match="no serving release is selected"):
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
    with pytest.raises(ValueError, match="no serving release is selected"):
        repository.create_run_snapshot(
            run_id="serving", binding=ServingDatasetBinding(release_id="release", corpus_version_id="corpus", dataset_revision=1),
            configuration_version_id="config", configuration=configuration,
        )
    assert store(client).create_run_snapshot(run_id="run", binding=binding, configuration_version_id="config",
                                            configuration=configuration, item_ids=("one",)) == snapshot
    with pytest.raises(GenerationIntegrityError):
        store(client).create_run_snapshot(run_id="run", binding=binding, configuration_version_id="different",
                                          configuration=configuration, item_ids=("one",))


def test_candidate_operation_replay_returns_same_head_and_rejects_other_corpus() -> None:
    from app.contracts import EvaluationRowV2
    from app.evaluation.generations import EvaluationGenerationError

    repository = store(FakeFirestore())
    row = EvaluationRowV2(
        item_id="one", origin="generated", question="Out of scope?", category="study",
        expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
        expected_refusal=True, scope_rationale="Outside the corpus.",
    )
    head = repository.stage_revision("corpus-v2", 1, (row,))

    published = repository.activate_revision(head, expected_revision=None, operation_key="corpus-v2:generator-v1")
    replay = repository.activate_revision(head, expected_revision=None, operation_key="corpus-v2:generator-v1")

    assert replay == published == head
    assert repository.operation_head("corpus-v2:generator-v1", corpus_version_id="corpus-v2") == head
    with pytest.raises(EvaluationGenerationError, match="another corpus"):
        repository.operation_head("corpus-v2:generator-v1", corpus_version_id="corpus-other")


def test_candidate_operation_lease_serializes_generation_and_allows_expired_recovery() -> None:
    from datetime import timedelta

    repository = store(FakeFirestore())
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    arguments = {"operation_key": "corpus-v2:generator-v1", "corpus_version_id": "corpus-v2"}

    assert repository.claim_operation(
        **arguments, owner_id="worker-one", lease_seconds=60, now=now
    ) is True
    assert repository.claim_operation(
        **arguments, owner_id="worker-two", lease_seconds=60, now=now + timedelta(seconds=30)
    ) is False
    assert repository.claim_operation(
        **arguments, owner_id="worker-two", lease_seconds=60, now=now + timedelta(seconds=61)
    ) is True
    repository.release_operation(arguments["operation_key"], owner_id="worker-one", status="failed")
    assert repository.claim_operation(
        **arguments, owner_id="worker-three", lease_seconds=60, now=now + timedelta(seconds=62)
    ) is False
    repository.release_operation(arguments["operation_key"], owner_id="worker-two", status="awaiting_budget")
    assert repository.claim_operation(
        **arguments, owner_id="worker-three", lease_seconds=60, now=now + timedelta(seconds=63)
    ) is True


def test_candidate_revision_publication_requires_the_current_lease_owner() -> None:
    from datetime import timedelta

    from app.contracts import EvaluationRowV2
    from app.evaluation.generations import ConcurrentActivationError

    repository = store(FakeFirestore())
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    arguments = {"operation_key": "corpus-v2:generator-v1", "corpus_version_id": "corpus-v2"}
    row = EvaluationRowV2(
        item_id="one", origin="generated", question="Out of scope?", category="study",
        expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
        expected_refusal=True, scope_rationale="Outside the corpus.",
    )
    head = repository.stage_revision("corpus-v2", 1, (row,))

    assert repository.claim_operation(
        **arguments, owner_id="stale-worker", lease_seconds=10, now=now
    ) is True
    assert repository.claim_operation(
        **arguments, owner_id="current-worker", lease_seconds=10, now=now + timedelta(seconds=11)
    ) is True
    with pytest.raises(ConcurrentActivationError, match="no longer owns the publication lease"):
        repository.activate_revision(
            head, expected_revision=None, operation_key=arguments["operation_key"], owner_id="stale-worker"
        )

    assert repository.activate_revision(
        head, expected_revision=None, operation_key=arguments["operation_key"], owner_id="current-worker"
    ) == head


def test_candidate_exploration_publishes_inactive_dataset_and_dispatches_one_full_run() -> None:
    from hashlib import sha256
    from app.contracts import EvaluationCandidateSnapshot, EvaluationRowV2, ServingDatasetBinding
    from app.evaluation.automatic import CandidateExplorationPublisher
    from app.evaluation.job import EvaluationJobService, EvaluationRunError, FirestoreEvaluationRunStore, InMemoryEvaluationJobDispatcher
    from app.evaluation.worker import FirestoreEvaluationCandidateSnapshotResolver
    from app.release.firestore import ReleaseManifest

    client = FakeFirestore()
    repository = store(client)
    serving = EvaluationDataset(
        dataset_version_id="serving-dataset",
        items=[EvaluationItem(item_id="serving-item", question="Serving?", dataset_version_id="serving-dataset")],
    )
    generation_id = repository.stage(serving)
    repository.activate(generation_id, expected_active_generation_id=None)

    row = EvaluationRowV2(
        item_id="candidate-item", origin="generated", question="Outside the DMBOK scope?",
        category="study", expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
        expected_refusal=True, scope_rationale="This subject is outside the corpus.",
    )
    head = repository.stage_revision("corpus-v2", 1, (row,))
    repository.activate_revision(head, expected_revision=None)

    class Resolver:
        def resolve(self, *, corpus_version_id, configuration_version_id, model_version_id):
            return EvaluationCandidateSnapshot(
                corpus_binding=CorpusServingBinding(
                    corpus_version_id=corpus_version_id, vector_collection="vectors",
                    vector_schema_version="2", embedding_model="embedding",
                    ingestion_fingerprint="a" * 64, configuration_fingerprint="b" * 64,
                ),
                project="project", location="region", answer_provider="openrouter", primary_model="model",
                json_schema_supported=True, request_timeout_ms=1000, generation_timeout_ms=500,
                prompt_version="prompt", retrieval_policy_version="retrieval", evidence_policy_version="evidence",
                scorer_version="scorer", dependency_lock_sha256="c" * 64,
                integration_artifact="artifact", runtime_version="runtime",
            )

    dispatcher = InMemoryEvaluationJobDispatcher()
    run_store = FirestoreEvaluationRunStore(client)
    jobs = EvaluationJobService(repository, run_store, dispatcher, Resolver())
    publisher = CandidateExplorationPublisher(repository, jobs)

    snapshot = publisher.launch(
        attempt_id="attempt-v2", head=head, configuration_version_id="config-v2", model_version_id="model-v2",
    )
    candidate_subset = publisher.launch(
        attempt_id="attempt-v2", head=head, configuration_version_id="config-v2", model_version_id="model-v2",
        item_ids=["candidate-item"],
    )
    replay = publisher.launch(
        attempt_id="attempt-v2", head=head, configuration_version_id="config-v2", model_version_id="model-v2",
    )

    run = jobs.status(snapshot.run_id)
    assert snapshot == replay
    assert snapshot.binding.attempt_id == "attempt-v2"
    assert snapshot.binding.corpus_version_id == "corpus-v2"
    assert snapshot.selection == "full"
    candidate_subset_record, _ = repository.get_run_snapshot(candidate_subset.run_id)
    assert candidate_subset.run_id != snapshot.run_id
    assert candidate_subset_record.selection == "subset"
    assert run.dataset_version_id.startswith("candidate-")
    assert run.selected_item_ids == ["candidate-item"]
    assert run.report_eligibility is not None
    assert run.report_eligibility.eligibility == "exploratory"
    assert repository.active().generation_id == "serving-dataset"
    assert set(dispatcher.dispatched) == {snapshot.run_id, candidate_subset.run_id}

    completed = run_store.update(run.model_copy(update={"status": "completed"}))
    candidate_digest = sha256(completed.candidate_snapshot.model_dump_json().encode("utf-8")).hexdigest()
    manifest = ReleaseManifest(
        release_id="release-v2",
        evaluation_run_id=completed.run_id,
        dataset_version_id=completed.dataset_version_id,
        corpus_version_id=completed.corpus_version_id,
        configuration_version_id=completed.configuration_version_id,
        provider_version_id="openrouter-v1",
        model_version_id=completed.model_version_id,
        scorer_version_id=completed.candidate_snapshot.scorer_version,
        candidate_digest=candidate_digest,
        gate_report_ids=["gates-v1"],
    )
    client.collection("release_manifests").document(manifest.release_id).set(manifest.model_dump(mode="json"))
    client.collection("release_decisions").document(manifest.release_id).set({
        "decision": {
            **manifest.model_dump(exclude={"provider_version_id", "scorer_version_id", "candidate_digest", "gate_report_ids"}),
            "decision": "approved",
            "provider_version_id": manifest.provider_version_id,
            "scorer_version_id": manifest.scorer_version_id,
            "gate_report_ids": manifest.gate_report_ids,
            "rationale": "Approved for serving.",
            "exception_approved": False,
        },
        "candidate_digest": candidate_digest,
        "actor_user_id": "admin",
        "decided_at": "2026-11-11T12:00:00+00:00",
    })
    client.collection("release_state").document("serving").set({
        "release_id": manifest.release_id,
        "artifact_id": candidate_digest,
        "configuration_version_id": manifest.configuration_version_id,
        "corpus_version_id": manifest.corpus_version_id,
        "vector_collection": completed.candidate_snapshot.corpus_binding.vector_collection,
        "vector_schema_version": completed.candidate_snapshot.corpus_binding.vector_schema_version,
        "source_uri": "gs://corpus/dmbok.pdf",
        "published_at": "2026-11-11T12:00:00+00:00",
        "previous": None,
    })
    # A newer approval that is not yet published must not replace the serving selector.
    client.collection("release_state").document("approved").set({
        "release_id": "future-approved-release", "candidate_digest": "f" * 64,
    })
    serving_jobs = EvaluationJobService(
        repository, run_store, dispatcher, FirestoreEvaluationCandidateSnapshotResolver(client)
    )
    served_subset = serving_jobs.launch(item_ids=["candidate-item"])
    served_replay = serving_jobs.launch(item_ids=["candidate-item"])
    served_snapshot, _ = repository.get_run_snapshot(served_subset.run_id)
    assert served_subset == served_replay
    assert served_snapshot.binding.release_id == manifest.release_id
    assert served_snapshot.binding.corpus_version_id == manifest.corpus_version_id
    assert served_snapshot.selection == "subset"
    served_full = serving_jobs.launch(item_ids=None)
    full_snapshot, _ = repository.get_run_snapshot(served_full.run_id)
    assert served_full.run_id != served_subset.run_id
    assert full_snapshot.selection == "full"
    assert set(dispatcher.dispatched) == {
        snapshot.run_id, candidate_subset.run_id, served_subset.run_id, served_full.run_id,
    }
    with pytest.raises(EvaluationRunError, match="unknown evaluation item IDs"):
        serving_jobs.launch(item_ids=["not-in-approved-dataset"])
    with pytest.raises(EvaluationRunError, match="resolved by the server"):
        serving_jobs.launch(
            corpus_version_id="caller-corpus",
            configuration_version_id="caller-config",
            model_version_id="caller-model",
            item_ids=None,
        )
    with pytest.raises(GenerationIntegrityError, match="does not match the selected release"):
        repository.create_run_snapshot(
            run_id="mismatched-serving-binding",
            binding=ServingDatasetBinding(
                release_id="other-release", corpus_version_id=manifest.corpus_version_id, dataset_revision=head.revision
            ),
            configuration_version_id=manifest.configuration_version_id,
            configuration=completed.candidate_snapshot,
            item_ids=None,
        )

    def switch_release_before_commit(transaction, callback):  # type: ignore[no-untyped-def]
        transaction.data["release_state/serving"]["release_id"] = "replacement-release"
        return callback(transaction)

    repository._transaction_runner = switch_release_before_commit
    with pytest.raises(ConcurrentActivationError, match="serving release changed"):
        repository.create_run_snapshot(
            run_id="pointer-switch-during-serving-snapshot",
            binding=full_snapshot.binding,
            configuration_version_id=manifest.configuration_version_id,
            configuration=completed.candidate_snapshot,
            item_ids=None,
        )


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


def test_v2_concurrent_proposals_have_one_winner_and_loser_can_retry() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from app.contracts import EvaluationRowV2

    client = FakeFirestore()
    lock = Lock()
    barrier = Barrier(2)

    def serialized_transaction(transaction, callback):
        with lock:
            transaction.data = deepcopy(client.data)
            result = callback(transaction)
            client.data = transaction.data
            return result

    repository = FirestoreEvaluationGenerationStore(client, transaction_runner=serialized_transaction)
    row = EvaluationRowV2(item_id="one", origin="generated", question="Outside?", category="study",
                          expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                          expected_refusal=True, scope_rationale="Outside corpus")
    proposals = [repository.stage_revision("corpus", 1, (row.model_copy(update={"question": f"Question {n}?"}),))
                 for n in range(2)]

    def activate(head):
        barrier.wait(timeout=5)
        try:
            return repository.activate_revision(head, expected_revision=None)
        except ConcurrentActivationError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(activate, proposals))
    winner = repository.corpus_head("corpus")
    assert results.count(None) == 1
    assert next(result for result in results if result is not None) == winner
    loser = next(head for head in proposals if head != winner)
    assert repository.get_revision(loser)[0].question != repository.get_revision(winner)[0].question
    retry = repository.stage_revision("corpus", 2, repository.get_revision(loser))
    assert repository.activate_revision(retry, expected_revision=1) == retry
