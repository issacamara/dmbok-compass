from __future__ import annotations

from hashlib import sha256

import pytest

from app.contracts import (
    CorpusServingBinding,
    DocumentChunk,
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItem,
    GoldAnnotation,
)
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
from app.evaluation.generations import ConcurrentActivationError, FirestoreEvaluationGenerationStore
from app.evaluation.job import (
    EvaluationJobService,
    EvaluationOutcome,
    EvaluationWorker,
    FirestoreEvaluationRunStore,
    InMemoryEvaluationJobDispatcher,
)
from app.release.decision import (
    InMemoryReleaseDecisionStore,
    ReleaseDecisionError,
    ReleaseDecisionRequest,
    ReleaseDecisionService,
)
from test_evaluation_generations import FakeFirestore, run_fake_transaction


def _snapshot(corpus_version_id: str) -> EvaluationCandidateSnapshot:
    return EvaluationCandidateSnapshot(
        corpus_binding=CorpusServingBinding(
            corpus_version_id=corpus_version_id,
            vector_collection="document_chunks_v2",
            vector_schema_version="2",
            embedding_model="gemini-embedding-001",
            ingestion_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
        ),
        project="dmbok-compass",
        location="europe-west1",
        answer_provider="openrouter",
        primary_model="meta-llama/llama-3.1-8b-instruct",
        fallback_model="mistralai/mistral-nemo",
        json_schema_supported=True,
        request_timeout_ms=14_000,
        generation_timeout_ms=8_000,
        prompt_version="langchain-rag-v3",
        retrieval_policy_version="retrieval-v2",
        evidence_policy_version="grounded-evidence-v1",
        scorer_version="evaluation-gates-v1",
        dependency_lock_sha256="c" * 64,
        integration_artifact="langchain-google-firestore==0.5.0.post2",
        runtime_version="candidate-image@sha256:" + "d" * 64,
    )


def _dataset(version: str) -> tuple[EvaluationDataset, list[DocumentChunk]]:
    items = [
        EvaluationItem(
            item_id=f"item-{index}",
            question=f"Authored evaluation question {index}",
            dataset_version_id=version,
            category=("scenarios", "definitions", "explanations", "comparisons", "study")[index % 5],
        )
        for index in range(30)
    ]
    annotations = [
        GoldAnnotation(
            item_id=item.item_id,
            expected_answer_or_rubric="Pass the authored rubric.",
            relevant_chunk_ids=[f"legacy-{item.item_id}"],
            review_status="candidate",
            supporting_passage=f"Candidate support record [{index:02d}]",
        )
        for index, item in enumerate(items)
    ]
    chunks = [
        DocumentChunk(
            chunk_id=f"candidate-{item.item_id}",
            corpus_version_id="corpus-v1",
            page=index + 1,
            section="Candidate section",
            chunk_ordinal=0,
            content_hash=sha256(item.item_id.encode()).hexdigest(),
            text=f"Candidate support record [{index:02d}]",
            embedding=[0.0] * 768,
        )
        for index, item in enumerate(items)
    ]
    return EvaluationDataset(dataset_version_id=version, items=items, gold_annotations=annotations), chunks


class _Snapshots:
    def resolve(self, *, corpus_version_id: str, configuration_version_id: str, model_version_id: str):
        del configuration_version_id, model_version_id
        return _snapshot(corpus_version_id)


def _decision_request(run, *, release_id: str) -> ReleaseDecisionRequest:
    return ReleaseDecisionRequest(
        release_id=release_id,
        evaluation_run_id=run.run_id,
        dataset_version_id=run.dataset_version_id,
        corpus_version_id=run.corpus_version_id,
        configuration_version_id=run.configuration_version_id,
        provider_version_id="openrouter-candidate-v1",
        model_version_id=run.model_version_id,
        scorer_version_id="evaluation-gates-v1",
        gate_report_ids=["candidate-gates-v1"],
        decision="approved",
        rationale="All candidate gates passed.",
    )


def test_evaluation_to_release_journey_is_version_bound_and_restart_safe() -> None:
    client = FakeFirestore()
    generations = FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)
    dataset, chunks = _dataset("generation-v1")
    snapshot = _snapshot("corpus-v1")
    review = remap_gold_annotations(
        dataset,
        chunks,
        CandidateGoldBinding(
            corpus_version_id="corpus-v1",
            candidate_snapshot_sha256=sha256(snapshot.model_dump_json().encode()).hexdigest(),
            source_provenance_sha256="e" * 64,
        ),
    )
    generations.stage(dataset, candidate_review=review)
    generations.activate("generation-v1", expected_active_generation_id=None)
    for item in dataset.items:
        generations.review("generation-v1", item.item_id, "approved")

    runs = FirestoreEvaluationRunStore(client)
    dispatcher = InMemoryEvaluationJobDispatcher()
    jobs = EvaluationJobService(generations, runs, dispatcher, _Snapshots())
    full = jobs.launch(
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        item_ids=None,
    )
    assert full.report_eligibility is not None
    assert full.report_eligibility.eligibility == "release_evidence"

    worker = EvaluationWorker(runs)
    completed = worker.execute(
        full.run_id,
        dataset,
        lambda item_id: EvaluationOutcome(
            retrieval_success=item_id != "item-0",
            grounded=True,
            citation_correct=True,
            answer_quality=True,
            refusal_correct=item_id == "item-0",
            response_time_ms=1_000,
            applicable_metrics=frozenset({"refusal_correctness"}) if item_id == "item-0" else frozenset({
                "retrieval_success", "grounded_claims", "citation_correctness", "answer_quality",
            }),
        ),
    )
    assert completed.status == "completed"

    decisions = InMemoryReleaseDecisionStore()
    release = ReleaseDecisionService(runs, decisions, generations)
    recorded = release.record(_decision_request(completed, release_id="release-v1"))
    assert recorded.decision == "approved"

    subset = jobs.launch(
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        item_ids=["item-1"],
    )
    assert subset.report_eligibility is not None
    assert subset.report_eligibility.eligibility == "exploratory"
    subset_completed = worker.execute(
        subset.run_id,
        dataset,
        lambda _: EvaluationOutcome(
            retrieval_success=True,
            grounded=True,
            citation_correct=True,
            answer_quality=True,
            refusal_correct=True,
            response_time_ms=1_000,
        ),
    )
    with pytest.raises(ReleaseDecisionError, match="exploratory"):
        release.record(_decision_request(subset_completed, release_id="release-subset"))

    dataset_v2, chunks_v2 = _dataset("generation-v2")
    review_v2 = remap_gold_annotations(
        dataset_v2,
        chunks_v2,
        CandidateGoldBinding("corpus-v1", "f" * 64, "1" * 64),
    )
    generations.stage(dataset_v2, candidate_review=review_v2)
    generations.activate("generation-v2", expected_active_generation_id="generation-v1")
    assert generations.report_eligibility("generation-v1").is_superseded is True
    with pytest.raises(ReleaseDecisionError, match="exploratory"):
        release.record(_decision_request(completed, release_id="release-stale"))

    generations.stage(_dataset("generation-v3")[0])
    with pytest.raises(ConcurrentActivationError):
        generations.activate("generation-v3", expected_active_generation_id="generation-v1")
    assert generations.active().generation_id == "generation-v2"  # type: ignore[union-attr]

    restarted_runs = FirestoreEvaluationRunStore(client)
    assert restarted_runs.get(full.run_id) == completed
    persisted_run = client.data[f"evaluation_runs/{full.run_id}"]
    assert not {"question", "answer", "prompt", "passage", "trace"}.intersection(persisted_run)
