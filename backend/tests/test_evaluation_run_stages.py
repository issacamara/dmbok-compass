from __future__ import annotations

import pytest

from app.contracts import (
    CandidateIdentity,
    CorpusServingBinding,
    EvaluationCandidateSnapshot,
    EvaluationDataset,
    EvaluationItem,
    RetrievalResult,
)
from app.evaluation.job import (
    EvaluationJobService,
    EvaluationOutcome,
    EvaluationReviewError,
    EvaluationStageError,
    EvaluationWorker,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
    InMemoryEvaluationStageStore,
)
from app.evaluation.dataset import InMemoryEvaluationDatasetStore


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        artifact_id="artifact-v1",
        installed_lock_sha256="a" * 64,
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )


def _snapshot() -> EvaluationCandidateSnapshot:
    return EvaluationCandidateSnapshot(
        corpus_binding=CorpusServingBinding(
            corpus_version_id="corpus-v1", vector_collection="chunks",
            vector_schema_version="v1", embedding_model="text-embedding",
            ingestion_fingerprint="b" * 64, configuration_fingerprint="c" * 64,
        ),
        project="project", location="location", answer_provider="openrouter",
        primary_model="primary", fallback_model=None, json_schema_supported=True,
        request_timeout_ms=1000, generation_timeout_ms=1000,
        prompt_version="prompt-v1", retrieval_policy_version="retrieval-v1",
        evidence_policy_version="evidence-v1", scorer_version="scorer-v1",
        dependency_lock_sha256="d" * 64, integration_artifact="artifact-v1",
        runtime_version="runtime-v1",
    )


class Resolver:
    def resolve(self, **_: str) -> EvaluationCandidateSnapshot:
        return _snapshot()


def _job() -> tuple[EvaluationJobService, InMemoryEvaluationStageStore]:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[EvaluationItem(item_id="item-1", question="Question", dataset_version_id="dataset-v1")],
    ))
    return (
        EvaluationJobService(
            datasets, InMemoryEvaluationRunStore(), InMemoryEvaluationJobDispatcher(), Resolver()
        ),
        InMemoryEvaluationStageStore(),
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        candidate=_candidate(), ordered_chunk_ids=["chunk-1"], scores=[0.91],
        score_semantics="cosine_similarity", strategy="dense",
        stage_status="succeeded", search_ready=True,
    )


def test_worker_persists_retrieval_before_pending_answer_review() -> None:
    job, stages = _job()
    run = job.launch(
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        model_version_id="model-v1", item_ids=None,
    )
    outcome = EvaluationOutcome(
        True, True, True, True, True, 10,
        retrieval=_retrieval(), output_handle="attempt-1", output_digest="e" * 64,
        assessment_status="awaiting_review",
    )

    completed = EvaluationWorker(job.runs, stages).execute(run.run_id, job.datasets.get("dataset-v1"), lambda _: outcome)  # type: ignore[arg-type]

    assert completed.status == "completed"
    assert completed.metrics == []
    stage = stages.get(run.run_id, "item-1")
    assert stage is not None
    assert stage.retrieval is not None
    assert stage.retrieval.ordered_chunk_ids == ["chunk-1"]
    assert stage.assessment_status == "awaiting_review"


def test_generation_failure_retains_retrieval_and_never_receives_refusal_credit() -> None:
    job, stages = _job()
    run = job.launch(
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        model_version_id="model-v1", item_ids=None,
    )

    def fail(_: str) -> EvaluationOutcome:
        raise EvaluationStageError("provider unavailable", retrieval=_retrieval())

    failed = EvaluationWorker(job.runs, stages).execute(run.run_id, job.datasets.get("dataset-v1"), fail)  # type: ignore[arg-type]
    stage = stages.get(run.run_id, "item-1")

    assert failed.status == "failed"
    assert stage is not None
    assert stage.retrieval is not None
    assert stage.generation_status == "failed"
    assert stage.assessment_status == "awaiting_review"
    assert stage.refusal_correct is False


def test_finalization_requires_exact_digest_and_recomputes_all_six_metrics() -> None:
    job, stages = _job()
    run = job.launch(
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        model_version_id="model-v1", item_ids=None,
    )
    outcome = EvaluationOutcome(
        True, False, True, True, True, 10,
        retrieval=_retrieval(), output_handle="attempt-1", output_digest="e" * 64,
        assessment_status="awaiting_review",
    )
    worker = EvaluationWorker(job.runs, stages)
    worker.execute(run.run_id, job.datasets.get("dataset-v1"), lambda _: outcome)  # type: ignore[arg-type]

    with pytest.raises(EvaluationReviewError, match="digest"):
        worker.review_item(
            run.run_id, "item-1", output_digest="f" * 64,
            assessment=outcome,
        )

    worker.review_item(run.run_id, "item-1", output_digest="e" * 64, assessment=outcome)
    finalized = worker.finalize(run.run_id, cleanup_output=lambda _: None)

    assert finalized.status == "completed"
    assert len(finalized.metrics) == 6
    assert {metric.metric_name for metric in finalized.metrics} == {
        "retrieval_success", "grounded_claims", "citation_correctness",
        "answer_quality", "refusal_correctness", "response_time",
    }
