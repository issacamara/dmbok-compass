from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from app.api.evaluations import evaluation_service
from app.contracts import CorpusServingBinding, EvaluationCandidateSnapshot, EvaluationDataset, EvaluationItem, EvaluationRun, UserProfile
from app.evaluation import (
    EvaluationJobService,
    InMemoryEvaluationDatasetStore,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
    EvaluationOutcome,
    EvaluationInterrupted,
    EvaluationWorker,
)
from app.identity import IdentityError, get_admin_user
from app.main import app


def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset_version_id="dataset-v1",
        items=[
            EvaluationItem(item_id=f"item-{i}", question=f"Question {i}", dataset_version_id="dataset-v1")
            for i in range(3)
        ],
    )


class SnapshotResolver:
    runtime_version = "revision-a"

    def resolve(
        self,
        *,
        corpus_version_id: str,
        configuration_version_id: str,
        model_version_id: str,
    ) -> EvaluationCandidateSnapshot:
        return EvaluationCandidateSnapshot(
            corpus_binding=CorpusServingBinding(
                corpus_version_id=corpus_version_id,
                vector_collection="document_chunks_v2",
                vector_schema_version="2",
                embedding_model="gemini-embedding-001",
                ingestion_fingerprint="a" * 64,
                configuration_fingerprint="b" * 64,
            ),
            project="project", location="europe-west1", answer_provider="openrouter",
            primary_model="model-primary", fallback_model="model-fallback",
            json_schema_supported=True,
            request_timeout_ms=14_000, generation_timeout_ms=8_000,
            prompt_version="grounded-answer-policy-v1",
            retrieval_policy_version="top-five-cosine-v1",
            evidence_policy_version="grounded-evidence-v1", scorer_version="evaluation-gates-v1",
            dependency_lock_sha256="c" * 64,
            integration_artifact="langchain-google-firestore==0.5.0.post2",
            runtime_version=self.runtime_version,
        )


def service() -> tuple[EvaluationJobService, InMemoryEvaluationJobDispatcher]:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset())
    dispatcher = InMemoryEvaluationJobDispatcher()
    return EvaluationJobService(datasets, InMemoryEvaluationRunStore(), dispatcher, SnapshotResolver()), dispatcher


def test_full_and_subset_runs_are_version_bound_and_retry_idempotent() -> None:
    job, dispatcher = service()
    full = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    retry = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    subset = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-2"],
    )
    assert full == retry
    assert full.selected_item_ids == ["item-0", "item-1", "item-2"]
    assert subset.selected_item_ids == ["item-2"]
    assert set(dispatcher.dispatched) == {full.run_id, subset.run_id}
    assert full.report_eligibility is not None
    assert full.report_eligibility.eligibility == "exploratory"
    assert full.candidate_snapshot is not None
    assert full.candidate_snapshot.corpus_binding.corpus_version_id == "corpus-v2"


def test_run_status_reclassifies_a_superseded_generation_as_exploratory() -> None:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset())
    job = EvaluationJobService(datasets, InMemoryEvaluationRunStore(), InMemoryEvaluationJobDispatcher(), SnapshotResolver())
    run = job.launch(
        corpus_version_id="corpus-v2", configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    datasets.create(
        EvaluationDataset(
            dataset_version_id="dataset-v2",
            items=[EvaluationItem(item_id="item-new", question="New question", dataset_version_id="dataset-v2")],
        )
    )

    status_result = job.status(run.run_id)

    assert status_result.report_eligibility is not None
    assert status_result.report_eligibility.eligibility == "exploratory"
    assert status_result.report_eligibility.is_current_generation is False
    assert status_result.report_eligibility.is_superseded is True


def test_admin_api_launch_and_status_require_admin_and_dispatch_once() -> None:
    job, dispatcher = service()
    admin = UserProfile(
        user_id="admin", email="admin@example.com", username="admin",
        approval_state="approved", role="admin", email_verified=True,
    )
    app.dependency_overrides[get_admin_user] = lambda: admin
    app.dependency_overrides[evaluation_service] = lambda: job
    try:
        client = TestClient(app)
        payload = {
            "corpus_version_id": "corpus-v2",
            "configuration_version_id": "config-v3", "model_version_id": "model-v4",
            "item_ids": ["item-1"],
        }
        created = client.post("/api/evaluations", json=payload)
        retried = client.post("/api/evaluations", json=payload)
        assert created.status_code == 202
        assert retried.json() == created.json()
        run_id = created.json()["run_id"]
        assert client.get(f"/api/evaluations/{run_id}").json() == created.json()
        assert len(dispatcher.dispatched) == 1
    finally:
        app.dependency_overrides.clear()


def test_evaluation_api_rejects_non_admin_callers() -> None:
    job, _ = service()

    def reject_non_admin() -> UserProfile:
        raise IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)

    app.dependency_overrides[get_admin_user] = reject_non_admin
    app.dependency_overrides[evaluation_service] = lambda: job
    try:
        response = TestClient(app).post(
            "/api/evaluations",
            json={
                "corpus_version_id": "corpus-v2",
                "configuration_version_id": "config-v3", "model_version_id": "model-v4",
            },
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
    finally:
        app.dependency_overrides.clear()


def test_worker_completes_with_six_aggregate_metrics_and_is_idempotent() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-0", "item-1"],
    )
    worker = EvaluationWorker(job.runs)
    result = worker.execute(
        run.run_id,
        dataset(),
        lambda _: EvaluationOutcome(True, True, True, False, True, 10),
    )
    assert result.status == "completed"
    assert len(result.metrics) == 6
    assert result.metrics[0].numerator == 2
    assert result.metrics[0].threshold == 90
    assert result.metrics[0].passed is True
    assert result.metrics[3].threshold == 85
    assert result.metrics[3].passed is False
    assert [item.item_id for item in result.item_results] == ["item-0", "item-1"]
    assert result.item_results[0].answer_quality is False
    assert worker.execute(run.run_id, dataset(), lambda _: (_ for _ in ()).throw(AssertionError())).metrics == result.metrics


def test_worker_marks_each_release_gate_from_gold_evidence_scores() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    result = EvaluationWorker(job.runs).execute(
        run.run_id,
        dataset(),
        lambda item_id: EvaluationOutcome(
            retrieval_success=item_id != "item-2",
            grounded=True,
            citation_correct=True,
            answer_quality=True,
            refusal_correct=True,
            response_time_ms=100,
        ),
    )

    assert [
        (metric.metric_name, metric.numerator, metric.denominator, metric.threshold, metric.passed)
        for metric in result.metrics
    ] == [
        ("retrieval_success", 2, 3, 90, False),
        ("grounded_claims", 3, 3, 95, True),
        ("citation_correctness", 3, 3, 95, True),
        ("answer_quality", 3, 3, 85, True),
        ("refusal_correctness", 3, 3, 95, True),
        ("response_time", 3, 3, 95, True),
    ]


def test_worker_uses_metric_specific_denominators_and_never_passes_an_empty_gate() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    result = EvaluationWorker(job.runs).execute(
        run.run_id,
        dataset(),
        lambda item_id: EvaluationOutcome(
            retrieval_success=True,
            grounded=True,
            citation_correct=True,
            answer_quality=True,
            refusal_correct=item_id == "item-2",
            response_time_ms=100,
            applicable_metrics=(
                frozenset({"refusal_correctness"}) if item_id == "item-2"
                else frozenset({"retrieval_success", "grounded_claims", "citation_correctness", "answer_quality"})
            ),
        ),
    )
    metrics = {metric.metric_name: metric for metric in result.metrics}

    assert (metrics["refusal_correctness"].numerator, metrics["refusal_correctness"].denominator) == (1, 1)
    assert (metrics["grounded_claims"].numerator, metrics["grounded_claims"].denominator) == (2, 2)
    assert metrics["refusal_correctness"].passed is True

    refusal_only = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-2"],
    )
    empty_gate = EvaluationWorker(job.runs).execute(
        refusal_only.run_id,
        dataset(),
        lambda _: EvaluationOutcome(
            retrieval_success=False, grounded=False, citation_correct=False, answer_quality=False,
            refusal_correct=True, response_time_ms=100,
            applicable_metrics=frozenset({"refusal_correctness"}),
        ),
    )
    empty_metrics = {metric.metric_name: metric for metric in empty_gate.metrics}
    assert (empty_metrics["grounded_claims"].denominator, empty_metrics["grounded_claims"].passed) == (0, False)


def test_worker_failure_is_content_free_and_persisted() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-0"],
    )

    failed = EvaluationWorker(job.runs).execute(
        run.run_id,
        dataset(),
        lambda _: (_ for _ in ()).throw(RuntimeError("Question 0 and generated answer must not persist")),
    )

    assert failed.status == "failed"
    assert failed.error == "RuntimeError: evaluation failed"


def test_worker_interruption_is_persisted_and_rethrown_for_cloud_run_retry() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )

    with pytest.raises(EvaluationInterrupted, match="interrupted"):
        EvaluationWorker(job.runs).execute(
            run.run_id,
            dataset(),
            lambda _: EvaluationOutcome(True, True, True, True, True, 10),
            termination_requested=lambda: True,
        )

    interrupted = job.runs.get(run.run_id)
    assert interrupted is not None
    assert interrupted.status == "failed"
    assert interrupted.error == "EvaluationInterrupted: evaluation interrupted"


def test_worker_rejects_a_dataset_other_than_the_run_pinned_generation() -> None:
    job, _ = service()
    run = job.launch(
        corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-0"],
    )
    other = dataset().model_copy(update={"dataset_version_id": "dataset-v2"})

    failed = EvaluationWorker(job.runs).execute(run.run_id, other, lambda _: EvaluationOutcome(True, True, True, True, True, 10))

    assert failed.status == "failed"
    assert failed.error == "EvaluationRunError: evaluation failed"


def test_launch_persists_snapshot_and_configuration_drift_creates_a_new_run() -> None:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset())
    resolver = SnapshotResolver()
    job = EvaluationJobService(datasets, InMemoryEvaluationRunStore(), InMemoryEvaluationJobDispatcher(), resolver)

    first = job.launch(
        corpus_version_id="corpus-v2", configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    resolver.runtime_version = "revision-b"
    second = job.launch(
        corpus_version_id="corpus-v2", configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )

    assert first.run_id != second.run_id
    assert first.candidate_snapshot is not None
    assert first.candidate_snapshot.runtime_version == "revision-a"
    assert second.candidate_snapshot is not None
    assert second.candidate_snapshot.runtime_version == "revision-b"


def test_worker_marks_legacy_run_without_a_snapshot_as_failed() -> None:
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(EvaluationRun(
        run_id="legacy", dataset_version_id="dataset-v1", corpus_version_id="corpus-v1", status="queued",
        selected_item_ids=["item-0"], configuration_version_id="config-v1", model_version_id="model-v1",
    ))

    failed = EvaluationWorker(runs).execute("legacy", dataset(), lambda _: EvaluationOutcome(True, True, True, True, True, 1))

    assert failed.status == "failed"
    assert failed.error == "EvaluationRunError: immutable candidate snapshot is unavailable"
