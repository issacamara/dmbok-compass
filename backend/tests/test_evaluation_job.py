from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient

from app.api.evaluations import evaluation_service
from app.contracts import EvaluationDataset, EvaluationItem, UserProfile
from app.evaluation import (
    EvaluationJobService,
    InMemoryEvaluationDatasetStore,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
    EvaluationOutcome,
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


def service() -> tuple[EvaluationJobService, InMemoryEvaluationJobDispatcher]:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset())
    dispatcher = InMemoryEvaluationJobDispatcher()
    return EvaluationJobService(datasets, InMemoryEvaluationRunStore(), dispatcher), dispatcher


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


def test_run_status_reclassifies_a_superseded_generation_as_exploratory() -> None:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(dataset())
    job = EvaluationJobService(datasets, InMemoryEvaluationRunStore(), InMemoryEvaluationJobDispatcher())
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
