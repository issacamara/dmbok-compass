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
        dataset_version_id="dataset-v1", corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    retry = job.launch(
        dataset_version_id="dataset-v1", corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=None,
    )
    subset = job.launch(
        dataset_version_id="dataset-v1", corpus_version_id="corpus-v2",
        configuration_version_id="config-v3", model_version_id="model-v4", item_ids=["item-2"],
    )
    assert full == retry
    assert full.selected_item_ids == ["item-0", "item-1", "item-2"]
    assert subset.selected_item_ids == ["item-2"]
    assert set(dispatcher.dispatched) == {full.run_id, subset.run_id}


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
            "dataset_version_id": "dataset-v1", "corpus_version_id": "corpus-v2",
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
                "dataset_version_id": "dataset-v1", "corpus_version_id": "corpus-v2",
                "configuration_version_id": "config-v3", "model_version_id": "model-v4",
            },
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
    finally:
        app.dependency_overrides.clear()


def test_worker_completes_with_six_aggregate_metrics_and_is_idempotent() -> None:
    job, _ = service()
    run = job.launch(
        dataset_version_id="dataset-v1", corpus_version_id="corpus-v2",
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
    assert [item.item_id for item in result.item_results] == ["item-0", "item-1"]
    assert result.item_results[0].answer_quality is False
    assert worker.execute(run.run_id, dataset(), lambda _: (_ for _ in ()).throw(AssertionError())).metrics == result.metrics
