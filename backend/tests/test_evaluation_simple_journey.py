from __future__ import annotations

import json

from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.api.evaluation_datasets import evaluation_generation_store
from app.contracts import EvaluationDataset, EvaluationItem, EvaluationRun
from app.corpus.selection import uploaded_corpus_version_id
from app.evaluation import (
    EvaluationJobService,
    EvaluationOutcome,
    EvaluationWorker,
    InMemoryEvaluationDatasetStore,
    InMemoryEvaluationJobDispatcher,
    InMemoryEvaluationRunStore,
)
from app.identity import get_admin_user
from app.main import app
from test_evaluation_import import FakeFirestore, _admin, _record, _service, _store
from test_evaluation_job import SnapshotResolver


def test_upload_to_run_is_one_content_safe_server_owned_journey() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    service = _service()
    try:
        response = TestClient(app).post(
            "/api/admin/evaluation-datasets/upload-and-run",
            content=json.dumps([
                _record(question="What is metadata?"),
                _record(category="unsupported"),
                _record(question="  "),
            ]),
            headers={"Content-Type": "application/json"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert (body["submitted_count"], body["imported_count"], body["skipped_count"]) == (3, 1, 2)
    assert body["run_id"] == "run-1"
    assert service.calls == 1
    assert not {"corpus_version_id", "configuration_version_id", "model_version_id", "item_ids"} & body.keys()
    assert "What is metadata?" not in response.text


def test_all_invalid_and_unavailable_corpus_paths_are_safe_and_non_dispatching() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    service = _service()
    try:
        rejected = TestClient(app).post(
            "/api/admin/evaluation-datasets/upload-and-run",
            json=[_record(question="", supporting_passage="private evidence")],
        )
    finally:
        app.dependency_overrides.clear()

    assert rejected.status_code == status.HTTP_200_OK
    assert rejected.json()["run_id"] is None
    assert service.calls == 0
    assert "private evidence" not in rejected.text

    def unavailable() -> str:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "uploaded_corpus_unavailable", "message": "The current evaluation corpus is unavailable."},
        )

    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(FakeFirestore())
    app.dependency_overrides[uploaded_corpus_version_id] = unavailable
    service = _service()
    app.dependency_overrides[uploaded_corpus_version_id] = unavailable
    try:
        unavailable_response = TestClient(app).post(
            "/api/admin/evaluation-datasets/upload-and-run",
            json=[_record()],
        )
    finally:
        app.dependency_overrides.clear()

    assert unavailable_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert unavailable_response.json()["detail"]["code"] == "uploaded_corpus_unavailable"
    assert service.calls == 0


def test_worker_restart_replays_completed_run_without_reexecuting_items() -> None:
    runs = InMemoryEvaluationRunStore()
    dataset = EvaluationDataset(
        dataset_version_id="dataset-journey",
        items=[EvaluationItem(item_id="item-1", question="Question", dataset_version_id="dataset-journey")],
    )
    runs.create_or_get(EvaluationRun(
        run_id="run-journey",
        dataset_version_id=dataset.dataset_version_id,
        corpus_version_id="server-selected",
        status="queued",
        selected_item_ids=[],
        configuration_version_id="server-selected",
        model_version_id="server-selected",
    ))
    calls = 0

    def evaluate(_: str) -> EvaluationOutcome:
        nonlocal calls
        calls += 1
        return EvaluationOutcome(True, True, True, True, True, 10)

    first = EvaluationWorker(runs).execute("run-journey", dataset, evaluate)
    replay = EvaluationWorker(runs).execute(
        "run-journey", dataset, lambda _: (_ for _ in ()).throw(AssertionError("replayed"))
    )

    assert first.status == replay.status == "completed"
    assert first.metrics == replay.metrics
    assert calls == 1
    persisted = runs.status("run-journey")
    assert persisted is not None and persisted.status == "completed"
    assert persisted.completed_at is not None


def test_server_resolved_full_run_is_exploratory_and_not_release_authorisation() -> None:
    datasets = InMemoryEvaluationDatasetStore()
    datasets.create(EvaluationDataset(
        dataset_version_id="dataset-journey",
        items=[EvaluationItem(item_id="item-1", question="Question", dataset_version_id="dataset-journey")],
    ))
    service = EvaluationJobService(
        datasets,
        InMemoryEvaluationRunStore(),
        InMemoryEvaluationJobDispatcher(),
        SnapshotResolver(),
    )

    run = service.launch(item_ids=None)

    assert run.selected_item_ids == ["item-1"]
    assert run.report_eligibility is not None
    assert run.report_eligibility.eligibility == "exploratory"
    assert run.report_eligibility.approved_gold_count == 0
