from __future__ import annotations

import json
import logging

from fastapi import status
from fastapi.testclient import TestClient

from app.api.evaluation_datasets import evaluation_generation_store
from app.contracts import UserProfile
from app.evaluation import FirestoreEvaluationGenerationStore
from app.identity import IdentityError, get_admin_user
from app.main import app
from test_evaluation_generations import FakeFirestore, run_fake_transaction


def _admin() -> UserProfile:
    return UserProfile(
        user_id="admin", email="admin@example.com", username="admin",
        approval_state="approved", role="admin", email_verified=True,
    )


def _store(client: FakeFirestore) -> FirestoreEvaluationGenerationStore:
    return FirestoreEvaluationGenerationStore(client, transaction_runner=run_fake_transaction)


def _record(**overrides: object) -> dict[str, object]:
    return {
        "question": "What is metadata?",
        "category": "definitions",
        "expected_answer_or_rubric": "A definition of metadata.",
        "supporting_passage": "Metadata describes data.",
        **overrides,
    }


def _post(client: TestClient, records: object):
    return client.post(
        "/api/admin/evaluation-datasets/import",
        content=json.dumps(records),
        headers={"Content-Type": "application/json"},
    )


def test_imports_valid_records_and_reports_invalid_indices_without_content() -> None:
    firestore = FakeFirestore()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(firestore)
    try:
        response = _post(TestClient(app), [_record(), _record(category="other"), _record(question="  ")])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert body["status"] == "accepted"
    assert (body["submitted_count"], body["imported_count"], body["skipped_count"]) == (3, 1, 2)
    assert [(error["index"], error["code"]) for error in body["validation_errors"]] == [
        (1, "invalid_category"), (2, "missing_question")
    ]
    assert "What is metadata?" not in json.dumps(body)
    active_id = firestore.data["evaluation_dataset_state/active"]["generation_id"]
    annotations = [data for path, data in firestore.data.items() if "/annotations/" in path]
    assert active_id == body["generation_id"]
    assert annotations[0]["supporting_passage"] == "Metadata describes data."


def test_rejections_and_storage_failures_leave_existing_active_generation_unchanged() -> None:
    firestore = FakeFirestore()
    repository = _store(firestore)
    first = _record()
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: repository
    try:
        created = _post(TestClient(app), [first])
        active_id = created.json()["generation_id"]
        rejected = _post(TestClient(app), [_record(question="")])
        firestore.fail_next_batch = True
        failed = _post(TestClient(app), [_record(question="A different question")])
    finally:
        app.dependency_overrides.clear()

    assert rejected.status_code == status.HTTP_200_OK
    assert rejected.json()["status"] == "rejected"
    assert failed.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert firestore.data["evaluation_dataset_state/active"]["generation_id"] == active_id


def test_import_enforces_bounds_and_administrator_access() -> None:
    app.dependency_overrides[get_admin_user] = lambda: (_ for _ in ()).throw(
        IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)
    )
    try:
        denied = _post(TestClient(app), [_record()])
    finally:
        app.dependency_overrides.clear()
    assert denied.status_code == status.HTTP_403_FORBIDDEN

    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(FakeFirestore())
    try:
        malformed = TestClient(app).post("/api/admin/evaluation-datasets/import", content=b"{")
        over_limit = _post(TestClient(app), [_record()] * 501)
    finally:
        app.dependency_overrides.clear()
    assert malformed.status_code == status.HTTP_400_BAD_REQUEST
    assert over_limit.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_import_error_messages_and_logs_do_not_expose_evaluator_text(caplog) -> None:
    secret_question = "secret evaluator question"
    secret_passage = "secret evaluator passage"
    app.dependency_overrides[get_admin_user] = _admin
    app.dependency_overrides[evaluation_generation_store] = lambda: _store(FakeFirestore())
    with caplog.at_level(logging.INFO):
        try:
            response = _post(TestClient(app), [_record(question=secret_question, supporting_passage=secret_passage, category="bad")])
        finally:
            app.dependency_overrides.clear()
    rendered = response.text + caplog.text
    assert response.status_code == status.HTTP_200_OK
    assert secret_question not in rendered
    assert secret_passage not in rendered
