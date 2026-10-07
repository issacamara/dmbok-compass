import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.contracts import EvaluationMetric, EvaluationRun, UserProfile
from app.evaluation import InMemoryEvaluationRunStore
from app.identity import IdentityError, get_admin_user
from app.main import app, release_decision_service
from app.release import InMemoryReleaseDecisionStore, ReleaseDecisionRequest, ReleaseDecisionService


def completed_run(*, passed: bool = True) -> EvaluationRun:
    return EvaluationRun(
        run_id="eval-1",
        dataset_version_id="dataset-v1",
        corpus_version_id="corpus-v1",
        configuration_version_id="config-v1",
        model_version_id="model-v1",
        status="completed",
        selected_item_ids=["item-1"],
        metrics=[EvaluationMetric(
            metric_name="retrieval_success", numerator=1 if passed else 0, denominator=1,
            percentage=100 if passed else 0, threshold=90, passed=passed,
        )],
    )


def request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "release_id": "release-1", "evaluation_run_id": "eval-1",
        "dataset_version_id": "dataset-v1", "corpus_version_id": "corpus-v1",
        "configuration_version_id": "config-v1", "provider_version_id": "openrouter-v1",
        "model_version_id": "model-v1", "scorer_version_id": "scorer-v1",
        "gate_report_ids": ["gate-report-1"], "decision": "approved", "rationale": "All gates passed.",
    }
    payload.update(overrides)
    return payload


def service(run: EvaluationRun) -> ReleaseDecisionService:
    runs = InMemoryEvaluationRunStore()
    runs.create_or_get(run)
    return ReleaseDecisionService(runs, InMemoryReleaseDecisionStore())


def test_decision_binds_exact_evidence_and_retry_is_idempotent() -> None:
    release_service = service(completed_run())
    first = release_service.record(ReleaseDecisionRequest(**request()))
    result = release_service.record(ReleaseDecisionRequest(**request()))
    assert result == first
    assert result.provider_version_id == "openrouter-v1"
    assert release_service.get("release-1") == first


def test_stale_or_missing_evidence_is_rejected() -> None:
    release_service = service(completed_run())
    with pytest.raises(ValueError, match="stale"):
        release_service.record(ReleaseDecisionRequest(**request(dataset_version_id="dataset-old")))
    with pytest.raises(ValidationError):
        ReleaseDecisionRequest(**request(gate_report_ids=[]))


def test_release_id_cannot_be_reused_for_different_evidence() -> None:
    release_service = service(completed_run())
    release_service.record(ReleaseDecisionRequest(**request()))
    with pytest.raises(ValueError, match="already bound"):
        release_service.record(ReleaseDecisionRequest(**request(rationale="Changed after approval.")))


def test_failed_gate_requires_explicit_approved_exception() -> None:
    release_service = service(completed_run(passed=False))

    try:
        release_service.record(ReleaseDecisionRequest(**request()))
    except ValueError as exc:
        assert "gate" in str(exc)
    else:
        raise AssertionError("failed evidence must block approval")
    result = release_service.record(ReleaseDecisionRequest(**request(
        exception_approved=True, exception_rationale="Sponsor accepted the documented risk.",
    )))
    assert result.exception_approved is True


def test_release_api_requires_admin_and_persists_decision() -> None:
    admin = UserProfile(user_id="admin", email="admin@example.com", username="admin",
                        approval_state="approved", role="admin", email_verified=True)
    app.dependency_overrides[get_admin_user] = lambda: admin
    app.dependency_overrides[release_decision_service] = lambda: service(completed_run())
    try:
        response = TestClient(app).post("/api/releases", json=request())
        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["gate_report_ids"] == ["gate-report-1"]
    finally:
        app.dependency_overrides.clear()


def test_release_api_rejects_non_admin() -> None:
    app.dependency_overrides[get_admin_user] = lambda: (_ for _ in ()).throw(
        IdentityError("Administrator access is required.", status.HTTP_403_FORBIDDEN)
    )
    try:
        response = TestClient(app).post("/api/releases", json=request())
        assert response.status_code == status.HTTP_403_FORBIDDEN
    finally:
        app.dependency_overrides.clear()
