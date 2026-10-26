from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import status
from fastapi.testclient import TestClient

from app.api.evaluation_assessments import assessment_repository, evaluation_run_store, output_store
from app.contracts import CandidateIdentity, EvaluationRun, UserProfile
from app.evaluation.reviews import InMemoryAssessmentRepository
from app.identity import get_admin_user
from app.main import app


def admin() -> UserProfile:
    return UserProfile(
        user_id="reviewer-1", email="reviewer@example.com", username="reviewer",
        approval_state="approved", role="admin", email_verified=True,
    )


def candidate() -> CandidateIdentity:
    return CandidateIdentity(
        artifact_id="artifact-v1", installed_lock_sha256="a" * 64,
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )


@dataclass
class Audit:
    evaluation_run_id: str = "run-1"
    evaluation_item_id: str = "item-1"
    candidate_id: str = "candidate-1"
    candidate_identity: CandidateIdentity = field(default_factory=candidate)
    scorer_version: str = "scorer-v1"
    output_digest: str = "b" * 64


class Runs:
    def get(self, run_id: str) -> EvaluationRun | None:
        return EvaluationRun(
            run_id=run_id, dataset_version_id="generation-1", corpus_version_id="corpus-v1",
            status="completed", selected_item_ids=["item-1"], configuration_version_id="config-v1",
            model_version_id="model-v1",
        ) if run_id == "run-1" else None


class Outputs:
    def get_audit(self, attempt_id: str) -> Audit | None:
        return Audit() if attempt_id == "attempt-1" else None


def payload(**overrides: object) -> dict[str, object]:
    return {
        "generation_id": "generation-1", "attempt_id": "attempt-1", "candidate_id": "candidate-1",
        "scorer_version": "scorer-v1", "output_digest": "b" * 64,
        "claim_support": True, "citation_support": True, "rubric_passed": True,
        "refusal_judgment": False, **overrides,
    }


def test_assessment_persists_server_actor_and_is_idempotent() -> None:
    repository = InMemoryAssessmentRepository()
    app.dependency_overrides.update({
        get_admin_user: admin, evaluation_run_store: Runs, output_store: Outputs,
        assessment_repository: lambda: repository,
    })
    try:
        client = TestClient(app)
        first = client.put("/api/admin/evaluations/run-1/items/item-1/assessment", json=payload())
        second = client.put("/api/admin/evaluations/run-1/items/item-1/assessment", json=payload())
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == status.HTTP_200_OK, first.text
    assert second.status_code == status.HTTP_200_OK
    assert first.json() == second.json()
    assert first.json()["reviewer_actor"] == "reviewer-1"
    assert datetime.fromisoformat(first.json()["reviewed_at"]).tzinfo == timezone.utc


def test_assessment_rejects_stale_binding_wrong_digest_and_unknown_output() -> None:
    app.dependency_overrides.update({get_admin_user: admin, evaluation_run_store: Runs, output_store: Outputs, assessment_repository: InMemoryAssessmentRepository})
    try:
        client = TestClient(app)
        stale = client.put("/api/admin/evaluations/run-1/items/item-1/assessment", json=payload(generation_id="old-generation"))
        wrong_digest = client.put("/api/admin/evaluations/run-1/items/item-1/assessment", json=payload(output_digest="c" * 64))
        missing = client.put("/api/admin/evaluations/run-1/items/item-1/assessment", json=payload(attempt_id="missing"))
    finally:
        app.dependency_overrides.clear()

    assert stale.status_code == status.HTTP_409_CONFLICT
    assert wrong_digest.status_code == status.HTTP_409_CONFLICT
    assert missing.status_code == status.HTTP_404_NOT_FOUND
