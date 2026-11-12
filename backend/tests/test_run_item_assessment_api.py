from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import status
from fastapi.testclient import TestClient

from app.api.evaluation_assessments import assessment_repository, evaluation_run_store, evaluation_stage_store, output_store
from app.contracts import CandidateIdentity, EvaluationCandidateSnapshot, EvaluationRun, UserProfile
from app.contracts.v29 import CandidateDatasetBinding, EvaluationRowV2, EvaluationRunSnapshot
from app.evaluation.job import EvaluationItemStage
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
        artifact_id="artifact-v1", installed_lock_sha256="d" * 64,
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )


@dataclass
class Audit:
    evaluation_run_id: str = "run-1"
    evaluation_item_id: str = "item-1"
    candidate_id: str = "candidate-1"
    attempt_id: str = "attempt-1"
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

    def get_run_snapshot(self, run_id: str):
        snapshot = EvaluationRunSnapshot(
            run_id=run_id,
            binding=CandidateDatasetBinding(attempt_id="attempt", corpus_version_id="corpus-v1", dataset_revision=1),
            configuration_version_id="config-v1",
            selection="full",
            items=(EvaluationRowV2(
                item_id="item-1", origin="generated", question="Question?", category="study",
                expected_outcome="evidence_refusal", expected_answer_or_rubric="Refuse",
                expected_refusal=True, scope_rationale="Outside scope.",
            ),),
            snapshot_sha256="c" * 64,
        )
        configuration = EvaluationCandidateSnapshot(
            corpus_binding={"corpus_version_id": "corpus-v1", "vector_collection": "chunks", "vector_schema_version": "2", "embedding_model": "embedding", "ingestion_fingerprint": "a" * 64, "configuration_fingerprint": "e" * 64},
            project="project", location="location", answer_provider="vertex", primary_model="model", json_schema_supported=True,
            request_timeout_ms=1000, generation_timeout_ms=1000, prompt_version="prompt", retrieval_policy_version="retrieval",
            evidence_policy_version="evidence", scorer_version="scorer-v1", dependency_lock_sha256="d" * 64,
            integration_artifact="artifact-v1", runtime_version="runtime-v1",
        )
        return snapshot, configuration


class Outputs:
    def get_audit(self, attempt_id: str) -> Audit | None:
        return Audit() if attempt_id == "attempt-1" else None

    def get_raw(self, *, actor: UserProfile, attempt_id: str) -> str | None:
        return "Generated answer for review." if actor.role == "admin" and attempt_id == "attempt-1" else None


class Stages:
    def get(self, run_id: str, item_id: str) -> EvaluationItemStage | None:
        return EvaluationItemStage(run_id=run_id, item_id=item_id, output_handle="attempt-1", output_digest="b" * 64)


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


def test_assessment_context_returns_exact_run_output_and_expected_refusal() -> None:
    app.dependency_overrides.update({
        get_admin_user: admin,
        evaluation_run_store: Runs,
        evaluation_stage_store: Stages,
        output_store: Outputs,
        assessment_repository: InMemoryAssessmentRepository,
    })
    try:
        response = TestClient(app).get("/api/admin/evaluations/run-1/items/item-1/assessment")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["expected_refusal"] is True
    assert body["output_text"] == "Generated answer for review."
    assert body["attempt_id"] == "attempt-1"
    assert body["output_digest"] == "b" * 64
    assert "source_spans" not in response.text
    assert "relevant_child_ids" not in response.text
