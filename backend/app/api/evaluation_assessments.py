"""Administrator-only exact-run human assessment endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.contracts import UserProfile
from app.evaluation.candidate_outputs import FirestoreCandidateOutputStore
from app.evaluation.job import EvaluationStageStore, FirestoreEvaluationRunStore
from app.evaluation.reviews import (
    AssessmentConflictError,
    AssessmentError,
    AssessmentRepository,
    AssessmentNotFoundError,
    OutputStore,
    RunItemAssessment,
    RunItemAssessmentRecord,
    assessment_key,
    utc_now,
)
from app.identity import get_admin_user

router = APIRouter(prefix="/api/admin/evaluations", tags=["evaluation assessments"])


def assessment_repository() -> AssessmentRepository:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "assessment_repository_unavailable", "message": "Assessment storage is not configured."})


def output_store() -> FirestoreCandidateOutputStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_output_store_unavailable", "message": "Evaluation output storage is not configured."})


def evaluation_run_store() -> FirestoreEvaluationRunStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_run_store_unavailable", "message": "Evaluation run storage is not configured."})


def evaluation_stage_store() -> EvaluationStageStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_stage_store_unavailable", "message": "Evaluation stage storage is not configured."})


class AssessmentDecisionProjection(BaseModel):
    assessment_id: str
    claim_support: bool | None
    citation_support: bool | None
    rubric_passed: bool | None
    refusal_judgment: bool | None
    reviewed_at: str


class RunItemAssessmentProjection(BaseModel):
    """Admin-only exact-output view with run-bound fields needed to review it."""

    run_id: str
    item_id: str
    generation_id: str
    expected_refusal: bool
    output_text: str = Field(min_length=1, max_length=20000)
    attempt_id: str
    candidate_id: str
    scorer_version: str
    output_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    assessment: AssessmentDecisionProjection | None = None


@router.get("/{run_id}/items/{item_id}/assessment", response_model=RunItemAssessmentProjection)
def get_run_item_assessment_context(
    run_id: str,
    item_id: str,
    admin: UserProfile = Depends(get_admin_user),
    runs: FirestoreEvaluationRunStore = Depends(evaluation_run_store),
    stages: EvaluationStageStore = Depends(evaluation_stage_store),
    outputs: FirestoreCandidateOutputStore = Depends(output_store),
    repository: AssessmentRepository = Depends(assessment_repository),
) -> RunItemAssessmentProjection:
    run = runs.get(run_id)
    if run is None or run.status != "completed" or item_id not in run.selected_item_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "assessment_target_not_found", "message": "The exact run output is unavailable."})
    try:
        snapshot, configuration = runs.get_run_snapshot(run_id)
    except (RuntimeError, ValueError, KeyError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "assessment_binding_conflict", "message": "The run snapshot is unavailable."}) from None
    row = next((candidate for candidate in snapshot.items if candidate.item_id == item_id), None)
    stage = stages.get(run_id, item_id)
    if row is None or stage is None or not stage.output_handle or not stage.output_digest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "assessment_target_not_found", "message": "The exact run output is unavailable."})
    audit = outputs.get_audit(stage.output_handle)
    if (
        audit is None
        or (audit.evaluation_run_id, audit.evaluation_item_id) != (run_id, item_id)
        or audit.attempt_id != stage.output_handle
        or audit.output_digest != stage.output_digest
        or audit.scorer_version != configuration.scorer_version
        or audit.candidate_identity.corpus_version_id != run.corpus_version_id
        or audit.candidate_identity.configuration_version_id != run.configuration_version_id
        or audit.candidate_identity.scorer_version != configuration.scorer_version
        or audit.candidate_identity.installed_lock_sha256 != configuration.dependency_lock_sha256
        or audit.candidate_identity.artifact_id != configuration.integration_artifact
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "assessment_binding_conflict", "message": "The exact run output is stale."})
    output_text = outputs.get_raw(actor=admin, attempt_id=stage.output_handle)
    if not output_text:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "assessment_output_expired", "message": "The exact run output is no longer available for review."})
    binding = RunItemAssessment(
        generation_id=run.dataset_version_id,
        attempt_id=audit.attempt_id,
        candidate_id=audit.candidate_id,
        scorer_version=audit.scorer_version,
        output_digest=audit.output_digest,
    )
    record = repository.get(assessment_key(binding))
    return RunItemAssessmentProjection(
        run_id=run_id,
        item_id=item_id,
        generation_id=run.dataset_version_id,
        expected_refusal=row.expected_refusal,
        output_text=output_text,
        attempt_id=audit.attempt_id,
        candidate_id=audit.candidate_id,
        scorer_version=audit.scorer_version,
        output_digest=audit.output_digest,
        assessment=AssessmentDecisionProjection(
            assessment_id=record.assessment_id,
            claim_support=record.claim_support,
            citation_support=record.citation_support,
            rubric_passed=record.rubric_passed,
            refusal_judgment=record.refusal_judgment,
            reviewed_at=record.reviewed_at.isoformat(),
        ) if record else None,
    )


@router.put("/{run_id}/items/{item_id}/assessment", response_model=RunItemAssessmentRecord)
def assess_run_item(
    run_id: str,
    item_id: str,
    assessment: RunItemAssessment,
    admin: UserProfile = Depends(get_admin_user),
    runs: FirestoreEvaluationRunStore = Depends(evaluation_run_store),
    outputs: OutputStore = Depends(output_store),
    repository: AssessmentRepository = Depends(assessment_repository),
) -> RunItemAssessmentRecord:
    if assessment.attempt_id.strip() == "":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "invalid_assessment_attempt", "message": "An exact output attempt is required."})
    try:
        run = runs.get(run_id)
        if run is None:
            raise AssessmentNotFoundError("evaluation run was not found")
        if run.dataset_version_id != assessment.generation_id:
            raise AssessmentConflictError("assessment generation does not match the run")
        if item_id not in run.selected_item_ids:
            raise AssessmentConflictError("assessment item is not part of the pinned run")
        audit = outputs.get_audit(assessment.attempt_id)
        if audit is None:
            raise AssessmentNotFoundError("exact candidate output was not found")
        if (audit.evaluation_run_id, audit.evaluation_item_id) != (run_id, item_id):
            raise AssessmentConflictError("exact output is bound to another run item")
        if audit.candidate_id != assessment.candidate_id or audit.scorer_version != assessment.scorer_version:
            raise AssessmentConflictError("candidate or scorer does not match the exact output")
        if audit.output_digest != assessment.output_digest:
            raise AssessmentConflictError("output digest does not match the exact output")
        return repository.save(
            assessment,
            actor=admin,
            candidate_identity=audit.candidate_identity,
            reviewed_at=utc_now(),
        )
    except AssessmentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "assessment_target_not_found", "message": str(exc)}) from exc
    except AssessmentConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "assessment_binding_conflict", "message": str(exc)}) from exc
    except AssessmentError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "assessment_storage_failed", "message": str(exc)}) from exc
