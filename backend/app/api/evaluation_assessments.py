"""Administrator-only exact-run human assessment endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.contracts import UserProfile
from app.evaluation.candidate_outputs import FirestoreCandidateOutputStore
from app.evaluation.job import FirestoreEvaluationRunStore
from app.evaluation.reviews import (
    AssessmentConflictError,
    AssessmentError,
    AssessmentRepository,
    AssessmentNotFoundError,
    FirestoreAssessmentRepository,
    OutputStore,
    RunItemAssessment,
    RunItemAssessmentRecord,
    utc_now,
)
from app.identity import get_admin_user

router = APIRouter(prefix="/api/admin/evaluations", tags=["evaluation assessments"])


def assessment_repository() -> AssessmentRepository:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "assessment_repository_unavailable", "message": "Assessment storage is not configured."})


def output_store() -> OutputStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_output_store_unavailable", "message": "Evaluation output storage is not configured."})


def evaluation_run_store() -> FirestoreEvaluationRunStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_run_store_unavailable", "message": "Evaluation run storage is not configured."})


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
