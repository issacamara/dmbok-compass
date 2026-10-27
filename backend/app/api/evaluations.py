"""Administrator-only evaluation launch and status endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from app.contracts import EvaluationRun, EvaluationRunSelection, UserProfile
from app.evaluation import EvaluationJobService, EvaluationRunError
from app.identity import get_admin_user

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


class EvaluationLaunchRequest(EvaluationRunSelection):
    """The active dataset is resolved on the server, never selected by the client."""


def evaluation_service() -> EvaluationJobService:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_service_unavailable", "message": "Evaluation jobs are not configured."},
    )


@router.post("", response_model=EvaluationRun, status_code=status.HTTP_202_ACCEPTED)
def launch_evaluation(
    request: EvaluationLaunchRequest,
    _: UserProfile = Depends(get_admin_user),
    service: EvaluationJobService = Depends(evaluation_service),
) -> EvaluationRun:
    try:
        return service.launch(
            item_ids=request.item_ids,
        )
    except EvaluationRunError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "invalid_evaluation_run", "message": str(exc)}) from exc


@router.get("/{run_id}", response_model=EvaluationRun)
def get_evaluation_status(
    run_id: str,
    _: UserProfile = Depends(get_admin_user),
    service: EvaluationJobService = Depends(evaluation_service),
) -> EvaluationRun:
    try:
        return service.status(run_id)
    except EvaluationRunError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "evaluation_run_not_found", "message": str(exc)}) from exc
