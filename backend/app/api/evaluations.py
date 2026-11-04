"""Administrator-only evaluation launch and status endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from google.api_core.exceptions import GoogleAPICallError

from app.contracts import EvaluationRun, EvaluationRunSelection, UserProfile
from app.evaluation import (
    EvaluationCorpusReadinessError,
    EvaluationDispatchError,
    EvaluationGenerationError,
    EvaluationJobService,
    EvaluationRunError,
)
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
    except (
        EvaluationCorpusReadinessError,
        EvaluationDispatchError,
        EvaluationGenerationError,
        GoogleAPICallError,
        RuntimeError,
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_run_unavailable", "message": "The evaluation run could not be started."},
        ) from None
    except EvaluationRunError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_evaluation_run", "message": "The evaluation request is invalid or stale."},
        ) from None


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
