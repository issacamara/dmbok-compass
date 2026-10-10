"""Administrator-only, bounded evaluation dataset JSON imports."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Protocol
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.contracts import (
    EvaluationDataset,
    EvaluationImportResponse,
    EvaluationItem,
    ActiveEvaluationDataset,
    GoldAnnotation,
    ImportValidationError,
    ItemReview,
    MAX_EVALUATION_IMPORT_BYTES,
    MAX_EVALUATION_IMPORT_ITEMS,
    UserProfile,
    ReviewDecision,
)
from app.evaluation import ConcurrentActivationError, EvaluationGenerationError, ReviewStateError
from app.identity import get_admin_user

router = APIRouter(prefix="/api/admin/evaluation-datasets", tags=["evaluation datasets"])

_ALLOWED_CATEGORIES = frozenset({"definitions", "explanations", "comparisons", "study", "scenarios"})


class EvaluationGenerationStore(Protocol):
    def active(self) -> Any: ...

    def stage(self, dataset: EvaluationDataset) -> str: ...

    def activate(self, generation_id: str, *, expected_active_generation_id: str | None) -> Any: ...

    def active_dataset(self) -> EvaluationDataset | None: ...

    def review(self, generation_id: str, item_id: str, review_status: str) -> GoldAnnotation: ...


def evaluation_generation_store() -> EvaluationGenerationStore:
    """Configured in :mod:`app.main`; a default here is safe for test imports."""

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_generation_store_unavailable", "message": "Evaluation imports are not configured."},
    )


@router.post("/import", response_model=EvaluationImportResponse, status_code=status.HTTP_201_CREATED)
async def import_evaluation_dataset(
    request: Request,
    response: Response,
    _: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
) -> EvaluationImportResponse:
    """Validate each JSON record, then stage and atomically activate valid data."""

    raw = await request.body()
    if len(raw) > MAX_EVALUATION_IMPORT_BYTES:
        _reject(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "evaluation_import_too_large", "The evaluation import exceeds the allowed size.")

    try:
        submitted = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _reject(status.HTTP_400_BAD_REQUEST, "malformed_evaluation_import", "The evaluation import must contain valid JSON.")
    if not isinstance(submitted, list):
        _reject(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_evaluation_import", "The evaluation import must be a JSON array.")
    if len(submitted) > MAX_EVALUATION_IMPORT_ITEMS:
        _reject(status.HTTP_422_UNPROCESSABLE_ENTITY, "evaluation_import_too_many_items", "The evaluation import exceeds the allowed item count.")

    generation_id = f"generation-{uuid4().hex}"
    valid_items: list[EvaluationItem] = []
    valid_annotations: list[GoldAnnotation] = []
    errors: list[ImportValidationError] = []
    for index, record in enumerate(submitted):
        parsed, error = _validate_record(record, index, generation_id)
        if error is not None:
            errors.append(error)
        else:
            item, annotation = parsed
            valid_items.append(item)
            valid_annotations.append(annotation)

    if not valid_items:
        response.status_code = status.HTTP_200_OK
        return EvaluationImportResponse(
            status="rejected",
            submitted_count=len(submitted),
            imported_count=0,
            skipped_count=len(errors),
            validation_errors=errors,
        )

    dataset = EvaluationDataset(
        dataset_version_id=generation_id,
        items=valid_items,
        gold_annotations=valid_annotations,
    )
    try:
        active = store.active()
        expected_active_id = active.generation_id if active is not None else None
        store.stage(dataset)
        store.activate(generation_id, expected_active_generation_id=expected_active_id)
    except ConcurrentActivationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "evaluation_import_conflict", "message": "The active evaluation dataset changed; retry the import."},
        ) from exc
    except (EvaluationGenerationError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_import_storage_failed", "message": "The evaluation import could not be stored."},
        ) from exc

    return EvaluationImportResponse(
        status="accepted",
        generation_id=generation_id,
        submitted_count=len(submitted),
        imported_count=len(valid_items),
        skipped_count=len(errors),
        validation_errors=errors,
    )


@router.get("/active", response_model=ActiveEvaluationDataset)
def get_active_evaluation_dataset(
    _: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
) -> ActiveEvaluationDataset:
    """Expose only active-generation IDs, categories, and review state."""

    try:
        dataset = store.active_dataset()
    except (EvaluationGenerationError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_generation_store_failed", "message": "The active evaluation dataset is unavailable."},
        ) from exc
    if dataset is None:
        _reject(status.HTTP_404_NOT_FOUND, "no_active_evaluation_generation", "No active evaluation dataset exists.")

    reviews = [
        ItemReview(generation_id=dataset.dataset_version_id, item_id=item.item_id, review_status=annotation.review_status)
        for item in dataset.items
        for annotation in dataset.gold_annotations
        if annotation.item_id == item.item_id
    ]
    return ActiveEvaluationDataset(
        generation_id=dataset.dataset_version_id,
        item_count=len(dataset.items),
        approved_gold_count=sum(review.review_status == "approved" for review in reviews),
        item_reviews=reviews,
    )


@router.patch("/{generation_id}/items/{item_id}/review", response_model=ItemReview)
def review_active_evaluation_item(
    generation_id: str,
    item_id: str,
    decision: ReviewDecision,
    _: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
) -> ItemReview:
    """Make the single permitted candidate-to-final review transition."""

    try:
        annotation = store.review(generation_id, item_id, decision.review_status)
    except ReviewStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_evaluation_review_transition", "message": str(exc)},
        ) from exc
    except EvaluationGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "unknown_evaluation_review_target", "message": "The evaluation generation or item does not exist."},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_generation_store_failed", "message": "The review could not be stored."},
        ) from exc
    return ItemReview(generation_id=generation_id, item_id=item_id, review_status=annotation.review_status)


def _validate_record(
    record: object, index: int, generation_id: str
) -> tuple[tuple[EvaluationItem, GoldAnnotation], None] | tuple[None, ImportValidationError]:
    if not isinstance(record, dict):
        return None, _validation_error(index, "invalid_record", "Record must be an object.")
    question = record.get("question")
    if not _is_present_string(question):
        return None, _validation_error(index, "missing_question", "Record requires a question.")
    category = record.get("category")
    if category not in _ALLOWED_CATEGORIES:
        return None, _validation_error(index, "invalid_category", "Record requires an allowed category.")
    expected = record.get("expected_answer_or_rubric")
    if not _is_present_string(expected):
        return None, _validation_error(
            index, "missing_expected_answer_or_rubric", "Record requires an expected answer or rubric."
        )
    passage = record.get("supporting_passage")
    if not _is_present_string(passage):
        return None, _validation_error(index, "missing_supporting_passage", "Record requires a supporting passage.")

    item_id = sha256(f"{generation_id}:{index}".encode()).hexdigest()
    return (
        EvaluationItem(
            item_id=item_id,
            question=question.strip(),
            dataset_version_id=generation_id,
            category=category,
        ),
        GoldAnnotation(
            item_id=item_id,
            expected_answer_or_rubric=expected.strip(),
            relevant_chunk_ids=[f"imported-passage-{item_id}"],
            review_status="candidate",
            supporting_passage=passage.strip(),
        ),
    ), None


def _is_present_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validation_error(index: int, code: str, message: str) -> ImportValidationError:
    return ImportValidationError(index=index, code=code, message=message)  # type: ignore[arg-type]


def _reject(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})
