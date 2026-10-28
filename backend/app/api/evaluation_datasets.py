"""Administrator-only, bounded evaluation dataset JSON imports."""

from __future__ import annotations

from hashlib import sha256
import json
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, model_validator

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
    DocumentChunk,
)
from app.evaluation import (
    ConcurrentActivationError,
    EvaluationGenerationError,
    EvaluationJobService,
    EvaluationRunError,
    ReviewStateError,
)
from app.api.evaluations import evaluation_service
from app.evaluation.dataset import (
    CandidateBindingError,
    CandidateChunkLookup,
    CandidateGoldBinding,
    CandidateGoldReview,
    GoldAnnotationRemap,
)
from app.identity import get_admin_user
from app.corpus.selection import uploaded_corpus_version_id

router = APIRouter(prefix="/api/admin/evaluation-datasets", tags=["evaluation datasets"])

_ALLOWED_CATEGORIES = frozenset({"definitions", "explanations", "comparisons", "study", "scenarios"})
_ALLOWED_RECORD_FIELDS = frozenset(
    {"question", "category", "expected_answer_or_rubric", "supporting_passage"}
)


class EvaluationUploadRunResponse(EvaluationImportResponse):
    """Import summary plus the asynchronously dispatched run identity."""

    run_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_run_id(self) -> "EvaluationUploadRunResponse":
        if self.status == "accepted" and self.run_id is None:
            raise ValueError("accepted uploads require a run_id")
        if self.status == "rejected" and self.run_id is not None:
            raise ValueError("rejected uploads cannot have a run_id")
        return self


class EvaluationGenerationStore(Protocol):
    def active(self) -> Any: ...

    def stage(self, dataset: EvaluationDataset) -> str: ...

    def activate(self, generation_id: str, *, expected_active_generation_id: str | None) -> Any: ...

    def active_dataset(self) -> EvaluationDataset | None: ...

    def review(self, generation_id: str, item_id: str, review_status: str) -> GoldAnnotation: ...

    def bind_candidate_review(
        self, generation_id: str, item_id: str, candidate_review: CandidateGoldReview
    ) -> GoldAnnotation: ...


class CandidateBindingPayload(BaseModel):
    candidate_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_provenance_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CandidateChunkBinding(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class CandidateBindingRequest(BaseModel):
    """Admin-selected support for one item in one immutable candidate corpus."""

    binding: CandidateBindingPayload | None = None
    candidate_snapshot_sha256: str | None = None
    source_provenance_sha256: str | None = None
    reviewed_chunk_ids: list[str] = Field(default_factory=list, max_length=5)
    reviewed_chunk_hashes: dict[str, str] = Field(default_factory=dict, max_length=5)
    reviewed_chunks: list[CandidateChunkBinding] = Field(default_factory=list, max_length=5)
    expected_refusal: bool = False
    binding_version: str = Field(default="candidate-binding-v1", min_length=1, max_length=64)

    @model_validator(mode="after")
    def normalise(self) -> "CandidateBindingRequest":
        binding = self.binding
        if binding is not None:
            values = (self.candidate_snapshot_sha256, self.source_provenance_sha256)
            if any(value is not None for value in values):
                raise ValueError("binding must be supplied either nested or at the top level")
            self.candidate_snapshot_sha256 = binding.candidate_snapshot_sha256
            self.source_provenance_sha256 = binding.source_provenance_sha256
        if not all((self.candidate_snapshot_sha256, self.source_provenance_sha256)):
            raise ValueError("candidate binding is required")
        listed = {chunk.chunk_id: chunk.content_hash for chunk in self.reviewed_chunks}
        if listed and self.reviewed_chunk_hashes and listed != self.reviewed_chunk_hashes:
            raise ValueError("reviewed chunk hashes disagree")
        if listed:
            self.reviewed_chunk_hashes = listed
        if self.reviewed_chunk_ids and set(self.reviewed_chunk_ids) != set(self.reviewed_chunk_hashes):
            raise ValueError("reviewed chunk IDs and hashes must match")
        self.reviewed_chunk_ids = list(self.reviewed_chunk_hashes)
        if self.expected_refusal and self.reviewed_chunk_ids:
            raise ValueError("expected-refusal items cannot bind relevant chunks")
        if not self.expected_refusal and not self.reviewed_chunk_ids:
            raise ValueError("answerable items require relevant chunks")
        if any(chunk_id.startswith("imported-passage-") for chunk_id in self.reviewed_chunk_ids):
            raise ValueError("placeholder passage IDs cannot be bound")
        return self


def evaluation_generation_store() -> EvaluationGenerationStore:
    """Configured in :mod:`app.main`; a default here is safe for test imports."""

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_generation_store_unavailable", "message": "Evaluation imports are not configured."},
    )


def candidate_chunk_lookup(
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
) -> CandidateChunkLookup:
    lookup = getattr(store, "get_candidate_chunk", None)
    if lookup is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "candidate_chunk_lookup_unavailable", "message": "Candidate chunk lookup is not configured."},
        )

    class _StoreLookup:
        def get(self, *, corpus_version_id: str, chunk_id: str) -> DocumentChunk | None:
            return lookup(corpus_version_id=corpus_version_id, chunk_id=chunk_id)

    return _StoreLookup()


@router.post("/upload-and-run", response_model=EvaluationUploadRunResponse, status_code=status.HTTP_201_CREATED)
@router.post("/import", response_model=EvaluationUploadRunResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_run_evaluation(
    request: Request,
    response: Response,
    _: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
    service: EvaluationJobService = Depends(evaluation_service),
    _uploaded_corpus_version_id: str = Depends(uploaded_corpus_version_id),
) -> EvaluationUploadRunResponse:
    """Validate, activate, and dispatch one full evaluation run atomically."""

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
        return EvaluationUploadRunResponse(
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

    try:
        run = service.launch(item_ids=None)
    except (EvaluationRunError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_run_unavailable", "message": "The evaluation run could not be started."},
        ) from exc

    return EvaluationUploadRunResponse(
        status="accepted",
        generation_id=generation_id,
        run_id=run.run_id,
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


@router.put("/{generation_id}/items/{item_id}/binding")
def bind_candidate_evaluation_item(
    generation_id: str,
    item_id: str,
    request: CandidateBindingRequest,
    admin: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
    chunks: CandidateChunkLookup = Depends(candidate_chunk_lookup),
    corpus_version_id: str = Depends(uploaded_corpus_version_id),
) -> dict[str, object]:
    """Persist reviewed real chunk support without approving the annotation."""

    binding = CandidateGoldBinding(
        corpus_version_id=corpus_version_id,
        candidate_snapshot_sha256=request.candidate_snapshot_sha256 or "",
        source_provenance_sha256=request.source_provenance_sha256 or "",
    )
    resolved: list[DocumentChunk] = []
    for chunk_id in request.reviewed_chunk_ids:
        chunk = chunks.get(corpus_version_id=binding.corpus_version_id, chunk_id=chunk_id)
        if chunk is None:
            _reject(status.HTTP_404_NOT_FOUND, "candidate_chunk_not_found", "The selected candidate chunk does not exist.")
        if chunk.corpus_version_id != binding.corpus_version_id:
            _reject(status.HTTP_409_CONFLICT, "candidate_chunk_wrong_corpus", "The selected chunk is outside the pinned corpus.")
        if chunk.content_hash != request.reviewed_chunk_hashes[chunk_id]:
            _reject(status.HTTP_409_CONFLICT, "candidate_chunk_hash_changed", "The selected candidate chunk changed; refresh the binding.")
        resolved.append(chunk)

    first = resolved[0] if len(resolved) == 1 else None
    remap = GoldAnnotationRemap(
        item_id=item_id,
        relevant_chunk_ids=tuple(request.reviewed_chunk_ids),
        remap_status="refusal" if request.expected_refusal else "matched",
        page=first.page if first else None,
        section=first.section if first else None,
        section_origin="explicit" if first else None,
        review_reasons=("expected_refusal",) if request.expected_refusal else ("reviewed_candidate_chunks",),
        content_hash=first.content_hash if first else None,
    )
    reviewed_at = datetime.now(timezone.utc).isoformat()
    candidate_review = CandidateGoldReview(
        binding=binding,
        remaps=(remap,),
        binding_version=request.binding_version,
        expected_refusal=request.expected_refusal,
        reviewed_by=admin.user_id,
        reviewed_at=reviewed_at,
        chunk_hashes=dict(request.reviewed_chunk_hashes),
    )
    try:
        annotation = store.bind_candidate_review(generation_id, item_id, candidate_review)
    except ReviewStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_evaluation_binding_transition", "message": str(exc)},
        ) from exc
    except (CandidateBindingError, EvaluationGenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_evaluation_binding", "message": str(exc)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "evaluation_binding_store_failed", "message": "The candidate binding could not be stored."},
        ) from exc
    return {
        "generation_id": generation_id,
        "item_id": item_id,
        "review_status": annotation.review_status,
        "binding_version": request.binding_version,
        "expected_refusal": request.expected_refusal,
        "reviewed_chunk_ids": list(request.reviewed_chunk_ids),
        "reviewed_by": admin.user_id,
        "reviewed_at": reviewed_at,
    }


def _validate_record(
    record: object, index: int, generation_id: str
) -> tuple[tuple[EvaluationItem, GoldAnnotation], None] | tuple[None, ImportValidationError]:
    if not isinstance(record, dict):
        return None, _validation_error(index, "invalid_record", "Record must be an object.")
    if set(record) - _ALLOWED_RECORD_FIELDS:
        return None, _validation_error(index, "invalid_record", "Record contains unsupported fields.")
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
