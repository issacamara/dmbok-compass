"""Administrator-only, bounded evaluation dataset JSON imports."""

from __future__ import annotations

from hashlib import sha256
import json
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
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
    DatasetHead,
    DatasetExclusion,
    EvaluationRowV2,
)
from app.evaluation import (
    ConcurrentActivationError,
    EvaluationGenerationError,
    EvaluationJobService,
    EvaluationDispatchError,
    EvaluationCorpusReadinessError,
    EvaluationRunError,
    ReviewStateError,
    GenerationIntegrityError,
)
from app.api.evaluations import evaluation_service
from app.evaluation.dataset import (
    CandidateBindingError,
    CandidateChunkLookup,
    CandidateGoldBinding,
    CandidateGoldReview,
    GoldAnnotationRemap,
)
from app.evaluation.legacy_import_binding import bind_legacy_imports
from app.evaluation.candidate_generation import is_near_duplicate_question, normalized_question_key
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
    row_v2_head: DatasetHead | None = None
    binding_errors: list["ImportBindingValidationError"] = Field(default_factory=list, max_length=MAX_EVALUATION_IMPORT_ITEMS)

    @model_validator(mode="after")
    def validate_run_id(self) -> "EvaluationUploadRunResponse":
        if self.status == "accepted" and self.run_id is None:
            raise ValueError("accepted uploads require a run_id")
        if self.status == "rejected" and self.run_id is not None:
            raise ValueError("rejected uploads cannot have a run_id")
        return self


class ImportBindingValidationError(BaseModel):
    """Content-free indexed guidance for a legacy passage that cannot be bound."""

    index: int = Field(ge=0)
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=500)


class EvaluationGenerationStore(Protocol):
    def active(self) -> Any: ...

    def stage(self, dataset: EvaluationDataset) -> str: ...

    def activate(self, generation_id: str, *, expected_active_generation_id: str | None) -> Any: ...

    def active_dataset(self) -> EvaluationDataset | None: ...

    def review(self, generation_id: str, item_id: str, review_status: str) -> GoldAnnotation: ...

    def bind_candidate_review(
        self, generation_id: str, item_id: str, candidate_review: CandidateGoldReview
    ) -> GoldAnnotation: ...

    def corpus_head(self, corpus_version_id: str) -> DatasetHead | None: ...

    def get_revision(self, head: DatasetHead) -> tuple[EvaluationRowV2, ...]: ...

    def stage_revision(
        self, corpus_version_id: str, revision: int, rows: tuple[EvaluationRowV2, ...]
    ) -> DatasetHead: ...

    def row_mutation_head(self, operation_key: str, *, fingerprint: str) -> DatasetHead | None: ...

    def publish_row_mutation(
        self, head: DatasetHead, *, expected_revision: int, operation_key: str,
        fingerprint: str, actor_id: str, mutation: str, item_id: str,
        exclusion: DatasetExclusion | None = None,
    ) -> DatasetHead: ...

    def get_parse_derived_records(
        self, *, corpus_version_id: str
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None: ...

    def get_dataset_exclusion_questions(self) -> tuple[tuple[DatasetExclusion, str], ...]: ...


class DatasetRowAdditionRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    operation_key: str = Field(min_length=1, max_length=256)
    question: str = Field(min_length=1, max_length=2000)
    category: Literal["definitions", "explanations", "comparisons", "study", "scenarios"]
    expected_outcome: Literal["answer", "qualified", "evidence_refusal"]
    expected_answer_or_rubric: str = Field(min_length=1, max_length=10000)
    supporting_passage: str | None = Field(default=None, min_length=1, max_length=20000)
    scope_rationale: str | None = Field(default=None, min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_support(self) -> "DatasetRowAdditionRequest":
        if not self.operation_key.strip():
            raise ValueError("operation key cannot be blank")
        if self.expected_outcome == "evidence_refusal":
            if not self.scope_rationale or self.supporting_passage is not None:
                raise ValueError("refusal additions require a scope rationale and no supporting passage")
        elif not self.supporting_passage:
            raise ValueError("answerable additions require a supporting passage")
        return self


class DatasetRowRemovalRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    operation_key: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_operation_key(self) -> "DatasetRowRemovalRequest":
        if not self.operation_key.strip():
            raise ValueError("operation key cannot be blank")
        return self


class DatasetRowMutationResponse(BaseModel):
    head: DatasetHead
    operation_key: str


class EvaluationRowV2Projection(BaseModel):
    """Browser-safe row details; support bindings remain server-owned."""

    schema_version: Literal["evaluation-row-v2"]
    item_id: str
    origin: Literal["generated", "admin", "imported"]
    question: str
    category: Literal["definitions", "explanations", "comparisons", "study", "scenarios"]
    expected_outcome: Literal["answer", "qualified", "evidence_refusal"]
    expected_answer_or_rubric: str
    expected_refusal: bool
    scope_rationale: str | None


class EvaluationDatasetRevisionResponse(BaseModel):
    """Current versioned dataset content without storage or evidence selectors."""

    revision: int = Field(ge=1)
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rows: list[EvaluationRowV2Projection] = Field(max_length=MAX_EVALUATION_IMPORT_ITEMS)


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
    valid_imports: list[tuple[int, EvaluationItem, GoldAnnotation]] = []
    errors: list[ImportValidationError] = []
    for index, record in enumerate(submitted):
        parsed, error = _validate_record(record, index, generation_id)
        if error is not None:
            errors.append(error)
        else:
            item, annotation = parsed
            valid_items.append(item)
            valid_annotations.append(annotation)
            valid_imports.append((index, item, annotation))

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

    # Keep the legacy exploratory dataset intact. Only uniquely matched rows
    # enter the corpus-scoped row-v2 publication path.
    binding_errors: list[ImportBindingValidationError] = []
    row_v2_head: DatasetHead | None = None
    bound_rows = ()
    record_loader = getattr(store, "get_parse_derived_records", None)
    if not callable(record_loader):
        binding_errors = [ImportBindingValidationError(
            index=index,
            code="parse_evidence_unavailable",
            message="Verified parse-derived evidence is unavailable for this selected corpus.",
        ) for index, _, _ in valid_imports]
    else:
        try:
            parsed_records = record_loader(corpus_version_id=_uploaded_corpus_version_id)
        except Exception:
            parsed_records = None
        if parsed_records is None:
            binding_errors = [ImportBindingValidationError(
                index=index,
                code="parse_evidence_unavailable",
                message="Verified parse-derived evidence is unavailable for this selected corpus.",
            ) for index, _, _ in valid_imports]
        else:
            parents, children = parsed_records
            import_rows = tuple(
                (index, item.item_id, item.question, item.category, annotation.expected_answer_or_rubric,
                 annotation.supporting_passage or "")
                for index, item, annotation in valid_imports
            )
            bound, unresolved = bind_legacy_imports(
                import_rows, parents=tuple(parents), children=tuple(children)
            )
            binding_errors = [
                ImportBindingValidationError(index=issue.index, code=issue.code, message=issue.message)
                for issue in unresolved
            ]
            bound_rows = bound
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

    # Publish v2 only after the compatible legacy import has committed, so a
    # failed legacy write cannot leave a new v2 head behind.
    if bound_rows:
        revision_methods = (
            getattr(store, "corpus_head", None),
            getattr(store, "stage_revision", None),
            getattr(store, "activate_revision", None),
        )
        if not all(callable(method) for method in revision_methods):
            binding_errors.extend(
                ImportBindingValidationError(
                    index=entry.index,
                    code="row_v2_publication_unavailable",
                    message="The row-v2 revision store is unavailable; retry these indexed rows.",
                )
                for entry in bound_rows
            )
        else:
            try:
                previous = revision_methods[0](_uploaded_corpus_version_id)
                next_revision = previous.revision + 1 if previous is not None else 1
                staged_head = revision_methods[1](
                    _uploaded_corpus_version_id,
                    next_revision,
                    tuple(entry.row for entry in bound_rows),
                )
                row_v2_head = revision_methods[2](
                    staged_head,
                    expected_revision=previous.revision if previous is not None else None,
                )
            except ConcurrentActivationError:
                row_v2_head = None
                binding_errors.extend(
                    ImportBindingValidationError(
                        index=entry.index,
                        code="row_v2_revision_conflict",
                        message="The row-v2 revision changed during import; retry these indexed rows.",
                    )
                    for entry in bound_rows
                )
            except Exception:
                row_v2_head = None
                binding_errors.extend(
                    ImportBindingValidationError(
                        index=entry.index,
                        code="row_v2_publication_failed",
                        message="The row-v2 revision could not be saved; retry these indexed rows.",
                    )
                    for entry in bound_rows
                )

    try:
        run = service.launch(item_ids=None)
    except EvaluationDispatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "evaluation_dispatch_failed",
                "message": "The evaluation run could not be started.",
                "provider_status": exc.status_code,
                "provider_code": exc.provider_code,
                "provider_message": exc.provider_message,
            },
        ) from exc
    except EvaluationCorpusReadinessError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "evaluation_corpus_not_ready",
                "message": "The evaluation corpus is not ready for provenance-bound evaluation.",
            },
        ) from exc
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
        row_v2_head=row_v2_head,
        binding_errors=binding_errors,
    )


@router.post("/rows", response_model=DatasetRowMutationResponse, status_code=status.HTTP_201_CREATED)
def add_evaluation_dataset_row(
    request: DatasetRowAdditionRequest,
    response: Response,
    admin: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
    corpus_version_id: str = Depends(uploaded_corpus_version_id),
) -> DatasetRowMutationResponse:
    """Bind and add one admin-authored row to a new corpus-scoped revision."""
    fingerprint = _row_operation_fingerprint("add", corpus_version_id, request.model_dump(mode="json"))
    try:
        replay = store.row_mutation_head(request.operation_key, fingerprint=fingerprint)
        if replay is not None:
            response.status_code = status.HTTP_200_OK
            return DatasetRowMutationResponse(head=replay, operation_key=request.operation_key)
        head = store.corpus_head(corpus_version_id)
        _require_expected_revision(head, request.expected_revision)
        current_rows = store.get_revision(head)
        exclusions = store.get_dataset_exclusion_questions()
        exclusion_questions = tuple(question for _, question in exclusions)
        question_key = normalized_question_key(request.question)
        exclusion_digest = sha256(question_key.encode("utf-8")).hexdigest()
        if is_near_duplicate_question(request.question, tuple(row.question for row in current_rows)):
            _reject(status.HTTP_409_CONFLICT, "duplicate_evaluation_question", "A matching or near-duplicate question already exists.")
        if exclusion_digest in {exclusion.normalized_question_sha256 for exclusion, _ in exclusions} or is_near_duplicate_question(
            request.question, exclusion_questions
        ):
            _reject(status.HTTP_409_CONFLICT, "excluded_evaluation_question", "A removed or near-duplicate question is excluded from future selections.")
        records = store.get_parse_derived_records(corpus_version_id=corpus_version_id)
        if records is None:
            _reject(status.HTTP_409_CONFLICT, "evaluation_corpus_not_ready", "The selected corpus is not ready for row support validation.")
        parents, children = records
        item_id = "admin:" + sha256(f"{corpus_version_id}\0{question_key}".encode("utf-8")).hexdigest()[:32]
        if request.expected_outcome == "evidence_refusal":
            row = EvaluationRowV2(
                item_id=item_id, origin="admin", question=request.question, category=request.category,
                expected_outcome=request.expected_outcome,
                expected_answer_or_rubric=request.expected_answer_or_rubric,
                expected_refusal=True, scope_rationale=request.scope_rationale,
            )
        else:
            bound, issues = bind_legacy_imports(
                ((0, item_id, request.question, request.category, request.expected_answer_or_rubric,
                  request.supporting_passage or ""),),
                parents=tuple(parents), children=tuple(children),
            )
            if issues or len(bound) != 1:
                _reject(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_evaluation_row_support", "The supporting passage must uniquely match the selected corpus.")
            row = EvaluationRowV2.model_validate({
                **bound[0].row.model_dump(mode="json"),
                "origin": "admin", "expected_outcome": request.expected_outcome,
            })
        staged = store.stage_revision(corpus_version_id, head.revision + 1, (*current_rows, row))
        published = store.publish_row_mutation(
            staged, expected_revision=request.expected_revision, operation_key=request.operation_key,
            fingerprint=fingerprint, actor_id=admin.user_id, mutation="add", item_id=item_id,
        )
    except ConcurrentActivationError:
        _reject(status.HTTP_409_CONFLICT, "dataset_revision_conflict", "The evaluation dataset changed; refresh it and retry.")
    except GenerationIntegrityError:
        _reject(status.HTTP_409_CONFLICT, "row_operation_conflict", "The operation key is already bound to a different row edit.")
    except (EvaluationGenerationError, RuntimeError):
        _reject(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_dataset_mutation_failed", "The evaluation row could not be added.")
    return DatasetRowMutationResponse(head=published, operation_key=request.operation_key)


@router.post("/rows/{item_id}/remove", response_model=DatasetRowMutationResponse)
def remove_evaluation_dataset_row(
    item_id: str,
    request: DatasetRowRemovalRequest,
    admin: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
    corpus_version_id: str = Depends(uploaded_corpus_version_id),
) -> DatasetRowMutationResponse:
    """Remove one row by publishing a revision and an auditable exclusion."""
    fingerprint = _row_operation_fingerprint(
        "remove", corpus_version_id,
        {"expected_revision": request.expected_revision, "item_id": item_id},
    )
    try:
        replay = store.row_mutation_head(request.operation_key, fingerprint=fingerprint)
        if replay is not None:
            return DatasetRowMutationResponse(head=replay, operation_key=request.operation_key)
        head = store.corpus_head(corpus_version_id)
        _require_expected_revision(head, request.expected_revision)
        current_rows = store.get_revision(head)
        row = next((candidate for candidate in current_rows if candidate.item_id == item_id), None)
        if row is None:
            _reject(status.HTTP_404_NOT_FOUND, "evaluation_row_not_found", "The selected evaluation row no longer exists.")
        if len(current_rows) == 1:
            _reject(status.HTTP_409_CONFLICT, "evaluation_dataset_cannot_be_empty", "At least one row must remain in the dataset.")
        key = normalized_question_key(row.question)
        exclusion = DatasetExclusion(
            corpus_version_id=corpus_version_id, item_id=row.item_id, lineage_id=row.item_id,
            normalized_question_sha256=sha256(key.encode("utf-8")).hexdigest(),
            revision=head.revision + 1, actor_id=admin.user_id, excluded_at=datetime.now(timezone.utc),
        )
        staged = store.stage_revision(
            corpus_version_id, head.revision + 1,
            tuple(candidate for candidate in current_rows if candidate.item_id != item_id),
        )
        published = store.publish_row_mutation(
            staged, expected_revision=request.expected_revision, operation_key=request.operation_key,
            fingerprint=fingerprint, actor_id=admin.user_id, mutation="remove", item_id=item_id,
            exclusion=exclusion,
        )
    except ConcurrentActivationError:
        _reject(status.HTTP_409_CONFLICT, "dataset_revision_conflict", "The evaluation dataset changed; refresh it and retry.")
    except GenerationIntegrityError:
        _reject(status.HTTP_409_CONFLICT, "row_operation_conflict", "The operation key is already bound to a different row edit.")
    except (EvaluationGenerationError, RuntimeError):
        _reject(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_dataset_mutation_failed", "The evaluation row could not be removed.")
    return DatasetRowMutationResponse(head=published, operation_key=request.operation_key)


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


@router.get("/revision", response_model=EvaluationDatasetRevisionResponse)
def get_evaluation_dataset_revision(
    _: UserProfile = Depends(get_admin_user),
    store: EvaluationGenerationStore = Depends(evaluation_generation_store),
    corpus_version_id: str = Depends(uploaded_corpus_version_id),
) -> EvaluationDatasetRevisionResponse:
    """Read the current dataset rows without exposing their support bindings."""
    try:
        head = store.corpus_head(corpus_version_id)
        if head is None:
            _reject(status.HTTP_404_NOT_FOUND, "evaluation_dataset_revision_not_found", "No evaluation dataset revision is available.")
        rows = store.get_revision(head)
    except (EvaluationGenerationError, RuntimeError):
        _reject(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_dataset_revision_unavailable", "The evaluation dataset revision is unavailable.")
    return EvaluationDatasetRevisionResponse(
        revision=head.revision,
        dataset_sha256=head.dataset_sha256,
        rows=[
            EvaluationRowV2Projection(
                schema_version=row.schema_version,
                item_id=row.item_id,
                origin=row.origin,
                question=row.question,
                category=row.category,
                expected_outcome=row.expected_outcome,
                expected_answer_or_rubric=row.expected_answer_or_rubric,
                expected_refusal=row.expected_refusal,
                scope_rationale=row.scope_rationale,
            )
            for row in rows
        ],
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


def _row_operation_fingerprint(mutation: str, corpus_version_id: str, request: dict[str, Any]) -> str:
    payload = {"mutation": mutation, "corpus_version_id": corpus_version_id, "request": request}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _require_expected_revision(head: DatasetHead | None, expected_revision: int) -> None:
    if head is None or head.revision != expected_revision:
        _reject(status.HTTP_409_CONFLICT, "dataset_revision_conflict", "The evaluation dataset changed; refresh it and retry.")
