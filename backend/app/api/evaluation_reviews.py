"""Admin-only browser projection and attestation for batch evidence reviews."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, status

from app.contracts import (
    BatchAttestationRequest,
    BatchAttestationResult,
    EvaluationDataset,
    EvaluationRun,
    ExceptionRecord,
    RepresentativeSampleEntry,
    ReviewDecisionMetadata,
    ReviewProjection,
    ReviewSummary,
    UserProfile,
)
from app.evaluation.batch_reviews import (
    BatchReview,
    BatchReviewConflictError,
    BatchReviewError,
    BatchReviewRepository,
    BatchReviewStaleError,
    DurableBatchAttestation,
    create_batch_review,
    NoEligibleCandidatesError,
)
from app.evaluation.job import EvaluationRunStore
from app.evaluation.support import (
    FirestoreRunPinnedChunkSource,
    SupportEvidenceStore,
    StoredSupportChunk,
)
from app.identity import get_admin_user


router = APIRouter(prefix="/api/admin/evaluations", tags=["evaluation reviews"])


class EvaluationGenerationReviewStore(Protocol):
    def active(self): ...

    def active_dataset(self) -> EvaluationDataset | None: ...


def evaluation_run_store() -> EvaluationRunStore:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_run_store_unavailable", "message": "Evaluation run storage is not configured."},
    )


def evaluation_generation_store() -> EvaluationGenerationReviewStore:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_generation_store_unavailable", "message": "Evaluation generation storage is not configured."},
    )


def support_evidence_store() -> SupportEvidenceStore:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "support_evidence_store_unavailable", "message": "Evaluation support storage is not configured."},
    )


def pinned_chunk_source() -> FirestoreRunPinnedChunkSource:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "support_chunk_source_unavailable", "message": "Pinned corpus support lookup is not configured."},
    )


def batch_review_repository() -> BatchReviewRepository:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "batch_review_repository_unavailable", "message": "Batch review storage is not configured."},
    )


def _run(run_id: str, runs: EvaluationRunStore) -> EvaluationRun:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "evaluation_run_not_found", "message": "The evaluation run was not found."},
        )
    if run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "evaluation_run_not_ready", "message": "The evaluation run is not ready for evidence review."},
        )
    if run.candidate_snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "stale_evaluation_run", "message": "The evaluation run has no immutable candidate snapshot."},
        )
    return run


def _current_dataset(
    run: EvaluationRun,
    generations: EvaluationGenerationReviewStore,
) -> EvaluationDataset:
    active = generations.active()
    if active is None or active.generation_id != run.dataset_version_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "stale_evaluation_run", "message": "The evaluation run is not the current dataset generation."},
        )
    dataset = generations.active_dataset()
    if dataset is None or dataset.dataset_version_id != run.dataset_version_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "stale_evaluation_run", "message": "The current evaluation dataset could not be verified."},
        )
    return dataset


def _review(
    run: EvaluationRun,
    dataset: EvaluationDataset,
    evidence_store: SupportEvidenceStore,
    repository: BatchReviewRepository,
) -> BatchReview:
    evidence = []
    for item_id in run.selected_item_ids:
        item = evidence_store.get_evidence(run_id=run.run_id, item_id=item_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "incomplete_batch_review", "message": "The run does not have complete review evidence."},
            )
        evidence.append(item)
    try:
        return repository.create_or_get(create_batch_review(run, dataset, evidence))
    except NoEligibleCandidatesError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "no_eligible_review_candidates", "message": "The run has no eligible evidence candidate for review."},
        ) from exc
    except BatchReviewStaleError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "stale_evaluation_run", "message": "The run-bound review evidence is stale."},
        ) from exc
    except BatchReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "invalid_batch_review", "message": "The run-bound review evidence could not be verified."},
        ) from exc


def _annotation_map(dataset: EvaluationDataset):
    return {annotation.item_id: annotation for annotation in dataset.gold_annotations}


def _item_map(dataset: EvaluationDataset):
    return {item.item_id: item for item in dataset.items}


def _safe_excerpt(value: str | None) -> str | None:
    if not value:
        return None
    return value[:5000]


def _chunk_for(
    evidence,
    source: FirestoreRunPinnedChunkSource,
) -> StoredSupportChunk | None:
    selected = evidence.selected_chunk
    if selected is None:
        return None
    return source.chunk(corpus_version_id=selected.corpus_version_id, chunk_id=selected.chunk_id)


def _decision_metadata(attestation: DurableBatchAttestation) -> ReviewDecisionMetadata:
    return ReviewDecisionMetadata(
        decision=attestation.decision,
        actor=attestation.actor,
        decided_at=attestation.decided_at,
        decision_version=attestation.decision_version,
    )


def _projection(
    review: BatchReview,
    dataset: EvaluationDataset,
    evidence_store: SupportEvidenceStore,
    source: FirestoreRunPinnedChunkSource,
    repository: BatchReviewRepository,
) -> tuple[ReviewSummary, list[RepresentativeSampleEntry], list[ExceptionRecord]]:
    items = _item_map(dataset)
    annotations = _annotation_map(dataset)
    attestation = getattr(repository, "attestation", lambda _: None)(review)
    decision = _decision_metadata(attestation) if attestation is not None else None
    status_value = attestation.decision if attestation is not None else review.summary.status
    summary = ReviewSummary(
        status=status_value,
        eligible_count=review.summary.eligible_count,
        exception_count=review.summary.exception_count,
        sampled_count=len(review.samples),
        attested_count=review.summary.eligible_count if attestation and attestation.decision == "attested" else 0,
        decision=decision,
    )
    samples: list[RepresentativeSampleEntry] = []
    for sample in review.samples:
        item = items.get(sample.item_id)
        annotation = annotations.get(sample.item_id)
        evidence = evidence_store.get_evidence(run_id=review.summary.run_id, item_id=sample.item_id)
        if item is None or annotation is None or evidence is None:
            raise HTTPException(status_code=409, detail={"code": "incomplete_batch_review", "message": "The review projection is incomplete."})
        chunk = _chunk_for(evidence, source)
        samples.append(
            RepresentativeSampleEntry(
                item_id=item.item_id,
                question=item.question,
                rubric=annotation.expected_answer_or_rubric,
                support_excerpt=_safe_excerpt(annotation.supporting_passage) or "Verified support excerpt unavailable.",
                page=chunk.chunk.page if chunk and chunk.chunk else evidence.selected_chunk.page,
                section=chunk.chunk.section if chunk and chunk.chunk else evidence.selected_chunk.section,
                status=status_value,
                decision=decision,
            )
        )
    exceptions: list[ExceptionRecord] = []
    for record in review.exceptions:
        item = items.get(record.item_id)
        annotation = annotations.get(record.item_id)
        evidence = evidence_store.get_evidence(run_id=review.summary.run_id, item_id=record.item_id)
        if item is None or annotation is None:
            raise HTTPException(status_code=409, detail={"code": "incomplete_batch_review", "message": "The review projection is incomplete."})
        exceptions.append(
            ExceptionRecord(
                item_id=record.item_id,
                question=item.question,
                rubric=annotation.expected_answer_or_rubric,
                support_excerpt=_safe_excerpt(annotation.supporting_passage),
                page=evidence.selected_chunk.page if evidence and evidence.selected_chunk else None,
                section=evidence.selected_chunk.section if evidence and evidence.selected_chunk else None,
                status=record.status,
                reason=record.reason,
                decision=decision,
            )
        )
    return summary, samples, exceptions


@router.get("/{run_id}/review-summary", response_model=ReviewProjection)
def get_review_summary(
    run_id: str,
    _: UserProfile = Depends(get_admin_user),
    runs: EvaluationRunStore = Depends(evaluation_run_store),
    generations: EvaluationGenerationReviewStore = Depends(evaluation_generation_store),
    evidence_store: SupportEvidenceStore = Depends(support_evidence_store),
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
    repository: BatchReviewRepository = Depends(batch_review_repository),
) -> ReviewProjection:
    run = _run(run_id, runs)
    dataset = _current_dataset(run, generations)
    review = _review(run, dataset, evidence_store, repository)
    summary, samples, exceptions = _projection(review, dataset, evidence_store, source, repository)
    return ReviewProjection(summary=summary, sample=samples, exceptions=exceptions)


@router.post("/{run_id}/batch-attestation", response_model=BatchAttestationResult)
def attest_review(
    run_id: str,
    request: BatchAttestationRequest,
    admin: UserProfile = Depends(get_admin_user),
    runs: EvaluationRunStore = Depends(evaluation_run_store),
    generations: EvaluationGenerationReviewStore = Depends(evaluation_generation_store),
    evidence_store: SupportEvidenceStore = Depends(support_evidence_store),
    repository: BatchReviewRepository = Depends(batch_review_repository),
) -> BatchAttestationResult:
    run = _run(run_id, runs)
    dataset = _current_dataset(run, generations)
    review = _review(run, dataset, evidence_store, repository)
    try:
        attestation = repository.attest(
            review,
            request,
            actor=admin.user_id,
            decided_at=datetime.now(timezone.utc),
        )
    except BatchReviewStaleError as exc:
        raise HTTPException(status_code=409, detail={"code": "stale_batch_review", "message": "The batch review snapshot is stale."}) from exc
    except BatchReviewConflictError as exc:
        raise HTTPException(status_code=409, detail={"code": "batch_attestation_conflict", "message": "The batch attestation conflicts with the current review state."}) from exc
    except BatchReviewError as exc:
        raise HTTPException(status_code=409, detail={"code": "invalid_batch_attestation", "message": "The batch attestation could not be applied."}) from exc
    return BatchAttestationResult(
        status=attestation.decision,
        decision=attestation.decision,
        exception_decisions=request.exception_decisions,
        decision_metadata=_decision_metadata(attestation),
    )
