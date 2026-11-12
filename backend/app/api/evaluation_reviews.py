"""Admin-only browser projection and attestation for batch evidence reviews."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.evaluation_assessments import assessment_repository, output_store
from app.contracts import (
    BatchAttestationRequest,
    BatchAttestationResult,
    ChildDocumentChunk,
    DocumentChunk,
    DurableSemanticAssessmentReference,
    EvaluationDataset,
    EvaluationRun,
    ExceptionRecord,
    IndividualEvidenceDecisionRequest,
    IndividualEvidenceDecisionResult,
    IndividualEvidenceReview,
    RepresentativeSampleEntry,
    RunBoundSupportEvidence,
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
from app.evaluation.job import EvaluationRunStore, EvaluationStageStore
from app.evaluation.reviews import AssessmentRepository, OutputStore
from app.evaluation.support import (
    FirestoreRunPinnedChunkSource,
    SupportEvidenceStore,
    StoredSupportChunk,
    SupportPreparationError,
    candidate_snapshot_sha256,
    verified_chunk_choice,
)
from app.evaluation.dataset import CandidateGoldBinding, remap_gold_annotations
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


def evaluation_stage_store() -> EvaluationStageStore:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "evaluation_stage_store_unavailable", "message": "Evaluation stage storage is not configured."},
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
    source: FirestoreRunPinnedChunkSource,
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
    evidence = _revalidate_batch_evidence(run, dataset, evidence, source)
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


def _revalidate_batch_evidence(
    run: EvaluationRun,
    dataset: EvaluationDataset,
    evidence: list[RunBoundSupportEvidence],
    source: FirestoreRunPinnedChunkSource,
) -> list[RunBoundSupportEvidence]:
    """Keep stale or mismatched parent/child mappings in the exception queue."""
    try:
        provenance = source.source_provenance_sha256(corpus_version_id=run.corpus_version_id)
        records = source.chunks(corpus_version_id=run.corpus_version_id)
    except (SupportPreparationError, ValueError, TypeError):
        return [_stale_support(item) for item in evidence]
    if any(item.source_provenance_sha256 != provenance for item in evidence):
        return [_stale_support(item) for item in evidence]

    valid_records = tuple(record for record in records if record.valid and record.chunk is not None)
    child_records = tuple(
        record for record in valid_records if isinstance(record.chunk, ChildDocumentChunk)
    )
    if child_records:
        parent_by_id = {
            record.parent.parent_id: record.parent
            for record in child_records if record.parent is not None
        }
        candidate_chunks = tuple(parent_by_id.values()) + tuple(record.chunk for record in child_records)
    else:
        candidate_chunks = tuple(
            record.chunk for record in valid_records if isinstance(record.chunk, DocumentChunk)
        )
    try:
        remapped = remap_gold_annotations(
            dataset,
            candidate_chunks,
            CandidateGoldBinding(
                run.corpus_version_id,
                candidate_snapshot_sha256(run),
                provenance,
            ),
        )
    except (ValueError, TypeError):
        return [_stale_support(item) if item.match_status in {"eligible", "review_approved"} else item for item in evidence]

    records_by_id = {record.chunk_id: record for record in valid_records}
    result: list[RunBoundSupportEvidence] = []
    for item in evidence:
        if item.expected_refusal or item.match_status not in {"eligible", "review_approved"}:
            result.append(item)
            continue
        mapping = remapped.remap_for(item.item_id)
        ids = tuple(item.candidate_chunk_ids)
        verified = tuple(
            verified_chunk_choice(records_by_id[chunk_id], provenance)
            for chunk_id in ids if chunk_id in records_by_id
        )
        selected = tuple(item.selected_chunks or ((item.selected_chunk,) if item.selected_chunk else ()))
        matches = (
            mapping.remap_status == "matched"
            and mapping.relevant_chunk_ids == ids
            and len(verified) == len(ids)
            and tuple(item.verified_chunks) == verified
            and (not selected or selected == verified)
            and item.parent_id == mapping.parent_id
            and item.parent_content_hash == mapping.parent_content_hash
            and item.parent_span_sha256 == mapping.parent_span_sha256
        )
        result.append(item if matches else _stale_support(item))
    return result


def _stale_support(evidence: RunBoundSupportEvidence) -> RunBoundSupportEvidence:
    payload = evidence.model_dump() | {
        "match_status": "ambiguous",
        "selected_chunk": None,
        "selected_chunks": [],
        "reason": "stale_source_evidence",
        "reviewer_actor": None,
        "reviewed_at": None,
    }
    return RunBoundSupportEvidence.model_validate(payload)


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
    selected = evidence.selected_chunk or (evidence.selected_chunks[0] if evidence.selected_chunks else None)
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


def _semantic_assessment_references(
    run: EvaluationRun,
    review: BatchReview,
    request: BatchAttestationRequest,
    *,
    assessments: AssessmentRepository,
    outputs: OutputStore,
    stages: EvaluationStageStore,
) -> tuple[DurableSemanticAssessmentReference, ...]:
    if request.decision == "rejected":
        if request.semantic_assessments:
            raise HTTPException(status_code=409, detail={
                "code": "invalid_batch_attestation",
                "message": "A rejected batch cannot attach semantic assessments.",
            })
        return ()

    refusal_ids = {
        record.item_id for record in review.exceptions
        if record.expected_refusal and record.status != "stale"
    }
    required_ids = set(review.summary.sampled_item_ids) | refusal_ids
    selected = {selection.item_id: selection for selection in request.semantic_assessments}
    if set(selected) != required_ids:
        raise HTTPException(status_code=409, detail={
            "code": "incomplete_semantic_attestation",
            "message": "Every sampled item and confirmed refusal requires an exact-output assessment.",
        })

    result: list[DurableSemanticAssessmentReference] = []
    for item_id in sorted(required_ids):
        selection = selected[item_id]
        record = assessments.get(selection.assessment_id)
        stage = stages.get(run.run_id, item_id)
        if record is None or stage is None:
            raise HTTPException(status_code=409, detail={
                "code": "stale_semantic_assessment",
                "message": "The exact run output assessment is unavailable.",
            })
        audit = outputs.get_audit(record.attempt_id)
        if audit is None:
            raise HTTPException(status_code=409, detail={
                "code": "stale_semantic_assessment",
                "message": "The exact run output assessment is unavailable.",
            })
        identity = record.candidate_identity
        snapshot = run.candidate_snapshot
        if (
            not record.complete
            or record.assessment_id != selection.assessment_id
            or record.generation_id != run.dataset_version_id
            or audit.evaluation_run_id != run.run_id
            or audit.evaluation_item_id != item_id
            or audit.attempt_id != record.attempt_id
            or audit.candidate_id != record.candidate_id
            or audit.scorer_version != record.scorer_version
            or audit.output_digest != record.output_digest
            or audit.candidate_identity != identity
            or stage.output_handle != record.attempt_id
            or stage.output_digest != record.output_digest
            or snapshot is None
            or identity.corpus_version_id != run.corpus_version_id
            or identity.configuration_version_id != run.configuration_version_id
            or identity.scorer_version != snapshot.scorer_version
            or identity.installed_lock_sha256 != snapshot.dependency_lock_sha256
            or identity.artifact_id != snapshot.integration_artifact
        ):
            raise HTTPException(status_code=409, detail={
                "code": "stale_semantic_assessment",
                "message": "The semantic assessment does not match the current immutable run output.",
            })
        if item_id in refusal_ids and record.refusal_judgment is not True:
            raise HTTPException(status_code=409, detail={
                "code": "invalid_refusal_assessment",
                "message": "A refusal confirmation requires a positive exact-output refusal assessment.",
            })
        result.append(DurableSemanticAssessmentReference(
            item_id=item_id,
            assessment_id=record.assessment_id,
            output_digest=record.output_digest,
            scorer_version=record.scorer_version,
            reviewer_actor=record.reviewer_actor,
            reviewed_at=record.reviewed_at,
        ))
    return tuple(result)


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
        selected = evidence.selected_chunk or (evidence.selected_chunks[0] if evidence.selected_chunks else None)
        page = selected.page if selected else None
        section = selected.section if selected else None
        if chunk and chunk.chunk:
            page = getattr(chunk.chunk, "page", getattr(chunk.chunk, "page_start", page))
            section = getattr(chunk.chunk, "section", section)
        samples.append(
            RepresentativeSampleEntry(
                item_id=item.item_id,
                question=item.question,
                rubric=annotation.expected_answer_or_rubric,
                support_excerpt=_safe_excerpt(annotation.supporting_passage) or "Verified support excerpt unavailable.",
                page=page,
                section=section,
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
                expected_refusal=record.expected_refusal,
                decision=decision,
            )
        )
    return summary, samples, exceptions


def _individual_review(
    run_id: str,
    item_id: str,
    runs: EvaluationRunStore,
    generations: EvaluationGenerationReviewStore,
    evidence_store: SupportEvidenceStore,
    source: FirestoreRunPinnedChunkSource,
) -> tuple[RunBoundSupportEvidence, IndividualEvidenceReview]:
    run = _run(run_id, runs)
    if item_id not in run.selected_item_ids:
        raise HTTPException(status_code=404, detail={"code": "evaluation_item_not_found", "message": "The item was not found in this run."})
    dataset = _current_dataset(run, generations)
    item = _item_map(dataset).get(item_id)
    annotation = _annotation_map(dataset).get(item_id)
    evidence = evidence_store.get_evidence(run_id=run_id, item_id=item_id)
    if item is None or annotation is None or evidence is None:
        raise HTTPException(status_code=409, detail={"code": "incomplete_individual_review", "message": "The item review is incomplete."})
    try:
        current_provenance = source.source_provenance_sha256(corpus_version_id=run.corpus_version_id)
        records = (
            source.chunks(corpus_version_id=run.corpus_version_id)
            if evidence.candidate_chunk_ids or evidence.match_status in {"eligible", "ambiguous"}
            else ()
        )
    except (SupportPreparationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review could not be verified against the pinned corpus."}) from exc
    if (
        evidence.run_id != run.run_id
        or evidence.item_id != item_id
        or evidence.corpus_version_id != run.corpus_version_id
        or evidence.candidate_snapshot_sha256 != candidate_snapshot_sha256(run)
        or evidence.source_provenance_sha256 != current_provenance
    ):
        raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review is stale."})
    record_by_id = {record.chunk_id: record for record in records}
    if any(record_by_id.get(chunk_id) is None for chunk_id in evidence.candidate_chunk_ids):
        raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review no longer matches verified corpus evidence."})
    if evidence.match_status in {"eligible", "ambiguous", "review_approved", "review_rejected"} and any(
        not record_by_id[chunk_id].valid for chunk_id in evidence.candidate_chunk_ids
    ):
        raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review no longer matches verified corpus evidence."})
    if evidence.match_status in {"eligible", "review_approved"} and not evidence.expected_refusal:
        if any(
            choice != verified_chunk_choice(record_by_id[choice.chunk_id], evidence.source_provenance_sha256)
            for choice in evidence.verified_chunks
        ):
            raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review no longer matches verified corpus evidence."})
    if evidence.match_status in {"eligible", "ambiguous", "review_approved", "review_rejected"} and not evidence.expected_refusal:
        try:
            binding = CandidateGoldBinding(
                run.corpus_version_id,
                candidate_snapshot_sha256(run),
                evidence.source_provenance_sha256,
            )
            current = remap_gold_annotations(
                EvaluationDataset(
                    dataset_version_id=dataset.dataset_version_id,
                    items=[item],
                    gold_annotations=[annotation],
                ),
                tuple({
                    record.parent.parent_id: record.parent
                    for record in records if record.parent is not None
                }.values()) + tuple(
                    record.chunk for record in records if record.valid and record.chunk is not None
                ),
                binding,
            ).remap_for(item_id)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review no longer matches verified corpus evidence."}) from exc
        expected_status = "matched" if evidence.match_status in {"eligible", "review_approved"} else "ambiguous"
        current_ids = current.candidate_chunk_ids or current.relevant_chunk_ids
        if not current_ids and expected_status == "ambiguous":
            normalized_support = " ".join((annotation.supporting_passage or "").casefold().split())
            current_ids = tuple(
                record.chunk_id for record in records
                if record.valid and record.chunk is not None
                and normalized_support in " ".join(record.chunk.text.casefold().split())
            )
        if (
            (evidence.match_status != "review_rejected" and current.remap_status != expected_status)
            or tuple(evidence.candidate_chunk_ids) != current_ids
            or (expected_status == "matched" and (
                evidence.parent_id != current.parent_id
                or evidence.parent_content_hash != current.parent_content_hash
                or evidence.parent_span_sha256 != current.parent_span_sha256
            ))
        ):
            raise HTTPException(status_code=409, detail={"code": "stale_individual_review", "message": "The item review no longer matches verified corpus evidence."})
    choices = [record_by_id[chunk_id] for chunk_id in evidence.candidate_chunk_ids if record_by_id[chunk_id].valid]
    first = next((record for record in choices if record.chunk is not None), None)
    chunk = first.chunk if first is not None else evidence.selected_chunk
    if first is not None and first.parent is not None and hasattr(chunk, "page_start"):
        page = chunk.page_start
        section = " / ".join(chunk.section_path)
    elif chunk is not None:
        page = chunk.page if hasattr(chunk, "page") else None
        section = chunk.section if hasattr(chunk, "section") else None
    else:
        page = section = None
    result = IndividualEvidenceReview(
        item_id=item.item_id,
        question=item.question,
        rubric=annotation.expected_answer_or_rubric,
        support_excerpt=_safe_excerpt(annotation.supporting_passage),
        page=page,
        section=section,
        status=evidence.match_status,
        expected_refusal=evidence.expected_refusal,
        reviewer_actor=evidence.reviewer_actor,
        reviewed_at=evidence.reviewed_at,
    )
    return evidence, result


@router.get("/{run_id}/items/{item_id}/review", response_model=IndividualEvidenceReview)
def get_individual_review(
    run_id: str,
    item_id: str,
    _: UserProfile = Depends(get_admin_user),
    runs: EvaluationRunStore = Depends(evaluation_run_store),
    generations: EvaluationGenerationReviewStore = Depends(evaluation_generation_store),
    evidence_store: SupportEvidenceStore = Depends(support_evidence_store),
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
) -> IndividualEvidenceReview:
    return _individual_review(run_id, item_id, runs, generations, evidence_store, source)[1]


@router.put("/{run_id}/items/{item_id}/review", response_model=IndividualEvidenceDecisionResult)
def decide_individual_review(
    run_id: str,
    item_id: str,
    request: IndividualEvidenceDecisionRequest,
    admin: UserProfile = Depends(get_admin_user),
    runs: EvaluationRunStore = Depends(evaluation_run_store),
    generations: EvaluationGenerationReviewStore = Depends(evaluation_generation_store),
    evidence_store: SupportEvidenceStore = Depends(support_evidence_store),
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
) -> IndividualEvidenceDecisionResult:
    evidence, _ = _individual_review(run_id, item_id, runs, generations, evidence_store, source)
    try:
        updated = evidence_store.decide_evidence(
            evidence,
            decision=request.decision,
            reviewer_actor=admin.user_id,
            reviewed_at=datetime.now(timezone.utc),
        )
    except SupportPreparationError as exc:
        raise HTTPException(status_code=409, detail={"code": "individual_review_conflict", "message": "The item decision could not be applied to the current review state."}) from exc
    return IndividualEvidenceDecisionResult(
        item_id=item_id,
        status=updated.match_status,
        decision=request.decision,
        reviewer_actor=updated.reviewer_actor or admin.user_id,
        reviewed_at=updated.reviewed_at or datetime.now(timezone.utc),
    )


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
    review = _review(run, dataset, evidence_store, repository, source)
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
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
    assessments: AssessmentRepository = Depends(assessment_repository),
    outputs: OutputStore = Depends(output_store),
    stages: EvaluationStageStore = Depends(evaluation_stage_store),
) -> BatchAttestationResult:
    run = _run(run_id, runs)
    dataset = _current_dataset(run, generations)
    review = _review(run, dataset, evidence_store, repository, source)
    try:
        semantic_assessments = _semantic_assessment_references(
            run, review, request, assessments=assessments, outputs=outputs, stages=stages,
        )
        attestation = repository.attest(
            review,
            request,
            actor=admin.user_id,
            decided_at=datetime.now(timezone.utc),
            semantic_assessments=semantic_assessments,
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
        semantic_assessments=attestation.semantic_assessments,
        refusal_confirmation_item_ids=attestation.refusal_confirmation_item_ids,
        decision_metadata=_decision_metadata(attestation),
    )
