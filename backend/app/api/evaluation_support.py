"""Administrator-only run-bound support exception review."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.contracts import EvaluationRun, RunBoundSupportEvidence, UserProfile, VerifiedChunkChoice
from app.evaluation.support import (
    FirestoreRunPinnedChunkSource,
    SupportEvidenceStore,
    SupportPreparationError,
    review_support,
)
from app.identity import get_admin_user

router = APIRouter(prefix="/api/admin/evaluations", tags=["evaluation support"])


class SupportReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str = Field(pattern=r"^(approve|reject)$")
    selected_chunk_id: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_decision(self) -> "SupportReviewRequest":
        if self.decision == "approve" and not self.selected_chunk_id:
            raise ValueError("approval requires a selected chunk")
        if self.decision == "reject" and self.selected_chunk_id is not None:
            raise ValueError("rejection cannot select a chunk")
        return self


class SupportStatusResponse(BaseModel):
    evidence: RunBoundSupportEvidence
    candidate_chunks: list[VerifiedChunkChoice] = Field(default_factory=list, max_length=5)


class SupportRunStore(Protocol):
    def get(self, run_id: str) -> EvaluationRun | None: ...


def evaluation_run_store() -> SupportRunStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "evaluation_run_store_unavailable", "message": "Evaluation run storage is not configured."})


def support_evidence_store() -> SupportEvidenceStore:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "support_evidence_store_unavailable", "message": "Evaluation support storage is not configured."})


def pinned_chunk_source() -> FirestoreRunPinnedChunkSource:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "support_chunk_source_unavailable", "message": "Pinned corpus support lookup is not configured."})


def _run_or_404(run_id: str, runs: SupportRunStore) -> EvaluationRun:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "evaluation_run_not_found", "message": "The evaluation run was not found."})
    return run


def _snapshot_digest(run: EvaluationRun) -> str:
    if run.candidate_snapshot is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "stale_evaluation_run", "message": "The evaluation run has no immutable candidate snapshot."})
    return sha256(run.candidate_snapshot.model_dump_json().encode("utf-8")).hexdigest()


def _status(evidence: RunBoundSupportEvidence, source: FirestoreRunPinnedChunkSource) -> SupportStatusResponse:
    candidates: list[VerifiedChunkChoice] = []
    for chunk_id in evidence.candidate_chunk_ids:
        record = source.chunk(corpus_version_id=evidence.corpus_version_id, chunk_id=chunk_id)
        if record is not None and record.valid and record.chunk is not None:
            candidates.append(VerifiedChunkChoice(
                chunk_id=record.chunk.chunk_id,
                corpus_version_id=record.chunk.corpus_version_id,
                content_hash=record.chunk.content_hash,
                page=record.chunk.page,
                section=record.chunk.section,
                source_provenance_sha256=record.source_provenance_sha256,
            ))
    return SupportStatusResponse(evidence=evidence, candidate_chunks=candidates)


def _get_evidence(
    run_id: str,
    item_id: str,
    runs: SupportRunStore,
    store: SupportEvidenceStore,
) -> tuple[EvaluationRun, RunBoundSupportEvidence]:
    run = _run_or_404(run_id, runs)
    evidence = store.get_evidence(run_id=run_id, item_id=item_id)
    if evidence is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "support_status_not_found", "message": "Support evidence was not prepared for this run item."})
    if evidence.candidate_snapshot_sha256 != _snapshot_digest(run):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "stale_evaluation_run", "message": "The evaluation run snapshot is stale."})
    return run, evidence


@router.get("/{run_id}/items/{item_id}/support", response_model=SupportStatusResponse)
def get_support_status(
    run_id: str,
    item_id: str,
    _: UserProfile = Depends(get_admin_user),
    runs: SupportRunStore = Depends(evaluation_run_store),
    store: SupportEvidenceStore = Depends(support_evidence_store),
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
) -> SupportStatusResponse:
    _, evidence = _get_evidence(run_id, item_id, runs, store)
    return _status(evidence, source)


@router.put("/{run_id}/items/{item_id}/support", response_model=SupportStatusResponse)
def review_support_status(
    run_id: str,
    item_id: str,
    request: SupportReviewRequest,
    admin: UserProfile = Depends(get_admin_user),
    runs: SupportRunStore = Depends(evaluation_run_store),
    store: SupportEvidenceStore = Depends(support_evidence_store),
    source: FirestoreRunPinnedChunkSource = Depends(pinned_chunk_source),
) -> SupportStatusResponse:
    _, evidence = _get_evidence(run_id, item_id, runs, store)
    candidate = None
    if request.selected_chunk_id is not None:
        candidate = source.chunk(corpus_version_id=evidence.corpus_version_id, chunk_id=request.selected_chunk_id)
        if candidate is None or not candidate.valid:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "candidate_chunk_invalid", "message": "The selected chunk is not valid in the pinned corpus."})
    try:
        updated = review_support(
            evidence,
            candidate=candidate,
            reviewer_actor=admin.user_id,
            reviewed_at=datetime.now(timezone.utc),
            store=store,
        )
    except SupportPreparationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "support_review_conflict", "message": str(exc)}) from exc
    return _status(updated, source)
